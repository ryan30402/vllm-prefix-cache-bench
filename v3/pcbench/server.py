"""Start and stop the vLLM server of one attempt.

`running_server` is a context manager. From the moment the server process
exists, every later step (writing its metadata, waiting for readiness, the
benchmark itself, a timeout) runs inside one try/finally that stops the
process group. Only the group this harness started is signalled; there is no
`pkill`. A server that cannot be stopped raises SweepAbort, because the next
run would otherwise measure it.
"""

import gzip
import os
import re
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import requests

from .store import (
    RunError,
    SweepAbort,
    atomic_write_bytes,
    atomic_write_json,
    now,
    read_json,
)


def build_server_cmd(cfg: dict) -> list[str]:
    s = cfg["server"]
    return [
        "vllm",
        "serve",
        cfg["model"],
        "--revision",
        cfg["model_revision"],
        "--port",
        str(s["port"]),
        "--max-model-len",
        str(s["max_model_len"]),
        "--gpu-memory-utilization",
        str(s["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(s["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(s["max_num_batched_tokens"]),
        # "vllm" = vLLM's neutral sampling defaults, not the model's own
        # generation_config.json (which sets temperature, top_p, top_k, ...).
        "--generation-config",
        str(s["generation_config"]),
        "--seed",
        str(s["seed"]),
        "--enable-prefix-caching"
        if s["prefix_caching"]
        else "--no-enable-prefix-caching",
    ]


def proc_start_time(pid: int) -> str:
    """Kernel start time of a process, used to detect a reused PID. '' if unknown."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return ""


def group_members(pgid: int) -> list[int]:
    """PIDs of live (non-zombie) processes in a process group."""
    proc = Path("/proc")
    if not proc.is_dir():  # not Linux: fall back to asking the kernel
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return []
        except PermissionError:
            pass
        return [pgid]
    members = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue  # the process exited while we were looking
        if int(fields[2]) == pgid and fields[0] != "Z":
            members.append(int(entry.name))
    return members


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def stop_process_group(
    pgid: int,
    proc: subprocess.Popen | None = None,
    grace_s: float = 60,
    kill_wait_s: float = 15,
) -> str:
    """SIGINT the group, escalate to SIGKILL if needed. Returns how it ended."""

    def wait_gone(seconds: float) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if proc is not None:
                proc.poll()  # reap our child so it does not linger as a zombie
            if not group_members(pgid):
                return True
            time.sleep(0.2)
        return not group_members(pgid)

    if proc is not None:
        proc.poll()
    if not group_members(pgid):
        return "already gone"
    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return "already gone"
    if wait_gone(grace_s):
        return "stopped with SIGINT"
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if wait_gone(kill_wait_s):
        return "stopped with SIGKILL"
    raise SweepAbort(f"process group {pgid} is still alive after SIGKILL")


def wait_port_free(port: int, seconds: float = 15) -> None:
    deadline = time.time() + seconds
    while port_in_use(port) and time.time() < deadline:
        time.sleep(0.5)
    if port_in_use(port):
        raise SweepAbort(
            f"port {port} still answers after stopping our server; "
            "another process is listening on it"
        )


def log_tail(log_path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def wait_until_ready(
    proc: subprocess.Popen, port: int, log_path: Path, timeout_s: float
) -> None:
    """Poll /health, but fail at once if our own process has exited."""
    url = f"http://localhost:{port}/health"
    started = time.time()
    deadline = started + timeout_s
    while time.time() < deadline:
        code = proc.poll()
        if code is not None:
            raise RunError(
                f"server exited with code {code} before becoming ready\n"
                + log_tail(log_path)
            )
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.2 if time.time() - started < 5 else 1)
    raise RunError(f"server on port {port} not ready after {timeout_s}s")


def verify_identity(cfg: dict, port: int) -> dict:
    """Check that the server on the port serves the model and length we asked for."""
    url = f"http://localhost:{port}/v1/models"
    cards = requests.get(url, timeout=10).json().get("data", [])
    card = next((c for c in cards if c.get("id") == cfg["model"]), None)
    if card is None:
        raise RunError(
            f"server on port {port} serves {[c.get('id') for c in cards]}, "
            f"expected {cfg['model']}"
        )
    want = cfg["server"]["max_model_len"]
    if "max_model_len" in card and card["max_model_len"] != want:
        raise RunError(
            f"server reports max_model_len={card['max_model_len']}, expected {want}"
        )
    return {"id": card.get("id"), "max_model_len": card.get("max_model_len")}


def parse_server_log(log_path: Path) -> dict:
    """Pull the settings the server itself printed at start-up."""
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    info: dict = {
        "non_default_args": "",
        "kv_cache_size_tokens": None,
        "chunked_prefill_max_num_batched_tokens": None,
    }
    m = re.search(r"non-default args: (.*)", text)
    if m:
        info["non_default_args"] = m.group(1).strip()
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        info["kv_cache_size_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)", text)
    if m:
        info["chunked_prefill_max_num_batched_tokens"] = int(m.group(1))
    return info


class ServerHandle:
    def __init__(self, proc: subprocess.Popen, info: dict):
        self.proc = proc
        self.info = info

    def ensure_alive(self, when: str) -> None:
        code = self.proc.poll()
        if code is not None:
            raise RunError(f"server exited with code {code} {when}")


@contextmanager
def running_server(cfg: dict, attempt_dir: Path, ready_timeout_s: float = 900):
    """Start the server, yield a handle, and always stop it again."""
    attempt_dir = Path(attempt_dir)
    port = cfg["server"]["port"]
    if port_in_use(port):
        raise RunError(
            f"port {port} is already in use; refusing to start. A stale server "
            "would answer the health check and be measured instead of ours."
        )
    cmd = build_server_cmd(cfg)
    log_path = attempt_dir / "server.log"
    proc = None
    try:
        with open(log_path, "wb") as log:
            # start_new_session puts the server and all its children in a
            # process group of their own, so stopping the group stops only them.
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        info = {
            "pid": proc.pid,
            "pgid": proc.pid,
            "proc_start_time": proc_start_time(proc.pid),
            "port": port,
            "cmd": cmd,
            "started_at": now(),
        }
        atomic_write_json(attempt_dir / "server.json", info)
        wait_until_ready(proc, port, log_path, ready_timeout_s)
        info["model_card"] = verify_identity(cfg, port)
        info.update(parse_server_log(log_path))
        info["ready_at"] = now()
        atomic_write_json(attempt_dir / "server.json", info)
        yield ServerHandle(proc, info)
    finally:
        if proc is not None:
            outcome = stop_process_group(proc.pid, proc)
            wait_port_free(port)
            print(f"  server {outcome}", flush=True)
        if log_path.is_file():
            atomic_write_bytes(
                attempt_dir / "server.log.gz",
                gzip.compress(log_path.read_bytes(), mtime=0),
            )
            log_path.unlink()


def stop_recorded_server(attempt_dir: Path) -> str:
    """Stop the server an interrupted attempt left behind, if it is still ours."""
    info_path = Path(attempt_dir) / "server.json"
    if not info_path.is_file():
        return "no server was recorded"
    try:
        info = read_json(info_path)
        pid, pgid, port = info["pid"], info["pgid"], info["port"]
    except (ValueError, KeyError, TypeError, OSError):
        return "server record unreadable"
    recorded, current = info.get("proc_start_time", ""), proc_start_time(pid)
    if not current and not group_members(pgid):
        return "already gone"
    if recorded and current and recorded != current:
        # The PID now belongs to some other process. Do not signal it.
        return f"PID {pid} was reused by another process, nothing stopped"
    outcome = stop_process_group(pgid)
    wait_port_free(port)
    return outcome
