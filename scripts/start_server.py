"""Start a vLLM server from a config, in its own process group, and verify that
the thing answering on the port is the server we just started.

Usage:
    python scripts/start_server.py configs/example.yaml results_v2/<run_dir>/attempt_1
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from common import RunError, load_config
from stop_server import port_in_use, proc_start_time, stop_process_group


def build_server_cmd(cfg: dict) -> list[str]:
    s = cfg["server"]
    cmd = [
        "vllm",
        "serve",
        cfg["model"],
        "--max-model-len",
        str(s["max_model_len"]),
        "--gpu-memory-utilization",
        str(s["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(s["max_num_seqs"]),
        "--port",
        str(s["port"]),
    ]
    if cfg.get("revision"):
        cmd += ["--revision", cfg["revision"]]
    if not s["prefix_caching"]:
        cmd.append("--no-enable-prefix-caching")
    return cmd


def log_tail(log_path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def wait_until_ready(
    proc: subprocess.Popen, port: int, log_path: Path, timeout_s: float = 900
) -> None:
    """Poll /health, but fail at once if our own process has exited."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout_s
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
        time.sleep(1)
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
    return card


def parse_server_log(log_path: Path) -> dict:
    """Pull the settings the server itself printed at start-up."""
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    info: dict = {"non_default_args": "", "kv_cache_size_tokens": None}
    m = re.search(r"non-default args: (.*)", text)
    if m:
        info["non_default_args"] = m.group(1).strip()
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        info["kv_cache_size_tokens"] = int(m.group(1).replace(",", ""))
    return info


def start_server(cfg: dict, run_dir: Path, timeout_s: float = 900) -> subprocess.Popen:
    """Start the server and return its process handle once it is verified."""
    run_dir = Path(run_dir)
    port = cfg["server"]["port"]
    if port_in_use(port):
        raise RunError(
            f"port {port} is already in use; refusing to start. A stale server "
            "would answer the health check and be measured instead of ours."
        )
    cmd = build_server_cmd(cfg)
    log_path = run_dir / "server.log"
    with open(log_path, "w") as log:
        # start_new_session puts the server and all its children in a process
        # group of their own, so stopping the group stops only our server.
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    info = {
        "pid": proc.pid,
        "pgid": proc.pid,
        "proc_start_time": proc_start_time(proc.pid),
        "port": port,
        "cmd": cmd,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "server.json").write_text(json.dumps(info, indent=2))
    try:
        wait_until_ready(proc, port, log_path, timeout_s)
        card = verify_identity(cfg, port)
    except Exception:
        stop_process_group(proc.pid, proc)
        raise
    info.update(model_card=card, **parse_server_log(log_path))
    (run_dir / "server_info.json").write_text(json.dumps(info, indent=2))
    return proc


def ensure_alive(proc: subprocess.Popen, when: str) -> None:
    code = proc.poll()
    if code is not None:
        raise RunError(f"server exited with code {code} {when}")


def main() -> None:
    cfg = load_config(sys.argv[1])
    run_dir = Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    proc = start_server(cfg, run_dir)
    print(f"Server ready, PID {proc.pid}. Stop it with:")
    print(f"  python scripts/stop_server.py {run_dir}")


if __name__ == "__main__":
    main()
