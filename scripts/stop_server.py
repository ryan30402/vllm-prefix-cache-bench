"""Stop the vLLM server that this harness started, and nothing else.

Usage:
    python scripts/stop_server.py results_v2/<run_dir>/attempt_1

The server is started in its own process group (see start_server.py). Stopping
it signals that group only. The v1 version fell back to `pkill -f "vllm serve"`,
which would also kill unrelated servers on a shared machine.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import RunError


def proc_start_time(pid: int) -> str:
    """Kernel start time of a process, used to detect a reused PID. '' if unknown."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return ""
    return stat.rsplit(")", 1)[1].split()[19]


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
        except OSError:
            continue  # the process exited while we were looking
        state, pgrp = fields[0], int(fields[2])
        if pgrp == pgid and state != "Z":
            members.append(int(entry.name))
    return members


def port_in_use(port: int) -> bool:
    import socket

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
    raise RunError(f"process group {pgid} is still alive after SIGKILL")


def stop_server(
    run_dir: Path, proc: subprocess.Popen | None = None, grace_s: float = 60
) -> str:
    """Stop the server recorded in <run_dir>/server.json."""
    info_path = Path(run_dir) / "server.json"
    if not info_path.is_file():
        return "no server.json, nothing to stop"
    info = json.loads(info_path.read_text())
    pid, pgid, port = info["pid"], info["pgid"], info["port"]

    recorded, current = info.get("proc_start_time", ""), proc_start_time(pid)
    if recorded and current and recorded != current:
        # The PID now belongs to some other process. Do not signal it.
        return f"PID {pid} was reused by another process, nothing to stop"

    outcome = stop_process_group(pgid, proc, grace_s)
    deadline = time.time() + 15
    while port_in_use(port) and time.time() < deadline:
        time.sleep(0.5)
    if port_in_use(port):
        raise RunError(
            f"port {port} still answers after stopping our server; "
            "another process is listening on it"
        )
    return outcome


def main() -> None:
    print(stop_server(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
