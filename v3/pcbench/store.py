"""Attempt folders, atomic writes and the attempt state machine.

Layout of one experiment:

    results_v3/<experiment>_b<batch>/
        manifest.json  environment.json  sessions.json
        workloads/<key>/{warmup.jsonl, main.jsonl, workload.json}
        runs/<run_id>/attempt_<n>/...
        analysis/{summary.csv, pairs.csv, audit.json}

An attempt is RUNNING from its first file until one atomic rename writes
status.json with state VALIDATED (or FAILED). A crash at any point leaves the
state at RUNNING, which is never treated as a result.
"""

import json
import os
import signal
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

RUNNING = "RUNNING"
FAILED = "FAILED"
INTERRUPTED = "INTERRUPTED"
VALIDATED = "VALIDATED"
STATES = (RUNNING, FAILED, INTERRUPTED, VALIDATED)


class RunError(RuntimeError):
    """The run produced something that must not be reported as a valid result."""


class SweepAbort(RunError):
    """Continuing would contaminate every later run (e.g. a server we cannot stop)."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temporary file in the same folder, flush it to disk, then rename.

    A reader sees either the old file or the complete new one, never a part.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def attempt_number(attempt_dir: Path) -> int:
    return int(attempt_dir.name.removeprefix("attempt_"))


def attempt_dirs(run_dir: Path) -> list[Path]:
    """Existing attempt folders of one run, oldest first."""
    found = []
    for d in Path(run_dir).glob("attempt_*"):
        suffix = d.name.removeprefix("attempt_")
        if d.is_dir() and suffix.isdigit():
            found.append((int(suffix), d))
    return [d for _, d in sorted(found)]


def read_state(attempt_dir: Path) -> dict:
    """status.json of an attempt. A missing or unreadable file reads as RUNNING,
    because only a complete status file can say that an attempt finished."""
    try:
        status = read_json(Path(attempt_dir) / "status.json")
    except (OSError, ValueError):
        return {"state": RUNNING, "unreadable": True}
    if not isinstance(status, dict) or status.get("state") not in STATES:
        return {"state": RUNNING, "unreadable": True}
    return status


def write_state(attempt_dir: Path, status: dict) -> None:
    atomic_write_json(Path(attempt_dir) / "status.json", status)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ExperimentLock:
    """One runner per experiment folder. A lock left by a dead runner is taken over."""

    def __init__(self, experiment_dir: Path):
        self.path = Path(experiment_dir) / "runner.lock"

    def __enter__(self):
        if self.path.is_file():
            try:
                pid = int(self.path.read_text().strip())
            except ValueError:
                pid = -1
            if pid > 0 and pid != os.getpid() and pid_alive(pid):
                raise RunError(
                    f"another runner (PID {pid}) is working on this experiment"
                )
        atomic_write_text(self.path, str(os.getpid()))
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def gzip_file(path: Path) -> None:
    """Replace a file by its gzip-compressed copy (name + .gz)."""
    import gzip

    path = Path(path)
    if path.is_file():
        atomic_write_bytes(
            path.with_name(path.name + ".gz"), gzip.compress(path.read_bytes(), mtime=0)
        )
        path.unlink()


@contextmanager
def terminate_as_interrupt():
    """Treat SIGTERM and SIGHUP like Ctrl+C while a run is in progress.

    Closing a terminal or `kill` would otherwise end the runner without running
    its cleanup, and the server it started would stay alive.
    """

    def handler(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, handler)
    except ValueError:
        pass  # not the main thread: keep the default behaviour
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)
