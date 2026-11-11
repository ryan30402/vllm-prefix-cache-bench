"""Shared helpers: config loading, config hashing, and run-directory bookkeeping.

A run lives in  <results_root>/<name>__<hash>/attempt_<n>/  where <hash> covers
the full resolved config plus the environment identity. Two runs that differ in
any server or workload setting can therefore never share a directory, and a
retry never overwrites an earlier attempt.
"""

import hashlib
import json
from pathlib import Path

import yaml

# Every one of these must exist and be non-empty for an attempt to count as done.
REQUIRED_FILES = [
    "config.yaml",
    "env.json",
    "server_info.json",
    "warmup_result.json",
    "metrics_before.json",
    "metrics_after.json",
    "bench_result.json",
    "checks.json",
    "status.json",
]


class RunError(RuntimeError):
    """The run produced something that must not be reported as a valid result."""


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict, identity: dict) -> str:
    """sha256 over the resolved config and the environment identity."""
    payload = json.dumps(
        {"config": cfg, "identity": identity}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def run_dir_name(cfg: dict, digest: str) -> str:
    return f"{cfg['name']}__{digest[:10]}"


def attempt_dirs(run_dir: Path) -> list[Path]:
    """Existing attempt directories, oldest first."""
    found = []
    for d in run_dir.glob("attempt_*"):
        suffix = d.name.removeprefix("attempt_")
        if d.is_dir() and suffix.isdigit():
            found.append((int(suffix), d))
    return [d for _, d in sorted(found)]


def next_attempt_dir(run_dir: Path) -> Path:
    existing = attempt_dirs(run_dir)
    n = int(existing[-1].name.removeprefix("attempt_")) + 1 if existing else 1
    return run_dir / f"attempt_{n}"


def check_complete(attempt_dir: Path, digest: str | None = None) -> list[str]:
    """Return the reasons this attempt is NOT a valid finished run (empty = valid)."""
    problems = []
    for name in REQUIRED_FILES:
        path = attempt_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing or empty {name}")
    if problems:
        return problems

    status = json.loads((attempt_dir / "status.json").read_text())
    if status.get("ok") is not True:
        problems.append("status.json does not say ok")
    if digest is not None and status.get("config_hash") != digest:
        problems.append("config hash differs from the requested run")

    cfg = yaml.safe_load((attempt_dir / "config.yaml").read_text())
    expected = {
        "bench_result.json": cfg["bench"]["num_prompts"],
        "warmup_result.json": cfg["bench"]["warmup_prompts"],
    }
    for name, want in expected.items():
        result = json.loads((attempt_dir / name).read_text())
        if result.get("completed") != want:
            problems.append(
                f"{name}: completed={result.get('completed')}, expected {want}"
            )
        if result.get("failed", 0) != 0:
            problems.append(f"{name}: {result.get('failed')} failed requests")
    return problems


def latest_complete_attempt(run_dir: Path, digest: str | None = None) -> Path | None:
    """Newest attempt that passes every completeness check, or None."""
    for attempt in reversed(attempt_dirs(run_dir)):
        if not check_complete(attempt, digest):
            return attempt
    return None
