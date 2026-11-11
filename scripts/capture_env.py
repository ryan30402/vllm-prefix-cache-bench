"""Record the hardware, software and source state of a benchmark run.

Usage:
    python scripts/capture_env.py results_v2/some_run/env.json [model_id]
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths that decide what a run measures. Result folders are deliberately not
# included: they change during a sweep and would make every run look dirty.
SOURCE_PATHS = ["scripts", "configs", "requirements.txt"]


def run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command and return its output as text. Empty string on failure."""
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL, cwd=cwd
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def pip_version(package: str) -> str:
    """Return the installed version of a pip package, or empty string."""
    out = run([sys.executable, "-m", "pip", "show", package])
    for line in out.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return ""


def pip_freeze() -> str:
    """Every installed package with its exact version."""
    return run([sys.executable, "-m", "pip", "freeze"])


def resolve_model_revision(model: str) -> str:
    """Commit hash of the locally cached Hugging Face snapshot of `model`, or ''."""
    hub = os.environ.get("HF_HUB_CACHE")
    if not hub:
        home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        hub = str(Path(home) / "hub")
    ref = Path(hub) / ("models--" + model.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.is_file() else ""


def source_state(repo_root: Path = REPO_ROOT) -> dict:
    """Commit, dirty flag and diff for the source paths only."""
    paths = [p for p in SOURCE_PATHS if (repo_root / p).exists()]
    porcelain = run(["git", "status", "--porcelain", "--", *paths], cwd=repo_root)
    return {
        "git_commit": run(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "source_dirty": porcelain != "",
        "source_status": porcelain,
        "source_diff": run(["git", "diff", "HEAD", "--", *paths], cwd=repo_root),
    }


def capture(model: str | None = None, repo_root: Path = REPO_ROOT) -> dict:
    """Collect everything we want to know about this machine into one dict."""
    gpu_query = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    nvcc = run(["nvcc", "--version"])
    freeze = pip_freeze()
    env = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu_query,
        "gpu_count": len(run(["nvidia-smi", "-L"]).splitlines()),
        "cuda_version": nvcc.splitlines()[-1] if nvcc else "",
        "vllm": pip_version("vllm"),
        "torch": pip_version("torch"),
        "pip_freeze": freeze,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "model": model or "",
        "model_revision": resolve_model_revision(model) if model else "",
    }
    env.update(source_state(repo_root))
    return env


def env_identity(env: dict) -> dict:
    """The parts of the environment that make two runs not comparable."""
    keys = ["gpu", "vllm", "torch", "python", "model_revision", "pip_freeze_sha256"]
    return {k: env.get(k, "") for k in keys}


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    env = capture(sys.argv[2] if len(sys.argv) == 3 else None)
    with open(sys.argv[1], "w") as f:
        json.dump(env, f, indent=2)
    print(f"Wrote {sys.argv[1]}")
    print(json.dumps({k: v for k, v in env.items() if k != "pip_freeze"}, indent=2))


if __name__ == "__main__":
    main()
