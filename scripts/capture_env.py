"""Record the hardware and software environment of a benchmark run.

Usage:
    python scripts/capture_env.py results/some_run/env.json
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone


def run(cmd: list[str]) -> str:
    """Run a shell command and return its output as text. Empty string on failure."""
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
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


def capture() -> dict:
    """Collect everything we want to know about this machine into one dict."""
    gpu_query = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    nvcc = run(["nvcc", "--version"])
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu_query,
        "gpu_count": len(run(["nvidia-smi", "-L"]).splitlines()),
        "cuda_version": nvcc.splitlines()[-1] if nvcc else "",
        "vllm": pip_version("vllm"),
        "torch": pip_version("torch"),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": run(["git", "status", "--porcelain"]) != "",
    }


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    out_path = sys.argv[1]
    env = capture()
    with open(out_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
