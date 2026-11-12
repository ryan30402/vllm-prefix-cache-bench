"""Record the hardware, software and source state of a session."""

import platform
import subprocess
import sys
from pathlib import Path

from .ident import sha256_bytes
from .store import now

REPO_ROOT = Path(__file__).resolve().parents[2]
# Paths whose uncommitted changes would make a result untraceable.
SOURCE_PATHS = ["v3/pcbench", "v3/protocol", "v3/bench.py"]
# Telemetry fields we would like; each is probed, because a field the driver
# does not know makes nvidia-smi refuse the whole query.
TELEMETRY_FIELDS = [
    "timestamp",
    "uuid",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "clocks.current.sm",
    "clocks.current.memory",
]


def run(cmd: list[str], cwd: Path | None = None, timeout: float = 120) -> str:
    """Run a command and return its output as text. Empty string on failure."""
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL, cwd=cwd, timeout=timeout
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def pip_version(package: str) -> str:
    out = run([sys.executable, "-m", "pip", "show", package])
    for line in out.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return ""


def gpus() -> list[dict]:
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    keys = ["name", "uuid", "memory_total", "driver_version"]
    return [
        dict(zip(keys, [v.strip() for v in line.split(",")]))
        for line in out.splitlines()
    ]


def telemetry_fields() -> list[str]:
    """The telemetry fields this driver can report."""
    ok = []
    for field in TELEMETRY_FIELDS:
        if run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"]):
            ok.append(field)
    return ok


def vllm_default_block_size() -> int | None:
    """vLLM's default KV cache block size, read from the installed package."""
    code = "from vllm.config.cache import CacheConfig as C; print(C.DEFAULT_BLOCK_SIZE)"
    out = run([sys.executable, "-c", code], timeout=300)
    last = out.splitlines()[-1] if out else ""
    return int(last) if last.isdigit() else None


def source_state(repo_root: Path = REPO_ROOT) -> dict:
    paths = [p for p in SOURCE_PATHS if (repo_root / p).exists()]
    porcelain = run(["git", "status", "--porcelain", "--", *paths], cwd=repo_root)
    return {
        "git_commit": run(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "source_dirty": porcelain != "",
        "source_status": porcelain,
    }


def capture() -> dict:
    """Everything we want to know about this machine and checkout."""
    freeze = run([sys.executable, "-m", "pip", "freeze"])
    nvcc = run(["nvcc", "--version"])
    env = {
        "captured_at": now(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpus": gpus(),
        "cuda_compiler": nvcc.splitlines()[-1] if nvcc else "",
        "vllm": pip_version("vllm"),
        "torch": pip_version("torch"),
        "transformers": pip_version("transformers"),
        "vllm_default_block_size": vllm_default_block_size(),
        "telemetry_fields": telemetry_fields(),
        "pip_freeze": freeze,
        "pip_freeze_sha256": sha256_bytes(freeze.encode()),
    }
    env.update(source_state())
    return env


def env_identity(env: dict) -> dict:
    """GPU model and the versions that decide behaviour. Part of every run
    identity: two runs that differ here are never the same experiment."""
    names = sorted({g["name"] for g in env.get("gpus", [])})
    return {
        "gpu_models": names,
        "gpu_count": len(env.get("gpus", [])),
        "vllm": env.get("vllm", ""),
        "torch": env.get("torch", ""),
        "transformers": env.get("transformers", ""),
        "python": env.get("python", ""),
    }


def host_identity(env: dict) -> dict:
    """The physical machine and the full package set. Recorded per session; a
    change needs an explicit yes and is listed in the audit, never hidden."""
    return {
        "hostname": env.get("hostname", ""),
        "gpu_uuids": sorted(g["uuid"] for g in env.get("gpus", [])),
        "driver_versions": sorted({g["driver_version"] for g in env.get("gpus", [])}),
        "pip_freeze_sha256": env.get("pip_freeze_sha256", ""),
    }
