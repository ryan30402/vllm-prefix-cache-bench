"""Start vLLM server from a config. Prints the PID so the caller can stop it.

Usage:
    python scripts/start_server.py configs/example.yaml results/<run_dir>
"""

import subprocess
import sys
from pathlib import Path

from common import load_config, wait_for_server


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
    if not s["prefix_caching"]:
        cmd.append("--no-enable-prefix-caching")
    return cmd


def main() -> None:
    cfg = load_config(sys.argv[1])
    run_dir = Path(sys.argv[2])
    cmd = build_server_cmd(cfg)
    log = open(run_dir / "server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"Started server PID {proc.pid}: {' '.join(cmd)}")
    wait_for_server(cfg["server"]["port"])
    print("Server ready")
    (run_dir / "server.pid").write_text(str(proc.pid))


if __name__ == "__main__":
    main()
