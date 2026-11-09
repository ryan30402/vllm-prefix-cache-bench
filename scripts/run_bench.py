"""Run vllm bench serve according to a config. Runs warmup first, discards it.

Usage:
    python scripts/run_bench.py configs/example.yaml results/<run_dir>
"""

import subprocess
import sys
from pathlib import Path

from common import load_config


def build_bench_cmd(cfg: dict, num_prompts: int, out_file: Path) -> list[str]:
    b = cfg["bench"]
    return [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        f"http://localhost:{cfg['server']['port']}",
        "--model",
        cfg["model"],
        "--dataset-name",
        "prefix_repetition",
        "--prefix-repetition-prefix-len",
        str(b["prefix_len"]),
        "--prefix-repetition-suffix-len",
        str(b["suffix_len"]),
        "--prefix-repetition-num-prefixes",
        str(b["num_prefixes"]),
        "--prefix-repetition-output-len",
        str(b["output_len"]),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(b["max_concurrency"]),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--seed",
        "0",
        "--save-result",
        "--result-filename",
        str(out_file),
    ]


def main() -> None:
    cfg = load_config(sys.argv[1])
    run_dir = Path(sys.argv[2])

    print("Warmup...")
    subprocess.run(
        build_bench_cmd(cfg, cfg["bench"]["warmup_prompts"], run_dir / "warmup.json"),
        check=True,
    )

    print("Main run...")
    subprocess.run(
        build_bench_cmd(
            cfg, cfg["bench"]["num_prompts"], run_dir / "bench_result.json"
        ),
        check=True,
    )


if __name__ == "__main__":
    main()
