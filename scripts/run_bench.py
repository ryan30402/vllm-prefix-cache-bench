"""Run `vllm bench serve` on one prompt file and validate what it wrote.

Usage:
    python scripts/run_bench.py <config.yaml> <prompts.jsonl> <num_prompts> <out.json>

The warm-up and the measured run are two separate calls on two different
prompt files (see make_workload.py), so the caller can read the server
counters in between.
"""

import json
import subprocess
import sys
from pathlib import Path

from common import RunError, load_config

# Per-request fields that are large and can be recomputed from what we keep.
DROPPED_FIELDS = ["itls", "generated_texts"]


def build_bench_cmd(
    cfg: dict, dataset_path: Path, num_prompts: int, out_file: Path
) -> list[str]:
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
        "custom",
        "--dataset-path",
        str(dataset_path),
        "--skip-chat-template",  # send the prompt exactly as generated
        "--disable-shuffle",  # request i is line i of the file
        "--no-oversample",  # never repeat a prompt to fill the count
        "--custom-output-len",
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
        "--save-detailed",  # keep per-request TTFT, latency and start time
        "--result-filename",
        str(out_file),
    ]


def validate_result(result: dict, num_prompts: int, what: str) -> None:
    """Refuse a result with missing, failed or errored requests."""
    if result.get("completed") != num_prompts:
        raise RunError(
            f"{what}: {result.get('completed')} requests completed, "
            f"expected {num_prompts}"
        )
    if result.get("failed", 0) != 0:
        raise RunError(f"{what}: {result['failed']} requests failed")
    errors = [e for e in result.get("errors", []) if e]
    if errors:
        raise RunError(f"{what}: {len(errors)} request errors, first: {errors[0]}")
    for field in ("ttfts", "latencies", "output_lens", "input_lens"):
        if len(result.get(field, [])) != num_prompts:
            raise RunError(f"{what}: per-request field {field} is missing or short")


def slim_result(result: dict) -> dict:
    """Add per-request TPOT, then drop the bulky fields it was derived from."""
    tpots = []
    for latency, ttft, n_out in zip(
        result["latencies"], result["ttfts"], result["output_lens"]
    ):
        tpots.append((latency - ttft) / (n_out - 1) if n_out > 1 else 0.0)
    result["tpots"] = tpots
    for field in DROPPED_FIELDS:
        result.pop(field, None)
    result["dropped_fields"] = DROPPED_FIELDS
    return result


def run_bench(
    cfg: dict, dataset_path: Path, num_prompts: int, out_file: Path, what: str
) -> dict:
    """Run one benchmark pass. Returns the validated, slimmed result."""
    cmd = build_bench_cmd(cfg, dataset_path, num_prompts, out_file)
    print("  $", " ".join(cmd), flush=True)
    done = subprocess.run(cmd)
    if done.returncode != 0:
        raise RunError(f"{what}: vllm bench serve exited with code {done.returncode}")
    if not Path(out_file).is_file():
        raise RunError(f"{what}: vllm bench serve wrote no result file")
    result = json.loads(Path(out_file).read_text())
    validate_result(result, num_prompts, what)
    result = slim_result(result)
    Path(out_file).write_text(json.dumps(result))
    return result


def main() -> None:
    cfg = load_config(sys.argv[1])
    result = run_bench(
        cfg, Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), "bench"
    )
    print(f"completed {result['completed']} requests")


if __name__ == "__main__":
    main()
