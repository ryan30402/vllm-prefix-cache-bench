"""Read every valid v2 run and write one row per run as a CSV.

Usage:
    python analysis/load_results_v2.py results_v2 analysis/runs_v2.csv

Unlike the v1 loader, percentiles come from the per-request data of each run,
and nothing is defaulted: a missing file, field or counter is an error.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from common import RunError, attempt_dirs, latest_complete_attempt  # noqa: E402
from scrape_metrics import counter_deltas  # noqa: E402


def percentiles_ms(seconds: list[float], prefix: str) -> dict:
    ms = np.asarray(seconds, dtype=float) * 1000
    return {
        f"{prefix}_mean_ms": float(ms.mean()),
        f"{prefix}_p50_ms": float(np.percentile(ms, 50)),
        f"{prefix}_p95_ms": float(np.percentile(ms, 95)),
        f"{prefix}_p99_ms": float(np.percentile(ms, 99)),
    }


def load_attempt(attempt: Path, n_attempts: int) -> dict:
    cfg = yaml.safe_load((attempt / "config.yaml").read_text())
    bench = json.loads((attempt / "bench_result.json").read_text())
    status = json.loads((attempt / "status.json").read_text())
    env = json.loads((attempt / "env.json").read_text())
    server = json.loads((attempt / "server_info.json").read_text())
    deltas = counter_deltas(
        json.loads((attempt / "metrics_before.json").read_text()),
        json.loads((attempt / "metrics_after.json").read_text()),
    )
    n = cfg["bench"]["num_prompts"]
    for field in ("ttfts", "tpots", "latencies", "queue_times"):
        if len(bench[field]) != n:
            raise RunError(
                f"{attempt}: {field} has {len(bench[field])} values, not {n}"
            )
    queries = deltas["prefix_cache_queries"]
    row = {
        "name": cfg["name"],
        "dir": attempt.parent.name,
        "attempt": attempt.name,
        "n_attempts": n_attempts,
        "order_index": status["order_index"],
        "tag": cfg["tag"],
        "cache": "on" if cfg["server"]["prefix_caching"] else "off",
        "prefix_len": cfg["bench"]["prefix_len"],
        "concurrency": cfg["bench"]["max_concurrency"],
        "repeat": cfg["repeat"],
        "gpu_mem_util": cfg["server"]["gpu_memory_utilization"],
        "max_model_len": cfg["server"]["max_model_len"],
        "kv_cache_size_tokens": server["kv_cache_size_tokens"],
        "model_revision": cfg["revision"],
        "git_commit": env["git_commit"],
        "source_dirty": env["source_dirty"],
        "n_requests": bench["completed"],
        "duration_s": bench["duration"],
        "output_tok_per_s": bench["output_throughput"],
        "request_per_s": bench["request_throughput"],
        "cache_hit_rate": deltas["prefix_cache_hits"] / queries if queries else 0.0,
        "preemptions": deltas["num_preemptions"],
    }
    row.update(percentiles_ms(bench["ttfts"], "ttft"))
    row.update(percentiles_ms(bench["tpots"], "tpot"))
    # latency = send to last token; it excludes the wait in the client-side
    # concurrency queue, which is reported separately.
    row.update(percentiles_ms(bench["latencies"], "e2e"))
    row.update(percentiles_ms(bench["queue_times"], "client_queue"))
    return row


def load_all(root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Return (one row per valid run, names of runs without a valid attempt)."""
    rows, invalid = [], []
    for run_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        attempts = attempt_dirs(run_dir)
        if not attempts:
            continue
        good = latest_complete_attempt(run_dir)
        if good is None:
            invalid.append(run_dir.name)
        else:
            rows.append(load_attempt(good, len(attempts)))
    return pd.DataFrame(rows), invalid


def main() -> int:
    root, out = Path(sys.argv[1]), sys.argv[2]
    df, invalid = load_all(root)
    df.to_csv(out, index=False)
    print(f"{len(df)} valid runs -> {out}")
    if len(df):
        cols = ["ttft_p50_ms", "ttft_p99_ms", "tpot_mean_ms", "output_tok_per_s"]
        cols += ["cache_hit_rate", "preemptions"]
        keys = ["prefix_len", "concurrency", "cache"]
        print(df.groupby(keys)[cols].median().round(3).to_string())
    if invalid:
        print(f"{len(invalid)} runs have NO valid attempt: {invalid}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
