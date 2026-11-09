"""Walk results/, read every completed run, produce one row per run as a CSV.

Usage:
    python analysis/load_results.py results analysis/runs.csv

Works with the default vllm bench serve output (aggregate fields only) and
with --save-detailed output (per-request ttfts), whichever is present.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def pct(values: list, q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def load_run(run_dir: Path) -> dict | None:
    status_path = run_dir / "status.json"
    if not status_path.exists() or not json.loads(status_path.read_text()).get("ok"):
        return None
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    bench = json.loads((run_dir / "bench_result.json").read_text())
    m0 = json.loads((run_dir / "metrics_before.json").read_text())
    m1 = json.loads((run_dir / "metrics_after.json").read_text())

    def delta(key: str) -> float:
        return m1.get(key, 0) - m0.get(key, 0)

    queries = delta("prefix_cache_queries")
    row = {
        "name": cfg["name"],
        "tag": cfg.get("tag", "main"),
        "cache": "on" if cfg["server"]["prefix_caching"] else "off",
        "prefix_len": cfg["bench"]["prefix_len"],
        "concurrency": cfg["bench"]["max_concurrency"],
        "repeat": cfg.get("repeat", 0),
        "gpu_mem_util": cfg["server"]["gpu_memory_utilization"],
        "n_requests": bench.get("completed"),
        "duration_s": bench.get("duration"),
        "output_tok_per_s": bench.get("output_throughput"),
        "request_per_s": bench.get("request_throughput"),
        "cache_hit_rate": delta("prefix_cache_hits") / queries if queries else 0.0,
        "preemptions": delta("num_preemptions"),
    }
    if bench.get("ttfts"):
        ttft_ms = [t * 1000 for t in bench["ttfts"]]
        row.update(
            ttft_mean_ms=float(np.mean(ttft_ms)),
            ttft_p50_ms=pct(ttft_ms, 50),
ttft_p95_ms=pct(ttft_ms, 95),
            ttft_p99_ms=pct(ttft_ms, 99),
        )
    else:
        row.update(
            ttft_mean_ms=bench.get("mean_ttft_ms"),
            ttft_p50_ms=bench.get("median_ttft_ms"),
            ttft_p95_ms=bench.get("p95_ttft_ms"),
            ttft_p99_ms=bench.get("p99_ttft_ms"),
        )
    row.update(
        tpot_mean_ms=bench.get("mean_tpot_ms"),
        tpot_p50_ms=bench.get("median_tpot_ms"),
        tpot_p99_ms=bench.get("p99_tpot_ms"),
        e2e_p50_ms=bench.get("median_e2el_ms"),
        e2e_p99_ms=bench.get("p99_e2el_ms"),
    )
    return row

def main() -> None:
    root, out = Path(sys.argv[1]), sys.argv[2]
    rows = [r for r in (load_run(d) for d in sorted(root.iterdir()) if d.is_dir()) if r]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"{len(df)} runs -> {out}")
    cols = ["ttft_p50_ms", "ttft_p99_ms", "output_tok_per_s", "cache_hit_rate"]
    print(df[df.tag == "main"].groupby(["cache", "prefix_len", "concurrency"])[cols].median().round(1))


if __name__ == "__main__":
    main()
