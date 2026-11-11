"""Compare the v1 sweep with the clean v2 runs for the conditions both cover.

Usage:
    python analysis/compare_v1_v2.py analysis/runs.csv analysis/runs_v2.csv

Prints medians of 3 repeats. A "gain" is cache on relative to cache off:
on / off for throughput, off / on for latencies.
"""

import sys

import pandas as pd

KEYS = ["prefix_len", "concurrency", "cache"]
METRICS = ["output_tok_per_s", "tpot_mean_ms", "ttft_p50_ms", "ttft_p99_ms"]
METRICS += ["e2e_p50_ms", "cache_hit_rate", "preemptions"]


def medians(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(KEYS)[METRICS].median()


def gains(m: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for prefix_len, concurrency in sorted({i[:2] for i in m.index}):
        off, on = (
            m.loc[(prefix_len, concurrency, "off")],
            m.loc[(prefix_len, concurrency, "on")],
        )
        rows[(prefix_len, concurrency)] = {
            "throughput": on.output_tok_per_s / off.output_tok_per_s,
            "tpot": off.tpot_mean_ms / on.tpot_mean_ms,
            "ttft_p50": off.ttft_p50_ms / on.ttft_p50_ms,
            "ttft_p99": off.ttft_p99_ms / on.ttft_p99_ms,
            "e2e_p50": off.e2e_p50_ms / on.e2e_p50_ms,
        }
    return pd.DataFrame(rows).T.rename_axis(["prefix_len", "concurrency"])


def main() -> None:
    pd.set_option("display.width", 200)
    v1 = pd.read_csv(sys.argv[1])
    v2 = pd.read_csv(sys.argv[2])
    clean = v2[v2.tag == "clean"]
    cells = pd.MultiIndex.from_frame(clean[KEYS[:2]].drop_duplicates())
    in_cells = pd.MultiIndex.from_frame(v1[KEYS[:2]]).isin(cells)
    v1_same = v1[(v1.tag == "main") & in_cells]

    m1, m2 = medians(v1_same), medians(clean)
    print("== Absolute values, v1 then v2 (median of 3) ==")
    both = pd.concat({"v1": m1, "v2": m2}, axis=1).swaplevel(axis=1)
    for metric in METRICS:
        print(f"-- {metric}")
        table = both[metric].copy()
        table["v2_vs_v1_pct"] = (table.v2 / table.v1 - 1) * 100
        print(table.round(3).to_string())

    print("\n== Gain from prefix caching, v1 then v2 ==")
    g = pd.concat({"v1": gains(m1), "v2": gains(m2)}, axis=1).swaplevel(axis=1)
    for metric in ["throughput", "tpot", "ttft_p50", "ttft_p99", "e2e_p50"]:
        print(f"-- {metric}")
        print(g[metric].round(3).to_string())

    print("\n== Memory pressure at prefix 2048, concurrency 128 ==")
    print(
        "v1: gpu_memory_utilization 0.5 and max_model_len 16384 (two settings changed)"
    )
    print(medians(v1[v1.tag == "pressure"]).loc[(2048, 128)].round(3).to_string())
    print("v2: gpu_memory_utilization 0.5 only; baseline is the 0.9 rows above")
    print(medians(v2[v2.tag == "clean_pressure"]).loc[(2048, 128)].round(3).to_string())

    print("\n== v2 spread across repeats, (max - min) / median, in percent ==")
    spread = v2.groupby(["tag", *KEYS])[METRICS[:4]].agg(
        lambda s: (s.max() - s.min()) / s.median() * 100
    )
    print(spread.round(2).to_string())


if __name__ == "__main__":
    main()
