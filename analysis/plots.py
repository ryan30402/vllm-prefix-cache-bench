"""Four figures from analysis/runs.csv.

Usage:
    python analysis/plots.py analysis/runs.csv analysis/figures
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm


def agg(df: pd.DataFrame) -> pd.DataFrame:
    """Median over repeats."""
    return df.groupby(["cache", "prefix_len", "concurrency"], as_index=False).median(
        numeric_only=True
    )


def fig_ttft_vs_prefix(a: pd.DataFrame, out: Path) -> None:
    concs = sorted(a["concurrency"].unique())
    fig, axes = plt.subplots(1, len(concs), figsize=(4 * len(concs), 3.6), sharey=True)
    for ax, c in zip(axes, concs):
        for cache, style in [("off", "o--"), ("on", "s-")]:
            s = a[(a.concurrency == c) & (a.cache == cache)].sort_values("prefix_len")
            ax.plot(s.prefix_len, s.ttft_p50_ms, style, label=f"cache {cache} p50")
            ax.plot(
                s.prefix_len,
                s.ttft_p99_ms,
                style,
                alpha=0.4,
                label=f"cache {cache} p99",
            )
        ax.set_yscale("log")  # TTFT spans 60 ms to 107 s across the panels
        ax.set_title(f"concurrency = {c}")
        ax.set_xlabel("shared prefix length (tokens)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("TTFT (ms, log scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Time to first token vs shared prefix length")
    fig.tight_layout()
    fig.savefig(out / "fig1_ttft_vs_prefix.png", dpi=150)


def fig_throughput_vs_concurrency(a: pd.DataFrame, out: Path) -> None:
    p = 2048  # a representative prefix length
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for cache, style in [("off", "o--"), ("on", "s-")]:
        s = a[(a.prefix_len == p) & (a.cache == cache)].sort_values("concurrency")
        ax.plot(s.concurrency, s.output_tok_per_s, style, label=f"cache {cache}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("output tokens / s")
    ax.set_title(f"Output throughput vs concurrency (prefix = {p} tokens)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_throughput_vs_concurrency.png", dpi=150)


def fig_gain_heatmap(
    a: pd.DataFrame, out: Path, col: str, lower_is_better: bool, title: str, fname: str
) -> None:
    """Gain from prefix caching for one metric, as a prefix_len x concurrency grid."""
    on = a[a.cache == "on"].set_index(["prefix_len", "concurrency"])[col]
    off = a[a.cache == "off"].set_index(["prefix_len", "concurrency"])[col]
    gain = (off / on if lower_is_better else on / off).unstack("concurrency")
    norm = LogNorm(vmin=1, vmax=gain.values.max())  # log colors: gains span 1x to ~40x
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(gain.values, cmap="YlGn", norm=norm)
    ax.set_xticks(range(len(gain.columns)), gain.columns)
    ax.set_yticks(range(len(gain.index)), gain.index)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("shared prefix length (tokens)")
    ax.set_title(title)
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            v = gain.values[i, j]
            color = "white" if norm(max(v, 1)) > 0.6 else "black"
            ax.text(
                j, i, f"{v:.2f}x", ha="center", va="center", fontsize=9, color=color
            )
    cbar = fig.colorbar(im, ax=ax, ticks=[1, 2, 5, 10, 20], format="%gx")
    cbar.ax.minorticks_off()
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)


def main() -> None:
    df = pd.read_csv(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    main_prefixes = [0, 256, 1024, 2048, 4096]
    main = df[(df.tag == "main") & (df.prefix_len.isin(main_prefixes))]
    a = agg(main)
    fig_ttft_vs_prefix(a, out)
    fig_throughput_vs_concurrency(a, out)
    fig_gain_heatmap(
        a,
        out,
        "ttft_p50_ms",
        True,
        "TTFT p50 speedup from prefix caching (off / on)",
        "fig3_ttft_speedup_heatmap.png",
    )
    fig_gain_heatmap(
        a,
        out,
        "output_tok_per_s",
        False,
        "Output throughput gain from prefix caching (on / off)",
        "fig4_throughput_gain_heatmap.png",
    )
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
