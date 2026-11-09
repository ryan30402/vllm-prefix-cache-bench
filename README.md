# vLLM Prefix Caching Benchmark Harness

A reproducible benchmark of vLLM prefix caching: the same model served with
the cache on and off, across shared-prefix length and concurrency, measuring
time to first token (TTFT), time per output token (TPOT) and output throughput.

## Results

Qwen2.5-7B-Instruct on one NVIDIA A40, vLLM 0.33.0. 150 runs of 500 requests,
medians of 3 repeats. Full analysis: [analysis/REPORT.md](analysis/REPORT.md).

| Shared prefix / concurrency | Metric | Cache off | Cache on | Change |
|---|---|---|---|---|
| 2,048 tokens / 32 | Output throughput | 277 tok/s | 778 tok/s | 2.8x |
| 2,048 tokens / 32 | p99 TTFT | 8.9 s | 1.2 s | −86% |
| 2,048 tokens / 32 | p50 TTFT | 1,540 ms | 660 ms | −57% |
| 4,096 tokens / 128 | Output throughput | 135 tok/s | 1,196 tok/s | 8.9x |
| 4,096 tokens / 128 | p50 TTFT | 67.5 s | 1.7 s | −97% |
| none / 32 | Output throughput | 837 tok/s | 855 tok/s | 1.02x |

![Throughput gain from prefix caching](analysis/figures/fig4_throughput_gain_heatmap.png)

What the data shows:

- The gain grows with prefix length and concurrency. Under load, prefill is
  the bottleneck, and the cache removes most of it.
- Decoding itself is not faster: TPOT is identical at concurrency 1. Under
  load TPOT still drops (102.8 → 35.7 ms at prefix 2048, concurrency 32)
  because other requests' prefill no longer delays decode steps.
- With `gpu_memory_utilization` lowered from 0.9 to 0.5, the cache-off server
  preempted requests at concurrency 128 and the cache-on server did not: a
  shared prefix is stored once.
- Greedy outputs are reproducible within a cache state but differ between
  cache states, so exact-match tests must pin the cache state.

Known limitations (a synthetic single-prefix workload, closed-loop load, and a
warm-up overlap that makes 10% of measured requests full-prompt hits) are
documented in the report.

## How it works

Each run starts a fresh server, sends 50 warm-up requests, then 500 measured
requests with `vllm bench serve`, and scrapes Prometheus counters before and
after to get the cache hit rate and preemption count. `run_matrix.py` expands
a sweep config into runs and skips runs that already completed.

    pip install -r requirements.txt
    python scripts/run_matrix.py configs/matrix_main.yaml
    python analysis/load_results.py results analysis/runs.csv
    python analysis/plots.py analysis/runs.csv analysis/figures

## Layout

| Path | Contents |
|---|---|
| `configs/` | sweep definitions: main, block-boundary, memory-pressure |
| `scripts/` | start/stop server, run benchmark, scrape metrics, run a sweep |
| `checks/` | greedy-output comparison across cache states |
| `analysis/` | results loader, plots, `runs.csv`, figures, report |
| `results/` | one folder per run: config, environment, benchmark output, metrics |
| `notes.md` | lab notes, including bugs found along the way |
