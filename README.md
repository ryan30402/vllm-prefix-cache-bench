# vLLM Prefix Caching Benchmark Harness

![tests](https://github.com/ryan30402/vllm-prefix-cache-bench/actions/workflows/tests.yml/badge.svg)

A reproducible benchmark of vLLM prefix caching: the same model served with
the cache on and off, across shared-prefix length and concurrency, measuring
time to first token (TTFT), time per output token (TPOT) and output throughput.

## Status

A full re-measurement under one written protocol (v3) is being prepared: new
harness, new data, nothing reused from the runs below. The harness and its CPU
tests are done; no v3 GPU run exists yet. See [v3/README.md](v3/README.md) and
[v3/PROTOCOL.md](v3/PROTOCOL.md). Everything below is the v1 and v2 work, kept
unchanged.

## Results (v1 and v2)

Qwen2.5-7B-Instruct on one NVIDIA A40, vLLM 0.33.0, 500 requests per run,
medians of 3 repeats. Full analysis: [analysis/REPORT.md](analysis/REPORT.md).

| Shared prefix / concurrency | Metric | Cache off | Cache on | Change |
|---|---|---|---|---|
| 2,048 tokens / 32 | Output throughput | 271 tok/s | 756 tok/s | 2.8x |
| 2,048 tokens / 32 | p50 end-to-end latency | 15.1 s | 5.3 s | −65% |
| 2,048 tokens / 32 | p50 TTFT | 1,276 ms | 634 ms | −50% |
| 4,096 tokens / 128 | Output throughput | 133 tok/s | 1,137 tok/s | 8.6x |
| 4,096 tokens / 128 | p50 TTFT | 68.4 s | 1.8 s | −97% |
| none / 32 | Output throughput | 832 tok/s | 832 tok/s | 1.00x |

These are the clean reference runs (42 runs, v2 harness). The figure shows the
earlier 20-condition sweep (150 runs, v1 harness), whose gains are up to 4%
higher because of a measurement flaw described in the report.

![Throughput gain from prefix caching, v1 sweep](analysis/figures/fig4_throughput_gain_heatmap.png)

What the data shows:

- The gain grows with prefix length and concurrency. Under load, prefill is
  the bottleneck, and the cache removes most of it.
- Decoding itself is not faster: TPOT is identical at concurrency 1. Under
  load TPOT still drops (105.9 → 36.9 ms at prefix 2048, concurrency 32)
  because other requests' prefill no longer delays decode steps.
- With `gpu_memory_utilization` lowered from 0.9 to 0.5, the cache-off server
  preempted requests at concurrency 128 and its median TTFT tripled. The
  cache-on server was unaffected: a shared prefix is stored once.
- Greedy outputs are reproducible within a cache state but differ between
  cache states, so exact-match tests must pin the cache state.

Known limitations (a synthetic single-prefix workload and closed-loop load
that starts with a burst) are documented in the report.

A review of the first harness (commit `1be108b`) found measurement and
reliability problems, including a warm-up overlap that made 10% of measured
requests full-prompt cache hits. The harness was rewritten and the main
conditions were measured again: the headline gain moved from 2.81x to 2.79x
and the zero-prefix control from 1.02x to 1.00x. Both data sets are kept and
reported side by side.

## How it works

Each run starts a fresh server in its own process group, checks that the
server answering on the port is the one it started, sends warm-up requests,
reads the Prometheus counters, sends the measured requests with
`vllm bench serve`, and reads the counters again. The counter difference
therefore covers the measured requests only.

- **Prompts.** `make_workload.py` writes the prompt files: every prompt has an
  exact token count, all prompts share the same prefix tokens, and no measured
  prompt repeats a warm-up prompt.
- **Resume.** A run folder is named after a hash of its full config and
  environment, each retry is kept as a separate attempt, and a run is skipped
  only if its files and request counts check out.
- **Failures.** A missing metric, a counter that went down, a failed request
  or a server that died is an error. The sweep exits non-zero unless every run
  has a valid result.
- **Provenance.** Each run records the model revision, `pip freeze`, the
  server's start-up settings and the source commit, and the sweep refuses to
  start if the scripts differ from the last commit.

    pip install -r requirements.txt
    python scripts/make_workload.py configs/matrix_clean24.yaml
    python scripts/run_matrix.py configs/matrix_clean24.yaml
    python analysis/load_results_v2.py results_v2 analysis/runs_v2.csv
    python analysis/compare_v1_v2.py analysis/runs.csv analysis/runs_v2.csv

## Tests

The harness is tested without a GPU against a stand-in for the `vllm` command
(`tests/fake_vllm.py`): process cleanup, changed settings, missing files,
missing metrics, failed requests and prompt generation.

    pip install -r requirements-dev.txt
    pytest -q

## Layout

| Path | Contents |
|---|---|
| `configs/` | sweep definitions |
| `scripts/` | generate prompts, start/stop server, run benchmark, scrape metrics, run a sweep |
| `tests/` | CPU-only tests and the fake `vllm` command |
| `checks/` | greedy-output comparison across cache states |
| `analysis/` | results loaders, plots, comparison, figures, report |
| `results/` | v1 sweep, one folder per run |
| `results_v2/` | clean reference runs, one folder per run and attempt |
| `workloads/` | manifests (checksums) of the generated prompt files |
| `notes.md` | lab notes, including bugs found along the way |
| `v3/` | v3 harness, protocol, specs and tests (independent of the paths above) |
| `results_v3/` | v3 data (empty until the first GPU run) |
