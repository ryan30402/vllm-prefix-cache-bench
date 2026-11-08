
vLLM Prefix Caching Benchmark Harness

Measures how vLLM prefix caching affects TTFT, end-to-end latency and

throughput under concurrent load. Compares the same model with prefix

caching enabled vs disabled across shared-prefix length and concurrency.

Status: work in progress.

Layout


configs/   one YAML per experiment (server + benchmark settings)

scripts/   start server, run benchmark, scrape metrics, run full matrix

checks/    correctness checks (outputs identical with/without cache)

analysis/  load results, plots, report

results/   one folder per run: config, env, per-request data, metrics

