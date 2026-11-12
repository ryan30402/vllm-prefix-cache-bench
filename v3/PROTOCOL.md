# v3 measurement protocol

Protocol version `3.0.0`. This file states the contract in words. The same
rules are code in `pcbench/schema.py` (`METRIC_DEFINITIONS`, `VALIDITY_RULES`)
and are copied into every frozen `manifest.json`, so the text a result was
measured under travels with the result.

Status: the protocol is written and tested on CPU against a fake server. No GPU
run has been made under it yet. Nothing in this file is a result.

## Question

How does vLLM prefix caching change throughput, latency and resource use of an
LLM server, as a function of shared-prefix length, concurrency and KV-cache
memory pressure? How reliable are those numbers, and for which workloads do
they hold?

## What is fixed

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28` (also the tokenizer revision) |
| Server | `vllm serve`, one fresh process per run |
| Server flags | `--max-model-len 8192 --max-num-seqs 256 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.9 --generation-config vllm --seed 0`, plus `--enable-prefix-caching` or `--no-enable-prefix-caching` |
| Load generator | `vllm bench serve`, `/v1/completions`, closed loop (`--request-rate inf --max-concurrency C`) |
| Decoding | greedy (`--temperature 0`), `--ignore-eos`, exactly 128 output tokens |
| Prompt | shared prefix of P tokens + 256 unique suffix tokens, one shared prefix per run |
| Requests | 50 warm-up + 500 measured per run |
| KV block size | 16 tokens (vLLM default, checked against the installed package and by the hit counters) |

`--max-num-batched-tokens 2048` and `--max-num-seqs 256` are what vLLM 0.33.0
chooses by default for a GPU with less than 70 GiB; they are written out so the
value is part of the record. `--generation-config vllm` stops the server from
adopting the model's own sampling defaults.

## One run

1. Refuse to start if the port is in use.
2. Start the server in its own process group. From that moment a `finally`
   block stops the group, whatever happens next.
3. Read `/metrics` (snapshot `fresh`). The server must have handled 0 requests.
4. Warm-up: 50 prompts that share the prefix with the measured prompts but have
   different suffixes. Same concurrency as the measured run. Cache off gets the
   same warm-up, so both modes start from an initialised GPU.
5. Read `/metrics` (snapshot `before`).
6. Measured run: 500 prompts.
7. Read `/metrics` (snapshot `after`), stop the server.
8. Recompute every number from the per-request records, compare with the
   counters, and only then mark the attempt `VALIDATED`.

`vllm bench serve` has its own warm-up (`--num-warmups`) and ready check. Both
re-send the first prompt of the file, which would put a measured prompt into
the cache. Both are switched off (`0`) and the counters confirm it: the server
must count exactly 500 requests between `before` and `after`.

## What "warm" means

Allowed: the warm-up puts the shared prefix into the cache, on purpose. The
measured run then shows the steady state of a server that already knows the
prefix.

Not allowed: a measured prompt whose complete token sequence was already sent.
That would be a full-prompt hit, beyond the designed shared prefix. It is
excluded twice: by token ids when the prompt files are written and re-verified
(`pcbench/workload.py`), and by the hit counter of every run.

A cold cache is not a measured condition. The warm-up records
(`warmup_requests.jsonl`) and the counter differences over the warm-up keep
what happened on the cold server, for reference only.

With prefix length 0 nothing is shared by design. That is not assumed: no two
prompts may share even their first 16-token block, and a cache-on run at
prefix 0 must show exactly 0 hit tokens. This is the zero-prefix control.

## Metrics

All times come from the client clock inside `vllm bench serve`.

| Name | Definition |
|---|---|
| `ttft_s` | request sent -> first output token received |
| `e2e_s` | request sent -> last output token received |
| `tpot_s` | `(e2e_s - ttft_s) / (output_tokens - 1)` |
| `client_queue_s` | time the request waited in the client for a free concurrency slot; not included in any latency |
| `duration_s` | last response end - first request start, from the records |
| `output_tok_per_s` | output tokens of completed requests / `duration_s` |
| percentiles | `numpy.percentile`, linear interpolation, over the 500 requests of one run |
| burst / steady | burst = the first C requests by start time (sent together at t=0); steady = all later requests. TTFT is reported for both. |
| `peak_in_flight` | largest number of requests between send and last token at any instant |
| `cache_hit_rate` | `prefix_cache_hits / prefix_cache_queries`, as differences between `before` and `after` |

p99 over the 500 requests of one run sits at about the 6th-slowest request. In a closed loop
at concurrency C the first C requests arrive together, so p99 TTFT of a run is
mostly that start-up burst. It is reported as such and is not a steady-state
service p99. Steady-state TTFT is reported separately.

The 500 requests of a run are not 500 independent experiments. Variation
between runs is shown by the pairs, not by the request count.

## When an attempt is invalid

An attempt is `VALIDATED` only if all of these hold. Otherwise it is `FAILED`,
kept on disk with its error, listed in `audit.json`, and never averaged in.

- every warm-up and measured request completed without error;
- every request generated exactly 128 tokens (from the server's usage report,
  not from `max_tokens`);
- every prompt has the expected token count on the client, and the server's
  `prompt_tokens` counter moved by exactly 500 x that count;
- `request_success` moved by exactly 500 and `generation_tokens` by 500 x 128;
- no counter went down, no required metric was missing (a missing metric is an
  error, never a 0);
- cache off: 0 hit tokens. Cache on with no preemption: exactly
  500 x floor(P / 16) x 16 hit tokens. With preemptions the hit count is
  recorded but not checked, because a preempted request asks the cache again;
- no more than C requests were ever in flight;
- timings are finite and `0 < ttft_s <= e2e_s`;
- GPU and server telemetry were recorded.

A request that fails or times out still has a record with its error text.

## Pairs, order and repeats

- A condition is (prefix length, concurrency, gpu_memory_utilization).
- A pair is cache off and cache on of one condition, run back to back.
- A round visits every condition once, in a new random order.
- Which mode goes first in a pair is random but balanced: about half of each
  condition's pairs and about half of each round's pairs start with cache on.
- The order comes from `order_seed` in the spec, fixed before the first run.
  The full order is in the manifest.
- A pair counts only if both of its runs are valid and belong to the same pair
  attempt. If one run fails, both are run again in new attempt folders.
- At most 3 failed attempts per pair. After that the pair is reported as
  unmeasured. A valid result is never re-measured to get a different number.

Three rounds give three pairs per condition. That is enough for effects of
tens of percent or more and not enough to resolve differences of about 1%.
The pilot measures run-to-run variation before the main matrix; more rounds
are added only under a rule written down before the main matrix starts.

## Statistics

- Primary metric per condition: geometric mean of the per-pair throughput
  ratios `on / off`, with the minimum and maximum of the pairs next to it.
- The three pair ratios are always shown. No confidence interval is given for
  n = 3.
- `speedup_ratio_of_medians` (median on / median off) is a secondary column for
  comparison with the older reports. It is a different quantity.
- Throughput, latency (TTFT, TPOT, E2E at p50, p95, p99), errors, preemptions,
  hit rate and telemetry are reported together. Higher throughput does not
  imply lower latency.
- Statements are made only about measured points.

## Experiments

| Spec | Runs | Purpose |
|---|---|---|
| `smoke.yaml` | 4 short runs | pipeline check on real vLLM, about 10 minutes, never reported |
| `pilot.yaml` | 20 | contract check on real vLLM, first look at variation and time per run |
| `main.yaml` | 120 | prefix 0/256/1024/2048/4096 x concurrency 1/8/32/128 x off/on x 3 rounds |
| `block_boundary.yaml` | 24 | prefix 2047/2048/2049/2064 at concurrency 32 |
| `memory_pressure.yaml` | 24 | prefix 2048/4096 at concurrency 128, gpu_memory_utilization 0.9 and 0.5 |

Each experiment has its own folder and manifest. They are never merged.
Deviation from the written plan: the plan had 12 memory-pressure runs at 0.5
and took the 0.9 arm from the main matrix. Here the 0.9 arm is repeated inside
the same randomized rounds (24 runs), so the comparison never crosses
experiments.

## Freezing

`bench.py freeze` writes the prompt files and `manifest.json`: spec, complete
ordered plan, identity hash of every planned run, workload hashes, metric
definitions, validity rules, measurement-code hash.

- A frozen experiment is never changed. A changed spec needs a higher `batch`,
  which gives a new folder and new run identities.
- `bench.py run` refuses to start if the measurement code, the GPU model or the
  vLLM / torch / transformers / Python versions differ from the frozen ones.
- A different machine, GPU, driver or package set needs `--accept-new-host`,
  is recorded as a new session and is listed in the audit. Pairs never span
  sessions.

## Traceability

From any number in `summary.csv`: `audit.json` lists the attempt folders it came
from -> each folder has the request records, counter snapshots, telemetry,
server log, config, identity and file hashes -> the manifest has the prompt
hashes, the code hash and the commit.

## Not claimed

No custom kernel, no change to the vLLM engine, no production performance, no
statement about answer quality. Prompts are random tokens with one shared
prefix; results describe that workload on one GPU model with one model.
