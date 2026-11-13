# v3 lab notes

## 2026-11-12 Stage 0 to 2: audit, harness, protocol

Decision: re-measure everything as v3 instead of adding points to v2. Reason:
the report had two data sets made with two methods, and every sentence needed a
qualifier saying which one it came from.

Audit of v2 at `31bd3f2` (file evidence in v3/README.md):

- Confirmed in the code: run identity without measurement code
  (`scripts/common.py: config_hash`); resume checks only file presence and the
  `completed` count (`check_complete`); in-place writes; in
  `scripts/start_server.py` the write of `server.json` sits between `Popen` and
  the `try`, so a failed write leaks the server.
- Found while reading vLLM 0.33.0: `--generation-config` defaults to `auto`,
  so the server takes its default sampling settings from the model's
  `generation_config.json`, and `vllm bench serve` sends no temperature unless
  told to. v1 and v2 therefore did not decode greedily. As far as I know that
  file sets temperature 0.7, top_p 0.8, top_k 20 and repetition_penalty 1.05
  for this model; v2 kept no server logs, so this is not confirmed from our own
  data. The pilot's server log will show what the server reports. With
  `--ignore-eos` the output length was 128 either way.
- Found in the v2 data: `max_concurrent_requests` from `vllm bench serve` was
  64 at concurrency 32 and up to 172 at 128. Recomputed from start and end
  times, the peak was exactly 1, 32 and 128 in all 42 runs. The tool's number
  is not an in-flight count.
- Checked on the v2 data before making it a rule: duration recomputed from the
  per-request records differs from the tool's `duration` by at most 0.012%.

Design choices:

- `vllm bench serve` stays the load generator and the clock. Writing my own
  client would add a second thing to validate.
- The pair is the unit of work and of analysis.
- Hit counts are checked exactly. In v2 all 42 runs matched the block
  arithmetic exactly, so a mismatch in v3 means something changed.
- Memory pressure: the plan took the 0.9 arm from the main matrix (12 runs at
  0.5). I run both arms in the same rounds (24 runs). The v1 conclusion "cache
  on loses 1.5% under pressure" came from exactly such a cross-batch
  comparison and was wrong.
- The full package list is recorded per session but is not part of the run
  identity; vLLM, torch, transformers, Python and the GPU model are. A changed
  package list needs `--accept-new-host` and shows up in the audit.

Not done yet: any GPU run. Output-reproducibility diagnostic, figures and the
report follow once there is data. The rule for adding rounds will be written
here after the pilot and before `main` is frozen.

## 2026-11-12 Telemetry sampling changed before the first GPU run

First version: one long-lived `nvidia-smi -lms` process writing CSV to a file.
Problem found by reading my own code, not by a failure: whether that process
flushes each line when its output is a file is not something I can check
without a GPU, and it is stopped with SIGTERM. If it buffers, a short run ends
with an empty file and fails the "telemetry was recorded" rule for a reason
that has nothing to do with the measurement.

Now: the sampling thread runs one short `nvidia-smi` query per sample and
stamps it with the harness clock. No dependence on buffering or on the
timestamp format of the tool; the first sample is taken immediately. Cost: one
process start every 2 s, the same for cache on and off.

Also hardened: the summary numbers of `vllm bench serve` are stored for
reference; a non-finite value anywhere in them is now written as null instead
of breaking the JSON file.

## 2026-11-13 Smoke test and pilot on the A40

Smoke (4 runs) and pilot (20 runs) at commit `5f0c011`, one session, one GPU.
All 24 runs valid on the first attempt. I re-validated every attempt from the
pushed files and re-ran the analysis: `runs.csv`, `pairs.csv` and `summary.csv`
come out byte-identical.

Contract on real vLLM, pilot: 10,000 measured requests, 0 failed; every request
128 output tokens; server counters match the client exactly; hit tokens exact
in 20 of 20 runs (hit rate 0.8889 at prefix 2048, 0.9412 at 4096, 0.8850 at
2040, 0 at prefix 0 and with the cache off); 0 preemptions; peak in-flight
equals the configured concurrency. Prefix 2040 gave 2,032 reusable tokens per
request, which confirms 16-token blocks on this machine.

Pilot pair ratios (cache on / off, throughput), round 1 | round 2:

| Condition | Round 1 | Round 2 |
|---|---|---|
| prefix 0, concurrency 32 | 0.9986 | 1.0001 |
| prefix 2040, concurrency 32 | 2.7818 | 2.7785 |
| prefix 2048, concurrency 1 | 1.0748 | 1.0747 |
| prefix 2048, concurrency 128 | 5.6397 | 5.6257 |
| prefix 4096, concurrency 128 | 8.7079 | 8.7249 |

These are pilot numbers. They are not part of the main matrix and are not
pooled with it.

Run-to-run variation (same condition, same mode, round 1 vs round 2):
throughput at most 0.25%, median 0.04%; TPOT p50 at most 2.6%; TTFT p99 at most
6.6%; TTFT p50 at most 19% (prefix 2048, concurrency 128, cache on: 1,781 vs
1,471 ms, while throughput differed by 0.25%). So throughput is very stable and
TTFT p50 at high concurrency is not: in a closed loop the requests arrive in
waves and the median moves with the wave pattern. The rule for more rounds in
PROTOCOL.md was written from these numbers.

Three things the pilot showed that I did not expect:

- Outputs. With greedy decoding and identical prompts I compared the hash of
  each of the 500 outputs between runs. Concurrency 1: 500 of 500 identical,
  between rounds and between cache modes. Higher concurrency: not always
  identical even within one mode (prefix 0, concurrency 32, cache off, round 1
  vs round 2: 349 of 500), and about as different between modes (356 to 358).
  So a difference between cache on and off cannot be blamed on the cache by
  this comparison alone. The dedicated diagnostic (token ids, first diverging
  position, logprob margin) is still to do.
- GPU utilisation as reported by nvidia-smi: 76 to 86% with the cache off at
  concurrency 128, 35 to 42% with the cache on. The cache-on runs are not
  limited by the GPU there. Worth a look in the report, with the caveat that
  this number is the share of time any kernel was running, not a load measure.
- Time. The pilot took 5.3 h, not the 4 h I estimated. Per run about 385 s are
  not measurement: about 225 s until the server is ready and about 75 s of
  start-up for each of the two `vllm bench serve` passes (a 9 s warm-up pass
  took 85 s of wall time). In v2 the same overhead was about 90 s per run,
  estimated from its total run time. The
  difference I can see is that the virtual environment was moved to
  /workspace, a volume, and each process start imports thousands of files from
  it. Not verified yet; the main matrix will run from a virtual environment on
  the local disk with the same package versions, and the import time of both
  is measured first. This does not touch the measurement: the clock starts
  when a request is sent.

Estimate for the rest, with the method that reproduces the pilot (5.25 h
predicted, 5.28 h actual): main 35 h at 385 s overhead per run, 26 h at 110 s.
