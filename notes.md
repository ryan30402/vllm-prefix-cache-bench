# Notes

## 2026-11-08 first manual run (Step 2)

Setup: Qwen2.5-7B-Instruct, RTX GPU on RunPod, vLLM 0.33.0.
Workload: prefix 1024 / suffix 256 / output 128, 100 prompts, concurrency 16.

Cache OFF: median TTFT 795 ms, p99 2653 ms, 291.5 tok/s, mean TPOT 44.7 ms
Cache ON:  median TTFT 491 ms, p99  789 ms, 415.6 tok/s, mean TPOT 31.8 ms

Observations:
- Biggest win is p99 TTFT (-70%): queued requests no longer wait for
  earlier requests' prefill.
- Throughput +43%, less than the TTFT win: decode work (128 tok/request)
  is unchanged, cache only removes prefill.
- Surprise: TPOT also dropped ~29%. Decode itself didn't change, so this
  must be prefill competing with decode for GPU time when cache is off.
- Single run, no repeats yet. Treat numbers as indicative.

Questions for later:
- Does the TPOT effect persist at concurrency 1? (no queue, less mixing)
- What does hit rate look like in /metrics? (expected ~1024/1280 = 80%)

## 2026-11-09 Resume-key collision (hit twice)

run_matrix.py resumes by skipping any run whose results/<name>/status.json
says ok. The name encoded prefix_len / concurrency / cache / repeat only.

1. Smoke test. The 8 smoke runs (num_prompts=50) shared names with main-matrix
   runs, so the main matrix silently reused them. Caught after the fact with
   `grep -L 'num_prompts: 500' results/p*/config.yaml`; deleted the 8 folders
   and resumed (only those 8 re-ran).
2. Pressure matrix. Different server settings (gpu_memory_utilization 0.5,
   max_model_len 16384) but identical names, so all 12 runs were skipped:
   "12 runs planned", 12 x "skip (done)", nothing executed.

Root cause: the resume key covered the workload but not the server config,
and server config is an experimental variable too.

Fix: matrix yaml takes a `tag`; names become `<tag>_p..._c..._cache_..._r..`.
Better fix for next time: key on a hash of the full resolved config.

Lessons:

- Anything that makes two runs non-interchangeable must be part of the key.
- Read the skip list after every resume. "Matrix complete" with zero new
  runs is a smell, not a success.
- Add a consistency check to the harvest step (the grep -L above).
- The block matrix shares server settings with main, so its 6 overlapping
  p2048_c32 runs legitimately reuse main-matrix results.
