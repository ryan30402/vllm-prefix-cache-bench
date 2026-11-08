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
