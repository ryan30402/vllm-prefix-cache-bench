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
