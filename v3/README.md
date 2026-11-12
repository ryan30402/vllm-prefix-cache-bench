# v3: an independent re-measurement

v1 (`results/`) explored 20 conditions and had a warm-up flaw. v2
(`results_v2/`) fixed the flaw and re-measured 6 of them. Reporting the two
side by side mixes two measurement methods. v3 measures everything again under
one written protocol, with new data only.

**Status: harness and protocol are written and tested on CPU. No GPU run has
been made under v3 yet. There are no v3 results.**

Rules that hold from the start:

- v1 and v2 code, data and reports stay in the repository unchanged.
- v3 code is in `v3/`, v3 data in `results_v3/`. The v3 analyzer reads one
  frozen v3 experiment folder and nothing else; it cannot be pointed at v1 or
  v2 data.
- Every v3 number comes from runs made under one frozen manifest.

The measurement contract is in [PROTOCOL.md](PROTOCOL.md).

## Stages

| Stage | What | State |
|---|---|---|
| 0 | Audit of the v2 code against the v3 plan | done, see "What changed" below |
| 1 | Harness hardening, CPU tests, failure injection | done: 125 tests |
| 2 | Protocol and experiment specs | written; frozen on the GPU machine |
| 3 | GPU smoke test and pilot | not run |
| 4 | Main matrix, 120 runs | not run |
| 5 | Block-boundary and memory-pressure diagnostics, output-reproducibility check, figures, report | not run |
| 6 | Optional: several prefixes, arrival rate, goodput under an SLO | not started |

## Commands

    python v3/bench.py plan    v3/protocol/main.yaml      # show the plan, no GPU
    python v3/bench.py freeze  v3/protocol/main.yaml      # prompt files + manifest
    python v3/bench.py run     results_v3/main_b1         # run or resume
    python v3/bench.py status  results_v3/main_b1
    python v3/bench.py verify  results_v3/main_b1         # re-validate every attempt
    python v3/bench.py analyze results_v3/main_b1         # tables + audit

`run` can be stopped (Ctrl+C, closed terminal, `kill`) and started again. It
stops its server before it exits, skips pairs that are complete and valid, and
runs everything else in new attempt folders.

## Layout

    v3/
      PROTOCOL.md              the measurement contract
      protocol/*.yaml          experiment specs (smoke, pilot, main, 2 diagnostics)
      bench.py                 command line entry
      pcbench/                 the harness
        spec.py                spec -> ordered, randomized, paired plan
        workload.py            prompt files with a token-id contract
        server.py              start/stop one server, cleanup guard
        bench.py               `vllm bench serve` wrapper, time limit
        metrics.py             counter snapshots, server and GPU telemetry
        schema.py              metric definitions and the one validator
        store.py               atomic writes, attempt states, lock
        ident.py               hashes, run identity, measurement-code hash
        envinfo.py             environment capture
        runner.py              freeze, sessions, pair attempts, resume
        analyze.py             runs.csv, pairs.csv, summary.csv, audit.json
      tests/                   CPU tests, fake `vllm` and fake `nvidia-smi`
    results_v3/<experiment>_b<batch>/
      manifest.json  environment.json  sessions.json
      workloads/<key>/workload.json          (prompt files are regenerated, not committed)
      runs/<run_id>/attempt_<n>/             one folder per attempt, never overwritten
      analysis/

## What changed from v2, and what stayed

Kept from v2 (already correct): warm-up and measured prompts share the prefix
and never repeat a prompt; a fresh server per run; counters read after the
warm-up; exact token counts; the server runs in its own process group and only
that group is stopped; a port in use is refused; retries get new folders.

| Problem in v2 (commit `31bd3f2`) | v3 | Test |
|---|---|---|
| Run identity covered config, environment and prompt files, but not the measurement code. Changing how a run is measured kept the same identity. | Identity includes a hash of the syntax tree of every measurement module. A frozen experiment refuses to run with changed code. | `test_measurement_code_hash_follows_measurement_logic_only`, `test_changed_measurement_code_stops_the_run` |
| Resume checked that files exist and that `completed` matches. A result with a per-request field removed still counted as done; a cut-off JSON file crashed the sweep. | Runner, resume and analyzer call the same `validate_attempt`: file hashes, every per-request record, recomputed summary, recomputed counter checks. It never raises. | `test_damaged_results_are_invalid` (12 kinds of damage, with and without refreshed hashes) |
| Files were written in place. `status.json` was written last, which worked as a marker but was not atomic. | Every file goes through a temp file, `fsync` and rename. An attempt is `RUNNING` until one rename writes `VALIDATED`. | `test_interrupting_any_write_never_leaves_a_false_success` (16 positions), `test_killed_runner_leaves_no_server_and_no_false_result` |
| If writing `server.json` failed right after the server started, nothing stopped the server. | One `try/finally` starts directly after the process exists and covers metadata, readiness, benchmark and timeout. | `test_server_is_stopped_when_writing_its_metadata_fails` |
| A killed runner left its server and load generator alive; the next start refused the port. | SIGTERM and SIGHUP are handled like Ctrl+C. After SIGKILL the next start stops the recorded processes (only if they are still the same ones) and marks the attempt `INTERRUPTED`. | `test_a_terminated_runner_cleans_up_before_it_exits`, `test_killed_runner_leaves_no_server_and_no_false_result` |
| No time limit on a benchmark pass. | Per-pass timeout; the whole process group is killed. | `test_timeout_kills_the_benchmark_and_the_server` |
| The v1 summary merged normal and memory-pressure runs. | `gpu_memory_utilization` is part of the condition id and of every table key. | `test_memory_conditions_are_never_merged_in_the_summary` |
| Reports mixed v1 and v2 numbers. | The analyzer needs a v3 manifest whose hash still matches. Every table row names its experiment, batch, manifest and source folder. | `test_analyzer_reads_only_frozen_v3_experiments` |
| Sampling was not pinned. `vllm bench serve` sends no temperature unless told to, and the server then takes its defaults from the model's `generation_config.json` (which for this model enables sampling). | `--generation-config vllm` and `--temperature 0`. | checked in `test_full_path_and_contract_numbers` |
| Output length was assumed from `max_tokens`. | Every request must report exactly 128 generated tokens, and the server's `generation_tokens` counter must agree. | `test_a_failing_run_is_recorded_and_never_counted[output shorter than agreed]` |
| Cache hits were recorded, not checked. | Exact expected hit tokens per run, including 0 at prefix 0. | `test_full_path_and_contract_numbers` |
| No telemetry. | GPU (UUID, temperature, power, utilisation, memory) and server gauges (running, waiting, KV usage) sampled every 2 s and stamped with the harness clock. | `test_gpu_samples_are_stamped_by_the_harness_and_survive_a_dead_server`, `test_failing_gpu_queries_fail_a_run_that_requires_telemetry` |
| In-flight requests were taken from the tool's `max_concurrent_requests`, which reported 64 at concurrency 32. | Computed from request start and end times. | `test_in_flight_count_handles_slots_handed_over_at_the_same_instant` |
| Pairs were randomized, but a failed run was retried alone, which broke the pair. | The pair is the unit: both runs are repeated together. | `test_a_transient_failure_is_retried_as_a_whole_pair` |
| Order: one shuffle per repeat, off/on order unbalanced. | Randomized blocks with balanced first-mode assignment, fixed by a seed in the spec. | `test_which_cache_mode_goes_first_is_balanced` |
| Gain = median(on) / median(off). | Geometric mean of the three pair ratios; the old ratio is a secondary column. | `test_full_path_and_contract_numbers` |

## Tests

    pip install -r requirements-dev.txt
    pytest -q v3/tests          # 125 tests, about 70 s, no GPU

They exercise the real code path (start, warm-up, measure, save, resume,
analyze) against `tests/fake_vllm3.py`, a small HTTP server with a 16-token
block cache and switchable failures. Passing these tests shows that the
harness handles the failures it was built for. It does not show that real vLLM
behaves as the fake does; that is what the smoke test and the pilot are for.
