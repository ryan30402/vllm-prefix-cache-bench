"""The measurement contract in code: what a valid attempt is.

One validator, used in three places:

  * the runner, before it marks an attempt VALIDATED;
  * resume, before it skips a run as done;
  * the analyzer, before it reads a number.

So a result that the runner would refuse can never be skipped by resume or
plotted by the analyzer. `validate_attempt` never raises: a file that is
missing, cut short or not valid JSON is reported as a problem.
"""

import json
import math
from pathlib import Path

import numpy as np

from .ident import sha256_file
from .metrics import counter_deltas
from .store import VALIDATED, read_json, read_state

METRIC_DEFINITIONS = {
    "clock": "client side, inside `vllm bench serve` (time.perf_counter)",
    "start_s": "when the request was sent, relative to the first request of the run",
    "client_queue_s": "time a request waited in the client for a free concurrency "
    "slot before it was sent; not part of any latency below",
    "ttft_s": "time from sending the request to receiving the first output token",
    "e2e_s": "time from sending the request to receiving the last output token",
    "tpot_s": "(e2e_s - ttft_s) / (output_tokens - 1): mean gap between output "
    "tokens after the first",
    "duration_s": "last response end minus first request start, from the rows",
    "output_tok_per_s": "sum of output_tokens of completed requests / duration_s",
    "requests_per_s": "completed requests / duration_s",
    "percentiles": "numpy.percentile with linear interpolation over the completed "
    "requests of one run (500 in the main matrix)",
    "burst": "the first `concurrency` requests by start time: they are sent "
    "together at t=0 and queue behind each other",
    "steady": "every request after the burst",
    "peak_in_flight": "largest number of requests between send and last token at "
    "any instant, from the rows; an end within 1 microsecond of a start counts "
    "as before it",
    "cache_hit_rate": "prefix_cache_hits / prefix_cache_queries, both as counter "
    "differences between the snapshot after warm-up and the snapshot after the "
    "measured requests",
    "counters": "request_success, prompt_tokens, generation_tokens, "
    "prefix_cache_queries, prefix_cache_hits, num_preemptions: differences over "
    "the measured requests only",
    "pair_speedup": "output_tok_per_s of the cache-on run / output_tok_per_s of "
    "the cache-off run of the same pair",
    "condition_speedup": "geometric mean of the pair speedups of a condition "
    "(primary metric); min and max of the pairs are reported next to it",
}

VALIDITY_RULES = {
    "requests": "every measured and warm-up request completed without error",
    "output_length": "every request generated exactly workload.output_len tokens",
    "input_length": "every prompt was counted as the expected number of tokens "
    "by the client and, in total, by the server",
    "fresh_server": "the server had handled no request before the warm-up",
    "counter_window": "request_success, prompt_tokens and generation_tokens moved "
    "by exactly the measured requests between the two snapshots",
    "cache_hits": "with no preemption: cache off gives 0 hit tokens; cache on "
    "gives exactly num_prompts x cacheable_tokens_per_request",
    "in_flight": "peak_in_flight never exceeds the configured concurrency",
    "timing": "0 < ttft_s <= e2e_s and all numbers are finite",
    "failure": "an attempt that breaks a rule is kept and reported as FAILED; "
    "it is never averaged in and never skipped by resume",
}

REQUIRED_FILES = [
    "config.json",
    "identity.json",
    "server.json",
    "server.log.gz",
    "metrics_fresh.json",
    "metrics_before.json",
    "metrics_after.json",
    "warmup_requests.jsonl",
    "requests.jsonl",
    "bench_warmup.json",
    "bench_main.json",
    "counters.json",
    "telemetry.json",
    "server_timeseries.csv",
    "summary.json",
]
ROW_FIELDS = {
    "i": int,
    "ok": bool,
    "error": str,
    "start_s": float,
    "client_queue_s": float,
    "ttft_s": float,
    "e2e_s": float,
    "input_tokens": int,
    "output_tokens": int,
    "output_sha256": str,
}
UNCOMMITTED = {"status.json", "files.json"}
HANDOVER_S = 1e-6


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rows_text(rows: list[dict]) -> str:
    return "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)


def check_rows(
    rows: list[dict], want_n: int, cfg: dict, wl: dict, what: str
) -> list[str]:
    """Reasons these per-request records are not a valid set (empty = valid)."""
    problems = []
    if len(rows) != want_n:
        return [f"{what}: {len(rows)} request records, expected {want_n}"]
    out_len = cfg["workload"]["output_len"]
    in_len = wl["server_prompt_tokens_each"]
    failed = 0
    for n, row in enumerate(rows):
        for field, kind in ROW_FIELDS.items():
            value = row.get(field)
            good = is_number(value) if kind is float else type(value) is kind
            if not good:
                return [f"{what}: record {n} has no valid `{field}`"]
            if kind is float and not math.isfinite(value):
                return [f"{what}: record {n} has a non-finite `{field}`"]
        if row["i"] != n:
            return [f"{what}: record {n} is out of order"]
        if not row["ok"] or row["error"]:
            failed += 1
            continue
        if row["output_tokens"] != out_len:
            problems.append(
                f"{what}: request {n} generated {row['output_tokens']} tokens, "
                f"expected {out_len}"
            )
        if row["input_tokens"] != in_len:
            problems.append(
                f"{what}: request {n} has {row['input_tokens']} prompt tokens, "
                f"expected {in_len}"
            )
        if not (0 < row["ttft_s"] <= row["e2e_s"]) or row["start_s"] < 0:
            problems.append(f"{what}: request {n} has impossible timings")
    if failed:
        problems.insert(0, f"{what}: {failed} of {want_n} requests failed")
    return problems[:20]


def _pct(values: np.ndarray, prefix: str, unit_scale: float = 1000.0) -> dict:
    if len(values) == 0:
        return {f"{prefix}_{k}_ms": None for k in ("mean", "p50", "p95", "p99", "max")}
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {
        f"{prefix}_mean_ms": float(values.mean() * unit_scale),
        f"{prefix}_p50_ms": float(p50 * unit_scale),
        f"{prefix}_p95_ms": float(p95 * unit_scale),
        f"{prefix}_p99_ms": float(p99 * unit_scale),
        f"{prefix}_max_ms": float(values.max() * unit_scale),
    }


def summarize(rows: list[dict], concurrency: int) -> dict:
    """Run-level metrics, computed only from the per-request records."""
    ok = [r for r in rows if r["ok"]]
    out: dict = {
        "requests": len(rows),
        "completed": len(ok),
        "failed": len(rows) - len(ok),
    }
    if not ok:
        return out
    start = np.array([r["start_s"] for r in ok], dtype=float)
    ttft = np.array([r["ttft_s"] for r in ok], dtype=float)
    e2e = np.array([r["e2e_s"] for r in ok], dtype=float)
    n_out = np.array([r["output_tokens"] for r in ok], dtype=float)
    queue = np.array([r["client_queue_s"] for r in ok], dtype=float)
    end = start + e2e
    duration = float(end.max() - start.min())
    tpot = (e2e - ttft)[n_out > 1] / (n_out[n_out > 1] - 1)
    # In-flight requests over time: +1 at every start, -1 at every end.
    # A slot that is handed over must not be counted twice. The client frees a
    # slot and starts the next request within microseconds, and adding two
    # floats can put an end a rounding error after the start that followed it,
    # so an end within 1 microsecond of a start counts as before it.
    events = sorted([(t, 1) for t in start] + [(t - HANDOVER_S, -1) for t in end])
    cur = peak = 0
    area, last_t = 0.0, events[0][0]
    for t, step in events:
        area += cur * (t - last_t)
        last_t = t
        cur += step
        peak = max(peak, cur)
    order = np.argsort(start, kind="stable")
    burst, steady = order[:concurrency], order[concurrency:]
    out.update(
        {
            "duration_s": duration,
            "input_tokens": int(sum(r["input_tokens"] for r in ok)),
            "output_tokens": int(n_out.sum()),
            "output_tok_per_s": float(n_out.sum() / duration),
            "requests_per_s": float(len(ok) / duration),
            "peak_in_flight": int(peak),
            "mean_in_flight": float(area / duration),
            "distinct_outputs": len({r["output_sha256"] for r in ok}),
            **_pct(ttft, "ttft"),
            **_pct(tpot, "tpot"),
            **_pct(e2e, "e2e"),
            **_pct(queue, "client_queue"),
            "burst_requests": int(len(burst)),
            "steady_requests": int(len(steady)),
            **{f"burst_{k}": v for k, v in _pct(ttft[burst], "ttft").items()},
            **{f"steady_{k}": v for k, v in _pct(ttft[steady], "ttft").items()},
        }
    )
    return out


def check_counters(cfg: dict, wl: dict, fresh: dict, before: dict, after: dict) -> dict:
    """Cross-check the server counters against what the run should have sent."""
    w = cfg["workload"]
    n, n_warm = w["num_prompts"], w["warmup_prompts"]
    each = wl["server_prompt_tokens_each"]
    warm = counter_deltas(fresh, before)
    main = counter_deltas(before, after)
    expected = {
        "request_success": n,
        "prompt_tokens": n * each,
        "generation_tokens": n * w["output_len"],
    }
    problems = []
    if fresh["request_success"] != 0 or fresh["prompt_tokens"] != 0:
        problems.append("the server had already handled requests before the warm-up")
    if warm["request_success"] != n_warm or warm["prompt_tokens"] != n_warm * each:
        problems.append(
            f"warm-up: server counted {warm['request_success']} requests and "
            f"{warm['prompt_tokens']} prompt tokens, expected {n_warm} and {n_warm * each}"
        )
    for key, want in expected.items():
        if main[key] != want:
            problems.append(
                f"measured run: server counted {key}={main[key]}, expected {want}"
            )
    cache_on = cfg["cache"] == "on"
    hits_expected = n * wl["cacheable_tokens_per_request"] if cache_on else 0
    if main["num_preemptions"] > 0 and cache_on:
        # A preempted request is scheduled again and asks the cache again, so
        # the hit counter no longer has one exact expected value.
        hit_check = "not checked: preemptions occurred"
    elif main["prefix_cache_hits"] != hits_expected:
        hit_check = "failed"
        problems.append(
            f"measured run: {main['prefix_cache_hits']} cache-hit tokens, "
            f"expected {hits_expected}"
        )
    else:
        hit_check = "exact"
    queries = main["prefix_cache_queries"]
    return {
        "fresh": {k: fresh[k] for k in ("request_success", "prompt_tokens")},
        "warmup_deltas": warm,
        "deltas": main,
        "expected": {**expected, "prefix_cache_hits": hits_expected},
        "cache_hit_rate": main["prefix_cache_hits"] / queries if queries else 0.0,
        "cache_hit_rate_expected": hits_expected / (n * each),
        "hit_check": hit_check,
        "problems": problems,
    }


def _close(a, b) -> bool:
    if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if is_number(a) and is_number(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def check_content(attempt_dir: Path) -> list[str]:
    """Recompute everything from the raw files and compare with what is stored."""
    d = Path(attempt_dir)
    problems = []
    try:
        config = read_json(d / "config.json")
        cfg, wl = config["run"], config["workload"]
        w = cfg["workload"]
        warm_rows = read_rows(d / "warmup_requests.jsonl")
        rows = read_rows(d / "requests.jsonl")
        problems += check_rows(warm_rows, w["warmup_prompts"], cfg, wl, "warm-up")
        problems += check_rows(rows, w["num_prompts"], cfg, wl, "measured run")
        if problems:
            return problems
        summary = summarize(rows, cfg["concurrency"])
        stored = read_json(d / "summary.json")
        for key in set(summary) | set(stored):
            if not _close(summary.get(key), stored.get(key)):
                problems.append(
                    f"summary.json: `{key}` does not match the request records"
                )
        if summary["peak_in_flight"] > cfg["concurrency"]:
            problems.append(
                f"{summary['peak_in_flight']} requests were in flight at once, "
                f"more than the configured concurrency {cfg['concurrency']}"
            )
        snaps = [
            read_json(d / f"metrics_{k}.json") for k in ("fresh", "before", "after")
        ]
        counters = check_counters(cfg, wl, *snaps)
        problems += counters["problems"]
        stored_counters = read_json(d / "counters.json")
        if stored_counters != json.loads(json.dumps(counters)):
            problems.append("counters.json does not match the metrics snapshots")
        if summary["output_tokens"] != counters["deltas"]["generation_tokens"]:
            problems.append("client and server disagree on the number of output tokens")
        if summary["input_tokens"] != counters["deltas"]["prompt_tokens"]:
            problems.append("client and server disagree on the number of prompt tokens")
        tel = read_json(d / "telemetry.json")
        if cfg["telemetry"]["required"]:
            if tel.get("gpu_samples_total", 0) < 1:
                problems.append("no GPU telemetry sample was recorded")
            if tel.get("server_samples_total", 0) < 1:
                problems.append("no server telemetry sample was recorded")
        server = read_json(d / "server.json")
        if server.get("model_card", {}).get("id") != cfg["model"]:
            problems.append("server.json does not show the expected model")
    except Exception as e:  # unreadable, cut short or malformed
        problems.append(f"unreadable result: {type(e).__name__}: {e}")
    return problems


def files_manifest(attempt_dir: Path) -> dict:
    """sha256 and size of every result file of an attempt."""
    out = {}
    for path in sorted(Path(attempt_dir).iterdir()):
        if (
            path.is_file()
            and path.name not in UNCOMMITTED
            and not path.name.endswith(".tmp")
        ):
            out[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return out


def validate_attempt(
    attempt_dir: Path, identity_sha256: str | None = None
) -> list[str]:
    """Reasons this attempt is NOT a valid finished run (empty list = valid)."""
    d = Path(attempt_dir)
    try:
        status = read_state(d)
        if status["state"] != VALIDATED:
            return [f"state is {status['state']}"]
        if (
            identity_sha256 is not None
            and status.get("identity_sha256") != identity_sha256
        ):
            return ["made with a different config, workload, environment or code"]
        if sha256_file(d / "files.json") != status.get("files_sha256"):
            return ["files.json does not match the hash in status.json"]
        listed = read_json(d / "files.json")
        problems = [f"missing {name}" for name in REQUIRED_FILES if name not in listed]
        for name, meta in listed.items():
            path = d / name
            if not path.is_file():
                problems.append(f"missing {name}")
            elif (
                path.stat().st_size != meta["bytes"]
                or sha256_file(path) != meta["sha256"]
            ):
                problems.append(f"{name} was changed after the run")
        if problems:
            return problems
    except Exception as e:
        return [f"unreadable attempt: {type(e).__name__}: {e}"]
    return check_content(d)
