"""Turn the validated runs of one frozen v3 experiment into tables.

Reads one experiment folder and nothing else. It needs that folder's
manifest.json (a v3 manifest whose hash still matches), so it cannot be pointed
at v1 or v2 results, and it uses a run only if `validate_attempt` accepts it.

Outputs, in <experiment>/analysis/:
    runs.csv      one row per run, with its provenance
    pairs.csv     one row per pair: cache on vs cache off, measured back to back
    summary.csv   one row per condition: geometric mean of the pair speedups
    audit.json    every attempt (valid or not), what is missing, which runs
                  each summary row came from
"""

import csv
import io
import math
from pathlib import Path

from . import ident, schema
from .runner import attempt_dir, load_manifest, pair_state, pairs_in_order
from .store import (
    atomic_write_json,
    atomic_write_text,
    attempt_dirs,
    now,
    read_json,
    read_state,
)

RUN_METRICS = [
    "requests",
    "completed",
    "failed",
    "duration_s",
    "output_tok_per_s",
    "requests_per_s",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "tpot_p99_ms",
    "e2e_mean_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "burst_requests",
    "burst_ttft_p50_ms",
    "burst_ttft_max_ms",
    "steady_requests",
    "steady_ttft_p50_ms",
    "steady_ttft_p95_ms",
    "steady_ttft_p99_ms",
    "peak_in_flight",
    "mean_in_flight",
    "distinct_outputs",
]
TELEMETRY = [
    "server_running_mean",
    "server_waiting_mean",
    "server_waiting_max",
    "kv_cache_usage_mean",
    "kv_cache_usage_max",
    "gpu_temperature_c_mean",
    "gpu_temperature_c_max",
    "gpu_power_w_mean",
    "gpu_utilization_pct_mean",
    "gpu_memory_used_mib_max",
]
CONDITION = ["condition_id", "prefix_len", "concurrency", "gpu_memory_utilization"]
# Ratios cache on / cache off for a pair. Above 1 is better only for throughput.
PAIR_RATIOS = [
    "output_tok_per_s",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "steady_ttft_p50_ms",
    "tpot_p50_ms",
    "e2e_p50_ms",
    "e2e_p99_ms",
]
SUMMARY_METRICS = [
    "output_tok_per_s",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "steady_ttft_p50_ms",
    "steady_ttft_p99_ms",
    "tpot_p50_ms",
    "e2e_p50_ms",
    "e2e_p99_ms",
]


def median(values: list[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def load_run(exp_dir: Path, entry: dict, n: int, base: dict) -> dict:
    d = attempt_dir(exp_dir, entry, n)
    cfg = entry["config"]
    summary = read_json(d / "summary.json")
    counters = read_json(d / "counters.json")
    tel = read_json(d / "telemetry.json")
    status = read_state(d)
    server = read_json(d / "server.json")
    row = {
        **base,
        "session_id": status["session_id"],
        "gpu_uuids": "|".join(tel.get("gpu_uuids", [])),
        "run_id": entry["run_id"],
        "pair_id": entry["pair_id"],
        "attempt": n,
        "round": entry["round"],
        "order": entry["order"],
        "first_in_pair": entry["first_in_pair"],
        **{k: cfg[k] for k in CONDITION},
        "cache": cfg["cache"],
        **{k: summary.get(k) for k in RUN_METRICS},
        "cache_hit_rate": counters["cache_hit_rate"],
        "cache_hit_rate_expected": counters["cache_hit_rate_expected"],
        "hit_check": counters["hit_check"],
        "preemptions": counters["deltas"]["num_preemptions"],
        "kv_cache_size_tokens": server.get("kv_cache_size_tokens"),
        **{k: tel.get(k) for k in TELEMETRY},
        "started_at": status.get("started_at"),
        "seconds": status.get("seconds"),
        "source": str(d.relative_to(exp_dir)),
    }
    return row


def to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def analyze(exp_dir: Path) -> dict:
    """Write the tables and the audit. Returns the audit."""
    exp_dir = Path(exp_dir)
    manifest = load_manifest(exp_dir)
    sessions_path = exp_dir / "sessions.json"
    sessions = read_json(sessions_path) if sessions_path.is_file() else []
    base = {
        "experiment": manifest["experiment"],
        "batch": manifest["batch"],
        "protocol_version": manifest["protocol_version"],
        "manifest_sha256": manifest["manifest_sha256"][:12],
    }
    runs, pairs, attempts, missing = [], [], [], []
    for pair in pairs_in_order(manifest):
        for entry in pair:
            for d in attempt_dirs(exp_dir / "runs" / entry["run_id"]):
                status = read_state(d)
                problems = (
                    schema.validate_attempt(d, entry["identity_sha256"])
                    if status["state"] == "VALIDATED"
                    else []
                )
                attempts.append(
                    {
                        "run_id": entry["run_id"],
                        "attempt": d.name,
                        "state": status["state"],
                        "valid": status["state"] == "VALIDATED" and not problems,
                        "error": status.get("error", ""),
                        "problems": problems,
                    }
                )
        state = pair_state(exp_dir, pair)
        if state["complete"] is None:
            missing.append(
                {"pair_id": pair[0]["pair_id"], "failed_attempts": state["failed"]}
            )
            continue
        by_cache = {
            e["config"]["cache"]: load_run(exp_dir, e, state["complete"], base)
            for e in pair
        }
        runs += [by_cache[e["config"]["cache"]] for e in pair]
        off, on = by_cache["off"], by_cache["on"]
        row = {
            **base,
            "pair_id": off["pair_id"],
            "round": off["round"],
            **{k: off[k] for k in CONDITION},
            "first_in_pair": off["first_in_pair"],
            "attempt": off["attempt"],
            "session_id": off["session_id"],
            "same_session": off["session_id"] == on["session_id"],
            "same_gpu": off["gpu_uuids"] == on["gpu_uuids"],
            "preemptions_off": off["preemptions"],
            "preemptions_on": on["preemptions"],
        }
        for m in PAIR_RATIOS:
            row[f"{m}_off"] = off[m]
            row[f"{m}_on"] = on[m]
            row[f"{m}_on_over_off"] = (
                on[m] / off[m] if off[m] and on[m] is not None else None
            )
        pairs.append(row)

    summary, sources = [], {}
    for cid in dict.fromkeys(p["condition_id"] for p in pairs):
        group = [p for p in pairs if p["condition_id"] == cid]
        speedups = [p["output_tok_per_s_on_over_off"] for p in group]
        row = {
            **base,
            **{k: group[0][k] for k in CONDITION},
            "pairs": len(group),
            "rounds": "|".join(str(p["round"]) for p in group),
            # Primary metric: geometric mean of the per-pair throughput ratios.
            "speedup_geomean": geomean(speedups),
            "speedup_min": min(speedups),
            "speedup_max": max(speedups),
            "speedup_pairs": "|".join(f"{s:.4f}" for s in speedups),
        }
        for m in SUMMARY_METRICS:
            for cache in ("off", "on"):
                values = [
                    r[m]
                    for r in runs
                    if r["condition_id"] == cid and r["cache"] == cache
                ]
                row[f"{m}_{cache}_median"] = median(values)
                row[f"{m}_{cache}_min"] = min(values)
                row[f"{m}_{cache}_max"] = max(values)
        # Secondary, for comparison with older reports: ratio of the medians.
        row["speedup_ratio_of_medians"] = (
            row["output_tok_per_s_on_median"] / row["output_tok_per_s_off_median"]
        )
        row["preemptions_off_total"] = sum(p["preemptions_off"] for p in group)
        row["preemptions_on_total"] = sum(p["preemptions_on"] for p in group)
        summary.append(row)
        sources[cid] = [r["source"] for r in runs if r["condition_id"] == cid]

    out = exp_dir / "analysis"
    out.mkdir(exist_ok=True)
    atomic_write_text(out / "runs.csv", to_csv(runs))
    atomic_write_text(out / "pairs.csv", to_csv(pairs))
    atomic_write_text(out / "summary.csv", to_csv(summary))
    planned = manifest["counts"]
    states: dict[str, int] = {}
    for a in attempts:
        key = (
            a["state"]
            if a["valid"] or a["state"] != "VALIDATED"
            else "VALIDATED_BUT_INVALID"
        )
        states[key] = states.get(key, 0) + 1
    audit = {
        "generated_at": now(),
        **base,
        "manifest_sha256": manifest["manifest_sha256"],
        "frozen_git_commit": manifest["git_commit"],
        "analysis_code_sha256": ident.code_fingerprint(Path(__file__).read_text()),
        "planned": planned,
        "complete_pairs": len(pairs),
        "missing_pairs": missing,
        "complete": not missing and len(pairs) == planned["pairs"],
        "attempts_by_state": states,
        "measured_requests_in_tables": sum(r["requests"] for r in runs),
        "failed_requests_in_tables": sum(r["failed"] for r in runs),
        "hit_checks": {
            k: sum(r["hit_check"] == k for r in runs)
            for k in {r["hit_check"] for r in runs}
        },
        "pairs_with_cache_on_first": sum(p["first_in_pair"] == "on" for p in pairs),
        "pairs_spanning_sessions": [
            p["pair_id"] for p in pairs if not p["same_session"]
        ],
        "pairs_spanning_gpus": [p["pair_id"] for p in pairs if not p["same_gpu"]],
        "sessions": sessions,
        "distinct_hosts": len({ident.sha256_json(s["host"]) for s in sessions}),
        "sources_by_condition": sources,
        "attempts": attempts,
    }
    atomic_write_json(out / "audit.json", audit)
    return audit
