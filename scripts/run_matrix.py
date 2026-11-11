"""Expand a sweep config into individual runs and execute them all.

Usage:
    python scripts/run_matrix.py configs/matrix_clean24.yaml
        [--results-root results_v2] [--workload-root workloads] [--allow-dirty]

Each run is stored in <results-root>/<name>__<config hash>/attempt_<n>/.
Re-running the command skips runs whose newest attempt passes every
completeness check, and gives every other run a new attempt folder.

Exit code: 0 only if every run in the sweep has a valid result, 1 if any run
failed, 2 if the sweep could not start.
"""

import argparse
import itertools
import json
import random
import sys
import time
import traceback
from pathlib import Path

import yaml

import capture_env
from common import (
    RunError,
    config_hash,
    latest_complete_attempt,
    load_config,
    next_attempt_dir,
    run_dir_name,
)
from make_workload import locate_workload
from run_bench import run_bench
from scrape_metrics import counter_deltas, scrape
from start_server import ensure_alive, start_server
from stop_server import stop_server


class SweepAbort(RunError):
    """Continuing the sweep would contaminate every later run."""


def make_cfg(matrix: dict, params: dict, rep: int) -> dict:
    bench = {
        **matrix["bench_base"],
        "prefix_len": params["prefix_len"],
        "max_concurrency": params["max_concurrency"],
    }
    bench.setdefault("workload_seed", 0)
    if bench.pop("num_prefixes", 1) != 1:
        raise RunError("the v2 harness supports exactly one shared prefix")
    cache = "on" if params["prefix_caching"] else "off"
    name = f"p{params['prefix_len']}_c{params['max_concurrency']}_cache_{cache}_r{rep}"
    if matrix.get("tag"):
        name = f"{matrix['tag']}_{name}"
    return {
        "name": name,
        "tag": matrix.get("tag", "main"),
        "model": matrix["model"],
        "revision": matrix.get("revision", ""),
        "server": {**matrix["server_base"], "prefix_caching": params["prefix_caching"]},
        "bench": bench,
        "repeat": rep,
    }


def expand(matrix: dict) -> list[dict]:
    """One config per combination per repeat, in the order they will run.

    order: sequential     every combination, then its repeats (the v1 order)
    order: paired_random  per repeat, shuffle the (prefix, concurrency) cells and
                          run cache off/on back to back in a random order, so
                          slow drift cannot line up with one cache setting
    """
    sweep = matrix["sweep"]
    order = matrix.get("order", "sequential")
    if order == "sequential":
        keys = list(sweep.keys())
        runs = []
        for combo in itertools.product(*[sweep[k] for k in keys]):
            for rep in range(matrix["repeats"]):
                runs.append(make_cfg(matrix, dict(zip(keys, combo)), rep))
        return runs
    if order != "paired_random":
        raise RunError(f"unknown order: {order}")
    rng = random.Random(matrix.get("order_seed", 0))
    runs = []
    for rep in range(matrix["repeats"]):
        cells = list(itertools.product(sweep["prefix_len"], sweep["max_concurrency"]))
        rng.shuffle(cells)
        for prefix_len, concurrency in cells:
            pair = list(sweep["prefix_caching"])
            rng.shuffle(pair)
            for caching in pair:
                params = {
                    "prefix_len": prefix_len,
                    "max_concurrency": concurrency,
                    "prefix_caching": caching,
                }
                runs.append(make_cfg(matrix, params, rep))
    return runs


def run_checks(
    cfg: dict, fresh: dict, deltas: dict, result: dict, workload: dict
) -> dict:
    """Cross-check the client-side result against the server-side counters."""
    n = cfg["bench"]["num_prompts"]
    tokens_each = workload["manifest"]["prompt_tokens_each"]
    queries = deltas["prefix_cache_queries"]
    prefix_len = cfg["bench"]["prefix_len"]
    checks = {
        "fresh_request_success": fresh.get("request_success", 0),
        "request_success_delta": deltas["request_success"],
        "prompt_tokens_delta": deltas["prompt_tokens"],
        "prompt_tokens_expected": n * tokens_each,
        "client_input_tokens": sum(result["input_lens"]),
        "cache_hit_rate": deltas["prefix_cache_hits"] / queries if queries else 0.0,
        # every measured request can reuse the whole blocks of the shared prefix
        "cache_hit_rate_expected": (
            (prefix_len // 16 * 16) / tokens_each
            if cfg["server"]["prefix_caching"]
            else 0.0
        ),
        "preemptions": deltas["num_preemptions"],
    }
    if checks["request_success_delta"] != n:
        raise RunError(
            f"server counted {checks['request_success_delta']} successful requests "
            f"during the measured run, expected {n}"
        )
    if checks["prompt_tokens_delta"] != checks["prompt_tokens_expected"]:
        raise RunError(
            f"server counted {checks['prompt_tokens_delta']} prompt tokens, expected "
            f"{checks['prompt_tokens_expected']}; the prompts were not tokenized "
            "as generated"
        )
    return checks


def run_one(
    cfg: dict, attempt_dir: Path, digest: str, env: dict, workload: dict
) -> None:
    """Execute one run. Raises on anything that makes the result untrustworthy."""
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    (attempt_dir / "env.json").write_text(json.dumps(env, indent=2))
    port = cfg["server"]["port"]
    bench = cfg["bench"]
    proc = None
    try:
        proc = start_server(cfg, attempt_dir)
        fresh = scrape(port, str(attempt_dir / "metrics_fresh"))
        if fresh["request_success"] != 0 or fresh["prompt_tokens"] != 0:
            raise RunError("the server has already handled requests; it is not fresh")

        run_bench(
            cfg,
            workload["warmup"],
            bench["warmup_prompts"],
            attempt_dir / "warmup_result.json",
            "warm-up",
        )
        ensure_alive(proc, "during warm-up")
        # Counters are read after the warm-up, so the deltas below describe the
        # measured requests and nothing else.
        before = scrape(port, str(attempt_dir / "metrics_before"))
        result = run_bench(
            cfg,
            workload["main"],
            bench["num_prompts"],
            attempt_dir / "bench_result.json",
            "measured run",
        )
        ensure_alive(proc, "during the measured run")
        after = scrape(port, str(attempt_dir / "metrics_after"))
        checks = run_checks(cfg, fresh, counter_deltas(before, after), result, workload)
        (attempt_dir / "checks.json").write_text(json.dumps(checks, indent=2))
    finally:
        try:
            print("  " + stop_server(attempt_dir, proc), flush=True)
        except RunError as e:
            # A server we cannot stop would be measured by the next run.
            raise SweepAbort(f"could not stop the server: {e}") from e


def write_status(attempt_dir: Path, status: dict) -> None:
    (attempt_dir / "status.json").write_text(json.dumps(status, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config")
    parser.add_argument("--results-root", default="results_v2")
    parser.add_argument("--workload-root", default="workloads")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run even if scripts/ or configs/ differ from the last commit",
    )
    args = parser.parse_args(argv)

    matrix = load_config(args.config)
    env = capture_env.capture(matrix["model"])
    if env["source_dirty"] and not args.allow_dirty:
        print("Source files differ from the last commit:\n" + env["source_status"])
        print("Commit them first, or pass --allow-dirty (the diff is then saved).")
        return 2
    matrix["revision"] = matrix.get("revision") or env["model_revision"]
    if not matrix["revision"]:
        print(
            "Cannot pin the model revision: set `revision:` in the config, or run "
            "scripts/make_workload.py first so the model is in the local cache."
        )
        return 2
    env["model_revision"] = matrix["revision"]
    identity = capture_env.env_identity(env)

    try:
        runs = expand(matrix)
        workloads = [locate_workload(cfg, Path(args.workload_root)) for cfg in runs]
    except RunError as e:
        print(f"Cannot start the sweep: {e}")
        return 2

    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs planned, model revision {matrix['revision']}")
    failed, plan = [], []
    for i, (cfg, workload) in enumerate(zip(runs, workloads), 1):
        files = {k: workload["manifest"][f"{k}_sha256"] for k in ("warmup", "main")}
        digest = config_hash(cfg, {**identity, "workload": files})
        run_dir = root / run_dir_name(cfg, digest)
        plan.append({"order_index": i, "name": cfg["name"], "dir": run_dir.name})
        if latest_complete_attempt(run_dir, digest) is not None:
            print(f"[{i}/{len(runs)}] skip {run_dir.name} (complete)")
            continue
        attempt_dir = next_attempt_dir(run_dir)
        print(f"[{i}/{len(runs)}] run  {run_dir.name}/{attempt_dir.name}", flush=True)
        others = [d.name for d in root.glob(f"{cfg['name']}__*") if d != run_dir]
        if others:
            print(
                f"  note: {len(others)} earlier result(s) for {cfg['name']} were made "
                "with a different config or environment and are not reused"
            )
        status = {"config_hash": digest, "order_index": i, "attempt": attempt_dir.name}
        t0 = time.time()
        abort = False
        try:
            run_one(cfg, attempt_dir, digest, env, workload)
            status["ok"] = True
        except Exception as e:
            status["ok"] = False
            status["error"] = traceback.format_exc()
            print(status["error"], flush=True)
            failed.append(run_dir.name)
            abort = isinstance(e, SweepAbort)
        status["seconds"] = round(time.time() - t0)
        write_status(attempt_dir, status)
        if abort:
            print("Aborting the sweep: later runs could not be trusted.")
            break

    plan_name = f"plan_{Path(args.config).stem}.json"
    (root / plan_name).write_text(json.dumps(plan, indent=2))
    if failed:
        print(f"{len(failed)} of {len(runs)} runs FAILED:")
        for name in failed:
            print(f"  {name}")
        return 1
    print(f"All {len(runs)} runs have a valid result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
