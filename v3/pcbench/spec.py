"""Experiment specs and the run plan.

A spec file (v3/protocol/*.yaml) lists the fixed settings and the conditions of
one experiment. `build_plan` turns it into the exact ordered list of runs:

  * every round visits every condition once, in a new random order
    (a randomized block design: slow drift is spread over all conditions);
  * inside a condition, cache off and cache on run back to back as a pair;
  * which of the two goes first is random, but balanced: over the rounds each
    condition has cache-on first in about half of its pairs, and so does each
    round over its conditions.

The plan depends only on the spec, so it can be generated and inspected before
any GPU time is spent.
"""

import itertools
import random
from pathlib import Path

import yaml

from .ident import sha256_json
from .store import RunError

CACHE_MODES = ("off", "on")
CONDITION_KEYS = ("prefix_len", "concurrency", "gpu_memory_utilization")
REQUIRED = {
    "server": [
        "port",
        "max_model_len",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "generation_config",
        "seed",
    ],
    "workload": [
        "suffix_len",
        "output_len",
        "num_prompts",
        "warmup_prompts",
        "seed",
        "block_size",
    ],
    "bench": ["temperature", "timeout_s", "warmup_timeout_s"],
    "telemetry": ["required", "interval_s"],
    "validity": ["max_pair_attempts"],
}
TOP_LEVEL = ["name", "batch", "model", "model_revision", "rounds", "order_seed"]


def load_spec(path: str | Path) -> dict:
    with open(path) as f:
        spec = yaml.safe_load(f)
    validate_spec(spec)
    return spec


def validate_spec(spec: dict) -> None:
    for key in TOP_LEVEL + list(REQUIRED) + ["conditions"]:
        if key not in spec:
            raise RunError(f"spec is missing `{key}`")
    for section, keys in REQUIRED.items():
        for key in keys:
            if key not in spec[section]:
                raise RunError(f"spec is missing `{section}.{key}`")
    if not spec["model_revision"]:
        raise RunError("spec must pin `model_revision` to a commit hash")
    w, s = spec["workload"], spec["server"]
    if w["suffix_len"] < w["block_size"]:
        raise RunError("workload.suffix_len must be at least one cache block")
    for cond in conditions(spec):
        total = cond["prefix_len"] + w["suffix_len"] + w["output_len"]
        if total > s["max_model_len"]:
            raise RunError(f"{cond}: {total} tokens exceed max_model_len")
        if cond["concurrency"] > s["max_num_seqs"]:
            raise RunError(f"{cond}: concurrency exceeds max_num_seqs")


def conditions(spec: dict) -> list[dict]:
    """The conditions of an experiment, in the order the spec lists them."""
    c = spec["conditions"]
    if ("grid" in c) == ("list" in c):
        raise RunError("conditions needs exactly one of `grid` or `list`")
    if "grid" in c:
        keys = [k for k in CONDITION_KEYS if k in c["grid"]]
        unknown = set(c["grid"]) - set(CONDITION_KEYS)
        combos = itertools.product(*[c["grid"][k] for k in keys])
        raw = [dict(zip(keys, combo)) for combo in combos]
    else:
        unknown = set().union(*[set(item) for item in c["list"]]) - set(CONDITION_KEYS)
        raw = [dict(item) for item in c["list"]]
    if unknown:
        raise RunError(f"unknown condition keys: {sorted(unknown)}")
    out = []
    for item in raw:
        if "prefix_len" not in item or "concurrency" not in item:
            raise RunError(f"condition {item} needs prefix_len and concurrency")
        item.setdefault(
            "gpu_memory_utilization", spec["server"]["gpu_memory_utilization"]
        )
        out.append(item)
    ids = [condition_id(item) for item in out]
    if len(set(ids)) != len(ids):
        raise RunError("the spec lists the same condition twice")
    return out


def condition_id(cond: dict) -> str:
    mem = round(cond["gpu_memory_utilization"] * 100)
    return f"p{cond['prefix_len']:04d}_c{cond['concurrency']:03d}_g{mem:02d}"


def workload_key(spec: dict, prefix_len: int) -> str:
    w = spec["workload"]
    return (
        f"p{prefix_len:04d}_s{w['suffix_len']}_o{w['output_len']}"
        f"_n{w['num_prompts']}_w{w['warmup_prompts']}_seed{w['seed']}"
    )


def experiment_dirname(spec: dict) -> str:
    return f"{spec['name']}_b{spec['batch']}"


def first_on_patterns(n_cond: int, rounds: int, rng: random.Random) -> list[list[bool]]:
    """For every condition and round: does cache-on run first in the pair?

    Balanced both ways where the numbers allow it: each condition is on-first in
    floor(R/2) or ceil(R/2) of its R rounds, and each round has on-first in
    floor(C/2) or ceil(C/2) of its C conditions.
    """
    lo_r, hi_r = rounds // 2, (rounds + 1) // 2
    lo_c, hi_c = n_cond // 2, (n_cond + 1) // 2
    last = None
    for _ in range(50_000):
        table = []
        for _ in range(n_cond):
            k = rng.choice([lo_r, hi_r])
            on_rounds = set(rng.sample(range(rounds), k))
            table.append([r in on_rounds for r in range(rounds)])
        last = table
        per_round = [sum(row[r] for row in table) for r in range(rounds)]
        if all(lo_c <= n <= hi_c for n in per_round):
            return table
    return last  # balanced per condition only; the counts are in the plan


def build_plan(spec: dict) -> list[dict]:
    """The ordered list of runs. Deterministic for a given spec."""
    conds = conditions(spec)
    rng = random.Random(spec["order_seed"])
    patterns = first_on_patterns(len(conds), spec["rounds"], rng)
    plan = []
    for rnd in range(1, spec["rounds"] + 1):
        order = list(range(len(conds)))
        rng.shuffle(order)
        for position, ci in enumerate(order, 1):
            cond = conds[ci]
            pair_id = f"r{rnd}_{condition_id(cond)}"
            first = "on" if patterns[ci][rnd - 1] else "off"
            for cache in (first, "off" if first == "on" else "on"):
                plan.append(
                    {
                        "order": len(plan) + 1,
                        "round": rnd,
                        "position_in_round": position,
                        "pair_id": pair_id,
                        "first_in_pair": first,
                        "run_id": f"{pair_id}_{cache}",
                        "config": run_config(spec, cond, cache, rnd),
                    }
                )
    return plan


def run_config(spec: dict, cond: dict, cache: str, rnd: int) -> dict:
    """Every setting of one run, resolved. This is what the run identity hashes."""
    server = {
        **spec["server"],
        "gpu_memory_utilization": cond["gpu_memory_utilization"],
        "prefix_caching": cache == "on",
    }
    return {
        "experiment": spec["name"],
        "batch": spec["batch"],
        "run_id": f"r{rnd}_{condition_id(cond)}_{cache}",
        "pair_id": f"r{rnd}_{condition_id(cond)}",
        "condition_id": condition_id(cond),
        "round": rnd,
        "cache": cache,
        "model": spec["model"],
        "model_revision": spec["model_revision"],
        "prefix_len": cond["prefix_len"],
        "concurrency": cond["concurrency"],
        "gpu_memory_utilization": cond["gpu_memory_utilization"],
        "server": server,
        "workload": dict(spec["workload"]),
        "workload_key": workload_key(spec, cond["prefix_len"]),
        "bench": dict(spec["bench"]),
        "telemetry": dict(spec["telemetry"]),
    }


def plan_summary(spec: dict, plan: list[dict]) -> dict:
    w = spec["workload"]
    pairs = {p["pair_id"]: p["first_in_pair"] for p in plan}
    on_first_by_round: dict[int, int] = {}
    for p in plan:
        if p["config"]["cache"] == "on" and p["first_in_pair"] == "on":
            on_first_by_round[p["round"]] = on_first_by_round.get(p["round"], 0) + 1
    return {
        "conditions": len(conditions(spec)),
        "rounds": spec["rounds"],
        "pairs": len(pairs),
        "runs": len(plan),
        "measured_requests": len(plan) * w["num_prompts"],
        "warmup_requests": len(plan) * w["warmup_prompts"],
        "pairs_with_cache_on_first": sum(v == "on" for v in pairs.values()),
        "cache_on_first_by_round": on_first_by_round,
    }


def spec_sha256(spec: dict) -> str:
    return sha256_json(spec)
