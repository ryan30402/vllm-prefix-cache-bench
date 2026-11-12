"""Command line for the v3 harness.

python v3/bench.py plan    v3/protocol/main.yaml
python v3/bench.py freeze  v3/protocol/main.yaml
python v3/bench.py run     results_v3/main_b1
python v3/bench.py status  results_v3/main_b1
python v3/bench.py verify  results_v3/main_b1
python v3/bench.py analyze results_v3/main_b1
"""

import argparse
from pathlib import Path

from . import analyze as analyze_mod, runner, schema, spec as specmod
from .store import RunError, attempt_dirs, read_state


def cmd_plan(args) -> int:
    spec = specmod.load_spec(args.spec)
    plan = specmod.build_plan(spec)
    for key, value in specmod.plan_summary(spec, plan).items():
        print(f"{key}: {value}")
    if args.full:
        for e in plan:
            print(f"{e['order']:4d}  round {e['round']}  {e['run_id']}")
    return 0


def cmd_freeze(args) -> int:
    runner.freeze(
        Path(args.spec), Path(args.results_root), allow_dirty=args.allow_dirty
    )
    return 0


def cmd_run(args) -> int:
    return runner.run_experiment(
        Path(args.experiment), accept_new_host=args.accept_new_host
    )


def cmd_status(args) -> int:
    exp = Path(args.experiment)
    manifest = runner.load_manifest(exp)
    pairs = runner.pairs_in_order(manifest)
    done = failed = 0
    for pair in pairs:
        state = runner.pair_state(exp, pair)
        done += state["complete"] is not None
        failed += state["failed"]
    print(f"{exp}: {done} of {len(pairs)} pairs complete, {failed} failed attempt(s)")
    return 0 if done == len(pairs) else 1


def cmd_verify(args) -> int:
    exp = Path(args.experiment)
    manifest = runner.load_manifest(exp)
    bad = 0
    for entry in manifest["plan"]:
        for d in attempt_dirs(exp / "runs" / entry["run_id"]):
            if read_state(d)["state"] != "VALIDATED":
                continue
            problems = schema.validate_attempt(d, entry["identity_sha256"])
            if problems:
                bad += 1
                print(f"INVALID {entry['run_id']}/{d.name}: {problems[0]}")
    code = cmd_status(args)
    print(
        "no VALIDATED attempt fails validation"
        if not bad
        else f"{bad} INVALID attempt(s)"
    )
    return 1 if bad else code


def cmd_analyze(args) -> int:
    audit = analyze_mod.analyze(Path(args.experiment))
    print(
        f"{audit['complete_pairs']} of {audit['planned']['pairs']} pairs in the tables; "
        f"attempts by state: {audit['attempts_by_state']}"
    )
    if audit["missing_pairs"]:
        print(f"MISSING: {[m['pair_id'] for m in audit['missing_pairs']]}")
    return 0 if audit["complete"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v3 prefix-caching benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="show the run plan of a spec (no GPU needed)")
    p.add_argument("spec")
    p.add_argument("--full", action="store_true", help="list every run in order")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("freeze", help="write prompt files and the frozen manifest")
    p.add_argument("spec")
    p.add_argument("--results-root", default="results_v3")
    p.add_argument("--allow-dirty", action="store_true", help="for tests only")
    p.set_defaults(func=cmd_freeze)
    p = sub.add_parser("run", help="run or resume a frozen experiment")
    p.add_argument("experiment")
    p.add_argument("--accept-new-host", action="store_true")
    p.set_defaults(func=cmd_run)
    for name, func, text in (
        ("status", cmd_status, "how many pairs are complete"),
        ("verify", cmd_verify, "re-validate every finished attempt"),
        ("analyze", cmd_analyze, "write runs.csv, pairs.csv, summary.csv, audit.json"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("experiment")
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RunError as e:
        print(f"ERROR: {e}")
        return 2
    except KeyboardInterrupt:
        print("interrupted")
        return 130
