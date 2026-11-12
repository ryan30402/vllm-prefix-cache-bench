"""Freeze an experiment, then run it pair by pair with safe resume.

freeze   writes the prompt files and manifest.json: the spec, the full ordered
         plan, the identity of every planned run, the metric definitions.
run      executes the plan under that manifest. It refuses to start if the
         measurement code or the software environment differs from the frozen
         one, so a result always belongs to exactly one protocol.

The unit of work is a pair (cache off and cache on of one condition, back to
back). A pair counts only if both runs of the same pair attempt are VALIDATED;
otherwise the whole pair is run again in new attempt folders. Earlier attempts
are kept, never overwritten.
"""

import os
import time
import traceback
from pathlib import Path

from . import PROTOCOL_VERSION, envinfo, ident, schema, spec as specmod, workload
from .bench import kill_recorded_bench, run_bench
from .metrics import Telemetry, snapshot
from .server import running_server, stop_recorded_server
from .store import (
    FAILED,
    INTERRUPTED,
    RUNNING,
    VALIDATED,
    ExperimentLock,
    RunError,
    SweepAbort,
    atomic_write_json,
    atomic_write_text,
    attempt_dirs,
    attempt_number,
    gzip_file,
    now,
    read_json,
    read_state,
    terminate_as_interrupt,
    write_state,
)

CORE_KEYS = [
    "protocol_version",
    "spec",
    "plan",
    "workloads",
    "env_identity",
    "measurement_code",
    "metric_definitions",
    "validity_rules",
]


def manifest_core_sha256(manifest: dict) -> str:
    return ident.sha256_json({k: manifest[k] for k in CORE_KEYS})


def freeze(
    spec_path: Path,
    results_root: Path,
    tok=None,
    allow_dirty: bool = False,
    env: dict | None = None,
) -> Path:
    """Write prompt files, environment.json and manifest.json for one experiment."""
    spec = specmod.load_spec(spec_path)
    exp_dir = Path(results_root) / specmod.experiment_dirname(spec)
    plan = specmod.build_plan(spec)
    env = env or envinfo.capture()
    if not allow_dirty:
        if not env["git_commit"]:
            raise RunError("not a git checkout: the source commit cannot be recorded")
        if env["source_dirty"]:
            raise RunError(
                "v3 source files differ from the last commit; commit them first:\n"
                + env["source_status"]
            )
    block = env.get("vllm_default_block_size")
    if block is not None and block != spec["workload"]["block_size"]:
        raise RunError(
            f"spec says block_size {spec['workload']['block_size']} but the installed "
            f"vLLM uses {block}"
        )
    code = ident.measurement_code()
    env_id = envinfo.env_identity(env)
    tok = tok or workload.load_tokenizer(spec["model"], spec["model_revision"])
    workloads: dict[str, dict] = {}
    for entry in plan:
        cfg = entry["config"]
        key = cfg["workload_key"]
        if key not in workloads:
            workloads[key] = workload.write_workload(
                tok, cfg, exp_dir / "workloads" / key
            )
            print(f"workload {key}: written and verified", flush=True)
        entry["identity_sha256"] = ident.run_identity(
            PROTOCOL_VERSION,
            cfg,
            workload.workload_identity(workloads[key]),
            env_id,
            code["sha256"],
        )["sha256"]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec,
        "plan": plan,
        "workloads": workloads,
        "env_identity": env_id,
        "measurement_code": code,
        "metric_definitions": schema.METRIC_DEFINITIONS,
        "validity_rules": schema.VALIDITY_RULES,
    }
    manifest["manifest_sha256"] = manifest_core_sha256(manifest)
    manifest.update(
        {
            "experiment": spec["name"],
            "batch": spec["batch"],
            "frozen_at": now(),
            "spec_file": str(spec_path),
            "spec_file_sha256": ident.sha256_file(Path(spec_path)),
            "git_commit": env["git_commit"],
            "allow_dirty": allow_dirty,
            "counts": specmod.plan_summary(spec, plan),
        }
    )
    path = exp_dir / "manifest.json"
    if path.is_file():
        old = read_json(path)
        if old.get("manifest_sha256") == manifest["manifest_sha256"]:
            print(f"{exp_dir} is already frozen with this exact protocol")
            return exp_dir
        changed = [k for k in CORE_KEYS if old.get(k) != manifest[k]]
        raise RunError(
            f"{exp_dir} is already frozen and {changed} differ now. A frozen "
            "experiment is never changed: raise `batch` in the spec to start a new one."
        )
    ignore = Path(results_root) / ".gitignore"
    if not ignore.is_file():
        # Prompt files are large and are regenerated from the manifest; their
        # hashes are in workload.json, which is committed.
        atomic_write_text(ignore, "*/workloads/*/*.jsonl\n*/runner.lock\n")
    atomic_write_json(exp_dir / "environment.json", env)
    atomic_write_json(path, manifest)
    print(f"frozen: {path}  ({manifest['counts']['runs']} runs planned)")
    return exp_dir


def load_manifest(exp_dir: Path) -> dict:
    path = Path(exp_dir) / "manifest.json"
    if not path.is_file():
        raise RunError(f"{path} not found: this is not a frozen v3 experiment")
    manifest = read_json(path)
    if not str(manifest.get("protocol_version", "")).startswith("3."):
        raise RunError(f"{path} is not a v3 manifest")
    if manifest_core_sha256(manifest) != manifest.get("manifest_sha256"):
        raise RunError(f"{path} was edited after it was frozen")
    return manifest


def pairs_in_order(manifest: dict) -> list[list[dict]]:
    pairs: dict[str, list[dict]] = {}
    for entry in manifest["plan"]:
        pairs.setdefault(entry["pair_id"], []).append(entry)
    return list(pairs.values())


def attempt_dir(exp_dir: Path, entry: dict, n: int) -> Path:
    return Path(exp_dir) / "runs" / entry["run_id"] / f"attempt_{n}"


def pair_state(exp_dir: Path, pair: list[dict]) -> dict:
    """Which attempt of a pair is complete (if any) and how many attempts failed."""
    numbers: set[int] = set()
    for entry in pair:
        for d in attempt_dirs(Path(exp_dir) / "runs" / entry["run_id"]):
            numbers.add(attempt_number(d))
    complete, failed = None, 0
    for n in sorted(numbers, reverse=True):
        dirs = [attempt_dir(exp_dir, e, n) for e in pair]
        states = [read_state(d)["state"] if d.is_dir() else None for d in dirs]
        if all(s == VALIDATED for s in states):
            problems = [
                p
                for d, e in zip(dirs, pair)
                for p in schema.validate_attempt(d, e["identity_sha256"])
            ]
            if not problems and complete is None:
                complete = n
                continue
            if problems:
                failed += 1
        elif FAILED in states:
            failed += 1
    return {"complete": complete, "failed": failed, "next": max(numbers, default=0) + 1}


def recover_interrupted(exp_dir: Path) -> list[str]:
    """Close attempts that a crash or kill left in state RUNNING.

    Their server may still be alive, so it is stopped first (only if the
    recorded process is still the same one).
    """
    notes = []
    for d in sorted(Path(exp_dir).glob("runs/*/attempt_*")):
        status = read_state(d)
        if status["state"] != RUNNING:
            continue
        outcome = "; ".join([stop_recorded_server(d), *kill_recorded_bench(d)])
        status = {k: v for k, v in status.items() if k != "unreadable"}
        status.update(
            state=INTERRUPTED,
            recovered_at=now(),
            error="the runner stopped before this attempt finished",
            server_cleanup=outcome,
        )
        write_state(d, status)
        notes.append(f"{d.parent.name}/{d.name}: interrupted earlier ({outcome})")
    return notes


def open_session(
    exp_dir: Path,
    manifest: dict,
    tok=None,
    accept_new_host: bool = False,
    env: dict | None = None,
) -> dict:
    """Check that this machine may continue the experiment, and record the session."""
    exp_dir = Path(exp_dir)
    spec = manifest["spec"]
    code = ident.measurement_code()
    if code["sha256"] != manifest["measurement_code"]["sha256"]:
        changed = [
            name
            for name, digest in code["files"].items()
            if manifest["measurement_code"]["files"].get(name) != digest
        ]
        raise RunError(
            f"the measurement code changed since this experiment was frozen ({changed}). "
            "Its results would not be comparable: raise `batch` in the spec and freeze again."
        )
    env = env or envinfo.capture()
    env_id = envinfo.env_identity(env)
    if env_id != manifest["env_identity"]:
        diff = {k: (manifest["env_identity"].get(k), v) for k, v in env_id.items()}
        diff = {k: v for k, v in diff.items() if v[0] != v[1]}
        raise RunError(
            f"the environment differs from the frozen one (frozen, now): {diff}. "
            "Install the frozen versions again, or raise `batch` in the spec to "
            "start a new experiment."
        )
    if env["source_dirty"] and not manifest["allow_dirty"]:
        raise RunError(
            "v3 source files differ from the last commit:\n" + env["source_status"]
        )
    sessions_path = exp_dir / "sessions.json"
    sessions = read_json(sessions_path) if sessions_path.is_file() else []
    host = envinfo.host_identity(env)
    if sessions and host != sessions[0]["host"] and not accept_new_host:
        raise RunError(
            "this is a different machine, GPU, driver or package set than session 1:\n"
            f"  session 1: {sessions[0]['host']}\n  now:       {host}\n"
            "Pairs never span sessions, so finished pairs stay valid, but the report "
            "must list both sessions. Pass --accept-new-host to continue."
        )
    tok = tok or workload.load_tokenizer(spec["model"], spec["model_revision"])
    for key, info in manifest["workloads"].items():
        facts = workload.verify_workload(tok, exp_dir / "workloads" / key, info)
        if any(info[k] != v for k, v in facts.items()):
            raise RunError(
                f"workload {key} no longer tokenizes as it did at freeze time"
            )
    session = {
        "session_id": len(sessions) + 1,
        "started_at": now(),
        "host": host,
        "git_commit": env["git_commit"],
        "telemetry_fields": env.get("telemetry_fields", []),
        "workloads_verified": len(manifest["workloads"]),
        "environment_file": f"environment_s{len(sessions) + 1}.json",
    }
    atomic_write_json(exp_dir / session["environment_file"], env)
    atomic_write_json(sessions_path, sessions + [session])
    return session


def run_attempt(
    exp_dir: Path, entry: dict, manifest: dict, session: dict, n: int
) -> None:
    """Execute one attempt of one run. Raises unless the attempt is VALIDATED."""
    cfg = entry["config"]
    wl = manifest["workloads"][cfg["workload_key"]]
    folder = Path(exp_dir) / "workloads" / cfg["workload_key"]
    d = attempt_dir(exp_dir, entry, n)
    d.mkdir(parents=True)
    t0 = time.time()
    status = {
        "state": RUNNING,
        "run_id": entry["run_id"],
        "pair_id": entry["pair_id"],
        "attempt": n,
        "order": entry["order"],
        "session_id": session["session_id"],
        "identity_sha256": entry["identity_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "runner_pid": os.getpid(),
        "started_at": now(),
    }
    write_state(d, status)
    try:
        atomic_write_json(
            d / "config.json",
            {
                "run": cfg,
                "workload": wl,
                "plan": {
                    k: entry[k]
                    for k in ("order", "round", "position_in_round", "first_in_pair")
                },
                "attempt": n,
                "session_id": session["session_id"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
        )
        atomic_write_json(
            d / "identity.json",
            ident.run_identity(
                PROTOCOL_VERSION,
                cfg,
                workload.workload_identity(wl),
                manifest["env_identity"],
                manifest["measurement_code"]["sha256"],
            ),
        )
        for kind in ("warmup", "main"):
            if ident.sha256_file(folder / f"{kind}.jsonl") != wl[f"{kind}_sha256"]:
                raise RunError(f"{folder}/{kind}.jsonl changed since it was verified")
        w, port = cfg["workload"], cfg["server"]["port"]
        with running_server(cfg, d) as server:
            tel = Telemetry(
                d, port, cfg["telemetry"]["interval_s"], session["telemetry_fields"]
            )
            tel.start()
            try:
                fresh = snapshot(port, d, "metrics_fresh")
                tel.mark("warmup_start")
                warm_rows, warm_agg = run_bench(
                    cfg,
                    folder / "warmup.jsonl",
                    w["warmup_prompts"],
                    d,
                    "warmup",
                    cfg["bench"]["warmup_timeout_s"],
                )
                tel.mark("warmup_end")
                atomic_write_text(
                    d / "warmup_requests.jsonl", schema.rows_text(warm_rows)
                )
                atomic_write_json(d / "bench_warmup.json", warm_agg)
                problems = schema.check_rows(
                    warm_rows, w["warmup_prompts"], cfg, wl, "warm-up"
                )
                if problems:
                    raise RunError("; ".join(problems))
                server.ensure_alive("during warm-up")
                # Counters are read after the warm-up, so the differences below
                # describe the measured requests and nothing else.
                before = snapshot(port, d, "metrics_before")
                tel.mark("main_start")
                rows, agg = run_bench(
                    cfg,
                    folder / "main.jsonl",
                    w["num_prompts"],
                    d,
                    "main",
                    cfg["bench"]["timeout_s"],
                )
                tel.mark("main_end")
                atomic_write_text(d / "requests.jsonl", schema.rows_text(rows))
                atomic_write_json(d / "bench_main.json", agg)
                server.ensure_alive("during the measured run")
                after = snapshot(port, d, "metrics_after")
            finally:
                atomic_write_json(d / "telemetry.json", tel.stop())
        atomic_write_json(
            d / "counters.json", schema.check_counters(cfg, wl, fresh, before, after)
        )
        atomic_write_json(
            d / "summary.json", schema.summarize(rows, cfg["concurrency"])
        )
        for name in ("bench_warmup.log", "bench_main.log"):
            gzip_file(d / name)
        problems = schema.check_content(d)
        if problems:
            raise RunError("; ".join(problems))
        # Commit: hash every file, then one atomic rename flips the state.
        atomic_write_json(d / "files.json", schema.files_manifest(d))
        status.update(
            state=VALIDATED,
            finished_at=now(),
            seconds=round(time.time() - t0),
            files_sha256=ident.sha256_file(d / "files.json"),
        )
        write_state(d, status)
    except BaseException as e:
        interrupted = isinstance(e, (KeyboardInterrupt, SystemExit))
        status.update(
            state=INTERRUPTED if interrupted else FAILED,
            finished_at=now(),
            seconds=round(time.time() - t0),
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        )
        for name in ("bench_warmup.log", "bench_main.log"):
            try:
                gzip_file(d / name)
            except OSError:
                pass
        write_state(d, status)
        raise


def run_experiment(
    exp_dir: Path,
    tok=None,
    accept_new_host: bool = False,
    env: dict | None = None,
    max_pairs: int | None = None,
) -> int:
    """Run or resume a frozen experiment. Returns the process exit code."""
    exp_dir = Path(exp_dir)
    manifest = load_manifest(exp_dir)
    max_attempts = manifest["spec"]["validity"]["max_pair_attempts"]
    pairs = pairs_in_order(manifest)
    with ExperimentLock(exp_dir), terminate_as_interrupt():
        for note in recover_interrupted(exp_dir):
            print("recovered: " + note)
        session = open_session(exp_dir, manifest, tok, accept_new_host, env)
        print(
            f"session {session['session_id']}: {len(pairs)} pairs in the plan",
            flush=True,
        )
        unfinished, ran = [], 0
        for i, pair in enumerate(pairs, 1):
            label = f"[pair {i}/{len(pairs)}] {pair[0]['pair_id']}"
            state = pair_state(exp_dir, pair)
            tries = 0
            # `tries` also bounds the loop when an attempt cannot even record
            # that it failed (for example a full disk).
            while (
                state["complete"] is None
                and state["failed"] < max_attempts
                and tries < max_attempts
            ):
                if max_pairs is not None and ran >= max_pairs:
                    break
                tries += 1
                n = state["next"]
                order = " then ".join(e["config"]["cache"] for e in pair)
                print(f"{label}: attempt {n} (cache {order})", flush=True)
                ran += 1
                for entry in pair:
                    try:
                        run_attempt(exp_dir, entry, manifest, session, n)
                    except SweepAbort as e:
                        print(f"ABORT: {e}\nLater runs could not be trusted.")
                        return 3
                    except RunError as e:
                        print(f"  FAILED {entry['run_id']}: {e}", flush=True)
                        break  # the pair is only valid as a whole
                    except Exception:
                        print(
                            f"  FAILED {entry['run_id']}:\n{traceback.format_exc()}",
                            flush=True,
                        )
                        break
                state = pair_state(exp_dir, pair)
            if state["complete"] is not None:
                print(f"{label}: complete (attempt {state['complete']})", flush=True)
            else:
                unfinished.append((pair[0]["pair_id"], state["failed"]))
    if unfinished:
        print(f"{len(unfinished)} of {len(pairs)} pairs have NO valid result:")
        for pair_id, failed in unfinished:
            print(f"  {pair_id}: {failed} failed attempt(s)")
        return 1
    print(f"All {len(pairs)} pairs have a valid result")
    return 0
