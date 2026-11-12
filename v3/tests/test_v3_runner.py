"""Start -> warm-up -> measure -> save -> resume -> analyze, and what happens
when each step is made to fail. Everything runs against the fake vLLM."""

import csv
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from v3_helpers import V3, WordTok, attempts, fake_env, freeze_and_run, write_spec

from pcbench import analyze, envinfo, ident, runner, schema, server, store
from pcbench.store import RunError


def states(exp) -> list[str]:
    return [
        f"{d.parent.name[-3:].strip('_')}/{d.name}:{store.read_state(d)['state']}"
        for d in attempts(exp)
    ]


def port_of(exp) -> int:
    return runner.load_manifest(exp)["spec"]["server"]["port"]


def rows_of(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_full_path_and_contract_numbers(tmp_path):
    exp, code = freeze_and_run(tmp_path, rounds=2)
    assert code == 0
    assert not server.port_in_use(port_of(exp))
    manifest = runner.load_manifest(exp)
    assert manifest["counts"]["runs"] == 4 and manifest["protocol_version"] == "3.0.0"
    for d in attempts(exp):
        assert schema.validate_attempt(d) == []
        counters = store.read_json(d / "counters.json")
        cache_on = d.parent.name.endswith("_on")
        # counters cover the 6 measured requests only, not the 3 warm-up ones
        assert counters["deltas"]["request_success"] == 6
        assert counters["warmup_deltas"]["request_success"] == 3
        assert counters["deltas"]["prompt_tokens"] == 6 * 48
        assert counters["deltas"]["generation_tokens"] == 6 * 4
        assert counters["hit_check"] == "exact"
        assert counters["deltas"]["prefix_cache_hits"] == (6 * 32 if cache_on else 0)
        assert counters["cache_hit_rate"] == pytest.approx(32 / 48 if cache_on else 0)
        tel = store.read_json(d / "telemetry.json")
        assert tel["gpu_samples_total"] >= 1 and tel["gpu_uuids"] == ["GPU-fake-0000"]
        assert tel["server_samples_total"] >= 1
        assert {"warmup_start", "main_start", "main_end"} <= set(tel["marks"])
        info = store.read_json(d / "server.json")
        assert info["kv_cache_size_tokens"] == 459936
        assert (
            "--generation-config" in info["cmd"]
            and "--max-num-batched-tokens" in info["cmd"]
        )
        assert ("--enable-prefix-caching" in info["cmd"]) == cache_on
        assert not list(d.glob("*.tmp")) and not (d / "server.log").exists()
    audit = analyze.analyze(exp)
    assert audit["complete"] and audit["complete_pairs"] == 2
    assert (
        audit["measured_requests_in_tables"] == 24
        and audit["failed_requests_in_tables"] == 0
    )
    assert audit["pairs_spanning_sessions"] == [] and audit["distinct_hosts"] == 1
    pairs = rows_of(exp / "analysis" / "pairs.csv")
    summary = rows_of(exp / "analysis" / "summary.csv")
    speedups = [float(p["output_tok_per_s_on_over_off"]) for p in pairs]
    for p, s in zip(pairs, speedups):
        assert s == pytest.approx(
            float(p["output_tok_per_s_on"]) / float(p["output_tok_per_s_off"])
        )
    # primary metric: geometric mean of the pair ratios, not a ratio of medians
    assert float(summary[0]["speedup_geomean"]) == pytest.approx(
        math.sqrt(speedups[0] * speedups[1])
    )
    assert (
        summary[0]["pairs"] == "2" and len(rows_of(exp / "analysis" / "runs.csv")) == 4
    )


def test_memory_conditions_are_never_merged_in_the_summary(tmp_path):
    conds = {
        "grid": {
            "prefix_len": [32],
            "concurrency": [2],
            "gpu_memory_utilization": [0.9, 0.5],
        }
    }
    exp, code = freeze_and_run(tmp_path, conditions=conds)
    assert code == 0
    analyze.analyze(exp)
    summary = rows_of(exp / "analysis" / "summary.csv")
    assert sorted(r["gpu_memory_utilization"] for r in summary) == ["0.5", "0.9"]
    assert {r["pairs"] for r in summary} == {"1"}
    for d in attempts(exp):
        want = "0.5" if "_g50_" in d.parent.name else "0.9"
        cmd = store.read_json(d / "server.json")["cmd"]
        assert cmd[cmd.index("--gpu-memory-utilization") + 1] == want


def test_frozen_experiment_cannot_be_changed_in_place(tmp_path):
    env = fake_env()
    spec = write_spec(tmp_path)
    freeze = lambda: runner.freeze(
        spec, tmp_path / "r", tok=WordTok(), allow_dirty=True, env=env
    )
    exp = freeze()
    first = (exp / "manifest.json").read_bytes()
    assert "workloads/*/*.jsonl" in (tmp_path / "r" / ".gitignore").read_text()
    assert (
        freeze() == exp and (exp / "manifest.json").read_bytes() == first
    )  # same spec: no-op
    text = spec.read_text()
    spec.write_text(text.replace("num_prompts: 6", "num_prompts: 7"))
    with pytest.raises(RunError, match="already frozen"):
        freeze()
    spec.write_text(spec.read_text().replace("batch: 1", "batch: 2"))
    assert freeze().name == "toy_b2"  # a new batch gets a new folder and new identities
    old = {e["run_id"]: e["identity_sha256"] for e in runner.load_manifest(exp)["plan"]}
    new = {
        e["run_id"]: e["identity_sha256"]
        for e in runner.load_manifest(tmp_path / "r" / "toy_b2")["plan"]
    }
    assert old.keys() == new.keys() and all(old[k] != new[k] for k in old)


def test_freeze_needs_a_clean_committed_source_tree(tmp_path):
    for env in (
        fake_env(source_dirty=True, source_status=" M v3/pcbench/bench.py"),
        fake_env(git_commit=""),
    ):
        with pytest.raises(RunError):
            runner.freeze(write_spec(tmp_path), tmp_path / "r", tok=WordTok(), env=env)
    with pytest.raises(RunError, match="block_size"):
        runner.freeze(
            write_spec(tmp_path),
            tmp_path / "r",
            tok=WordTok(),
            allow_dirty=True,
            env=fake_env(vllm_default_block_size=32),
        )


def test_edited_manifest_is_refused(tmp_path):
    exp, _ = freeze_and_run(tmp_path)
    manifest = store.read_json(exp / "manifest.json")
    manifest["spec"]["workload"]["num_prompts"] = 5
    store.atomic_write_json(exp / "manifest.json", manifest)
    with pytest.raises(RunError, match="edited after"):
        runner.run_experiment(exp, tok=WordTok(), env=fake_env())
    with pytest.raises(RunError, match="edited after"):
        analyze.analyze(exp)


def test_changed_measurement_code_stops_the_run(tmp_path, monkeypatch):
    exp, _ = freeze_and_run(tmp_path)
    real = ident.measurement_code()
    changed = {"files": {**real["files"], "bench.py": "0" * 64}, "sha256": "f" * 64}
    monkeypatch.setattr(ident, "measurement_code", lambda: changed)
    with pytest.raises(RunError, match=r"measurement code changed.*bench\.py"):
        runner.run_experiment(exp, tok=WordTok(), env=fake_env())


def test_changed_environment_stops_the_run(tmp_path):
    exp, _ = freeze_and_run(tmp_path)
    with pytest.raises(RunError, match="environment differs"):
        runner.run_experiment(exp, tok=WordTok(), env=fake_env(vllm="9.9.9"))
    with pytest.raises(RunError, match="environment differs"):
        runner.run_experiment(exp, tok=WordTok(), env=fake_env(torch="9.9.9"))
    # a different package set with the same key versions is a new session
    other = fake_env(pip_freeze_sha256="1" * 64)
    with pytest.raises(RunError, match="different machine"):
        runner.run_experiment(exp, tok=WordTok(), env=other)
    assert (
        runner.run_experiment(exp, tok=WordTok(), env=other, accept_new_host=True) == 0
    )


def test_another_machine_needs_an_explicit_yes_and_is_recorded(tmp_path):
    exp, _ = freeze_and_run(tmp_path)
    other = fake_env(uuid="GPU-fake-9999")
    with pytest.raises(RunError, match="different machine"):
        runner.run_experiment(exp, tok=WordTok(), env=other)
    assert (
        runner.run_experiment(exp, tok=WordTok(), env=other, accept_new_host=True) == 0
    )
    sessions = store.read_json(exp / "sessions.json")
    assert [s["host"]["gpu_uuids"] for s in sessions] == [
        ["GPU-fake-0000"],
        ["GPU-fake-9999"],
    ]
    assert analyze.analyze(exp)["distinct_hosts"] == 2


def test_analyzer_reads_only_frozen_v3_experiments(tmp_path):
    legacy = tmp_path / "results_v2"
    (legacy / "clean_p2048_c32_cache_on_r0__abc" / "attempt_1").mkdir(parents=True)
    with pytest.raises(RunError, match="not a frozen v3 experiment"):
        analyze.analyze(legacy)
    store.atomic_write_json(legacy / "manifest.json", {"protocol_version": "2.0"})
    with pytest.raises(RunError, match="not a v3 manifest"):
        analyze.analyze(legacy)
    repo_root = V3.parent
    assert analyze.__file__ and "results_v2" not in Path(
        analyze.__file__
    ).read_text().replace("v1 or v2 results", "")
    assert (repo_root / "v3").is_dir()


def test_a_transient_failure_is_retried_as_a_whole_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_FAIL_ONCE", str(tmp_path / "marker"))
    exp, code = freeze_and_run(tmp_path)
    assert code == 0
    first = (
        "on"
        if runner.load_manifest(exp)["plan"][0]["config"]["cache"] == "on"
        else "off"
    )
    by_state = [
        f"{d.parent.name.rsplit('_', 1)[1]}/{d.name}:{store.read_state(d)['state']}"
        for d in attempts(exp)
    ]
    # attempt 1: the first run failed, so its partner was not run at all
    assert f"{first}/attempt_1:FAILED" in by_state
    assert sorted(by_state) == sorted(
        [
            f"{first}/attempt_1:FAILED",
            "off/attempt_2:VALIDATED",
            "on/attempt_2:VALIDATED",
        ]
    )
    failed = next(d for d in attempts(exp, "FAILED"))
    assert "exited with code 1" in store.read_state(failed)["error"]
    audit = analyze.analyze(exp)
    assert (
        audit["attempts_by_state"] == {"FAILED": 1, "VALIDATED": 2}
        and audit["complete"]
    )


FAILURES = {
    "bench exits with an error": ({"FAKE_BENCH_EXIT_CODE": "3"}, "exited with code 3"),
    "requests fail (HTTP 500)": ({"FAKE_SERVE_HTTP_500_AFTER": "5"}, "requests failed"),
    "output shorter than agreed": (
        {"FAKE_BENCH_SHORT_OUTPUT": "1"},
        "generated 3 tokens",
    ),
    "per-request field missing": (
        {"FAKE_BENCH_DROP_FIELD": "ttfts"},
        "`ttfts` is missing",
    ),
    "result file cut short": ({"FAKE_BENCH_TRUNCATE": "1"}, "not valid JSON"),
    "counter reset mid-run": ({"FAKE_SERVE_RESET_AFTER": "8"}, "went down"),
    "counter reset hidden by later requests": (
        {"FAKE_SERVE_RESET_AFTER": "5"},
        "expected 6",
    ),
    "metric missing": ({"FAKE_SERVE_DROP_METRIC": "num_preemptions"}, "missing"),
    "wrong model on the port": ({"FAKE_SERVE_WRONG_MODEL": "1"}, "expected fake/model"),
    "server dies at start": ({"FAKE_SERVE_EXIT_CODE": "7"}, "exited with code 7"),
}


@pytest.mark.parametrize("name", FAILURES)
def test_a_failing_run_is_recorded_and_never_counted(tmp_path, monkeypatch, name):
    env_vars, message = FAILURES[name]
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    exp, code = freeze_and_run(tmp_path)
    assert code == 1
    assert not server.port_in_use(port_of(exp))
    found = attempts(exp)
    assert len(found) == 2  # max_pair_attempts = 2, the partner run is never started
    for d in found:
        status = store.read_state(d)
        assert status["state"] == "FAILED" and message in status["error"], status[
            "error"
        ]
        assert schema.validate_attempt(d)
    audit = analyze.analyze(exp)
    assert not audit["complete"] and audit["complete_pairs"] == 0
    assert audit["missing_pairs"] == [
        {"pair_id": "r1_p0032_c002_g90", "failed_attempts": 2}
    ]
    assert rows_of(exp / "analysis" / "runs.csv") == []
    # a later invocation does not keep retrying once the limit is reached
    assert runner.run_experiment(exp, tok=WordTok(), env=fake_env()) == 1
    assert len(attempts(exp)) == 2


def test_failed_requests_stay_on_record(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SERVE_HTTP_500_AFTER", "5")
    exp, _ = freeze_and_run(tmp_path)
    d = attempts(exp)[0]
    rows = schema.read_rows(d / "requests.jsonl")
    assert len(rows) == 6  # every request has a record, failed or not
    assert [r["ok"] for r in rows] == [True, True, False, False, False, False]
    assert rows[-1]["error"] == "HTTP 500"


def test_timeout_kills_the_benchmark_and_the_server(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_SLEEP", "30")
    t0 = time.time()
    exp, code = freeze_and_run(
        tmp_path,
        bench={"timeout_s": 1, "warmup_timeout_s": 1},
        validity={"max_pair_attempts": 1},
    )
    assert code == 1 and time.time() - t0 < 20
    assert "exceeded 1 s" in store.read_state(attempts(exp)[0])["error"]
    assert not server.port_in_use(port_of(exp))


def test_missing_gpu_telemetry_fails_a_run_that_requires_it(tmp_path):
    env = fake_env(fields=[])
    exp, code = freeze_and_run(tmp_path, env=env, validity={"max_pair_attempts": 1})
    assert code == 1
    assert "no GPU telemetry" in store.read_state(attempts(exp)[0])["error"]
    exp2, code2 = freeze_and_run(
        tmp_path / "b", env=env, name="notel", telemetry={"required": False}
    )
    assert code2 == 0


def test_a_port_in_use_is_refused_and_its_owner_left_alone(tmp_path):
    spec = write_spec(tmp_path, validity={"max_pair_attempts": 1})
    exp = runner.freeze(
        spec, tmp_path / "r", tok=WordTok(), allow_dirty=True, env=fake_env()
    )
    with socket.socket() as other:
        other.bind(("127.0.0.1", port_of(exp)))
        other.listen()
        assert runner.run_experiment(exp, tok=WordTok(), env=fake_env()) == 1
        assert server.port_in_use(port_of(exp))  # the other service still answers
    assert "already in use" in store.read_state(attempts(exp)[0])["error"]
    assert not (attempts(exp)[0] / "server.json").exists()


def test_server_is_stopped_when_writing_its_metadata_fails(tmp_path, monkeypatch):
    spec = write_spec(tmp_path, validity={"max_pair_attempts": 1})
    exp = runner.freeze(
        spec, tmp_path / "r", tok=WordTok(), allow_dirty=True, env=fake_env()
    )
    started = []
    real_popen = subprocess.Popen

    def recording_popen(cmd, *a, **kw):
        proc = real_popen(cmd, *a, **kw)
        if cmd[:2] == ["vllm", "serve"]:
            started.append(proc)
        return proc

    def failing_write(path, obj):
        raise OSError("No space left on device")

    monkeypatch.setattr(server.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(server, "atomic_write_json", failing_write)
    assert runner.run_experiment(exp, tok=WordTok(), env=fake_env()) == 1
    assert len(started) == 1 and started[0].poll() is not None  # started, then stopped
    assert server.group_members(started[0].pid) == []
    assert not server.port_in_use(port_of(exp))
    assert "No space left" in store.read_state(attempts(exp)[0])["error"]


def test_a_server_that_ignores_sigint_is_killed(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SERVE_IGNORE_SIGINT", "1")
    real = server.stop_process_group
    monkeypatch.setattr(
        server,
        "stop_process_group",
        lambda pgid, proc=None: real(pgid, proc, grace_s=1),
    )
    exp, code = freeze_and_run(tmp_path)
    assert code == 0 and not server.port_in_use(port_of(exp))


# Write 1-3 belong to the session, 4-27 to the first run, 28+ to its partner.
@pytest.mark.parametrize(
    "fail_at", [1, 3, 4, 5, 7, 8, 10, 12, 15, 20, 21, 23, 26, 27, 28, 40]
)
def test_interrupting_any_write_never_leaves_a_false_success(
    tmp_path, monkeypatch, fail_at
):
    spec = write_spec(tmp_path)
    exp = runner.freeze(
        spec, tmp_path / "r", tok=WordTok(), allow_dirty=True, env=fake_env()
    )
    calls = {"n": 0}
    real_bytes = store.atomic_write_bytes

    def flaky(path, data):
        calls["n"] += 1
        if calls["n"] == fail_at:
            raise OSError(f"interrupted at write {fail_at}")
        return real_bytes(path, data)

    monkeypatch.setattr(store, "atomic_write_bytes", flaky)
    try:
        runner.run_experiment(exp, tok=WordTok(), env=fake_env())
    except OSError:
        pass  # the runner itself died; nothing may be left behind either way
    assert calls["n"] >= fail_at
    monkeypatch.setattr(store, "atomic_write_bytes", real_bytes)
    assert not server.port_in_use(port_of(exp))
    for d in attempts(exp):
        state = store.read_state(d)["state"]
        assert state in ("VALIDATED", "FAILED", "RUNNING")
        if state == "VALIDATED":
            assert schema.validate_attempt(d) == []
        else:
            assert schema.validate_attempt(d)
    # resume: only complete, valid pairs count; the rest is run again
    assert runner.run_experiment(exp, tok=WordTok(), env=fake_env()) == 0
    assert analyze.analyze(exp)["complete"]


def test_killed_runner_leaves_no_server_and_no_false_result(tmp_path):
    spec = write_spec(tmp_path)
    exp = runner.freeze(
        spec, tmp_path / "r", tok=WordTok(), allow_dirty=True, env=fake_env()
    )
    port = port_of(exp)
    child = subprocess.Popen(
        [sys.executable, str(V3 / "tests" / "run_child.py"), str(exp)],
        env={**os.environ, "FAKE_BENCH_SLEEP": "60"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline and not list(
        exp.glob("runs/*/attempt_1/bench_warmup.proc.json")
    ):
        time.sleep(0.1)
    assert server.port_in_use(port)
    child.send_signal(signal.SIGKILL)  # the runner dies without any cleanup
    child.wait()
    orphan = store.read_json(attempts(exp)[0] / "server.json")["pid"]
    assert server.port_in_use(port) and server.group_members(orphan)
    assert store.read_state(attempts(exp)[0])["state"] == "RUNNING"
    bench_pid = store.read_json(attempts(exp)[0] / "bench_warmup.proc.json")["pid"]
    assert server.group_members(bench_pid)  # the load generator is orphaned too

    assert runner.run_experiment(exp, tok=WordTok(), env=fake_env()) == 0
    assert server.group_members(orphan) == [] and not server.port_in_use(port)
    assert server.group_members(bench_pid) == []
    first = attempts(exp)[0]
    status = store.read_state(first)
    assert status["state"] == "INTERRUPTED" and "stopped" in status["server_cleanup"]
    assert "benchmark process" in status["server_cleanup"]
    assert len(attempts(exp, "VALIDATED")) == 2
    audit = analyze.analyze(exp)
    assert audit["complete"] and audit["attempts_by_state"] == {
        "INTERRUPTED": 1,
        "VALIDATED": 2,
    }


def test_recovery_does_not_signal_a_reused_pid(tmp_path):
    d = tmp_path / "attempt_1"
    d.mkdir()
    store.atomic_write_json(
        d / "server.json",
        {"pid": os.getpid(), "pgid": os.getpid(), "port": 1, "proc_start_time": "1"},
    )
    assert "reused" in server.stop_recorded_server(d)  # we are still alive


def test_real_environment_capture_has_the_fields_the_harness_needs():
    env = envinfo.capture()
    assert env["gpus"] == [
        {
            "name": "Fake GPU",
            "uuid": "GPU-fake-0000",
            "memory_total": "46068 MiB",
            "driver_version": "0.0",
        }
    ]
    assert (
        "temperature.gpu" in env["telemetry_fields"]
        and "uuid" in env["telemetry_fields"]
    )
    assert len(env["pip_freeze_sha256"]) == 64
    assert envinfo.env_identity(env)["gpu_models"] == ["Fake GPU"]
    assert envinfo.host_identity(env)["gpu_uuids"] == ["GPU-fake-0000"]


def test_unknown_telemetry_fields_are_left_out_not_fatal(monkeypatch):
    monkeypatch.setenv("FAKE_SMI_BAD_FIELD", "power.draw")
    fields = envinfo.telemetry_fields()
    assert "power.draw" not in fields and "temperature.gpu" in fields


def test_command_line_end_to_end(tmp_path, monkeypatch, capsys):
    from pcbench import cli, workload

    monkeypatch.setattr(workload, "load_tokenizer", lambda model, revision: WordTok())
    monkeypatch.setattr(envinfo, "capture", fake_env)
    spec = write_spec(tmp_path)
    root = str(tmp_path / "results_v3")
    assert cli.main(["plan", str(spec), "--full"]) == 0
    assert cli.main(["freeze", str(spec), "--results-root", root, "--allow-dirty"]) == 0
    exp = str(tmp_path / "results_v3" / "toy_b1")
    assert cli.main(["status", exp]) == 1
    assert cli.main(["run", exp]) == 0
    assert cli.main(["status", exp]) == 0
    assert cli.main(["verify", exp]) == 0
    assert cli.main(["analyze", exp]) == 0
    assert cli.main(["run", str(tmp_path / "nothing")]) == 2
    out = capsys.readouterr().out
    assert "1 of 1 pairs complete" in out and "ERROR:" in out
    (attempts(exp)[0] / "summary.json").write_text("{}")
    assert cli.main(["verify", exp]) == 1


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_a_terminated_runner_cleans_up_before_it_exits(tmp_path, sig):
    spec = write_spec(tmp_path)
    exp = runner.freeze(
        spec, tmp_path / "r", tok=WordTok(), allow_dirty=True, env=fake_env()
    )
    child = subprocess.Popen(
        [sys.executable, str(V3 / "tests" / "run_child.py"), str(exp)],
        env={**os.environ, "FAKE_BENCH_SLEEP": "60"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline and not list(
        exp.glob("runs/*/attempt_1/bench_warmup.proc.json")
    ):
        time.sleep(0.1)
    d = attempts(exp)[0]
    server_pid = store.read_json(d / "server.json")["pid"]
    bench_pid = store.read_json(d / "bench_warmup.proc.json")["pid"]
    child.send_signal(sig)
    child.wait(timeout=90)
    assert (
        server.group_members(server_pid) == [] and server.group_members(bench_pid) == []
    )
    assert not server.port_in_use(port_of(exp))
    assert store.read_state(d)["state"] == "INTERRUPTED"
    assert not (exp / "runner.lock").exists()
    assert len(attempts(exp)) == 1  # it stopped; it did not go on to the next run
