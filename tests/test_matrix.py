"""The whole sweep against the fake vLLM: isolation, resume, failures, exit codes."""

import json
from pathlib import Path

import pytest
import yaml
from conftest import FAKE_ENV, MODEL, WordTok

import capture_env
import run_matrix
from load_results_v2 import load_all
from make_workload import distinct_workloads, write_workload
from stop_server import port_in_use


@pytest.fixture
def sweep(matrix, tmp_path, monkeypatch):
    """Returns run(**changes) -> exit code, plus the paths it uses."""
    monkeypatch.setattr(capture_env, "capture", lambda model=None: dict(FAKE_ENV))
    workloads = tmp_path / "workloads"
    for model, bench in distinct_workloads(run_matrix.expand(matrix)):
        write_workload(WordTok(), model, bench, workloads, {"name": MODEL})
    results = tmp_path / "results_v2"
    config = tmp_path / "matrix.yaml"

    def run(extra_args=(), **changes) -> int:
        m = json.loads(json.dumps(matrix))
        for dotted, value in changes.items():
            section, key = dotted.split("__")
            m[section][key] = value
        config.write_text(yaml.safe_dump(m))
        args = [str(config), "--results-root", str(results)]
        args += ["--workload-root", str(workloads), *extra_args]
        code = run_matrix.main(args)
        assert not port_in_use(matrix["server_base"]["port"])  # nothing left running
        return code

    run.results = results
    run.workloads = workloads
    return run


def attempts(results: Path) -> list[str]:
    return sorted(f"{a.parent.name}/{a.name}" for a in results.glob("*/attempt_*"))


def status_of(results: Path, fragment: str, attempt: str = "attempt_1") -> dict:
    (run_dir,) = [d for d in results.iterdir() if fragment in d.name]
    return json.loads((run_dir / attempt / "status.json").read_text())


def test_counters_cover_exactly_the_measured_requests(sweep):
    assert sweep() == 0
    df, invalid = load_all(sweep.results)
    assert invalid == [] and len(df) == 4
    rows = {(r.prefix_len, r.cache): r for r in df.itertuples()}

    # 6 measured requests each. The 3 warm-up requests are not in the counters.
    for d in sweep.results.glob("*/attempt_1"):
        checks = json.loads((d / "checks.json").read_text())
        assert checks["request_success_delta"] == 6
        assert checks["prompt_tokens_delta"] == checks["prompt_tokens_expected"]
        assert checks["fresh_request_success"] == 0

    # Hot shared prefix: every measured request hits the 32 prefix tokens of 48.
    assert rows[(32, "on")].cache_hit_rate == pytest.approx(32 / 48)
    # No shared prefix and unique prompts: not a single cached token.
    assert rows[(0, "on")].cache_hit_rate == 0.0
    assert rows[(32, "off")].cache_hit_rate == 0.0
    assert set(df.n_requests) == {6} and set(df.preemptions) == {0.0}
    assert df.kv_cache_size_tokens.eq(459936).all()
    assert {"ttft_p95_ms", "e2e_p99_ms", "tpot_p50_ms"} <= set(df.columns)


def test_a_finished_sweep_is_skipped_and_a_changed_setting_is_not(sweep, capsys):
    assert sweep() == 0
    first = attempts(sweep.results)
    assert len(first) == 4

    assert sweep() == 0
    assert attempts(sweep.results) == first  # nothing ran again
    assert capsys.readouterr().out.count("(complete)") == 4

    # Same workload, different server setting: must not reuse the old results.
    assert sweep(server_base__gpu_memory_utilization=0.5) == 0
    assert len(attempts(sweep.results)) == 8
    assert "different config or environment" in capsys.readouterr().out
    assert sweep(server_base__max_model_len=16384) == 0
    assert len(attempts(sweep.results)) == 12


def test_a_run_with_a_missing_file_gets_a_new_attempt(sweep):
    assert sweep() == 0
    victim = sorted(sweep.results.glob("*/attempt_1"))[0]
    (victim / "metrics_after.json").unlink()

    assert sweep() == 0
    after = attempts(sweep.results)
    assert len(after) == 5
    assert f"{victim.parent.name}/attempt_2" in after
    assert (victim / "status.json").exists()  # the old attempt is kept as it was


def test_failed_requests_fail_the_sweep_and_a_retry_recovers(sweep, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_FAIL", "1")
    assert sweep() == 1
    status = status_of(sweep.results, "p0_c2_cache_off")
    assert status["ok"] is False and "requests completed" in status["error"]
    df, invalid = load_all(sweep.results)
    assert len(df) == 0 and len(invalid) == 4  # nothing is reported as a result

    monkeypatch.delenv("FAKE_BENCH_FAIL")
    assert sweep() == 0
    assert len(attempts(sweep.results)) == 8
    df, invalid = load_all(sweep.results)
    assert len(df) == 4 and invalid == [] and set(df.n_attempts) == {2}


def test_a_missing_metric_fails_the_run_instead_of_reading_as_zero(sweep, monkeypatch):
    monkeypatch.setenv("FAKE_SERVE_DROP_METRIC", "num_preemptions")
    assert sweep() == 1
    status = status_of(sweep.results, "p32_c2_cache_on")
    assert status["ok"] is False and "num_preemptions" in status["error"]


def test_a_crashing_bench_or_server_fails_the_run(sweep, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_EXIT_CODE", "7")
    assert sweep() == 1
    assert "exited with code 7" in status_of(sweep.results, "p0_c2_cache_on")["error"]

    monkeypatch.delenv("FAKE_BENCH_EXIT_CODE")
    monkeypatch.setenv("FAKE_SERVE_EXIT_CODE", "3")
    assert sweep() == 1
    status = status_of(sweep.results, "p0_c2_cache_on", "attempt_2")
    assert "exited with code 3" in status["error"]


def test_uncommitted_source_changes_block_the_sweep_unless_allowed(sweep, monkeypatch):
    dirty = {**FAKE_ENV, "source_dirty": True, "source_status": " M scripts/x.py"}
    monkeypatch.setattr(capture_env, "capture", lambda model=None: dict(dirty))
    assert sweep() == 2
    assert not sweep.results.exists()

    assert sweep(["--allow-dirty"]) == 0
    (env_file,) = list(sweep.results.glob("*p0_c2_cache_on*/attempt_1/env.json"))
    assert json.loads(env_file.read_text())["source_dirty"] is True


def test_the_sweep_does_not_start_without_a_revision_or_a_workload(sweep, monkeypatch):
    assert sweep(bench_base__num_prompts=7) == 2  # no prompt file for this size

    unpinned = {**FAKE_ENV, "model_revision": ""}
    monkeypatch.setattr(capture_env, "capture", lambda model=None: dict(unpinned))
    assert sweep() == 2


def test_run_order_is_sequential_by_default_and_paired_when_asked(matrix):
    names = [c["name"] for c in run_matrix.expand(matrix)]
    assert names[0] == "p0_c2_cache_off_r0" and len(names) == 4

    matrix.update(order="paired_random", order_seed=1, repeats=3, tag="clean")
    runs = run_matrix.expand(matrix)
    assert runs == run_matrix.expand(matrix)  # same seed, same order
    assert len(runs) == 12 and all(c["name"].startswith("clean_") for c in runs)
    for a, b in zip(runs[0::2], runs[1::2]):  # cache off/on always back to back
        assert a["bench"] == b["bench"] and a["repeat"] == b["repeat"]
        assert a["server"]["prefix_caching"] != b["server"]["prefix_caching"]
    first_of_pair = {c["server"]["prefix_caching"] for c in runs[0::2]}
    assert first_of_pair == {True, False}  # neither setting always goes first
