"""One validator for runner, resume and analyzer: damaged results are INVALID.

Each test copies a finished toy experiment, damages one attempt, and checks
that (a) the validator rejects it, (b) resume does not skip it, and (c) the
analyzer does not read it.
"""

import json
import shutil

import pytest
from v3_helpers import WordTok, attempts, fake_env, recommit

from pcbench import analyze, runner, schema, store


@pytest.fixture
def exp(finished, tmp_path):
    return shutil.copytree(finished, tmp_path / "exp")


def on_attempt(exp):
    return next(d for d in attempts(exp) if d.parent.name.endswith("_on"))


def identity(exp, attempt):
    plan = runner.load_manifest(exp)["plan"]
    return next(
        e["identity_sha256"] for e in plan if e["run_id"] == attempt.parent.name
    )


def edit_rows(attempt, change):
    rows = schema.read_rows(attempt / "requests.jsonl")
    change(rows)
    (attempt / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_an_untouched_attempt_is_valid(exp):
    for d in attempts(exp):
        assert schema.validate_attempt(d, identity(exp, d)) == []


def drop_ttft(d):
    edit_rows(d, lambda rows: rows[2].pop("ttft_s"))


def nan_latency(d):
    edit_rows(d, lambda rows: rows[1].update(e2e_s=float("nan")))


def fewer_rows(d):
    edit_rows(d, lambda rows: rows.pop())


def failed_request(d):
    edit_rows(d, lambda rows: rows[0].update(ok=False, error="HTTP 500"))


def cut_json(name):
    def damage(d):
        text = (d / name).read_text()
        (d / name).write_text(text[: len(text) // 2])

    return damage


def completed_count(d):
    summary = store.read_json(d / "summary.json")
    summary["completed"] += 1
    (d / "summary.json").write_text(json.dumps(summary))


def faster(d):
    summary = store.read_json(d / "summary.json")
    summary["output_tok_per_s"] *= 2
    (d / "summary.json").write_text(json.dumps(summary))


def counter_edit(d):
    after = store.read_json(d / "metrics_after.json")
    after["request_success"] += 1
    (d / "metrics_after.json").write_text(json.dumps(after))


def short_output(d):
    edit_rows(d, lambda rows: rows[3].update(output_tokens=3))


def remove(name):
    return lambda d: (d / name).unlink()


DAMAGE = {
    "per-request field deleted": drop_ttft,
    "NaN inserted": nan_latency,
    "a request record removed": fewer_rows,
    "a request marked failed": failed_request,
    "requests.jsonl cut short": cut_json("requests.jsonl"),
    "summary.json cut short": cut_json("summary.json"),
    "completed count changed": completed_count,
    "throughput changed": faster,
    "server counter changed": counter_edit,
    "output shorter than agreed": short_output,
    "counters.json removed": remove("counters.json"),
    "server log removed": remove("server.log.gz"),
}


@pytest.mark.parametrize("name", DAMAGE)
@pytest.mark.parametrize("hide", [False, True], ids=["plain", "hashes-refreshed"])
def test_damaged_results_are_invalid(exp, name, hide):
    d = on_attempt(exp)
    DAMAGE[name](d)
    if hide:  # even if someone recomputes the file hashes, the content check fails
        recommit(d)
    problems = schema.validate_attempt(d, identity(exp, d))
    assert problems, name
    pair = runner.pairs_in_order(runner.load_manifest(exp))[0]
    assert runner.pair_state(exp, pair)["complete"] is None


@pytest.mark.parametrize("content", ["", "{not json", '{"state": "VALIDATED"}'])
def test_a_broken_status_file_is_never_a_success(exp, content):
    d = on_attempt(exp)
    (d / "status.json").write_text(content)
    assert schema.validate_attempt(d, identity(exp, d))


def test_a_result_made_under_another_identity_is_not_reused(exp):
    d = on_attempt(exp)
    assert schema.validate_attempt(d, "0" * 64) == [
        "made with a different config, workload, environment or code"
    ]


def test_resume_reruns_a_damaged_pair_and_keeps_the_old_attempt(exp):
    d = on_attempt(exp)
    drop_ttft(d)
    recommit(d)
    assert runner.run_experiment(exp, tok=WordTok(), env=fake_env()) == 0
    names = sorted(f"{a.parent.name}/{a.name}" for a in attempts(exp))
    assert names == [
        "r1_p0032_c002_g90_off/attempt_1",
        "r1_p0032_c002_g90_off/attempt_2",
        "r1_p0032_c002_g90_on/attempt_1",
        "r1_p0032_c002_g90_on/attempt_2",
    ]
    assert (
        d / "requests.jsonl"
    ).is_file()  # the damaged attempt is kept, not overwritten
    audit = analyze.analyze(exp)
    assert audit["complete"] and audit["attempts_by_state"] == {
        "VALIDATED": 3,
        "VALIDATED_BUT_INVALID": 1,
    }
    sources = audit["sources_by_condition"]["p0032_c002_g90"]
    assert sorted(sources) == [
        "runs/r1_p0032_c002_g90_off/attempt_2",
        "runs/r1_p0032_c002_g90_on/attempt_2",
    ]


def test_resume_skips_a_valid_pair_without_touching_it(exp):
    before = {d: (d / "status.json").read_bytes() for d in attempts(exp)}
    assert runner.run_experiment(exp, tok=WordTok(), env=fake_env()) == 0
    assert {d: (d / "status.json").read_bytes() for d in attempts(exp)} == before


def test_summary_is_recomputed_from_request_records(exp):
    d = on_attempt(exp)
    rows = schema.read_rows(d / "requests.jsonl")
    summary = schema.summarize(rows, 2)
    assert summary == store.read_json(d / "summary.json")
    end = max(r["start_s"] + r["e2e_s"] for r in rows)
    assert summary["duration_s"] == pytest.approx(end)
    assert summary["output_tok_per_s"] == pytest.approx(6 * 4 / end)
    assert summary["peak_in_flight"] == 2
    assert summary["burst_requests"] == 2 and summary["steady_requests"] == 4
    assert summary["ttft_p50_ms"] == pytest.approx(
        sorted(r["ttft_s"] for r in rows)[2] * 500
        + sorted(r["ttft_s"] for r in rows)[3] * 500
    )


def test_in_flight_count_handles_slots_handed_over_at_the_same_instant():
    rows = [
        {
            "ok": True,
            "start_s": float(i),
            "e2e_s": 1.0,
            "ttft_s": 0.5,
            "output_tokens": 3,
            "input_tokens": 1,
            "client_queue_s": 0.0,
            "output_sha256": "x",
        }
        for i in range(4)
    ]
    assert schema.summarize(rows, 1)["peak_in_flight"] == 1
    # float rounding: (start + latency) can land a hair after the next start
    t0, latency = 97531.123456789, 0.0162
    starts = [t0 + k * latency for k in range(200)]
    chained = [
        dict(rows[0], start_s=b - t0, e2e_s=latency, ttft_s=0.01) for b in starts
    ]
    assert any(
        a["start_s"] + latency > b["start_s"] for a, b in zip(chained, chained[1:])
    )
    assert schema.summarize(chained, 1)["peak_in_flight"] == 1
    rows[1]["start_s"] = 0.5
    assert schema.summarize(rows, 1)["peak_in_flight"] == 2
