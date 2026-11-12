"""Run identity, the measurement-code hash and the randomized paired plan."""

import shutil
from collections import Counter

import pytest
from v3_helpers import V3, spec_dict

from pcbench import ident, spec as specmod
from pcbench.store import RunError


def test_code_fingerprint_ignores_comments_and_layout_but_not_logic():
    base = "def f(x):\n    return x + 1\n"
    assert ident.code_fingerprint(base) == ident.code_fingerprint(
        "# a comment\ndef f(x):\n\n    return x + 1  # another\n"
    )
    assert ident.code_fingerprint(base) != ident.code_fingerprint(
        "def f(x):\n    return x + 2\n"
    )


def test_measurement_code_hash_follows_measurement_logic_only(tmp_path):
    pkg = tmp_path / "pcbench"
    shutil.copytree(V3 / "pcbench", pkg)
    before = ident.measurement_code(pkg)
    # A change to an analysis module or a text file must not invalidate results.
    (pkg / "analyze.py").write_text((pkg / "analyze.py").read_text() + "\nX = 1\n")
    (tmp_path / "README.md").write_text("docs")
    assert ident.measurement_code(pkg)["sha256"] == before["sha256"]
    # A comment in a measurement module does not either.
    (pkg / "bench.py").write_text((pkg / "bench.py").read_text() + "\n# note\n")
    assert ident.measurement_code(pkg)["sha256"] == before["sha256"]
    # A change to what is measured does.
    text = (pkg / "bench.py").read_text().replace('"--ignore-eos",', "")
    assert text != (pkg / "bench.py").read_text()
    (pkg / "bench.py").write_text(text)
    after = ident.measurement_code(pkg)
    assert after["sha256"] != before["sha256"]
    assert [k for k in after["files"] if after["files"][k] != before["files"][k]] == [
        "bench.py"
    ]


def test_run_identity_changes_with_every_ingredient():
    args = ["3.0.0", {"prefix_len": 32}, {"main_sha256": "a"}, {"vllm": "1"}, "code1"]
    base = ident.run_identity(*args)["sha256"]
    assert ident.run_identity(*args)["sha256"] == base
    for i, other in enumerate(
        ["3.0.1", {"prefix_len": 48}, {"main_sha256": "b"}, {"vllm": "2"}, "code2"]
    ):
        changed = list(args)
        changed[i] = other
        assert ident.run_identity(*changed)["sha256"] != base


def grid_spec(rounds: int = 3, seed: int = 7) -> dict:
    grid = {"prefix_len": [0, 256, 1024, 2048, 4096], "concurrency": [1, 8, 32, 128]}
    return spec_dict(rounds=rounds, order_seed=seed, conditions={"grid": grid})


def test_plan_is_deterministic_and_complete():
    spec = grid_spec()
    plan = specmod.build_plan(spec)
    assert plan == specmod.build_plan(spec)
    reordered = specmod.build_plan({**spec, "order_seed": 8})
    assert [e["run_id"] for e in reordered] != [e["run_id"] for e in plan]
    assert len(plan) == 5 * 4 * 2 * 3
    assert len({e["run_id"] for e in plan}) == len(plan)
    summary = specmod.plan_summary(spec, plan)
    assert summary["runs"] == 120 and summary["pairs"] == 60
    assert summary["measured_requests"] == 120 * spec["workload"]["num_prompts"]
    for rnd in (1, 2, 3):
        in_round = [e for e in plan if e["round"] == rnd]
        # every condition once per round, as an adjacent off/on pair
        assert Counter(e["config"]["condition_id"] for e in in_round) == Counter(
            {specmod.condition_id(c): 2 for c in specmod.conditions(spec)}
        )
        for a, b in zip(in_round[0::2], in_round[1::2]):
            assert a["pair_id"] == b["pair_id"]
            assert {a["config"]["cache"], b["config"]["cache"]} == {"off", "on"}
            assert a["config"]["cache"] == a["first_in_pair"]


def test_which_cache_mode_goes_first_is_balanced():
    plan = specmod.build_plan(grid_spec())
    firsts = {
        e["pair_id"]: (e["round"], e["config"]["condition_id"], e["first_in_pair"])
        for e in plan
    }
    per_round = Counter(r for r, _, first in firsts.values() if first == "on")
    assert per_round == {1: 10, 2: 10, 3: 10}
    per_condition = Counter(c for _, c, first in firsts.values() if first == "on")
    assert set(per_condition.values()) <= {1, 2} and len(per_condition) == 20
    orders = [
        [e["config"]["condition_id"] for e in plan if e["round"] == r]
        for r in (1, 2, 3)
    ]
    assert orders[0] != orders[1] != orders[2]


def test_memory_utilization_is_part_of_the_condition():
    conds = {
        "grid": {
            "prefix_len": [32],
            "concurrency": [2],
            "gpu_memory_utilization": [0.9, 0.5],
        }
    }
    plan = specmod.build_plan(spec_dict(conditions=conds))
    assert {e["config"]["condition_id"] for e in plan} == {
        "p0032_c002_g90",
        "p0032_c002_g50",
    }
    by_id = {e["run_id"]: e["config"]["server"]["gpu_memory_utilization"] for e in plan}
    assert by_id["r1_p0032_c002_g50_on"] == 0.5 and by_id["r1_p0032_c002_g90_on"] == 0.9


@pytest.mark.parametrize(
    "change, message",
    [
        ({"model_revision": ""}, "model_revision"),
        ({"workload": {"suffix_len": 8}}, "cache block"),
        (
            {"conditions": {"list": [{"prefix_len": 9000, "concurrency": 2}]}},
            "max_model_len",
        ),
        (
            {"conditions": {"list": [{"prefix_len": 32, "concurrency": 999}]}},
            "max_num_seqs",
        ),
        (
            {"conditions": {"list": [{"prefix_len": 32, "concurrency": 2, "typo": 1}]}},
            "unknown",
        ),
        ({"conditions": {"list": [{"prefix_len": 32, "concurrency": 2}] * 2}}, "twice"),
    ],
)
def test_bad_specs_are_refused(change, message):
    with pytest.raises(RunError, match=message):
        specmod.validate_spec(spec_dict(**change))


@pytest.mark.parametrize(
    "name, runs, pairs",
    [
        ("smoke.yaml", 4, 2),
        ("pilot.yaml", 20, 10),
        ("main.yaml", 120, 60),
        ("block_boundary.yaml", 24, 12),
        ("memory_pressure.yaml", 24, 12),
    ],
)
def test_protocol_files_expand_to_the_planned_sizes(name, runs, pairs):
    spec = specmod.load_spec(V3 / "protocol" / name)
    summary = specmod.plan_summary(spec, specmod.build_plan(spec))
    assert (summary["runs"], summary["pairs"]) == (runs, pairs)
    assert spec["model_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    if name != "smoke.yaml":
        assert spec["workload"]["num_prompts"] == 500
        assert spec["workload"]["warmup_prompts"] == 50
