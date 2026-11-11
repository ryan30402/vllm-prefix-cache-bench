"""Metrics parsing, counter arithmetic, config hashing and completeness checks."""

import json

import pytest
import yaml

from common import (
    REQUIRED_FILES,
    RunError,
    check_complete,
    config_hash,
    latest_complete_attempt,
    next_attempt_dir,
    run_dir_name,
)
from run_bench import build_bench_cmd, slim_result, validate_result
from scrape_metrics import COUNTERS, counter_deltas, parse, scrape
from start_server import build_server_cmd, start_server
from stop_server import stop_server

REAL_SAMPLE = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 0.25
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 1152000.0
vllm:prefix_cache_queries_created{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 1.79e+09
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 1024000.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 499.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 1.0
"""


def test_parse_reads_real_metric_lines_and_sums_labels():
    parsed = parse(REAL_SAMPLE)
    assert parsed["prefix_cache_queries"] == 1152000.0
    assert parsed["prefix_cache_hits"] == 1024000.0
    assert parsed["kv_cache_usage_perc"] == 0.25
    assert parsed["request_success"] == 500.0  # both finish reasons
    assert "num_preemptions" not in parsed  # absent stays absent, never 0


def test_scrape_raises_when_a_metric_is_missing(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SERVE_DROP_METRIC", "num_preemptions")
    proc = start_server(cfg, tmp_path, timeout_s=30)
    try:
        with pytest.raises(RunError, match="num_preemptions"):
            scrape(cfg["server"]["port"], str(tmp_path / "m"))
        assert (tmp_path / "m.txt").exists()  # the raw text is kept for debugging
    finally:
        stop_server(tmp_path, proc)


def test_counter_deltas_reject_a_reset_and_a_missing_counter():
    before = dict.fromkeys(COUNTERS, 10.0)
    after = dict.fromkeys(COUNTERS, 15.0)
    assert counter_deltas(before, after)["num_preemptions"] == 5.0

    restarted = {**after, "request_success": 2.0}
    with pytest.raises(RunError, match="went down"):
        counter_deltas(before, restarted)

    incomplete = {k: v for k, v in after.items() if k != "num_preemptions"}
    with pytest.raises(RunError, match="missing"):
        counter_deltas(before, incomplete)


def test_config_hash_changes_with_any_setting_or_environment(cfg):
    identity = {"gpu": "A40", "vllm": "0.33.0"}
    base = config_hash(cfg, identity)
    assert base == config_hash(json.loads(json.dumps(cfg)), dict(identity))

    changed = json.loads(json.dumps(cfg))
    changed["server"]["gpu_memory_utilization"] = 0.5
    assert config_hash(changed, identity) != base

    changed = json.loads(json.dumps(cfg))
    changed["server"]["max_model_len"] = 16384
    assert config_hash(changed, identity) != base

    assert config_hash(cfg, {**identity, "vllm": "0.34.0"}) != base
    assert run_dir_name(cfg, base) == f"{cfg['name']}__{base[:10]}"


def make_attempt(run_dir, cfg, digest, n=None):
    attempt = next_attempt_dir(run_dir)
    attempt.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (attempt / name).write_text("{}")
    (attempt / "config.yaml").write_text(yaml.safe_dump(cfg))
    done = cfg["bench"]["num_prompts"] if n is None else n
    (attempt / "bench_result.json").write_text(
        json.dumps({"completed": done, "failed": 0})
    )
    (attempt / "warmup_result.json").write_text(
        json.dumps({"completed": cfg["bench"]["warmup_prompts"], "failed": 0})
    )
    (attempt / "status.json").write_text(
        json.dumps({"ok": True, "config_hash": digest})
    )
    return attempt


def test_completeness_check_catches_each_kind_of_bad_attempt(cfg, tmp_path):
    good = make_attempt(tmp_path, cfg, "abc")
    assert good.name == "attempt_1"
    assert check_complete(good, "abc") == []
    assert latest_complete_attempt(tmp_path, "abc") == good

    assert "config hash differs" in check_complete(good, "other")[0]

    (good / "metrics_after.json").unlink()
    assert check_complete(good, "abc") == ["missing or empty metrics_after.json"]
    assert latest_complete_attempt(tmp_path, "abc") is None

    short = make_attempt(tmp_path, cfg, "abc", n=5)
    assert short.name == "attempt_2"  # a retry never reuses a folder
    assert "completed=5, expected 6" in check_complete(short, "abc")[0]

    (short / "status.json").write_text(json.dumps({"ok": False, "config_hash": "abc"}))
    assert "status.json does not say ok" in check_complete(short, "abc")


def test_bench_command_uses_our_prompt_file_and_keeps_per_request_data(cfg, tmp_path):
    cmd = build_bench_cmd(cfg, tmp_path / "main.jsonl", 6, tmp_path / "out.json")
    joined = " ".join(cmd)
    assert "--dataset-name custom" in joined
    assert f"--dataset-path {tmp_path / 'main.jsonl'}" in joined
    for flag in ("--skip-chat-template", "--disable-shuffle", "--no-oversample"):
        assert flag in cmd
    assert "--save-detailed" in cmd and "--ignore-eos" in cmd
    assert "prefix_repetition" not in joined


def test_server_command_pins_the_revision_and_the_cache_switch(cfg):
    cmd = build_server_cmd(cfg)
    assert cmd[cmd.index("--revision") + 1] == "testrev"
    assert "--no-enable-prefix-caching" not in cmd
    cfg["server"]["prefix_caching"] = False
    assert "--no-enable-prefix-caching" in build_server_cmd(cfg)


def good_result(n=6):
    return {
        "completed": n,
        "failed": 0,
        "errors": [""] * n,
        "ttfts": [0.1] * n,
        "latencies": [0.4] * n,
        "output_lens": [4] * n,
        "input_lens": [48] * n,
        "itls": [[0.1] * 3] * n,
        "generated_texts": ["x"] * n,
    }


def test_validate_result_refuses_failures_errors_and_short_data():
    validate_result(good_result(), 6, "run")
    with pytest.raises(RunError, match="5 requests completed"):
        validate_result({**good_result(), "completed": 5}, 6, "run")
    with pytest.raises(RunError, match="2 requests failed"):
        validate_result({**good_result(), "failed": 2}, 6, "run")
    with pytest.raises(RunError, match="request errors"):
        validate_result({**good_result(), "errors": ["boom"] + [""] * 5}, 6, "run")
    with pytest.raises(RunError, match="ttfts"):
        validate_result({**good_result(), "ttfts": [0.1]}, 6, "run")


def test_slim_result_keeps_per_request_tpot_and_drops_bulk_fields():
    slim = slim_result(good_result())
    assert slim["tpots"] == pytest.approx([0.1] * 6)  # (0.4 - 0.1) / (4 - 1)
    assert "itls" not in slim and "generated_texts" not in slim
    assert len(slim["ttfts"]) == 6
