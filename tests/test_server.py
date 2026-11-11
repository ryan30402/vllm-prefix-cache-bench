"""Server lifecycle: we start our own server, verify it, and stop only it."""

import json
import os
import subprocess
import time

import pytest
import requests
from conftest import MODEL, free_port

from common import RunError
from start_server import start_server
from stop_server import group_members, port_in_use, stop_server


def healthy(port: int) -> bool:
    try:
        return requests.get(f"http://localhost:{port}/health", timeout=2).ok
    except requests.RequestException:
        return False


@pytest.fixture
def other_server():
    """An unrelated `vllm serve` process that the harness must never touch."""
    port = free_port()
    proc = subprocess.Popen(
        ["vllm", "serve", "someone/elses-model", "--port", str(port)]
        + ["--max-model-len", "4096"],
        start_new_session=True,
    )
    deadline = time.time() + 20
    while not healthy(port) and time.time() < deadline:
        time.sleep(0.1)
    assert healthy(port)
    yield port, proc
    proc.kill()
    proc.wait()


def test_start_records_and_verifies_then_stop_frees_the_port(cfg, tmp_path):
    proc = start_server(cfg, tmp_path, timeout_s=30)
    port = cfg["server"]["port"]
    assert proc.poll() is None and healthy(port)

    info = json.loads((tmp_path / "server_info.json").read_text())
    assert info["pid"] == proc.pid == info["pgid"]
    assert info["model_card"]["id"] == MODEL
    assert info["kv_cache_size_tokens"] == 459936
    assert "non_default_args" in info and info["non_default_args"]

    assert stop_server(tmp_path, proc) == "stopped with SIGINT"
    assert proc.poll() is not None
    assert not port_in_use(port)


def test_stop_kills_the_whole_group_and_nothing_else(
    cfg, tmp_path, monkeypatch, other_server
):
    other_port, other_proc = other_server
    child_pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("FAKE_SERVE_CHILD", str(child_pid_file))
    proc = start_server(cfg, tmp_path, timeout_s=30)
    child_pid = int(child_pid_file.read_text())
    assert child_pid in group_members(proc.pid)

    stop_server(tmp_path, proc)

    assert group_members(proc.pid) == []  # our server and its child are gone
    assert other_proc.poll() is None and healthy(other_port)  # theirs is untouched


def test_refuses_to_start_when_the_port_is_taken(cfg, tmp_path, other_server):
    other_port, other_proc = other_server
    cfg["server"]["port"] = other_port
    with pytest.raises(RunError, match="already in use"):
        start_server(cfg, tmp_path, timeout_s=30)
    assert other_proc.poll() is None and healthy(other_port)
    assert not (tmp_path / "server.json").exists()


def test_server_that_dies_during_startup_is_reported_at_once(
    cfg, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_SERVE_EXIT_CODE", "3")
    t0 = time.time()
    with pytest.raises(RunError, match="exited with code 3"):
        start_server(cfg, tmp_path, timeout_s=60)
    assert time.time() - t0 < 20  # did not wait for the full timeout


def test_wrong_model_is_rejected_and_the_server_is_stopped(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SERVE_WRONG_MODEL", "1")
    with pytest.raises(RunError, match="expected fake/model"):
        start_server(cfg, tmp_path, timeout_s=30)
    assert not port_in_use(cfg["server"]["port"])


def test_wrong_max_model_len_is_rejected(cfg, tmp_path, monkeypatch):
    import start_server as mod

    real = mod.build_server_cmd

    def lying_cmd(c):  # the process is started with a different length
        cmd = real(c)
        cmd[cmd.index("--max-model-len") + 1] = "1234"
        return cmd

    monkeypatch.setattr(mod, "build_server_cmd", lying_cmd)
    with pytest.raises(RunError, match="max_model_len=1234"):
        start_server(cfg, tmp_path, timeout_s=30)
    assert not port_in_use(cfg["server"]["port"])


def test_escalates_to_sigkill_when_sigint_is_ignored(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SERVE_IGNORE_SIGINT", "1")
    proc = start_server(cfg, tmp_path, timeout_s=30)
    assert stop_server(tmp_path, proc, grace_s=1) == "stopped with SIGKILL"
    assert not port_in_use(cfg["server"]["port"])


def test_a_reused_pid_is_never_signalled(tmp_path):
    info = {
        "pid": os.getpid(),  # this test process stands in for "someone else"
        "pgid": os.getpgid(0),
        "proc_start_time": "1",  # does not match our real start time
        "port": free_port(),
    }
    (tmp_path / "server.json").write_text(json.dumps(info))
    assert "reused" in stop_server(tmp_path)


def test_stop_without_a_recorded_server_does_nothing(tmp_path):
    assert "nothing to stop" in stop_server(tmp_path)
