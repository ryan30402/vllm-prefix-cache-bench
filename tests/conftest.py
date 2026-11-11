"""Shared fixtures: a fake `vllm` on PATH, free ports, toy configs and workloads."""

import os
import socket
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

MODEL = "fake/model"
FAKE_ENV = {
    "gpu": "Fake GPU, 1 MiB, 0.0",
    "vllm": "0.0.0",
    "torch": "0.0.0",
    "python": "3.12",
    "pip_freeze": "",
    "pip_freeze_sha256": "0" * 64,
    "model_revision": "testrev",
    "git_commit": "0" * 40,
    "source_dirty": False,
    "source_status": "",
    "source_diff": "",
}


class WordTok:
    """A tokenizer whose tokens are whole words, matching the fake server."""

    vocab_size = 5000

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(w[1:]) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"w{i}" for i in ids)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def fake_vllm_on_path(tmp_path, monkeypatch):
    """Put a `vllm` command on PATH that runs tests/fake_vllm.py."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "vllm"
    fake = Path(__file__).with_name("fake_vllm.py")
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    for key in list(os.environ):
        if key.startswith(("FAKE_SERVE_", "FAKE_BENCH_")):
            monkeypatch.delenv(key)


@pytest.fixture
def cfg():
    """One run config pointing at a free port."""
    return {
        "name": "p32_c2_cache_on_r0",
        "tag": "main",
        "model": MODEL,
        "revision": "testrev",
        "server": {
            "prefix_caching": True,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 256,
            "port": free_port(),
        },
        "bench": {
            "prefix_len": 32,
            "suffix_len": 16,
            "output_len": 4,
            "num_prompts": 6,
            "warmup_prompts": 3,
            "max_concurrency": 2,
            "workload_seed": 0,
        },
        "repeat": 0,
    }


@pytest.fixture
def matrix():
    """A four-run sweep: prefix 0 and 32, cache off and on."""
    return {
        "model": MODEL,
        "repeats": 1,
        "server_base": {
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 256,
            "port": free_port(),
        },
        "bench_base": {
            "suffix_len": 16,
            "output_len": 4,
            "num_prompts": 6,
            "warmup_prompts": 3,
        },
        "sweep": {
            "prefix_caching": [False, True],
            "prefix_len": [0, 32],
            "max_concurrency": [2],
        },
    }
