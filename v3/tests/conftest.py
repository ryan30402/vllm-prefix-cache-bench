"""Fixtures for the v3 tests: fake `vllm` and `nvidia-smi` commands on PATH."""

import os
import stat
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # v3/, for `import pcbench`
sys.path.insert(0, str(HERE))  # for `import v3_helpers`


def make_bin(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, script in (
        ("vllm", "fake_vllm3.py"),
        ("nvidia-smi", "fake_nvidia_smi.py"),
    ):
        wrapper = folder / name
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{HERE / script}" "$@"\n'
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return folder


@pytest.fixture(scope="session")
def fake_bin(tmp_path_factory) -> Path:
    return make_bin(tmp_path_factory.mktemp("bin"))


@pytest.fixture(autouse=True)
def fake_commands_on_path(fake_bin, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    for key in list(os.environ):
        if key.startswith(("FAKE_SERVE_", "FAKE_BENCH_", "FAKE_SMI_")):
            monkeypatch.delenv(key)


@pytest.fixture(scope="module")
def finished(fake_bin, tmp_path_factory) -> Path:
    """One finished toy experiment (1 pair). Tests copy it before changing it."""
    from v3_helpers import freeze_and_run

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
        exp, code = freeze_and_run(tmp_path_factory.mktemp("finished"))
    assert code == 0
    return exp
