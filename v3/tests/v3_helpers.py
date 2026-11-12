"""Helpers shared by the v3 tests: a word tokenizer, toy specs, a fake environment."""

import socket
from pathlib import Path

import yaml

V3 = Path(__file__).resolve().parents[1]
MODEL = "fake/model"
SMI_FIELDS = [
    "timestamp",
    "uuid",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "memory.used",
]


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


def fake_env(
    uuid: str = "GPU-fake-0000", fields: list[str] | None = None, **over
) -> dict:
    """Shaped like envinfo.capture(), without calling pip, git or nvidia-smi."""
    env = {
        "captured_at": "2026-01-01T00:00:00+00:00",
        "hostname": "testhost",
        "os": "test",
        "python": "3.12",
        "gpus": [
            {
                "name": "Fake GPU",
                "uuid": uuid,
                "memory_total": "1 MiB",
                "driver_version": "0.0",
            }
        ],
        "cuda_compiler": "",
        "vllm": "0.0.0",
        "torch": "0.0.0",
        "transformers": "0.0.0",
        "vllm_default_block_size": 16,
        "telemetry_fields": SMI_FIELDS if fields is None else fields,
        "pip_freeze": "",
        "pip_freeze_sha256": "0" * 64,
        "git_commit": "0" * 40,
        "source_dirty": False,
        "source_status": "",
    }
    env.update(over)
    return env


def spec_dict(**over) -> dict:
    spec = {
        "name": "toy",
        "batch": 1,
        "model": MODEL,
        "model_revision": "testrev",
        "rounds": 1,
        "order_seed": 1,
        "server": {
            "port": free_port(),
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 2048,
            "generation_config": "vllm",
            "seed": 0,
        },
        "workload": {
            "suffix_len": 16,
            "output_len": 4,
            "num_prompts": 6,
            "warmup_prompts": 3,
            "seed": 0,
            "block_size": 16,
        },
        "bench": {"temperature": 0.0, "timeout_s": 60, "warmup_timeout_s": 60},
        "telemetry": {"required": True, "interval_s": 0.2},
        "validity": {"max_pair_attempts": 2},
        "conditions": {"list": [{"prefix_len": 32, "concurrency": 2}]},
    }
    for key, value in over.items():
        if (
            key != "conditions"
            and isinstance(value, dict)
            and isinstance(spec.get(key), dict)
        ):
            spec[key] = {**spec[key], **value}
        else:
            spec[key] = value
    return spec


def write_spec(folder: Path, **over) -> Path:
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(folder) / "spec.yaml"
    path.write_text(yaml.safe_dump(spec_dict(**over)))
    return path


def freeze_and_run(tmp_path: Path, env: dict | None = None, **over):
    """Freeze a toy spec and run it. Returns (experiment folder, exit code)."""
    from pcbench import runner

    env = env or fake_env()
    exp = runner.freeze(
        write_spec(tmp_path, **over),
        tmp_path / "results_v3",
        tok=WordTok(),
        allow_dirty=True,
        env=env,
    )
    return exp, runner.run_experiment(exp, tok=WordTok(), env=env)


def recommit(attempt_dir: Path) -> None:
    """Refresh files.json and its hash, as someone hiding an edit would."""
    from pcbench import ident, schema, store

    store.atomic_write_json(
        attempt_dir / "files.json", schema.files_manifest(attempt_dir)
    )
    status = store.read_json(attempt_dir / "status.json")
    status["files_sha256"] = ident.sha256_file(attempt_dir / "files.json")
    store.write_state(attempt_dir, status)


def attempts(exp: Path, state: str | None = None) -> list[Path]:
    from pcbench import store

    found = sorted(Path(exp).glob("runs/*/attempt_*"))
    return [d for d in found if state is None or store.read_state(d)["state"] == state]
