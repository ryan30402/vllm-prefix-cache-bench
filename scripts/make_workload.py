"""Generate the prompt files for a sweep: one warm-up file and one measured file
per distinct workload, with exact token counts.

Usage:
    python scripts/make_workload.py configs/matrix_clean24.yaml [workloads]

Why this exists: the v1 harness used `vllm bench serve --dataset-name
prefix_repetition` for both the warm-up and the measured run with the same
seed, so 50 measured prompts were exact repeats of warm-up prompts and became
full-prompt cache hits. Changing only the warm-up seed would change the shared
prefix too and leave the cache cold. Here the warm-up and measured prompts
share the same prefix tokens and have disjoint random suffixes.

Every prompt is checked so that decoding and re-encoding it gives back exactly
the intended tokens: the first `prefix_len` tokens are identical in every
prompt and each prompt is exactly `prefix_len + suffix_len` tokens long.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from common import RunError, load_config

BLOCK = 16  # vLLM caches KV in blocks of 16 tokens


def workload_key(model: str, bench: dict) -> str:
    """Folder name that identifies one workload."""
    return (
        f"{model.replace('/', '--')}_p{bench['prefix_len']}_s{bench['suffix_len']}"
        f"_n{bench['num_prompts']}_w{bench['warmup_prompts']}"
        f"_seed{bench.get('workload_seed', 0)}"
    )


def stable_ids(tok, rng: np.random.Generator, n: int) -> list[int]:
    """n random token ids whose decoded text re-encodes to exactly the same ids."""
    if n == 0:
        return []
    for _ in range(200):
        ids = rng.integers(0, tok.vocab_size, size=n).tolist()
        for _ in range(20):
            again = tok.encode(tok.decode(ids), add_special_tokens=False)
            if again == ids:
                return ids
            pad = rng.integers(0, tok.vocab_size, size=max(0, n - len(again)))
            ids = again[:n] + pad.tolist()
    raise RunError(f"could not build a stable sequence of {n} tokens")


def build_workload(
    tok, prefix_len: int, suffix_len: int, n_warmup: int, n_main: int, seed: int
) -> tuple[list[str], list[str], list[int]]:
    """Return (warm-up prompts, measured prompts, prefix token ids)."""
    if suffix_len < BLOCK:
        raise RunError(f"suffix_len must be at least {BLOCK} tokens")
    rng = np.random.default_rng(seed)
    prefix = stable_ids(tok, rng, prefix_len)
    # Suffix tokens that share a cache block with the end of the prefix (all 16
    # when the prefix ends exactly on a block boundary).
    head_len = BLOCK - prefix_len % BLOCK
    prompts: list[str] = []
    first_blocks: set[tuple[int, ...]] = set()
    while len(prompts) < n_warmup + n_main:
        for _ in range(200):
            suffix = stable_ids(tok, rng, suffix_len)
            text = tok.decode(prefix + suffix)
            # The prefix must survive the join, and no two prompts may share the
            # first block that contains suffix tokens, or they would get extra
            # cache hits beyond the shared prefix.
            head = tuple(suffix[:head_len])
            if (
                tok.encode(text, add_special_tokens=False) == prefix + suffix
                and head not in first_blocks
            ):
                first_blocks.add(head)
                prompts.append(text)
                break
        else:
            raise RunError("could not build a prompt that keeps the prefix intact")
    return prompts[:n_warmup], prompts[n_warmup:], prefix


def write_jsonl(path: Path, prompts: list[str], output_len: int) -> str:
    """Write one prompt per line in the vLLM custom dataset format. Returns sha256."""
    lines = [json.dumps({"prompt": p, "output_tokens": output_len}) for p in prompts]
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_back(path: Path) -> list[str]:
    """Read a prompt file the same way `vllm bench serve` does."""
    import pandas as pd

    return pd.read_json(path_or_buf=str(path), lines=True)["prompt"].tolist()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_workload(
    tok, model: str, bench: dict, root: Path, tokenizer_info: dict
) -> Path:
    out = root / workload_key(model, bench)
    out.mkdir(parents=True, exist_ok=True)
    warmup, main, prefix = build_workload(
        tok,
        bench["prefix_len"],
        bench["suffix_len"],
        bench["warmup_prompts"],
        bench["num_prompts"],
        bench.get("workload_seed", 0),
    )
    manifest = {
        "model": model,
        "prefix_len": bench["prefix_len"],
        "suffix_len": bench["suffix_len"],
        "output_len": bench["output_len"],
        "warmup_prompts": len(warmup),
        "num_prompts": len(main),
        "seed": bench.get("workload_seed", 0),
        "prompt_tokens_each": bench["prefix_len"] + bench["suffix_len"],
        "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
        "warmup_sha256": write_jsonl(out / "warmup.jsonl", warmup, bench["output_len"]),
        "main_sha256": write_jsonl(out / "main.jsonl", main, bench["output_len"]),
        "tokenizer": tokenizer_info,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name, prompts in (("warmup.jsonl", warmup), ("main.jsonl", main)):
        if read_back(out / name) != prompts:
            raise RunError(f"{out / name} does not read back as it was written")
    return out


def locate_workload(cfg: dict, root: Path) -> dict:
    """Find the files for a run and verify them against their manifest."""
    folder = root / workload_key(cfg["model"], cfg["bench"])
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise RunError(
            f"workload {folder} not found; run scripts/make_workload.py first"
        )
    manifest = json.loads(manifest_path.read_text())
    for kind in ("warmup", "main"):
        if sha256_file(folder / f"{kind}.jsonl") != manifest[f"{kind}_sha256"]:
            raise RunError(f"{folder}/{kind}.jsonl does not match its manifest")
    return {
        "warmup": folder / "warmup.jsonl",
        "main": folder / "main.jsonl",
        "manifest": manifest,
    }


def distinct_workloads(runs: list[dict]) -> list[tuple[str, dict]]:
    seen, out = set(), []
    for cfg in runs:
        key = workload_key(cfg["model"], cfg["bench"])
        if key not in seen:
            seen.add(key)
            out.append((cfg["model"], cfg["bench"]))
    return out


def main() -> None:
    # Imported here so the tests do not need transformers or a model download.
    from transformers import AutoTokenizer

    from capture_env import resolve_model_revision
    from run_matrix import expand

    matrix = load_config(sys.argv[1])
    root = Path(sys.argv[2] if len(sys.argv) > 2 else "workloads")
    tokenizers: dict = {}
    for model, bench in distinct_workloads(expand(matrix)):
        if model not in tokenizers:
            tokenizers[model] = AutoTokenizer.from_pretrained(
                model, revision=matrix.get("revision")
            )
        info = {
            "name": model,
            "revision": matrix.get("revision") or resolve_model_revision(model),
            "vocab_size": tokenizers[model].vocab_size,
        }
        out = write_workload(tokenizers[model], model, bench, root, info)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
