"""Prompt files with a token-level contract.

For one prefix length the generator writes a warm-up file and a measured file:

  * every prompt is exactly prefix_len + suffix_len tokens;
  * the first prefix_len token ids are identical in every prompt;
  * no prompt appears twice, in either file;
  * no two prompts share the cache block that holds the first suffix token, so
    a measured request can reuse the whole blocks of the shared prefix and
    nothing more. With prefix_len 0 this means no two prompts share any block.

The contract is about token ids, not characters. `verify_workload` therefore
reads the files back the way `vllm bench serve` does, tokenizes every prompt
again the way the server does, and checks each point. It runs when the files
are written and again at the start of every session.
"""

import json
from pathlib import Path

import numpy as np

from .ident import sha256_bytes, sha256_file, sha256_json
from .store import RunError, atomic_write_bytes, atomic_write_json, read_json


def load_tokenizer(model: str, revision: str):
    # Imported here so that the CPU tests do not need transformers or a download.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision)


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


def build_prompts(
    tok, prefix_len: int, suffix_len: int, n_total: int, seed: int, block: int
) -> tuple[list[str], list[list[int]]]:
    """Return (prompt texts, their token ids). All share the first prefix_len ids."""
    rng = np.random.default_rng(seed)
    prefix = stable_ids(tok, rng, prefix_len)
    # Suffix tokens that share a cache block with the end of the prefix (a whole
    # block when the prefix ends exactly on a block boundary).
    head_len = block - prefix_len % block
    texts: list[str] = []
    all_ids: list[list[int]] = []
    heads: set[tuple[int, ...]] = set()
    while len(texts) < n_total:
        for _ in range(200):
            suffix = stable_ids(tok, rng, suffix_len)
            text = tok.decode(prefix + suffix)
            head = tuple(suffix[:head_len])
            if (
                tok.encode(text, add_special_tokens=False) == prefix + suffix
                and head not in heads
            ):
                heads.add(head)
                texts.append(text)
                all_ids.append(prefix + suffix)
                break
        else:
            raise RunError("could not build a prompt that keeps the prefix intact")
    return texts, all_ids


def jsonl_bytes(prompts: list[str], output_len: int) -> bytes:
    """One prompt per line in the `vllm bench serve` custom dataset format."""
    lines = [json.dumps({"prompt": p, "output_tokens": output_len}) for p in prompts]
    return ("\n".join(lines) + "\n").encode()


def read_back(path: Path) -> list[str]:
    """Read a prompt file the same way `vllm bench serve` does."""
    import pandas as pd

    return pd.read_json(path_or_buf=str(path), lines=True)["prompt"].tolist()


def ids_sha256(ids: list[list[int]]) -> str:
    return sha256_json(ids)


def write_workload(tok, run_cfg: dict, folder: Path) -> dict:
    """Generate, write and verify the two prompt files of one workload."""
    w = run_cfg["workload"]
    prefix_len = run_cfg["prefix_len"]
    folder.mkdir(parents=True, exist_ok=True)
    n_warm, n_main = w["warmup_prompts"], w["num_prompts"]
    texts, ids = build_prompts(
        tok, prefix_len, w["suffix_len"], n_warm + n_main, w["seed"], w["block_size"]
    )
    files = {"warmup": texts[:n_warm], "main": texts[n_warm:]}
    info = {
        "key": run_cfg["workload_key"],
        "model": run_cfg["model"],
        "tokenizer_revision": run_cfg["model_revision"],
        "vocab_size": tok.vocab_size,
        "prefix_len": prefix_len,
        "suffix_len": w["suffix_len"],
        "output_len": w["output_len"],
        "warmup_prompts": n_warm,
        "num_prompts": n_main,
        "seed": w["seed"],
        "block_size": w["block_size"],
        "prompt_tokens_each": prefix_len + w["suffix_len"],
        "shared_prefix_ids_sha256": ids_sha256([ids[0][:prefix_len]]),
        "warmup_input_ids_sha256": ids_sha256(ids[:n_warm]),
        "main_input_ids_sha256": ids_sha256(ids[n_warm:]),
    }
    for kind, prompts in files.items():
        data = jsonl_bytes(prompts, w["output_len"])
        atomic_write_bytes(folder / f"{kind}.jsonl", data)
        info[f"{kind}_sha256"] = sha256_bytes(data)
    info.update(verify_workload(tok, folder, info))
    atomic_write_json(folder / "workload.json", info)
    return info


def verify_workload(tok, folder: Path, info: dict) -> dict:
    """Check the files on disk against the contract. Raises RunError on any breach.

    Returns the facts the run checks need: how many tokens the server will count
    per prompt and how many of them a measured request can take from the cache.
    """
    block, prefix_len = info["block_size"], info["prefix_len"]
    want_len = info["prompt_tokens_each"]
    plain: dict[str, list[list[int]]] = {}
    served: list[list[int]] = []
    seen: set[str] = set()
    for kind, count in (("warmup", "warmup_prompts"), ("main", "num_prompts")):
        path = folder / f"{kind}.jsonl"
        if not path.is_file():
            raise RunError(f"{path} is missing; run the freeze step again")
        if sha256_file(path) != info[f"{kind}_sha256"]:
            raise RunError(f"{path} does not match its recorded sha256")
        prompts = read_back(path)
        if len(prompts) != info[count]:
            raise RunError(f"{path}: {len(prompts)} prompts, expected {info[count]}")
        plain[kind] = []
        for text in prompts:
            if text in seen:
                raise RunError(f"{path}: a prompt appears more than once")
            seen.add(text)
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) != want_len:
                raise RunError(
                    f"{path}: a prompt has {len(ids)} tokens, not {want_len}"
                )
            plain[kind].append(ids)
            # The completions endpoint tokenizes with special tokens enabled.
            with_special = tok.encode(text, add_special_tokens=True)
            extra = len(with_special) - len(ids)
            if extra < 0 or with_special[extra:] != ids:
                raise RunError(f"{path}: special tokens are not a plain leading run")
            served.append(with_special)
        if ids_sha256(plain[kind]) != info[f"{kind}_input_ids_sha256"]:
            raise RunError(f"{path}: token ids differ from the recorded ones")
    every = plain["warmup"] + plain["main"]
    prefix = every[0][:prefix_len]
    if any(ids[:prefix_len] != prefix for ids in every):
        raise RunError("the prompts do not share the same prefix token ids")
    if ids_sha256([prefix]) != info["shared_prefix_ids_sha256"]:
        raise RunError("the shared prefix token ids differ from the recorded ones")
    extras = {len(s) - want_len for s in served}
    if len(extras) != 1:
        raise RunError("prompts get different numbers of special tokens")
    n_special = extras.pop()
    cacheable = (prefix_len + n_special) // block * block
    # Up to and including the first block that holds a suffix token, every
    # prompt must differ from every other one.
    firsts = {tuple(s[: cacheable + block]) for s in served}
    if len(firsts) != len(served):
        raise RunError("two prompts share a cache block beyond the shared prefix")
    return {
        "special_tokens_added": n_special,
        "server_prompt_tokens_each": want_len + n_special,
        "cacheable_tokens_per_request": cacheable,
        "verified_prompts": len(served),
    }


def load_workload(folder: Path) -> dict:
    path = folder / "workload.json"
    if not path.is_file():
        raise RunError(f"{path} is missing; run the freeze step first")
    return read_json(path)


def workload_identity(info: dict) -> dict:
    """The fields of workload.json that a run identity depends on."""
    keys = [
        "key",
        "tokenizer_revision",
        "warmup_sha256",
        "main_sha256",
        "warmup_input_ids_sha256",
        "main_input_ids_sha256",
        "shared_prefix_ids_sha256",
        "server_prompt_tokens_each",
        "cacheable_tokens_per_request",
    ]
    return {k: info[k] for k in keys}
