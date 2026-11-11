"""Prompt files: exact token counts, a shared prefix, and no repeated prompts."""

import json

import pytest
from conftest import MODEL, WordTok
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from common import RunError
from make_workload import (
    BLOCK,
    build_workload,
    locate_workload,
    workload_key,
    write_workload,
)


class BpeTok:
    """A real byte-level BPE tokenizer, where decode -> encode is not stable."""

    def __init__(self) -> None:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        corpus = [
            "the quick brown fox jumps over the lazy dog " * 3,
            "prefix caching reuses the key value cache of a shared prompt prefix",
            "time to first token and time per output token under concurrent load",
            "0123456789 !?.,;:-_()[]{} ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ] * 20
        trainer = trainers.BpeTrainer(
            vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        tok.train_from_iterator(corpus, trainer)
        self.tok = tok
        self.vocab_size = tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


@pytest.fixture(scope="module")
def bpe():
    return BpeTok()


@pytest.mark.parametrize("prefix_len", [0, 64, 70])
def test_prompts_have_exact_tokens_and_share_only_the_prefix(bpe, prefix_len):
    suffix_len = 32
    warmup, main, prefix = build_workload(bpe, prefix_len, suffix_len, 5, 20, seed=0)
    assert len(warmup) == 5 and len(main) == 20 and len(prefix) == prefix_len

    encoded = [bpe.encode(p) for p in warmup + main]
    for ids in encoded:
        assert len(ids) == prefix_len + suffix_len
        assert ids[:prefix_len] == prefix

    # No measured prompt repeats a warm-up prompt or another measured prompt.
    assert len(set(warmup + main)) == 25
    assert set(warmup).isdisjoint(main)

    # Beyond the whole blocks of the prefix, no two prompts share a cache block.
    shared = prefix_len // BLOCK * BLOCK
    next_blocks = [tuple(ids[shared : shared + BLOCK]) for ids in encoded]
    assert len(set(next_blocks)) == 25


def test_same_seed_gives_the_same_prompts_and_another_seed_does_not(bpe):
    a = build_workload(bpe, 32, 32, 2, 4, seed=7)
    assert a == build_workload(bpe, 32, 32, 2, 4, seed=7)
    assert a != build_workload(bpe, 32, 32, 2, 4, seed=8)


def test_written_files_match_the_manifest_and_tampering_is_detected(cfg, tmp_path):
    folder = write_workload(WordTok(), MODEL, cfg["bench"], tmp_path, {"name": MODEL})
    assert folder.name == workload_key(MODEL, cfg["bench"])

    lines = (folder / "main.jsonl").read_text().splitlines()
    assert len(lines) == cfg["bench"]["num_prompts"]
    first = json.loads(lines[0])
    assert first["output_tokens"] == cfg["bench"]["output_len"]
    assert len(first["prompt"].split()) == 48

    found = locate_workload(cfg, tmp_path)
    assert found["manifest"]["prompt_tokens_each"] == 48
    assert found["main"] == folder / "main.jsonl"

    with open(folder / "main.jsonl", "a") as f:
        f.write(lines[0] + "\n")
    with pytest.raises(RunError, match="does not match its manifest"):
        locate_workload(cfg, tmp_path)


def test_a_missing_workload_is_an_error_with_instructions(cfg, tmp_path):
    with pytest.raises(RunError, match="make_workload.py"):
        locate_workload(cfg, tmp_path)
