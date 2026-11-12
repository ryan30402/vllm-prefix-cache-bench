"""Atomic writes, attempt states, the experiment lock and the prompt contract."""

import json
import os

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from v3_helpers import MODEL, WordTok, spec_dict

from pcbench import spec as specmod, store, workload
from pcbench.store import RunError


def test_atomic_write_keeps_the_old_file_when_interrupted(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    store.atomic_write_json(path, {"state": "RUNNING"})

    def boom(*a):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.atomic_write_json(path, {"state": "VALIDATED"})
    assert store.read_json(path) == {"state": "RUNNING"}
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]  # no temp file left


@pytest.mark.parametrize("content", [None, "", "{", '{"state": "DONE"}', "[]"])
def test_only_a_complete_status_file_can_say_an_attempt_finished(tmp_path, content):
    if content is not None:
        (tmp_path / "status.json").write_text(content)
    assert store.read_state(tmp_path)["state"] == store.RUNNING


def test_lock_blocks_a_live_runner_and_takes_over_a_dead_one(tmp_path):
    (tmp_path / "runner.lock").write_text(str(os.getppid()))
    with pytest.raises(RunError, match="another runner"):
        with store.ExperimentLock(tmp_path):
            pass
    (tmp_path / "runner.lock").write_text("999999999")
    with store.ExperimentLock(tmp_path):
        assert (tmp_path / "runner.lock").read_text() == str(os.getpid())
    assert not (tmp_path / "runner.lock").exists()


class BpeTok:
    """A real byte-level BPE tokenizer, where decode -> encode is not stable."""

    def __init__(self) -> None:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        corpus = [
            "the quick brown fox jumps over the lazy dog " * 3,
            "prefix caching reuses the key value cache of a shared prompt prefix",
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


class BosTok(WordTok):
    """Adds one special token in front, like a tokenizer with a BOS token."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = super().encode(text)
        return [4999, *ids] if add_special_tokens else ids


def run_cfg(prefix_len: int, **workload_over) -> dict:
    spec = spec_dict(
        workload={
            "suffix_len": 24,
            "num_prompts": 12,
            "warmup_prompts": 5,
            **workload_over,
        },
        conditions={"list": [{"prefix_len": prefix_len, "concurrency": 2}]},
    )
    return specmod.build_plan(spec)[0]["config"]


@pytest.mark.parametrize("tok_cls", [WordTok, BpeTok])
@pytest.mark.parametrize("prefix_len", [0, 16, 30, 33, 64])
def test_prompt_files_keep_the_token_contract(tmp_path, tok_cls, prefix_len):
    tok = tok_cls()
    info = workload.write_workload(tok, run_cfg(prefix_len), tmp_path)
    assert info["prompt_tokens_each"] == prefix_len + 24
    assert info["cacheable_tokens_per_request"] == prefix_len // 16 * 16
    assert info["verified_prompts"] == 17 and info["special_tokens_added"] == 0
    prompts = workload.read_back(tmp_path / "warmup.jsonl") + workload.read_back(
        tmp_path / "main.jsonl"
    )
    ids = [tok.encode(p) for p in prompts]
    assert {len(i) for i in ids} == {prefix_len + 24}
    assert len({tuple(i[:prefix_len]) for i in ids}) == 1
    # beyond the whole blocks of the prefix, no two prompts share a block
    cut = prefix_len // 16 * 16 + 16
    assert len({tuple(i[:cut]) for i in ids}) == len(ids)
    # the same inputs give byte-identical files
    again = workload.write_workload(tok, run_cfg(prefix_len), tmp_path / "again")
    assert again["main_sha256"] == info["main_sha256"]
    assert again["main_input_ids_sha256"] == info["main_input_ids_sha256"]


def test_a_leading_special_token_shifts_the_block_arithmetic(tmp_path):
    info = workload.write_workload(BosTok(), run_cfg(15), tmp_path)
    assert info["special_tokens_added"] == 1
    assert info["server_prompt_tokens_each"] == 15 + 24 + 1
    assert info["cacheable_tokens_per_request"] == 16  # BOS + 15 prefix tokens


def rewrite(path, prompts):
    data = workload.jsonl_bytes(prompts, 4)
    path.write_bytes(data)
    return workload.sha256_bytes(data)


def test_verification_catches_each_breach_of_the_contract(tmp_path):
    tok = WordTok()
    info = workload.write_workload(tok, run_cfg(32), tmp_path)
    main = workload.read_back(tmp_path / "main.jsonl")
    assert workload.verify_workload(tok, tmp_path, info)["verified_prompts"] == 17

    # 1. a byte changed on disk
    (tmp_path / "main.jsonl").write_bytes(
        (tmp_path / "main.jsonl").read_bytes() + b"\n"
    )
    with pytest.raises(RunError, match="sha256"):
        workload.verify_workload(tok, tmp_path, info)

    def with_main(prompts):
        return {**info, "main_sha256": rewrite(tmp_path / "main.jsonl", prompts)}

    # 2. a measured prompt repeats a warm-up prompt (the v1 flaw)
    warm = workload.read_back(tmp_path / "warmup.jsonl")
    with pytest.raises(RunError, match="more than once"):
        workload.verify_workload(tok, tmp_path, with_main([warm[0], *main[1:]]))
    # 3. a prompt with the wrong number of tokens
    with pytest.raises(RunError, match="tokens, not"):
        workload.verify_workload(tok, tmp_path, with_main([main[0] + " w1", *main[1:]]))
    # 4. a prompt whose prefix differs
    words = main[0].split()
    other = " ".join(["w4998", *words[1:]])
    changed = with_main([other, *main[1:]])
    changed["main_input_ids_sha256"] = workload.ids_sha256(
        [tok.encode(p) for p in [other, *main[1:]]]
    )
    with pytest.raises(RunError, match="same prefix"):
        workload.verify_workload(tok, tmp_path, changed)
    # 5. two prompts that share the first block after the prefix
    twin = " ".join(main[0].split()[:48] + main[1].split()[48:])
    changed = with_main([main[0], twin, *main[2:]])
    changed["main_input_ids_sha256"] = workload.ids_sha256(
        [tok.encode(p) for p in [main[0], twin, *main[2:]]]
    )
    with pytest.raises(RunError, match="share a cache block"):
        workload.verify_workload(tok, tmp_path, changed)


def test_zero_prefix_prompts_share_no_block(tmp_path):
    tok = WordTok()
    info = workload.write_workload(tok, run_cfg(0), tmp_path)
    assert info["cacheable_tokens_per_request"] == 0
    main = workload.read_back(tmp_path / "main.jsonl")
    twin = " ".join(main[0].split()[:16] + main[1].split()[16:])
    info["main_sha256"] = rewrite(tmp_path / "main.jsonl", [main[0], twin, *main[2:]])
    info["main_input_ids_sha256"] = workload.ids_sha256(
        [tok.encode(p) for p in [main[0], twin, *main[2:]]]
    )
    with pytest.raises(RunError, match="share a cache block"):
        workload.verify_workload(tok, tmp_path, info)


def test_workload_json_is_written_last_and_complete(tmp_path):
    info = workload.write_workload(WordTok(), run_cfg(32), tmp_path)
    assert json.loads((tmp_path / "workload.json").read_text()) == info
    assert info["model"] == MODEL and info["tokenizer_revision"] == "testrev"
