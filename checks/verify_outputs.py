"""Send identical greedy requests to two servers (cache on / off) and compare outputs.

Usage:
    # Phase 1: server running with cache OFF
    python checks/verify_outputs.py collect 8000 checks/outputs_off.json
    # Phase 2: server running with cache ON
    python checks/verify_outputs.py collect 8000 checks/outputs_on.json
    # Compare
    python checks/verify_outputs.py compare checks/outputs_off.json checks/outputs_on.json checks/verify_report.json
"""
import json
import random
import sys

import requests

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_PROMPTS = 50
MAX_TOKENS = 64

# Shared prefix: simulates a long system prompt
SHARED_PREFIX = (
    "You are a careful technical assistant. Answer concisely and precisely. "
    "Do not speculate. If unsure, say so. The following is a question from a user. "
) * 8

QUESTIONS = [
    "Explain what a KV cache is in one paragraph.",
    "Why is decoding slower than prefill?",
    "What does p99 latency mean?",
        "Describe PagedAttention briefly.",
    "What is a token?",
]


def make_prompts() -> list[str]:
    """50 prompts: shared prefix + a question + a unique number so each is different."""
    random.seed(0)
    prompts = []
    for i in range(N_PROMPTS):
        q = QUESTIONS[i % len(QUESTIONS)]
        prompts.append(f"{SHARED_PREFIX}Question #{i}: {q}")
    return prompts


def query(port: int, prompt: str) -> dict:
    """One greedy completion. Returns text and token ids."""
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "seed": 0,
            "logprobs": 0,          # make the response include token info
        },
        timeout=120,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return {
        "text": choice["text"],
        "tokens": choice.get("logprobs", {}).get("tokens", []),
    }

def collect(port: int, out_path: str) -> None:
    prompts = make_prompts()
    results = []
    for i, p in enumerate(prompts):
        out = query(port, p)
        results.append({"prompt": p, **out})
        print(f"{i + 1}/{len(prompts)}", end="\r", flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


def first_divergence(a: list, b: list) -> int:
    """Index of first differing token, or -1 if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))



def compare(path_a: str, path_b: str, report_path: str) -> None:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    assert len(a) == len(b), "different number of prompts"
    identical = 0
    diffs = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        assert ra["prompt"] == rb["prompt"], f"prompt {i} differs"
        if ra["text"] == rb["text"]:
            identical += 1
        else:
            diffs.append({
                "index": i,
                "first_divergence_token": first_divergence(ra["tokens"], rb["tokens"]),
                "text_a": ra["text"],
                "text_b": rb["text"],
            })
    report = {
        "n": len(a),
        "identical": identical,
        "identical_ratio": identical / len(a),
        "diffs": diffs,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"{identical}/{len(a)} identical ({report['identical_ratio']:.1%})")
    for d in diffs:
        print(f"  prompt {d['index']}: diverges at token {d['first_divergence_token']}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "collect":
        collect(int(sys.argv[2]), sys.argv[3])
    elif mode == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
