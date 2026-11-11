"""Scrape vLLM /metrics and save both the raw text and parsed key numbers.

Usage:
    python scripts/scrape_metrics.py <port> <out_prefix>
    -> writes <out_prefix>.txt (raw) and <out_prefix>.json (parsed)

A metric that is missing is an error, never a silent zero: "0 preemptions"
must mean the counter was read and did not move.
"""

import json
import re
import sys

import requests

from common import RunError

# Metrics we care about. Left: our key, right: the vLLM metric name prefix.
WANTED = {
    "prefix_cache_queries": "vllm:prefix_cache_queries",
    "prefix_cache_hits": "vllm:prefix_cache_hits",
    "num_preemptions": "vllm:num_preemptions",
    "kv_cache_usage_perc": "vllm:kv_cache_usage_perc",
    "prompt_tokens": "vllm:prompt_tokens",
    "generation_tokens": "vllm:generation_tokens",
    "request_success": "vllm:request_success",
}

# Everything except the usage gauge only ever goes up while a server lives.
COUNTERS = [k for k in WANTED if k != "kv_cache_usage_perc"]

# Each line looks like: name{labels} value   or   name value
LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$")


def parse(text: str) -> dict:
    """Turn Prometheus text into {our_name: value} for the metrics we want."""
    found = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        name, _labels, value = m.groups()
        for ours, theirs in WANTED.items():
            # tolerate the _total suffix being present or absent
            if name == theirs or name == theirs + "_total":
                found[ours] = found.get(ours, 0.0) + float(value)
    return found


def scrape(port: int | str, out_prefix: str) -> dict:
    """Fetch, save and parse /metrics. Raises if any wanted metric is absent."""
    text = requests.get(f"http://localhost:{port}/metrics", timeout=10).text
    with open(out_prefix + ".txt", "w") as f:
        f.write(text)
    parsed = parse(text)
    with open(out_prefix + ".json", "w") as f:
        json.dump(parsed, f, indent=2)
    missing = [k for k in WANTED if k not in parsed]
    if missing:
        raise RunError(f"/metrics is missing {missing}")
    return parsed


def counter_deltas(before: dict, after: dict) -> dict:
    """after - before for every counter. Raises on a missing or reset counter."""
    deltas = {}
    for key in COUNTERS:
        if key not in before or key not in after:
            raise RunError(f"counter {key} is missing from a metrics snapshot")
        deltas[key] = after[key] - before[key]
        if deltas[key] < 0:
            raise RunError(
                f"counter {key} went down ({before[key]} -> {after[key]}); "
                "the server restarted during the run"
            )
    return deltas


def main() -> None:
    port, out_prefix = sys.argv[1], sys.argv[2]
    print(json.dumps(scrape(port, out_prefix), indent=2))


if __name__ == "__main__":
    main()
