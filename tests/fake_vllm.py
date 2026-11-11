#!/usr/bin/env python3
"""Stand-in for the `vllm` command so the harness can be tested without a GPU.

    vllm serve MODEL --port P ...       a tiny HTTP server with /health,
                                        /v1/models, /metrics, /v1/completions
    vllm bench serve --dataset-path F   sends every prompt to that server and
                                        writes a result file shaped like vLLM's

A "token" here is a whitespace-separated word. The server keeps a prefix cache
of 16-token blocks like vLLM does, so cache hit counters behave realistically.

Environment variables switch on failure modes:
    FAKE_SERVE_EXIT_CODE=N       exit with code N before becoming ready
    FAKE_SERVE_WRONG_MODEL=1     /v1/models reports a different model
    FAKE_SERVE_DROP_METRIC=key   leave one metric out of /metrics
    FAKE_SERVE_IGNORE_SIGINT=1   only SIGKILL stops the server
    FAKE_SERVE_CHILD=path        start a child process, write its PID to path
    FAKE_BENCH_FAIL=N            report N failed requests
    FAKE_BENCH_EXIT_CODE=N       exit with code N without writing a result
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BLOCK = 16
METRIC_NAMES = {
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "num_preemptions": "vllm:num_preemptions_total",
    "kv_cache_usage_perc": "vllm:kv_cache_usage_perc",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "request_success": "vllm:request_success_total",
}


def serve(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--no-enable-prefix-caching", action="store_true")
    args, _ = parser.parse_known_args(argv)

    if os.environ.get("FAKE_SERVE_EXIT_CODE"):
        print("fake server: failing on purpose", flush=True)
        sys.exit(int(os.environ["FAKE_SERVE_EXIT_CODE"]))
    if os.environ.get("FAKE_SERVE_IGNORE_SIGINT"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if os.environ.get("FAKE_SERVE_CHILD"):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        with open(os.environ["FAKE_SERVE_CHILD"], "w") as f:
            f.write(str(child.pid))

    counters = dict.fromkeys(METRIC_NAMES, 0.0)
    cached_blocks: set[str] = set()
    lock = threading.Lock()
    caching = not args.no_enable_prefix_caching
    dropped = os.environ.get("FAKE_SERVE_DROP_METRIC", "")
    model = (
        "some-other-model" if os.environ.get("FAKE_SERVE_WRONG_MODEL") else args.model
    )

    def handle_completion(body: dict) -> dict:
        tokens = body["prompt"].split()
        with lock:
            counters["prompt_tokens"] += len(tokens)
            counters["generation_tokens"] += body["max_tokens"]
            counters["request_success"] += 1
            if caching:
                counters["prefix_cache_queries"] += len(tokens)
                chain, still_hitting = "", True
                for i in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK):
                    block = " ".join(tokens[i : i + BLOCK])
                    chain = hashlib.sha256((chain + block).encode()).hexdigest()
                    if still_hitting and chain in cached_blocks:
                        counters["prefix_cache_hits"] += BLOCK
                    else:
                        still_hitting = False
                        cached_blocks.add(chain)
        return {"choices": [{"text": "x " * body["max_tokens"]}]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

        def reply(self, payload, content_type="application/json") -> None:
            data = (
                payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.reply({})
            elif self.path == "/v1/models":
                card = {"id": model, "max_model_len": args.max_model_len}
                self.reply({"data": [card]})
            elif self.path == "/metrics":
                lines = []
                with lock:
                    for key, name in METRIC_NAMES.items():
                        if key != dropped:
                            lines.append(
                                f'{name}{{engine="0",model_name="m"}} {counters[key]}'
                            )
                self.reply(("\n".join(lines) + "\n").encode(), "text/plain")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.reply(handle_completion(body))

    print(f"non-default args: {{'port': {args.port}, 'model': '{args.model}'}}")
    print("GPU KV cache size: 459,936 tokens", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def bench(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--custom-output-len", type=int, required=True)
    parser.add_argument("--result-filename", required=True)
    args, _ = parser.parse_known_args(argv)

    if os.environ.get("FAKE_BENCH_EXIT_CODE"):
        sys.exit(int(os.environ["FAKE_BENCH_EXIT_CODE"]))
    with open(args.dataset_path) as f:
        prompts = [json.loads(line)["prompt"] for line in f][: args.num_prompts]
    n_out = args.custom_output_len
    t0 = time.perf_counter()
    start_times, latencies, ttfts = [], [], []
    for prompt in prompts:
        body = json.dumps({"prompt": prompt, "max_tokens": n_out}).encode()
        req = urllib.request.Request(
            args.base_url + "/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        urllib.request.urlopen(req).read()
        latency = time.perf_counter() - start
        start_times.append(start)
        latencies.append(latency + 0.001 * n_out)
        ttfts.append(latency)
    duration = time.perf_counter() - t0
    failed = int(os.environ.get("FAKE_BENCH_FAIL", "0"))
    n = len(prompts)
    result = {
        "completed": n - failed,
        "failed": failed,
        "duration": duration,
        "output_throughput": n * n_out / duration,
        "request_throughput": n / duration,
        "input_lens": [len(p.split()) for p in prompts],
        "output_lens": [n_out] * n,
        "ttfts": ttfts,
        "itls": [[0.001] * (n_out - 1) for _ in prompts],
        "latencies": latencies,
        "start_times": start_times,
        "queue_times": [0.0] * n,
        "generated_texts": ["x " * n_out] * n,
        "errors": [""] * n,
    }
    with open(args.result_filename, "w") as f:
        json.dump(result, f)


def main() -> None:
    if sys.argv[1] == "serve":
        serve(sys.argv[2:])
    elif sys.argv[1:3] == ["bench", "serve"]:
        bench(sys.argv[3:])
    else:
        sys.exit(f"fake vllm: unsupported command {sys.argv[1:]}")


if __name__ == "__main__":
    main()
