"""Stand-in for the `vllm` command so the v3 harness can be tested without a GPU.

    vllm serve MODEL --port P ...       a tiny HTTP server with /health,
                                        /v1/models, /metrics, /v1/completions
    vllm bench serve --dataset-path F   sends every prompt to that server and
                                        writes a result file shaped like vLLM's

A "token" here is a whitespace-separated word. The server keeps a prefix cache
of 16-token blocks like vLLM does, so the hit counters behave realistically.

Environment variables switch on failure modes:
    FAKE_SERVE_EXIT_CODE=N       exit with code N before becoming ready
    FAKE_SERVE_WRONG_MODEL=1     /v1/models reports a different model
    FAKE_SERVE_DROP_METRIC=key   leave one metric out of /metrics
    FAKE_SERVE_IGNORE_SIGINT=1   only SIGKILL stops the server
    FAKE_SERVE_RESET_AFTER=N     counters jump back to 0 after N requests
    FAKE_SERVE_HTTP_500_AFTER=N  completions answer 500 after N requests
    FAKE_BENCH_EXIT_CODE=N       exit with code N without writing a result
    FAKE_BENCH_FAIL_ONCE=path    exit 1 the first time (path is the marker file)
    FAKE_BENCH_SLEEP=S           sleep S seconds before sending anything
    FAKE_BENCH_SHORT_OUTPUT=1    report one request with too few output tokens
    FAKE_BENCH_DROP_FIELD=name   leave one per-request field out of the result
    FAKE_BENCH_TRUNCATE=1        write only the first half of the result file
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.error
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
    "num_requests_running": "vllm:num_requests_running",
    "num_requests_waiting": "vllm:num_requests_waiting",
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
    counters = dict.fromkeys(METRIC_NAMES, 0.0)
    cached_blocks: set[str] = set()
    lock = threading.Lock()
    caching = not args.no_enable_prefix_caching
    dropped = os.environ.get("FAKE_SERVE_DROP_METRIC", "")
    reset_after = int(os.environ.get("FAKE_SERVE_RESET_AFTER", "0"))
    fail_after = int(os.environ.get("FAKE_SERVE_HTTP_500_AFTER", "0"))
    seen = {"n": 0}
    model = (
        "some-other-model" if os.environ.get("FAKE_SERVE_WRONG_MODEL") else args.model
    )

    def handle_completion(body: dict) -> dict | None:
        tokens = body["prompt"].split()
        with lock:
            seen["n"] += 1
            if fail_after and seen["n"] > fail_after:
                return None
            if reset_after and seen["n"] == reset_after + 1:
                for key in counters:
                    counters[key] = 0.0
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
                self.reply(
                    {"data": [{"id": model, "max_model_len": args.max_model_len}]}
                )
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
            answer = handle_completion(body)
            if answer is None:
                self.send_error(500, "fake server: failing on purpose")
            else:
                self.reply(answer)

    print(f"non-default args: {{'port': {args.port}, 'model': '{args.model}'}}")
    print("Chunked prefill is enabled with max_num_batched_tokens=2048.")
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
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--result-filename", required=True)
    args, _ = parser.parse_known_args(argv)
    if os.environ.get("FAKE_BENCH_EXIT_CODE"):
        sys.exit(int(os.environ["FAKE_BENCH_EXIT_CODE"]))
    marker = os.environ.get("FAKE_BENCH_FAIL_ONCE")
    if marker and not os.path.exists(marker):
        open(marker, "w").close()
        sys.exit(1)
    time.sleep(float(os.environ.get("FAKE_BENCH_SLEEP", "0")))
    with open(args.dataset_path) as f:
        prompts = [json.loads(line)["prompt"] for line in f][: args.num_prompts]
    n_out, n = args.custom_output_len, len(prompts)
    # Timestamps are synthetic but consistent with a closed loop of
    # --max-concurrency slots: a request starts when a slot becomes free.
    t0 = time.perf_counter()
    slot_free = [t0] * args.max_concurrency
    out = {
        k: [] for k in ("start_times", "queue_times", "ttfts", "latencies", "errors")
    }
    out["output_lens"], out["generated_texts"] = [], []
    for i, prompt in enumerate(prompts):
        body = json.dumps({"prompt": prompt, "max_tokens": n_out}).encode()
        req = urllib.request.Request(
            args.base_url + "/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        slot = min(range(len(slot_free)), key=slot_free.__getitem__)
        start = slot_free[slot]
        try:
            text = json.loads(urllib.request.urlopen(req).read())["choices"][0]["text"]
            ttft = 0.010 + 0.001 * (i % 7)
            latency = ttft + 0.002 * (n_out - 1)
            error, tokens = "", n_out
        except urllib.error.HTTPError as e:
            text, ttft, latency, error, tokens = "", 0.0, 0.0, f"HTTP {e.code}", 0
        slot_free[slot] = start + latency
        out["start_times"].append(start)
        out["queue_times"].append(start - t0)
        out["ttfts"].append(ttft)
        out["latencies"].append(latency)
        out["errors"].append(error)
        out["output_lens"].append(tokens)
        out["generated_texts"].append(text)
    if os.environ.get("FAKE_BENCH_SHORT_OUTPUT"):
        out["output_lens"][-1] = n_out - 1
    failed = sum(1 for e in out["errors"] if e)
    duration = max(slot_free) - t0
    result = {
        "date": "20260101-000000",
        "backend": "openai",
        "model_id": "fake/model",
        "num_prompts": n,
        "request_rate": "inf",
        "max_concurrency": args.max_concurrency,
        "completed": n - failed,
        "failed": failed,
        "duration": duration,
        "output_throughput": sum(out["output_lens"]) / duration if duration else 0.0,
        "request_throughput": (n - failed) / duration if duration else 0.0,
        "max_concurrent_requests": args.max_concurrency,
        "input_lens": [len(p.split()) for p in prompts],
        "itls": [[0.002] * (n_out - 1) for _ in prompts],
        **out,
    }
    result.pop(os.environ.get("FAKE_BENCH_DROP_FIELD", ""), None)
    data = json.dumps(result)
    if os.environ.get("FAKE_BENCH_TRUNCATE"):
        data = data[: len(data) // 2]
    with open(args.result_filename, "w") as f:
        f.write(data)


def main() -> None:
    if sys.argv[1] == "serve":
        serve(sys.argv[2:])
    elif sys.argv[1:3] == ["bench", "serve"]:
        bench(sys.argv[3:])
    else:
        sys.exit(f"fake vllm: unsupported command {sys.argv[1:]}")


if __name__ == "__main__":
    main()
