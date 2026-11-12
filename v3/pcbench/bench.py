"""Run `vllm bench serve` on one prompt file and turn its output into rows.

`vllm bench serve` is the load generator and the clock: it sends the prompts,
times every request and writes one JSON file. This module pins every option
that could change what is sent, enforces a time limit, and converts the
per-request arrays into one record per request (requests.jsonl).
"""

import json
import math
import os
import signal
import subprocess
from pathlib import Path

from .ident import sha256_bytes
from .server import group_members, proc_start_time
from .store import RunError, atomic_write_json, read_json

PER_REQUEST_FIELDS = [
    "start_times",
    "queue_times",
    "ttfts",
    "latencies",
    "input_lens",
    "output_lens",
    "errors",
    "generated_texts",
]


def build_bench_cmd(
    cfg: dict, dataset_path: Path, num_prompts: int, out_file: Path
) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--endpoint",
        "/v1/completions",
        "--base-url",
        f"http://localhost:{cfg['server']['port']}",
        "--model",
        cfg["model"],
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_path),
        "--skip-chat-template",  # send the prompt exactly as generated
        "--disable-shuffle",  # request i is line i of the file
        "--no-oversample",  # never repeat a prompt to fill the count
        "--custom-output-len",
        str(cfg["workload"]["output_len"]),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(cfg["concurrency"]),
        "--request-rate",
        "inf",  # closed loop: a new request starts as soon as a slot is free
        "--ignore-eos",  # every request generates exactly output_len tokens
        "--temperature",
        str(cfg["bench"]["temperature"]),
        # The tool's own warm-up and ready check re-send the first prompt, which
        # would put a measured prompt into the cache. Both are switched off;
        # the harness warms up with its own, different prompts.
        "--num-warmups",
        "0",
        "--ready-check-timeout-sec",
        "0",
        "--seed",
        "0",
        "--save-result",
        "--save-detailed",
        "--result-filename",
        str(out_file),
    ]


def run_bench_cmd(cmd: list[str], log_path: Path, timeout_s: float, what: str) -> None:
    """Run the benchmark in its own process group and enforce the time limit.

    The process is recorded in a small file while it lives, so that a later
    invocation can stop it if this runner is killed.
    """
    record = log_path.with_suffix(".proc.json")
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            atomic_write_json(
                record, {"pid": proc.pid, "proc_start_time": proc_start_time(proc.pid)}
            )
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            code = None
        finally:
            # Also reached on Ctrl+C or any other error: never leave it running.
            _kill_group(proc)
            record.unlink(missing_ok=True)
    if code is None:
        raise RunError(f"{what}: vllm bench serve exceeded {timeout_s} s")
    if code != 0:
        raise RunError(f"{what}: vllm bench serve exited with code {code}")


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def kill_recorded_bench(attempt_dir: Path) -> list[str]:
    """Stop benchmark processes an interrupted attempt left behind."""
    notes = []
    for record in sorted(Path(attempt_dir).glob("bench_*.proc.json")):
        try:
            info = read_json(record)
            pid, recorded = info["pid"], info.get("proc_start_time", "")
        except (ValueError, KeyError, TypeError, OSError):
            continue
        current = proc_start_time(pid)
        if recorded and current and recorded != current:
            notes.append(f"benchmark PID {pid} was reused, nothing stopped")
        elif group_members(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            notes.append(f"benchmark process {pid} killed")
        record.unlink(missing_ok=True)
    return notes


def to_rows(raw: dict, num_prompts: int, what: str) -> list[dict]:
    """One record per request, in the order of the prompt file.

    Every per-request array of the tool must have exactly num_prompts entries:
    a failed request still has an entry (with an error text), so nothing is
    silently dropped.
    """
    for field in PER_REQUEST_FIELDS:
        value = raw.get(field)
        if not isinstance(value, list) or len(value) != num_prompts:
            raise RunError(
                f"{what}: per-request field `{field}` is missing or does not have "
                f"{num_prompts} entries"
            )
    t0 = min(raw["start_times"])
    rows = []
    for i in range(num_prompts):
        error = raw["errors"][i] or ""
        text = raw["generated_texts"][i] or ""
        rows.append(
            {
                "i": i,
                "ok": error == "",
                "error": str(error),
                "start_s": raw["start_times"][i] - t0,
                "client_queue_s": raw["queue_times"][i],
                "ttft_s": raw["ttfts"][i],
                "e2e_s": raw["latencies"][i],
                "input_tokens": raw["input_lens"][i],
                "output_tokens": raw["output_lens"][i],
                "output_sha256": sha256_bytes(text.encode())[:16],
            }
        )
    return rows


def _finite(value):
    """Replace non-finite numbers by null, at any depth."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def aggregates(raw: dict) -> dict:
    """The tool's own summary numbers (everything that is not a per-request list).

    Kept for reference only: the harness recomputes its metrics from the rows.
    A non-finite number is stored as null.
    """
    return {
        key: _finite(value) for key, value in raw.items() if not isinstance(value, list)
    }


def run_bench(
    cfg: dict,
    dataset_path: Path,
    num_prompts: int,
    out_dir: Path,
    kind: str,
    timeout_s: float,
) -> tuple[list[dict], dict]:
    """Run one pass. Returns (rows, the tool's aggregates)."""
    raw_path = Path(out_dir) / f"bench_{kind}.raw.json"
    cmd = build_bench_cmd(cfg, dataset_path, num_prompts, raw_path)
    print("  $ " + " ".join(cmd), flush=True)
    run_bench_cmd(cmd, Path(out_dir) / f"bench_{kind}.log", timeout_s, kind)
    if not raw_path.is_file():
        raise RunError(f"{kind}: vllm bench serve wrote no result file")
    try:
        raw = json.loads(raw_path.read_text())
    except ValueError as e:
        raise RunError(f"{kind}: the result file is not valid JSON ({e})") from e
    rows = to_rows(raw, num_prompts, kind)
    agg = aggregates(raw)
    agg["cmd"] = cmd
    # The raw file repeats every generated text and every inter-token gap; the
    # rows keep what the analysis needs plus a hash of each text.
    raw_path.unlink()
    return rows, agg
