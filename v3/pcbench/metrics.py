"""Server counters (Prometheus /metrics) and telemetry during a run.

Counters are read just before and just after the measured requests; their
difference describes those requests and nothing else. A counter that is missing
is an error, never a silent zero: "0 preemptions" must mean the counter was
read and did not move.
"""

import csv
import gzip
import io
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

from .store import RunError, atomic_write_bytes, atomic_write_json, atomic_write_text

# Left: our key, right: the vLLM metric name (a _total suffix is tolerated).
COUNTERS = {
    "request_success": "vllm:request_success",
    "prompt_tokens": "vllm:prompt_tokens",
    "generation_tokens": "vllm:generation_tokens",
    "prefix_cache_queries": "vllm:prefix_cache_queries",
    "prefix_cache_hits": "vllm:prefix_cache_hits",
    "num_preemptions": "vllm:num_preemptions",
}
REQUIRED_GAUGES = {"kv_cache_usage_perc": "vllm:kv_cache_usage_perc"}
# Recorded when present. Absent values are stored as null, never as 0.
OPTIONAL_GAUGES = {
    "num_requests_running": "vllm:num_requests_running",
    "num_requests_waiting": "vllm:num_requests_waiting",
}
WANTED = {**COUNTERS, **REQUIRED_GAUGES, **OPTIONAL_GAUGES}
# Each line looks like: name{labels} value   or   name value
LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$")


def parse(text: str) -> dict:
    """Turn Prometheus text into {our_name: value} for the metrics we want."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        name, _labels, value = m.groups()
        for ours, theirs in WANTED.items():
            if name == theirs or name == theirs + "_total":
                found[ours] = found.get(ours, 0.0) + float(value)
    return found


def fetch(port: int, timeout: float = 10) -> str:
    return requests.get(f"http://localhost:{port}/metrics", timeout=timeout).text


def snapshot(port: int, attempt_dir: Path, name: str) -> dict:
    """Fetch /metrics, save the raw text (gzipped) and the parsed values."""
    text = fetch(port)
    atomic_write_bytes(
        Path(attempt_dir) / f"{name}.txt.gz", gzip.compress(text.encode(), mtime=0)
    )
    parsed = parse(text)
    missing = [k for k in {**COUNTERS, **REQUIRED_GAUGES} if k not in parsed]
    if missing:
        raise RunError(f"/metrics is missing {sorted(missing)}")
    values = {k: parsed.get(k) for k in WANTED}
    atomic_write_json(Path(attempt_dir) / f"{name}.json", values)
    return values


def counter_deltas(before: dict, after: dict) -> dict:
    """after - before for every counter. Raises on a missing or reset counter."""
    deltas = {}
    for key in COUNTERS:
        if before.get(key) is None or after.get(key) is None:
            raise RunError(f"counter {key} is missing from a metrics snapshot")
        deltas[key] = after[key] - before[key]
        if deltas[key] < 0:
            raise RunError(
                f"counter {key} went down ({before[key]} -> {after[key]}); "
                "the server restarted during the run"
            )
    return deltas


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _numbers(rows: list[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        try:
            out.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            pass  # "[N/A]" or an absent column
    return out


class Telemetry:
    """Sample server gauges and GPU state while an attempt runs.

    The server is sampled by a thread that reads /metrics every interval_s
    seconds. The GPU is sampled by one long-lived nvidia-smi process writing CSV.
    Neither is part of the timed path; both run the same way for cache on and off.
    """

    def __init__(
        self, attempt_dir: Path, port: int, interval_s: float, gpu_fields: list[str]
    ):
        self.dir = Path(attempt_dir)
        self.port = port
        self.interval_s = interval_s
        self.gpu_fields = gpu_fields
        self.rows: list[dict] = []
        self.marks: dict[str, float] = {}
        self.scrape_errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._smi: subprocess.Popen | None = None
        self._smi_out = None

    def mark(self, name: str) -> None:
        self.marks[name] = time.time()

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.time()
            try:
                parsed = parse(fetch(self.port, timeout=5))
                self.rows.append(
                    {"t": round(t, 3), **{k: parsed.get(k) for k in WANTED}}
                )
            except requests.RequestException:
                self.scrape_errors += 1
            self._stop.wait(max(0.0, self.interval_s - (time.time() - t)))

    def start(self) -> None:
        self.mark("telemetry_start")
        if self.gpu_fields:
            self._smi_out = open(self.dir / "gpu_telemetry.csv", "wb")
            try:
                self._smi = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=" + ",".join(self.gpu_fields),
                        "--format=csv,nounits",
                        "-lms",
                        str(int(self.interval_s * 1000)),
                    ],
                    stdout=self._smi_out,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                self._smi = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """Stop sampling, write the files and return a summary."""
        self.mark("telemetry_stop")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        if self._smi is not None:
            self._smi.terminate()
            try:
                self._smi.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._smi.kill()
                self._smi.wait()
        if self._smi_out is not None:
            self._smi_out.close()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["t", *WANTED])
        writer.writeheader()
        writer.writerows(self.rows)
        atomic_write_text(self.dir / "server_timeseries.csv", buf.getvalue())
        return self.summary()

    def gpu_rows(self) -> list[dict]:
        path = self.dir / "gpu_telemetry.csv"
        if not path.is_file():
            return []
        rows = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                # Column titles look like "temperature.gpu" or "power.draw [W]".
                clean = {
                    k.split(" [")[0].strip(): (v or "").strip()
                    for k, v in row.items()
                    if k
                }
                try:
                    stamp = datetime.strptime(
                        clean["timestamp"], "%Y/%m/%d %H:%M:%S.%f"
                    )
                    clean["t"] = stamp.timestamp()
                except (KeyError, ValueError):
                    clean["t"] = None
                rows.append(clean)
        return rows

    def summary(self) -> dict:
        lo, hi = self.marks.get("main_start"), self.marks.get("main_end")

        def in_window(row: dict) -> bool:
            return (
                lo is not None
                and hi is not None
                and row.get("t")
                and lo <= row["t"] <= hi
            )

        server = [r for r in self.rows if in_window(r)]
        gpu_all = self.gpu_rows()
        gpu = [r for r in gpu_all if in_window(r)]
        running = _numbers(server, "num_requests_running")
        waiting = _numbers(server, "num_requests_waiting")
        kv = _numbers(server, "kv_cache_usage_perc")
        temp = _numbers(gpu, "temperature.gpu")
        power = _numbers(gpu, "power.draw")
        util = _numbers(gpu, "utilization.gpu")
        mem = _numbers(gpu, "memory.used")
        return {
            "marks": self.marks,
            "interval_s": self.interval_s,
            "gpu_fields": self.gpu_fields,
            "server_samples_total": len(self.rows),
            "server_samples_in_main": len(server),
            "server_scrape_errors": self.scrape_errors,
            "gpu_samples_total": len(gpu_all),
            "gpu_samples_in_main": len(gpu),
            "gpu_uuids": sorted({r["uuid"] for r in gpu_all if r.get("uuid")}),
            "server_running_mean": _mean(running),
            "server_running_max": max(running) if running else None,
            "server_waiting_mean": _mean(waiting),
            "server_waiting_max": max(waiting) if waiting else None,
            "kv_cache_usage_mean": _mean(kv),
            "kv_cache_usage_max": max(kv) if kv else None,
            "gpu_temperature_c_mean": _mean(temp),
            "gpu_temperature_c_max": max(temp) if temp else None,
            "gpu_power_w_mean": _mean(power),
            "gpu_utilization_pct_mean": _mean(util),
            "gpu_memory_used_mib_max": max(mem) if mem else None,
        }
