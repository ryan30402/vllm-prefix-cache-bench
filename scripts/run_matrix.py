"""Expand a sweep config into individual runs and execute them all.

Usage:
    python scripts/run_matrix.py configs/matrix_main.yaml

Each run gets its own folder under results/ containing config.yaml, env.json,
server.log, metrics_before/after, warmup.json, bench_result.json, status.json.
Safe to re-run: completed runs (status.json says ok) are skipped.
"""

import itertools
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

from common import load_config

PY = sys.executable
SCRIPTS = Path(__file__).parent


def expand(matrix: dict) -> list[dict]:
    """Turn the sweep lists into one flat config dict per combination per repeat."""
    sweep = matrix["sweep"]
    keys = list(sweep.keys())
    runs = []
    for combo in itertools.product(*[sweep[k] for k in keys]):
        params = dict(zip(keys, combo))
        for rep in range(matrix["repeats"]):
            cfg = {
                "model": matrix["model"],
                "server": {
                    **matrix["server_base"],
                    "prefix_caching": params["prefix_caching"],
                },
                "bench": {
                    **matrix["bench_base"],
                    "prefix_len": params["prefix_len"],
                    "max_concurrency": params["max_concurrency"],
                },
                "repeat": rep,
            }
            cache = "on" if params["prefix_caching"] else "off"
            cfg["name"] = (
                f"p{params['prefix_len']}_c{params['max_concurrency']}_cache_{cache}_r{rep}"
            )
            cfg["tag"] = matrix.get("tag", "main")
            if matrix.get("tag"):
                cfg["name"] = f"{matrix['tag']}_{cfg['name']}"
            runs.append(cfg)
    return runs


def already_done(run_dir: Path) -> bool:
    status = run_dir / "status.json"
    return status.exists() and json.loads(status.read_text()).get("ok") is True


def sh(args: list[str]) -> None:
    """Run a command, raise if it fails."""
    print("  $", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def run_one(cfg: dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    port = str(cfg["server"]["port"])
    t0 = time.time()
    try:
        sh([PY, str(SCRIPTS / "capture_env.py"), str(run_dir / "env.json")])
        sh([PY, str(SCRIPTS / "start_server.py"), str(cfg_path), str(run_dir)])
        sh(
            [
                PY,
                str(SCRIPTS / "scrape_metrics.py"),
                port,
                str(run_dir / "metrics_before"),
            ]
        )
        sh([PY, str(SCRIPTS / "run_bench.py"), str(cfg_path), str(run_dir)])
        sh(
            [
                PY,
                str(SCRIPTS / "scrape_metrics.py"),
                port,
                str(run_dir / "metrics_after"),
            ]
        )
        status = {"ok": True, "seconds": round(time.time() - t0)}
    except Exception:
        status = {
            "ok": False,
            "error": traceback.format_exc(),
            "seconds": round(time.time() - t0),
        }
        print(status["error"])
    finally:
        if (run_dir / "server.pid").exists():
            subprocess.run([PY, str(SCRIPTS / "stop_server.py"), str(run_dir)])
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))


def main() -> None:
    matrix = load_config(sys.argv[1])
    runs = expand(matrix)
    results_root = Path("results")
    print(f"{len(runs)} runs planned")
    for i, cfg in enumerate(runs, 1):
        run_dir = results_root / cfg["name"]
        if already_done(run_dir):
            print(f"[{i}/{len(runs)}] skip {cfg['name']} (done)")
            continue
        print(f"[{i}/{len(runs)}] run  {cfg['name']}")
        run_one(cfg, run_dir)
    print("Matrix complete")


if __name__ == "__main__":
    main()
