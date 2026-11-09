"""Shared helpers: load config, build result directory, wait for server."""

import time
from datetime import datetime
from pathlib import Path

import requests
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_result_dir(cfg: dict, root: str = "results") -> Path:
    """results/20261108_143022_p1024_c16_cache_on/"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path(root) / f"{stamp}_{cfg['name']}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def wait_for_server(port: int, timeout_s: int = 600) -> None:
    """Poll /health until the server answers, or raise after timeout."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(3)
    raise TimeoutError(f"Server on port {port} not ready after {timeout_s}s")
