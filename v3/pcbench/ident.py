"""Hashes and identities.

A run identity covers everything that decides what a run measures: the resolved
run config, the prompt files, the model revision, the software environment and
the measurement code itself. Changing any of them gives a different identity, so
an old result can never be reused for a new measurement.
"""

import ast
import hashlib
import json
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

# Modules that decide what is sent, how it is timed and when a run is valid.
# The analysis and CLI modules are deliberately not listed: changing a plot or a
# help text must not invalidate measured data.
MEASUREMENT_MODULES = [
    "ident.py",
    "store.py",
    "spec.py",
    "workload.py",
    "envinfo.py",
    "server.py",
    "metrics.py",
    "bench.py",
    "schema.py",
    "runner.py",
]


def canonical_json(obj) -> str:
    """One fixed text form per JSON value: sorted keys, no extra whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint(source: str) -> str:
    """Hash of what a Python source file does, ignoring comments and layout.

    The source is parsed and the syntax tree is hashed, so reformatting or
    editing a comment keeps the hash while any change to the logic (or to a
    docstring, which is part of the tree) changes it.
    """
    tree = ast.parse(source)
    return sha256_bytes(ast.dump(tree, include_attributes=False).encode())


def measurement_code(package_dir: Path = PACKAGE_DIR) -> dict:
    """Fingerprint of every measurement module, plus one combined hash."""
    files = {}
    for name in MEASUREMENT_MODULES:
        files[name] = code_fingerprint((package_dir / name).read_text())
    return {"files": files, "sha256": sha256_json(files)}


def run_identity(
    protocol_version: str,
    run_config: dict,
    workload: dict,
    env_identity: dict,
    code_sha256: str,
) -> dict:
    """Everything a result depends on, and its hash."""
    body = {
        "protocol_version": protocol_version,
        "run_config": run_config,
        "workload": workload,
        "env_identity": env_identity,
        "measurement_code_sha256": code_sha256,
    }
    return {**body, "sha256": sha256_json(body)}
