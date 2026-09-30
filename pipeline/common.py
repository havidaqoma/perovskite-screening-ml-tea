"""Run bookkeeping: run directories, hashing, stage manifests, idempotent skip."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Iterable

from . import config

TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "pymatgen", "matminer", "xgboost",
                    "scikit-learn", "joblib", "optuna")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hash(modules: Iterable[str]) -> str:
    """Hash of the source files that define a stage (so code edits invalidate caches)."""
    h = hashlib.sha256()
    for name in sorted(modules):
        h.update((config.ROOT / "pipeline" / f"{name}.py").read_bytes())
    return h.hexdigest()


def env_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def git_commit() -> str | None:
    try:
        rev = subprocess.run(["git", "-C", str(config.ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(config.ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return None


class Stage:
    """Context for one stage: declares inputs, records outputs, writes stage_<name>.json.

    A stage is skipped when its manifest exists, all outputs still exist with the recorded
    hashes, and the input+code fingerprint is unchanged.
    """

    def __init__(self, run_dir: Path, name: str, inputs: dict[str, Path], modules: Iterable[str],
                 params: dict | None = None):
        self.run_dir = run_dir
        self.name = name
        self.inputs = inputs
        self.params = params or {}
        self.manifest_path = run_dir / f"stage_{name}.json"
        self.fingerprint = sha256_text(json.dumps({
            "inputs": {k: sha256_file(p) for k, p in sorted(inputs.items())},
            "code": code_hash(modules),
            "params": self.params,
        }, sort_keys=True))
        self.outputs: dict[str, Path] = {}
        self.metrics: dict = {}
        self._t0 = time.time()

    def up_to_date(self) -> bool:
        if not self.manifest_path.exists():
            return False
        m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if m.get("fingerprint") != self.fingerprint:
            return False
        for rel, digest in m.get("outputs", {}).items():
            p = self.run_dir / rel
            if not p.exists() or sha256_file(p) != digest:
                return False
        return True

    def path(self, rel: str) -> Path:
        p = self.run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        self.outputs[rel] = p
        return p

    def finish(self) -> dict:
        m = {
            "stage": self.name,
            "fingerprint": self.fingerprint,
            "inputs": {k: {"path": str(p.relative_to(config.ROOT)) if p.is_relative_to(config.ROOT) else str(p),
                           "sha256": sha256_file(p)} for k, p in self.inputs.items()},
            "params": self.params,
            "outputs": {rel: sha256_file(p) for rel, p in self.outputs.items()},
            "metrics": self.metrics,
            "env": env_versions(),
            "git": git_commit(),
            "seconds": round(time.time() - self._t0, 2),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        return m


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
