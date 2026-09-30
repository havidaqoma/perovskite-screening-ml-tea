"""Materials Project access: one cached, fail-loud client for every query in the pipeline.

* The API key is read from the MP_API_KEY environment variable or a local, gitignored `.env`.
  It is never logged or written to disk. Anyone reproducing this work must register their own
  key at https://next-gen.materialsproject.org/api
* Every query result is cached under data/mp_cache/<name>.csv, with a sidecar <name>.meta.json
  recording the query, MP database version, UTC timestamp, row count and sha256. Re-runs read
  the cache and never hit the network unless refresh=True.
* A failure raises. A query can never silently return "0 results".
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd

from . import config
from .common import sha256_file

CACHE_DIR = config.DATA_DIR / "mp_cache"


def api_key() -> str:
    key = os.environ.get("MP_API_KEY", "").strip()
    if not key:
        env = config.ROOT / ".env"
        if env.exists():
            m = re.search(r"^MP_API_KEY=(.*)$", env.read_text(encoding="utf-8"), re.M)
            if m:
                key = m.group(1).strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("MP_API_KEY not set. Register at https://next-gen.materialsproject.org/api "
                           "and put MP_API_KEY=<key> in the environment or in ./.env (gitignored).")
    return key


def _rester():
    from mp_api.client import MPRester
    return MPRester(api_key(), mute_progress_bars=True)


def cached_summary(name: str, query: dict, fields: list[str], refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """materials.summary.search(**query, fields=fields) -> DataFrame, cached as CSV + meta."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv, meta_path = CACHE_DIR / f"{name}.csv", CACHE_DIR / f"{name}.meta.json"
    if csv.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("query") != _jsonable(query) or meta.get("fields") != fields:
            raise RuntimeError(f"cache {name} was built with a different query; rerun with refresh=True")
        if sha256_file(csv) != meta["sha256"]:
            raise RuntimeError(f"cache {name} was modified on disk (sha256 mismatch)")
        return pd.read_csv(csv), meta

    t0 = time.time()
    with _rester() as mpr:
        db_version = mpr.get_database_version()
        docs = mpr.materials.summary.search(**query, fields=fields)
    rows = []
    for d in docs:
        row = {}
        for f in fields:
            v = getattr(d, f, None)
            if f == "material_id" and v is not None:
                v = str(v)
            elif f in ("elements",) and v is not None:
                v = "-".join(sorted(str(e) for e in v))
            elif hasattr(v, "value"):
                v = v.value
            row[f] = v
        rows.append(row)
    if not rows:
        raise RuntimeError(f"MP query {name} returned 0 documents: treating as failure, not as a result")
    df = pd.DataFrame(rows, columns=fields)
    df.to_csv(csv, index=False)
    meta = {"name": name, "query": _jsonable(query), "fields": fields, "mp_database_version": db_version,
            "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_rows": int(len(df)),
            "seconds": round(time.time() - t0, 1), "sha256": sha256_file(csv)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df, meta


def _jsonable(q: dict) -> dict:
    return json.loads(json.dumps(q, default=list))
