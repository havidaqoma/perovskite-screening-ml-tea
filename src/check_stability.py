"""Materials Project energy-above-hull lookup and stability gating.

The default path is deliberately offline: cached values are used when present
and every missing value is reported as unknown.  The Materials Project branch
is available for an explicitly requested online run, but is never selected by
default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# Small deterministic fixture used by offline tests and examples.
SAMPLE_EHULL = {
    "Cs2AgBiBr6": 12.0,
    "Cs2SnI6": 40.0,
    "Xu2ZrCl6": None,
}


def _read_cache(cache_path: str | Path) -> dict[str, Any]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _record_ehull(record: Any) -> float | None:
    """Extract a numeric eV/atom or meV value from a cache record.

    The on-disk contract stores ``ehull`` as meV.  A numeric shorthand is
    accepted as a convenience for small hand-written fixtures.
    """

    if record is None:
        return None
    if isinstance(record, dict):
        value = record.get("ehull")
    else:
        value = record
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _doc_value(doc: Any, name: str, default: Any = None) -> Any:
    if isinstance(doc, dict):
        return doc.get(name, default)
    return getattr(doc, name, default)


def _status_for_ehull(ehull_meV: float | None, threshold_meV: float = 35.0) -> str:
    if ehull_meV is None or not np.isfinite(ehull_meV):
        return "unknown"
    return "stable" if ehull_meV <= threshold_meV else "unstable"


def _write_cache(cache_path: str | Path, cache: dict[str, Any]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def load_or_query_ehull(
    formulas: Iterable[str],
    api_key: str | None = None,
    cache_path: str | Path = "data/ehull_cache.json",
    online: bool = False,
) -> pd.DataFrame:
    """Load energy-above-hull values, optionally querying Materials Project.

    Offline operation is selected whenever ``online`` is false or no API key
    is supplied.  Cache values are interpreted as meV/atom and are classified
    using the canonical 35 meV threshold for the returned status.  The gate
    function below recomputes the classification for any requested threshold.
    """

    requested = list(formulas)
    cache = _read_cache(cache_path)
    values: dict[str, float | None] = {}

    if online and api_key is not None:
        # Keep the dependency and network-capable branch lazy so offline users
        # do not need to initialize or contact the Materials Project client.
        try:
            from mp_api.client import MPRester

            unique_formulas = list(dict.fromkeys(requested))
            with MPRester(api_key) as mpr:
                docs = list(mpr.materials.summary.search(
                    formula=unique_formulas,
                    fields=["formula_pretty", "energy_above_hull"],
                ))

                # Some MP client deployments expose hull data through the
                # chemenv endpoint.  Use it only as a compatible fallback.
                if not docs:
                    chemenv = getattr(mpr.materials, "chemenv", None)
                    chemenv_search = getattr(chemenv, "search", None)
                    if callable(chemenv_search):
                        try:
                            docs = list(chemenv_search(
                                formula=unique_formulas,
                                fields=["formula_pretty", "energy_above_hull"],
                            ))
                        except Exception:
                            docs = []

            # MP reports energy_above_hull in eV/atom; this module exposes meV.
            for doc in docs:
                formula = _doc_value(doc, "formula_pretty") or _doc_value(doc, "formula")
                ehull_eV = _doc_value(doc, "energy_above_hull")
                if formula is None or ehull_eV is None:
                    continue
                try:
                    ehull_meV = float(ehull_eV) * 1000.0
                except (TypeError, ValueError):
                    continue
                if np.isfinite(ehull_meV):
                    # If polymorphs are returned, retain the lowest hull value.
                    previous = values.get(str(formula))
                    values[str(formula)] = (
                        ehull_meV if previous is None else min(previous, ehull_meV)
                    )

            # Material Project formula matching can return a canonical formula
            # spelling.  Also accept exact requested-formula keys.
            for formula in unique_formulas:
                if formula not in values:
                    values[formula] = None

            cache = {
                formula: {
                    "ehull": values.get(formula),
                    "status": "cached" if values.get(formula) is not None else "unknown",
                }
                for formula in unique_formulas
            }
            _write_cache(cache_path, cache)
        except Exception:
            # A failed online lookup must not turn into a stability pass.
            values = {formula: None for formula in dict.fromkeys(requested)}
    else:
        # Missing cache entries intentionally remain unknown.
        for formula in dict.fromkeys(requested):
            values[formula] = _record_ehull(cache.get(formula)) if formula in cache else None

    rows = [
        {
            "Formula": formula,
            "ehull_meV": values.get(formula),
            "status": _status_for_ehull(values.get(formula)),
        }
        for formula in requested
    ]
    result = pd.DataFrame(rows, columns=["Formula", "ehull_meV", "status"])
    if not result.empty:
        result["ehull_meV"] = pd.to_numeric(result["ehull_meV"], errors="coerce").astype(float)
    return result


def apply_stability_gate(
    df: pd.DataFrame, threshold_meV: float = 35.0
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only candidates at or below an energy-above-hull threshold.

    Missing hull energies are unknown and therefore fail closed at every
    threshold.  The summary always includes the standard 20/35/50 meV counts
    so scenario comparisons use the same denominator.
    """

    if "ehull_meV" not in df.columns:
        raise KeyError("apply_stability_gate requires an 'ehull_meV' column")

    gated = df.copy()
    ehull = pd.to_numeric(gated["ehull_meV"], errors="coerce")
    unknown = ehull.isna()
    if "status" in gated.columns:
        unknown = unknown | gated["status"].astype(str).eq("unknown")
    gated["ehull_meV"] = ehull.astype(float)
    gated["status"] = np.select(
        [unknown, ehull <= float(threshold_meV)],
        ["unknown", "stable"],
        default="unstable",
    )

    summary = {
        "n_total": int(len(gated)),
        "n_kept_20": int((~unknown & (ehull <= 20.0)).sum()),
        "n_kept_35": int((~unknown & (ehull <= 35.0)).sum()),
        "n_kept_50": int((~unknown & (ehull <= 50.0)).sum()),
        "n_unknown": int(unknown.sum()),
    }
    return gated[gated["status"] == "stable"].copy(), summary
