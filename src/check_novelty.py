"""Fail-closed Materials Project novelty validation.

Offline cache lookup is the default.  A candidate is novel only when an
explicit, successfully resolved cache record says that no MP material ID was
found.  Missing records and lookup errors remain unknown and fail closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pymatgen.core import Composition


SAMPLE_NOVELTY = {
    "Cs2AgBiBr6": {"mp_id": "mp-123", "is_synthesized": True},
    "Xu2ZrCl6": None,
}


REPORT_COLUMNS = [
    "Formula",
    "normalized_formula",
    "mp_id",
    "is_synthesized",
    "status",
]


def _read_cache(cache_path: str | Path) -> dict[str, Any]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _write_cache(cache_path: str | Path, cache: dict[str, Any]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def _doc_value(doc: Any, name: str, default: Any = None) -> Any:
    if isinstance(doc, dict):
        return doc.get(name, default)
    return getattr(doc, name, default)


def _normalize_formula(formula: Any) -> str | None:
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:
        return None


def _row_from_record(
    formula: Any,
    normalized_formula: str | None,
    record: Any,
    resolved: bool,
) -> dict[str, Any]:
    """Convert a cache record to a report row without optimistic inference."""

    if not resolved or normalized_formula is None:
        return {
            "Formula": formula,
            "normalized_formula": normalized_formula,
            "mp_id": None,
            "is_synthesized": None,
            "status": "unknown",
        }

    # ``None`` is accepted as a compact explicit no-match fixture.
    if record is None:
        mp_id = None
        synthesized = False
    elif isinstance(record, dict):
        mp_id = record.get("mp_id")
        synthesized = bool(record.get("is_synthesized", False))
    else:
        mp_id = None
        synthesized = False

    if mp_id is not None and str(mp_id) != "":
        status = "known"
        mp_id = str(mp_id)
    else:
        status = "novel"
        mp_id = None

    return {
        "Formula": formula,
        "normalized_formula": normalized_formula,
        "mp_id": mp_id,
        "is_synthesized": synthesized,
        "status": status,
    }


def validate_novelty(
    formulas: Iterable[str],
    mp_api_key: str | None = None,
    cache_path: str | Path = "data/mp_novelty_cache.json",
    online: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Resolve formulas against Materials Project and fail closed.

    ``online`` is honored only when ``mp_api_key`` is present.  In offline
    mode, a cache entry with ``mp_id=None`` is an explicit resolved no-match
    (``novel``); a missing cache entry is ``unknown`` and causes exit code 1.
    """

    requested = list(formulas)
    normalized = [_normalize_formula(formula) for formula in requested]
    cache: dict[str, Any] = {}
    api_error = False

    if online and mp_api_key is not None:
        try:
            from mp_api.client import MPRester

            unique_normalized = list(dict.fromkeys(value for value in normalized if value is not None))
            resolved_records: dict[str, Any] = {}
            with MPRester(mp_api_key) as mpr:
                docs = mpr.summary.search(
                    formula=unique_normalized,
                    fields=["material_id", "formula_pretty", "theoretical"],
                )

            for doc in docs:
                doc_formula = _doc_value(doc, "formula_pretty") or _doc_value(doc, "formula")
                doc_normalized = _normalize_formula(doc_formula) if doc_formula is not None else None
                if doc_normalized is None or doc_normalized in resolved_records:
                    continue
                resolved_records[doc_normalized] = {
                    "mp_id": _doc_value(doc, "material_id"),
                    "is_synthesized": _doc_value(doc, "theoretical") is False,
                }

            # A successful query resolves an empty result as explicitly novel.
            cache = {
                formula: resolved_records.get(formula, {"mp_id": None, "is_synthesized": False})
                for formula in unique_normalized
            }
            _write_cache(cache_path, cache)
        except Exception:
            api_error = True
            cache = {}
    else:
        try:
            cache = _read_cache(cache_path)
        except Exception:
            # A corrupt/unreadable cache is indistinguishable from an
            # unresolved lookup for fail-closed purposes.
            cache = {}

    rows = []
    for formula, normalized_formula in zip(requested, normalized):
        resolved = normalized_formula is not None and normalized_formula in cache and not api_error
        record = cache.get(normalized_formula) if normalized_formula is not None else None
        rows.append(_row_from_record(formula, normalized_formula, record, resolved))

    report_df = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    exit_code = 1 if api_error or (report_df["status"] == "unknown").any() else 0
    return report_df, exit_code
