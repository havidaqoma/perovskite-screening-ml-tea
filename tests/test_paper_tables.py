"""The ESI workbook must hold the same data that the figures plot (round-5 review: Fig1_funnel sheet had drifted)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))


def test_fig1_funnel_ends_at_stability_gated_pool_and_front():
    from make_figures_v2 import funnel_counts
    pa = json.loads((ROOT / "runs/v2/metrics/pareto.json").read_text(encoding="utf-8"))
    legacy, v2 = funnel_counts()
    assert [s for s, _, _ in v2[-2:]] == ["stability-gated pool", "stability-gated Pareto front"]
    assert v2[-2][2] == pa["n_stable_gated_pool"] and v2[-1][2] == pa["n_front_stable"]
    assert all(a[2] >= b[2] for a, b in zip(v2, v2[1:])), "funnel counts must not increase"
    assert all(a[2] >= b[2] for a, b in zip(legacy, legacy[1:]))


def test_fig1_sheet_matches_figure_data():
    xlsx = ROOT / "paper/build/SI_data_v2.xlsx"
    if not xlsx.exists():
        pytest.skip("workbook not built (run paper/make_si_excel.py)")
    openpyxl = pytest.importorskip("openpyxl")
    from make_figures_v2 import funnel_counts
    legacy, v2 = funnel_counts()
    want = [("legacy", s, n) for s, _, n in legacy] + [("v2", s, n) for s, _, n in v2]
    rows = list(openpyxl.load_workbook(xlsx, read_only=True, data_only=True)["Fig1_funnel"].iter_rows(values_only=True))
    assert [tuple(r) for r in rows[1:]] == want
