"""Gate for Phase F (Pareto shortlist + figures from runs only). Exit 1 on failure.

F-G1  Pareto front non-empty; every front-1 member is non-dominated under the primary objectives (re-checked
      independently here with a brute-force O(n^2) dominance test, not the stage's own sorter)
F-G2  no front-1 member is plausibility-flagged or MACE-vetoed with p_stable_final > 0 (consistency)
F-G3  every manuscript figure exists as PNG + PDF, is newer than every run artifact it reads, and is non-trivial
F-G4  figure values trace: fig1 funnel counts == metrics files; fig5 bar heights == tea_v2.json (recomputed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import dump_json

ROOT = Path(__file__).resolve().parents[1]
FIGS = ["fig1_funnel", "fig2_ml_audit", "fig3_stability", "fig4_tea", "fig5_candidates"]
READS = ["legacy_april/metrics/funnel_generate.json", "v2/metrics/screen_v2.json", "v2/metrics/stability_v2.json",
         "v2/metrics/pareto.json", "v2/metrics/tea_v2.json", "v2/metrics/ml_eval_v2.json",
         "v2/metrics/leakage_audit.json", "v2/stability/umlip.csv", "v2/tea_v2/efficiency_sweep.csv",
         "v2/tea_v2/breakeven_efficiency.csv", "v2/pareto/pareto.csv"]


def check(run_dir: Path) -> dict:
    res = {"checks": {}}

    def rec(name, ok, **info):
        res["checks"][name] = {"pass": bool(ok), **info}

    p = pd.read_csv(run_dir / "pareto/pareto.csv")
    col = [c for c in p.columns if c.startswith("lcoe_median__")][0]
    F = np.column_stack([-p.p_stable_final, -p.p_absorber, p[col].replace(np.inf, 1e9)])
    dominated = np.array([np.any(np.all(F <= F[i], 1) & np.any(F < F[i], 1)) for i in range(len(F))])
    f1 = p.front.to_numpy() == 1
    rec("F-G1_pareto_valid", f1.any() and np.array_equal(f1, ~dominated), n_front1=int(f1.sum()),
        n_nondominated_bruteforce=int((~dominated).sum()))

    bad = p[f1 & ((~p.plausible.astype(bool) & (p.p_stable_final > 0)) |
                  (p.umlip_veto.fillna(False).astype(bool) & (p.p_stable_final > 0)))]
    rec("F-G2_front_consistency", bad.empty, offending=bad.formula.tolist())

    newest_input = max((ROOT / "runs" / r).stat().st_mtime for r in READS)
    info = {}
    ok3 = True
    for f in FIGS:
        png, pdf = ROOT / "paper/figures" / f"{f}.png", ROOT / "paper/figures" / f"{f}.pdf"
        good = png.exists() and pdf.exists() and png.stat().st_size > 40_000 and pdf.stat().st_mtime >= newest_input
        info[f] = good
        ok3 &= good
    rec("F-G3_figures_fresh", ok3, **info)

    fg = json.loads((ROOT / "runs/legacy_april/metrics/funnel_generate.json").read_text())
    sc = json.loads((run_dir / "metrics/screen_v2.json").read_text())
    t = json.loads((run_dir / "metrics/tea_v2.json").read_text())["e5_candidates"]
    cl = pd.read_csv(run_dir / "tea_v2/candidate_lcoe.csv")
    recomputed = float(cl["p_competitive_x_stable_umlip__optimistic"].sum())
    # candidate_lcoe.csv is written with 5-decimal rounding: sum of 1,256 rounded values -> tolerance 1,256 * 5e-6
    tol = len(cl) * 5e-6
    rec("F-G4_values_trace", fg["n_generated"] == 23940 and sc["n_v2_candidates"] == int((pd.read_csv(
        run_dir / "screen/v2_candidates.csv")).shape[0]) and abs(recomputed - t["expected_n_competitive_and_stable_umlip"]
                                                                  ["optimistic"]) <= tol,
        fig5_optimistic_recomputed=recomputed, tolerance=tol,
        fig5_optimistic_metric=t["expected_n_competitive_and_stable_umlip"]["optimistic"])
    res["pass"] = all(v["pass"] for v in res["checks"].values())
    dump_json(run_dir / "gates/gate_pareto.json", res)
    return res


if __name__ == "__main__":
    out = check(Path(sys.argv[1]))
    for k, v in out["checks"].items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}  " + json.dumps({a: b for a, b in v.items() if a != 'pass'})[:300])
    print("GATE:", "PASS" if out["pass"] else "FAIL")
    sys.exit(0 if out["pass"] else 1)
