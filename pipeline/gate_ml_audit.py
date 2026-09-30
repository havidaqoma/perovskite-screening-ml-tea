"""Gate for Phase C (ML audit). Exit 1 on failure. Writes runs/<id>/gates/gate_ml_audit.json.

C-G1  every model x split x task has metrics (mean, ridge, rf, xgb; random/chemsys/lofo_anion)
C-G2  XGB beats the mean baseline on every split/task (a model that cannot beat the mean is not a result)
C-G3  gatekeeper exists and ROC-AUC >= 0.90 on chemsys split (it was a dummy in April)
C-G4  conformal coverage within +-3 pp of nominal (90%) on random AND chemsys splits, every fold within +-5 pp
C-G5  leakage audit present: April-protocol MAE and formula-grouped MAE both recorded
C-G6  split indices and out-of-fold predictions persisted
Roost is reported when present (optional: it runs in the WSL GPU env).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import dump_json


def check(run_dir: Path) -> dict:
    res = {"checks": {}}

    def rec(name, ok, **info):
        res["checks"][name] = {"pass": bool(ok), **info}

    ev = json.loads((run_dir / "metrics/ml_eval_v2.json").read_text())
    rows = ev["results"]
    reg = [r for r in rows if r["task"] in ("gap_semi", "gap_all", "ef") and r["model"] in ("mean", "ridge", "rf", "xgb")]
    need = {(s, t, m) for s in ("random", "chemsys", "lofo_anion") for t in ("gap_semi", "gap_all", "ef")
            for m in ("mean", "ridge", "rf", "xgb")}
    have = {(r["split"], r["task"], r["model"]) for r in reg}
    rec("C-G1_coverage", need <= have, missing=sorted(map(list, need - have)))

    mae = {(r["split"], r["task"], r["model"]): r["mae_pooled"] for r in reg}
    beat = {f"{s}/{t}": (mae[(s, t, "xgb")], mae[(s, t, "mean")]) for s in ("random", "chemsys", "lofo_anion")
            for t in ("gap_semi", "gap_all", "ef") if (s, t, "xgb") in mae}
    rec("C-G2_beats_mean", all(x < m for x, m in beat.values()), xgb_vs_mean=beat)

    cls = {r["split"]: r for r in rows if r["task"] == "metal_cls"}
    rec("C-G3_gatekeeper", "chemsys" in cls and cls["chemsys"]["roc_auc"] >= 0.90,
        auc={k: v["roc_auc"] for k, v in cls.items()})

    conf = ev["conformal"]
    ok4 = all(abs(conf[s]["coverage_pooled"] - 0.9) <= 0.03 and all(abs(c - 0.9) <= 0.05 for c in conf[s]["coverage_folds"])
              for s in ("random", "chemsys"))
    rec("C-G4_conformal", ok4, **{s: {"pooled": conf[s]["coverage_pooled"], "folds": conf[s]["coverage_folds"],
                                      "width_median": conf[s]["width_median"]} for s in conf})

    la = run_dir / "metrics/leakage_audit.json"
    leak = json.loads(la.read_text()) if la.exists() else {}
    rec("C-G5_leakage_audit", {"random_row", "formula", "chemsys"} <= set(leak),
        **{k: v["bg_mae_eV"] for k, v in leak.items()})

    rec("C-G6_artifacts", (run_dir / "splits/v2_splits.json").exists() and (run_dir / "predictions/oof_v2.csv").exists())

    roost = {p.stem: json.loads(p.read_text())["mae_pooled"] for p in (run_dir / "metrics").glob("roost_*.json")}
    res["roost_reported"] = roost
    res["pass"] = all(c["pass"] for c in res["checks"].values())
    dump_json(run_dir / "gates/gate_ml_audit.json", res)
    return res


if __name__ == "__main__":
    out = check(Path(sys.argv[1]))
    for k, v in out["checks"].items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}  " + json.dumps({a: b for a, b in v.items() if a != 'pass'})[:400])
    print("roost:", out["roost_reported"])
    print("GATE:", "PASS" if out["pass"] else "FAIL")
    sys.exit(0 if out["pass"] else 1)
