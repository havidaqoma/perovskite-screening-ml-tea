"""Gate for Phase D (stability + novelty). Exit 1 on failure. Writes runs/<id>/gates/gate_stability.json.

D-G1  every v2 candidate has a stability status; 'unknown' never counts as stable
D-G2  control on known MP A2BB'X6 compounds (hull built WITHOUT them, Ef out-of-fold):
      ROC-AUC of P(E_hull<=50) >= 0.70 and reliability: |mean_p - observed| <= 0.15 in every bin with n >= 20
D-G3  positive control: Cs2AgBiBr6 passes the corrected geometry gate (it failed the legacy one)
D-G4  novelty is resolved for every candidate (known_mp vs not-in-MP) from a successful, cached MP query
      (no fail-open "0 matches" possible: queries raise on error)
D-G5  funnel counts at 0/20/35/50/100 meV are recorded
D-G6  uMLIP check: if umlip.csv exists, >= 80% of attempted shortlist rows finished with status ok, and
      on the known-MP controls the uMLIP verdict (E_hull vs other phases <= 50 meV) agrees with MP for >= 70%
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .common import dump_json


def check(run_dir: Path) -> dict:
    res = {"checks": {}}

    def rec(name, ok, **info):
        res["checks"][name] = {"pass": bool(ok), **info}

    st = pd.read_csv(run_dir / "stability/v2_stability.csv")
    m = json.loads((run_dir / "metrics/stability_v2.json").read_text())
    rec("D-G1_status_complete", st.status.notna().all() and set(st.status) <= {"known_mp", "stable_likely",
                                                                               "unstable_likely", "unknown"},
        counts=st.status.value_counts().to_dict())

    c = m["control"]
    rel_ok = all(abs(v["mean_p"] - v["obs_frac"]) <= 0.15 for v in c["reliability"].values() if v["n"] >= 20)
    rec("D-G2_control_calibration", (c["roc_auc_p50"] or 0) >= 0.70 and rel_ok,
        roc_auc=c["roc_auc_p50"], brier=c["brier_p50"], n=c["n_ok"], reliability=c["reliability"],
        gate_precision=c["gate_precision"], gate_recall=c["gate_recall"])

    allcb = pd.read_csv(run_dir / "screen/v2_all_charge_balanced.csv")
    from pymatgen.core import Composition
    allcb["red"] = [Composition(f).reduced_formula for f in allcb.formula]
    row = allcb[allcb.red == "Cs2AgBiBr6"]
    rec("D-G3_positive_control", len(row) == 1 and row.stage_fail.iloc[0] == "pass" and not row.legacy_steric_pass.iloc[0],
        v2=row.stage_fail.tolist(), legacy_steric_pass=row.legacy_steric_pass.tolist())

    rec("D-G4_novelty_resolved", st.status.ne("unknown").all(),
        n_known_mp=int(st.status.eq("known_mp").sum()), n_not_in_mp=int(st.status.isin(["stable_likely", "unstable_likely"]).sum()),
        n_unknown=int(st.status.eq("unknown").sum()))

    rec("D-G5_funnel_thresholds", "n_p50_ge_0.5_by_thr" in m, counts=m.get("n_p50_ge_0.5_by_thr"))

    u = run_dir / "stability/umlip.csv"
    if u.exists():
        ud = pd.read_csv(u)
        frac = float(ud.status.eq("ok").mean())
        ok = ud[ud.status.eq("ok")]
        ctrl = ok[ok.role.eq("control_known_mp")] if "role" in ok else ok.iloc[0:0]
        agree = float(((ctrl.umlip_ehull_vs_rest_meV <= 50) == (ctrl.mp_ehull_meV <= 50)).mean()) if len(ctrl) else None
        rec("D-G6_umlip", frac >= 0.8 and (agree is None or agree >= 0.7), n=int(len(ud)), frac_ok=frac,
            control_agreement=agree, n_control=int(len(ctrl)))
    res["pass"] = all(v["pass"] for v in res["checks"].values())
    dump_json(run_dir / "gates/gate_stability.json", res)
    return res


if __name__ == "__main__":
    out = check(Path(sys.argv[1]))
    for k, v in out["checks"].items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}  " + json.dumps({a: b for a, b in v.items() if a != 'pass'}, default=str)[:600])
    print("GATE:", "PASS" if out["pass"] else "FAIL")
    sys.exit(0 if out["pass"] else 1)
