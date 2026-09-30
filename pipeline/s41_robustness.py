"""Stage 41: robustness numbers requested by the manuscript review (Codex, 2026-09-29).

1. Dependence-free bounds for the joint "competitive AND stable" expectation. The shipped numbers multiply
   P(LCOE <= c-Si) (which already contains P(semiconductor) and the gap posterior jointly, by Monte Carlo) with
   P(stable) under an independence assumption. For ANY dependence structure the joint probability of two events lies
   in the Frechet interval [max(0, a + b - 1), min(a, b)]; summing gives bounds on the expected count.
2. Conformal coverage restricted to MIXED-ANION compounds in the chemical-system out-of-fold predictions (the
   regime the candidates occupy), and the anion composition of the stability control set.
3. Number of usable (featurized) training rows.
Output: runs/<id>/metrics/robustness.json
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .common import Stage, dump_json

MODULES = ("s41_robustness",)
ANION_FAM = {"O": "O", "S": "Ch", "Se": "Ch", "Te": "Ch", "F": "F", "Cl": "Cl", "Br": "Br", "I": "I"}


def n_anion_families(chemsys: str) -> int:
    return len({ANION_FAM[e] for e in chemsys.split("-") if e in ANION_FAM})


def run(run_dir, force: bool = False) -> dict:
    cl, oof = run_dir / "tea_v2/candidate_lcoe.csv", run_dir / "predictions/oof_v2.csv"
    ctrl, feats = run_dir / "stability/control_known_a2bbx6.csv", run_dir / "features/train_v2_X.npz"
    st = Stage(run_dir, "s41_robustness", {"candidate_lcoe": cl, "oof": oof, "control": ctrl}, MODULES)
    if st.up_to_date() and not force:
        print("[s41] up to date, skipped")
        return {}
    c = pd.read_csv(cl)
    m = {"frechet": {}}
    scen = {"demonstrated": "f0.21_low_L35_d0.007", "realistic": "f0.61_med_L15_d0.02",
            "optimistic": "f0.61_low_L35_d0.007", "ceiling": "f0.82_low_L35_d0.007"}
    ps = (c.p_stable50 * c.plausible).where(~c.umlip_veto.fillna(False).astype(bool), 0.0)
    for k, t in scen.items():
        a = c[f"p_lcoe_le_csi__{t}"]
        m["frechet"][k] = {"independent": float((a * ps).sum()), "lower": float(np.maximum(0, a + ps - 1).sum()),
                           "upper": float(np.minimum(a, ps).sum()),
                           "n_candidates_competitive_p_gt_0": int((a > 0).sum())}
    sc = pd.read_csv(run_dir / "stability/v2_screen_scored.csv")
    nov = sc[(sc.status != "known_mp") & sc.plausible]
    a2, b2, c2 = nov.p_semi, nov.p_gap_pv, nov.p_stable50
    m["viable_absorber"] = {"independent": float((a2 * b2 * c2).sum()),
                            "upper": float(np.minimum(np.minimum(a2, b2), c2).sum()),
                            "lower": float(np.maximum(0, a2 + b2 + c2 - 2).sum())}

    o = pd.read_csv(oof, usecols=["chemsys", "band_gap", "is_semi", "chemsys__conformal__lo", "chemsys__conformal__hi"])
    o = o[o.is_semi == 1 & o["chemsys__conformal__lo"].notna()]
    o = o[o["chemsys__conformal__lo"].notna()]
    o["n_anion_fam"] = o.chemsys.map(n_anion_families)
    cov = ((o.band_gap >= o["chemsys__conformal__lo"]) & (o.band_gap <= o["chemsys__conformal__hi"]))
    m["conformal_by_anion_mix"] = {str(k): {"n": int((o.n_anion_fam == k).sum()), "coverage": float(cov[o.n_anion_fam == k].mean())}
                                   for k in (1, 2) if (o.n_anion_fam == k).any()}
    alk = o.chemsys.str.contains(r"(?:^|-)(?:Li|Na|K|Rb|Cs)(?:-|$)")
    mix_alk = (o.n_anion_fam == 2) & alk
    m["conformal_mixed_anion_alkali"] = {"n": int(mix_alk.sum()), "coverage": float(cov[mix_alk].mean())}

    cc = pd.read_csv(ctrl).dropna(subset=["true_ehull_meV"])
    cc["n_anion_fam"] = [n_anion_families("-".join(re.findall(r"[A-Z][a-z]?", f))) for f in cc.formula]
    m["control_anion_mix"] = {"n": int(len(cc)), "n_mixed_anion": int((cc.n_anion_fam >= 2).sum())}
    if (cc.n_anion_fam >= 2).sum() >= 10:
        from sklearn.metrics import roc_auc_score
        mm = cc[cc.n_anion_fam >= 2]
        y = (mm.true_ehull_meV <= 50).astype(int)
        m["control_anion_mix"]["mixed_auc"] = float(roc_auc_score(y, mm.p_ehull_le_50)) if y.nunique() == 2 else None

    X = np.load(feats)["X"]
    m["n_train_featurized"] = int(np.isfinite(X).all(axis=1).sum())
    m["n_train_featurize_failed"] = int((~np.isfinite(X).all(axis=1)).sum())
    dump_json(st.path("metrics/robustness.json"), m)
    st.metrics = m
    st.finish()
    print(f"[s41] {json.dumps(m, default=float)}")
    return m
