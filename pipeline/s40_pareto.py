"""Stage 40: Pareto shortlist (Phase F).

Objectives per v2 candidate (1,256), all calibrated on held-out data in earlier stages:
  maximise  p_stable_final = P(E_hull <= 50 meV) x plausible, set to 0 when the MACE check vetoes (> 50 meV)
  maximise  p_absorber     = P(semiconductor) x P(gap in 1.0-1.8 eV)
  minimise  LCOE median    under the named TEA scenario (default: optimistic, f_SQ 0.61, $50/m2, 35 y)
Non-dominated sorting gives front ranks (1 = Pareto optimal). Rank stability: the front-1 set is recomputed
under every TEA scenario and with/without the MACE veto; each candidate's front-1 frequency is reported.
Output: runs/<id>/pareto/pareto.csv, metrics/pareto.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import Stage, dump_json

MODULES = ("s40_pareto",)
SCENARIOS = {"optimistic": "f0.61_low_L35_d0.007", "ceiling": "f0.82_low_L35_d0.007",
             "realistic": "f0.61_med_L15_d0.02", "demonstrated": "f0.21_low_L35_d0.007"}
PRIMARY = "optimistic"


def front_ranks(F: np.ndarray) -> np.ndarray:
    """Non-dominated sorting for MINIMISATION of every column of F (n x k). Returns rank >= 1."""
    n = len(F)
    rank = np.zeros(n, int)
    remaining = np.arange(n)
    r = 1
    while remaining.size:
        sub = F[remaining]
        dominated = np.zeros(len(sub), bool)
        for i in range(len(sub)):
            le = np.all(sub <= sub[i], axis=1)
            lt = np.any(sub < sub[i], axis=1)
            dominated[i] = np.any(le & lt)
        rank[remaining[~dominated]] = r
        remaining = remaining[dominated]
        r += 1
    return rank


def objectives(df: pd.DataFrame, scen: str, veto: bool) -> np.ndarray:
    ps = df.p_stable50 * df.plausible
    if veto:
        ps = ps.where(~df.umlip_veto.fillna(False).astype(bool), 0.0)
    lcoe = df[f"lcoe_median__{SCENARIOS[scen]}"].replace([np.inf], 1e9).fillna(1e9)
    return np.column_stack([-ps.to_numpy(float), -df.p_absorber.to_numpy(float), lcoe.to_numpy(float)])


def run(run_dir, force: bool = False) -> dict:
    cl = run_dir / "tea_v2/candidate_lcoe.csv"
    sc = run_dir / "stability/v2_screen_scored.csv"
    st = Stage(run_dir, "s40_pareto", {"candidate_lcoe": cl, "scored": sc}, MODULES,
               params={"scenarios": SCENARIOS, "primary": PRIMARY})
    if st.up_to_date() and not force:
        print("[s40] up to date, skipped")
        return {}
    df = pd.read_csv(cl)
    extra = pd.read_csv(sc)[["formula", "p_gap_pv", "ox_states", "tau", "ehull_pred_meV", "mp_ehull_meV"]]
    df = df.merge(extra, on="formula", how="left")
    df["p_absorber"] = df.p_semi * df.p_gap_pv
    df["p_stable_final"] = (df.p_stable50 * df.plausible).where(~df.umlip_veto.fillna(False).astype(bool), 0.0)

    df["front"] = front_ranks(objectives(df, PRIMARY, veto=True))
    # stability-gated front (review 2026-09-29): Pareto sorting among compositions with P(stable) >= 0.5 only,
    # so a front member cannot sit there by trading away thermodynamic stability entirely
    gate = df.p_stable_final >= 0.5
    df["front_stable"] = 0
    df.loc[gate, "front_stable"] = front_ranks(objectives(df[gate].reset_index(drop=True), PRIMARY, veto=True))
    freq = np.zeros(len(df))
    variants = [(s, v) for s in SCENARIOS for v in (True, False)]
    for s, v in variants:
        freq += front_ranks(objectives(df, s, v)) == 1
    df["front1_frequency"] = freq / len(variants)

    col_l = f"lcoe_median__{SCENARIOS[PRIMARY]}"
    col_p = f"p_lcoe_le_csi__{SCENARIOS[PRIMARY]}"
    keep = ["formula", "status", "front", "front_stable", "front1_frequency", "p_stable_final", "p_stable50", "plausible",
            "umlip_ehull_meV", "umlip_veto", "p_absorber", "p_semi", "p_gap_pv", "gap_pred_eV", "gap_lo90_eV",
            "gap_hi90_eV", col_l, col_p, "ox_states", "tau", "ehull_pred_meV", "mp_ehull_meV", "absorber_cost_m2"]
    out = df[keep].sort_values(["front", "p_stable_final", "p_absorber"], ascending=[True, False, False])
    out.to_csv(st.path("pareto/pareto.csv"), index=False, float_format="%.5f")

    f1 = out[out.front == 1]
    fs = out[out.front_stable == 1]
    robust = out[out.front1_frequency >= 0.75]
    m = {"n": int(len(out)), "n_front1": int(len(f1)), "n_front1_novel": int((f1.status != "known_mp").sum()),
         "n_front1_known": int((f1.status == "known_mp").sum()),
         "n_stable_gated_pool": int((out.p_stable_final >= 0.5).sum()),
         "n_front_stable": int(len(fs)), "n_front_stable_novel": int((fs.status != "known_mp").sum()),
         "front_stable": fs[["formula", "status", "p_stable_final", "p_absorber", col_l, col_p, "umlip_ehull_meV"]]
         .round(3).to_dict("records"),
         "front_stable_umlip_checked": int(fs.umlip_ehull_meV.notna().sum()),
         "front_stable_max_p_competitive": float(fs[col_p].max()) if len(fs) else None,
         "n_robust_front1_ge_0.75": int(len(robust)),
         "robust_formulas": robust.formula.tolist(),
         "front1": f1[["formula", "status", "p_stable_final", "p_absorber", col_l, col_p, "front1_frequency"]]
         .round(3).to_dict("records"),
         "front1_min_lcoe_usd_mwh": float(f1[col_l].min()),
         "front1_max_p_competitive": float(f1[col_p].max()),
         "variants": [f"{s}{'+veto' if v else ''}" for s, v in variants]}
    dump_json(st.path("metrics/pareto.json"), m)
    st.metrics = {k: v for k, v in m.items() if k not in ("front1",)}
    st.finish()
    print(f"[s40] {st.metrics}")
    return m
