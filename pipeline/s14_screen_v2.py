"""Stage 14: v2 screen of the full A2BB'X6 space (Phase C/D).

  23,940 generated (same space as April)
  -> charge balance (same rule as April)                                        [s03 output]
  -> corrected geometry: oxidation-state-aware Shannon radii, Bartel tau < 4.18  (replaces legacy t/mu moat)
  -> v2 ML: P(semiconductor), band gap + 90% conformal interval, P(gap in PV window), Ef
  -> toxic filter
Every charge-balanced formula gets a row with all descriptors and a `stage_fail` reason, so the
funnel can be recounted and the April candidates traced through the new gates.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition

from . import config
from .common import Stage, dump_json
from .features import featurize_many
from .geometry import descriptors

MODULES = ("s14_screen_v2", "geometry", "features")
PV_WINDOW_EV = (1.0, 1.8)       # SQ >= ~28% (legacy SQ table); PV-relevant single-junction window
P_SEMI_MIN = 0.5


def gap_posterior_prob(pred: np.ndarray, sigma: np.ndarray, signed: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """P(true gap in [lo, hi]) from the empirical distribution of signed normalised residuals."""
    s = np.sort(signed)
    # true = pred + s*sigma  ->  lo <= true <= hi  <=>  (lo-pred)/sigma <= s <= (hi-pred)/sigma
    a = (lo - pred) / sigma
    b = (hi - pred) / sigma
    return (np.searchsorted(s, b, side="right") - np.searchsorted(s, a, side="left")) / len(s)


def run(run_dir, force: bool = False, legacy_run: str = "legacy_april") -> dict:
    charge = config.RUNS_DIR / legacy_run / "screen/charge_balanced.csv"
    model = run_dir / "models/v2_final.joblib"
    st = Stage(run_dir, "s14_screen_v2", {"charge_balanced": charge, "model": model}, MODULES,
               params={"pv_window": PV_WINDOW_EV, "p_semi_min": P_SEMI_MIN, "toxic": config.TOXIC})
    if st.up_to_date() and not force:
        print("[s14] up to date, skipped")
        return {}
    cb = pd.read_csv(charge)
    m = joblib.load(model)

    geo = []
    for f in cb.formula:
        d = descriptors(f, _a_site(f))
        ox = d.pop("ox_states", None)
        d["ox_states"] = ";".join(f"{k}{int(v):+d}" for k, v in ox.items()) if ox else ""
        geo.append(d)
    g = pd.DataFrame(geo)
    df = pd.concat([cb[["formula", "steric_pass"]].rename(columns={"steric_pass": "legacy_steric_pass"}), g], axis=1)

    X, failed = featurize_many(df.formula.tolist(), n_jobs=8)
    ok = np.isfinite(X).all(axis=1)
    p_semi = np.full(len(df), np.nan)
    gap = np.full(len(df), np.nan)
    sig = np.full(len(df), np.nan)
    ef = np.full(len(df), np.nan)
    p_semi[ok] = m["cls"].predict_proba(X[ok])[:, 1]
    gap[ok] = m["spec"].predict(X[ok])
    sig[ok] = np.clip(m["diff"].predict(X[ok]), m["sigma_floor"], None)
    ef[ok] = m["ef"].predict(X[ok])
    df["p_semi"], df["gap_pred_eV"], df["gap_sigma_eV"], df["ef_pred_eV_atom"] = p_semi, gap, sig, ef
    df["gap_lo90_eV"] = gap - m["q_gap"] * sig
    df["gap_hi90_eV"] = gap + m["q_gap"] * sig
    df["p_gap_pv"] = np.nan
    df.loc[ok, "p_gap_pv"] = gap_posterior_prob(gap[ok], sig[ok], m["gap_cal_signed_scores"], *PV_WINDOW_EV)
    df["p_pv_absorber"] = df.p_semi * df.p_gap_pv
    df["toxic"] = [any(e.symbol in config.TOXIC for e in Composition(f).elements) for f in df.formula]

    fail = []
    for r in df.itertuples():
        if not r.ox_ok:
            fail.append("no_oxidation_state_assignment")
        elif not r.tau_perovskite:
            fail.append("tau_not_perovskite")
        elif not np.isfinite(r.p_semi):
            fail.append("featurize_failed")
        elif r.p_semi < P_SEMI_MIN:
            fail.append("predicted_metal")
        elif r.toxic:
            fail.append("toxic")
        else:
            fail.append("pass")
    df["stage_fail"] = fail
    df.to_csv(st.path("screen/v2_all_charge_balanced.csv"), index=False, float_format="%.5f")
    passed = df[df.stage_fail == "pass"].reset_index(drop=True)
    passed.to_csv(st.path("screen/v2_candidates.csv"), index=False, float_format="%.5f")

    april = set(pd.read_csv(config.RUNS_DIR / legacy_run / "screen/candidates.csv").formula)
    trace = df[df.formula.isin(april)].stage_fail.value_counts().to_dict()
    st.metrics = {"n_charge_balanced": int(len(df)), "funnel_fail_counts": df.stage_fail.value_counts().to_dict(),
                  "n_v2_candidates": int(len(passed)), "n_legacy_steric_pass": int(df.legacy_steric_pass.sum()),
                  "april_candidates_trace": trace,
                  "n_v2_candidates_in_april_set": int(passed.formula.isin(april).sum()),
                  "p_pv_absorber_ge_0.5": int((passed.p_pv_absorber >= 0.5).sum()),
                  "cs2agbibr6": df[df.formula == "Cs2AgBiBr6"][["stage_fail", "tau", "gap_pred_eV", "p_semi"]]
                  .to_dict("records")}
    dump_json(st.path("metrics/screen_v2.json"), st.metrics)
    st.finish()
    print(f"[s14] {st.metrics}")
    return st.metrics


def _a_site(formula: str) -> str:
    """Generated formulas always start with the A element (e.g. 'Na2FeMnO3S3')."""
    for a in config.A_SITE:
        if formula.startswith(a + "2"):
            return a
    raise ValueError(formula)
