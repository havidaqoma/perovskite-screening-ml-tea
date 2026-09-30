"""Stage 13: final v2 models used for screening (trained on ALL v2 rows), plus calibration objects.

Outputs models/v2_final.joblib:
  cls   : XGB metal/semiconductor gatekeeper (is_semi = gap >= 0.1 eV)
  spec  : XGB band-gap specialist (semiconductors only)
  ef    : XGB formation-energy model
  diff  : XGB difficulty model for normalised conformal intervals
  q_gap : conformal quantile (90%), calibrated on held-out chemical systems
  ef_residuals : out-of-fold (chemsys split) Ef residuals pred-true for 4-5 element compounds,
                 used by s20 to turn a predicted hull distance into P(stable)
Calibration protocol: one GroupShuffle split by chemsys (80% fit / 20% calibration); the
difficulty model is fit on residuals of an auxiliary model that did not see its rows. The final
spec/cls/ef are then refit on 100% of rows (standard split-conformal with refit caveat, logged).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .common import Stage, dump_json
from .s11_ml_eval import CONFORMAL_ALPHA, DEVICE, SEED, _xgb_reg

MODULES = ("s13_final_models", "s11_ml_eval")


def run(run_dir, force: bool = False) -> dict:
    data = run_dir / "data/train_v2.csv"
    feats = run_dir / "features/train_v2_X.npz"
    oof = run_dir / "predictions/oof_v2.csv"
    st = Stage(run_dir, "s13_final_models", {"train_v2": data, "X": feats, "oof": oof}, MODULES,
               params={"alpha": CONFORMAL_ALPHA, "seed": SEED})
    if st.up_to_date() and not force:
        print("[s13] up to date, skipped")
        return {}
    df = pd.read_csv(data)
    X = np.load(feats)["X"]
    ok = np.isfinite(X).all(axis=1)
    df, X = df[ok].reset_index(drop=True), X[ok]
    y, yef, semi = df.band_gap.to_numpy(float), df.formation_energy_per_atom.to_numpy(float), df.is_semi.to_numpy(int)

    # --- calibration on held-out chemical systems ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    fit_idx, cal_idx = next(gss.split(X, groups=df.chemsys))
    fit_s, cal_s = fit_idx[semi[fit_idx] == 1], cal_idx[semi[cal_idx] == 1]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(fit_s)
    a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
    spec_fit = _xgb_reg().fit(X[fit_s], y[fit_s])
    aux = _xgb_reg().fit(X[a], y[a])
    diff = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, device=DEVICE,
                            n_jobs=-1, random_state=SEED).fit(X[b], np.abs(y[b] - aux.predict(X[b])))
    sig = np.clip(diff.predict(X[cal_s]), 0.05, None)
    signed = (y[cal_s] - spec_fit.predict(X[cal_s])) / sig          # signed normalised residuals
    scores = np.abs(signed)
    k = int(np.ceil((len(cal_s) + 1) * (1 - CONFORMAL_ALPHA)))
    q = float(np.sort(scores)[min(k, len(scores)) - 1])

    # --- final models on all rows ---
    cls = xgb.XGBClassifier(**config.CLS_PARAMS_LEGACY, device=DEVICE, n_jobs=-1, random_state=SEED).fit(X, semi)
    spec = _xgb_reg().fit(X[semi == 1], y[semi == 1])
    ef = xgb.XGBRegressor(**config.FE_PARAMS_LEGACY, device=DEVICE, n_jobs=-1, random_state=SEED).fit(X, yef)
    for m in (cls, spec, ef, diff):
        m.set_params(device="cpu")          # portable inference

    o = pd.read_csv(oof)
    o = o[o.row.isin(df.row)]
    nel = o.chemsys.str.count("-") + 1
    sel = nel.isin([4, 5]) & o["chemsys__ef__xgb"].notna()
    ef_res = (o.loc[sel, "chemsys__ef__xgb"] - o.loc[sel, "formation_energy_per_atom"]).to_numpy(float)

    joblib.dump({"cls": cls, "spec": spec, "ef": ef, "diff": diff, "q_gap": q, "alpha": CONFORMAL_ALPHA,
                 "gap_cal_signed_scores": signed, "sigma_floor": 0.05,
                 "ef_residuals": ef_res, "metal_gap_eV": 0.1}, st.path("models/v2_final.joblib"))
    st.metrics = {"n_rows": int(len(df)), "n_semi": int(semi.sum()), "q_gap": q, "n_cal": int(len(cal_s)),
                  "ef_residual_n": int(len(ef_res)), "ef_residual_mae_4_5el": float(np.mean(np.abs(ef_res))),
                  "ef_residual_bias": float(np.mean(ef_res))}
    dump_json(st.path("metrics/final_models_v2.json"), st.metrics)
    st.finish()
    print(f"[s13] {st.metrics}")
    return st.metrics
