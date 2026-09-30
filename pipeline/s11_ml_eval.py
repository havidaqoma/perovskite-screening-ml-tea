"""Stage 11: audit-grade ML evaluation on the v2 dataset (Phase C).

Splits (all persisted as row-index lists):
  random      : 5-fold KFold over formulas (v2 has one row per formula, so no formula leakage)
  chemsys     : 5-fold GroupKFold over chemical system (all polymorph/stoichiometry variants of
                e.g. Cs-Ag-Bi-Br are held out together)
  lofo_anion  : leave-one-anion-family-out (O, S/Se/Te, F, Cl, Br, I). Only rows with exactly one
                of these anions participate; the candidates' mixed-anion chemistry is the extrapolation case.

Models (identical folds, same 139 features):
  mean (DummyRegressor), ridge (standardised), rf (Magpie-RF), xgb (legacy params).
  Roost runs separately in WSL (s12_roost.py) on the same persisted folds.

Tasks:
  gap_all   : band_gap regression on all rows (metals = 0)             -> MAE
  gap_semi  : band_gap regression on is_semi rows only                 -> MAE (the legacy "specialist" task)
  metal_cls : is_semi classification                                   -> ROC-AUC, balanced accuracy
  ef        : formation_energy_per_atom regression                     -> MAE
Cascade: gap = cls_prob>=0.5 ? specialist : 0, scored on all rows.

Uncertainty: split-conformal intervals for the XGB gap specialist, calibrated on out-of-fold
residuals, normalised by a quantile-gradient-boosting difficulty model; coverage checked per split.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .common import Stage, dump_json
from .features import featurize_many, feature_names

MODULES = ("s11_ml_eval", "features")
N_FOLDS = 5
ANION_FAMILIES = {"O": ["O"], "chalcogen": ["S", "Se", "Te"], "F": ["F"], "Cl": ["Cl"], "Br": ["Br"], "I": ["I"]}
ALL_ANIONS = sorted({a for v in ANION_FAMILIES.values() for a in v})
CONFORMAL_ALPHA = 0.10           # 90% intervals
SEED = 42


DEVICE = "cuda"   # xgboost GPU (RTX 5060 Ti); results logged per device in the manifest


def _xgb_reg(**over):
    return xgb.XGBRegressor(**{**config.BG_PARAMS_LEGACY, **over}, device=DEVICE, n_jobs=-1, random_state=SEED)


def _models():
    return {
        "mean": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "rf": lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.33,
                                            max_samples=0.3, n_jobs=-1, random_state=SEED),
        "xgb": lambda: _xgb_reg(),
    }


def anion_family(formula_elements: list[str]) -> str | None:
    fams = {fam for fam, els in ANION_FAMILIES.items() if any(e in formula_elements for e in els)}
    return fams.pop() if len(fams) == 1 else None


def make_splits(df: pd.DataFrame) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    idx = np.arange(len(df))
    out = {"random": [(tr, te) for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(idx)]}
    # GroupKFold is deterministic given group order; shuffle group labels with a seed for balance.
    rng = np.random.default_rng(SEED)
    uniq = df.chemsys.unique()
    perm = dict(zip(uniq, rng.permutation(len(uniq))))
    groups = df.chemsys.map(perm).to_numpy()
    out["chemsys"] = [(tr, te) for tr, te in GroupKFold(N_FOLDS).split(idx, groups=groups)]
    fam = df.chemsys.str.split("-").map(anion_family)
    lofo = []
    for f in ANION_FAMILIES:
        te = idx[fam.to_numpy() == f]
        tr = idx[fam.notna().to_numpy() & (fam.to_numpy() != f)]
        lofo.append((tr, te))
    out["lofo_anion"] = lofo
    return out


def _fit_predict(model_factory, X, y, tr, te, mask_tr=None):
    tr_fit = tr if mask_tr is None else tr[mask_tr[tr]]
    m = model_factory()
    m.fit(X[tr_fit], y[tr_fit])
    return m.predict(X[te])


def conformal(X, y, semi, folds, rng_seed=SEED):
    """Normalised split-conformal on out-of-fold XGB specialist predictions.

    For each fold: fit specialist + a difficulty model (|residual| predictor, trained on an inner
    split of the training fold), compute normalised scores on a calibration part of the training
    fold, take the finite-sample (1-alpha) quantile, then apply to the test fold. Returns per-row
    lower/upper and fold coverage.
    """
    lo = np.full(len(y), np.nan)
    hi = np.full(len(y), np.nan)
    pred = np.full(len(y), np.nan)
    cover = []
    rng = np.random.default_rng(rng_seed)
    for tr, te in folds:
        tr_s = tr[semi[tr] == 1]
        te_s = te[semi[te] == 1]
        perm = rng.permutation(tr_s)
        n_cal = max(200, int(0.2 * len(perm)))
        cal, fit = perm[:n_cal], perm[n_cal:]
        half = len(fit) // 2
        f_a, f_b = fit[:half], fit[half:]
        spec = _xgb_reg().fit(X[fit], y[fit])
        # difficulty model trained on residuals of a model that did not see those rows
        aux = _xgb_reg().fit(X[f_a], y[f_a])
        diff = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                device=DEVICE, n_jobs=-1, random_state=SEED).fit(X[f_b], np.abs(y[f_b] - aux.predict(X[f_b])))
        sig = lambda Z: np.clip(diff.predict(Z), 0.05, None)
        scores = np.abs(y[cal] - spec.predict(X[cal])) / sig(X[cal])
        k = int(np.ceil((len(cal) + 1) * (1 - CONFORMAL_ALPHA)))
        q = np.sort(scores)[min(k, len(scores)) - 1]
        p = spec.predict(X[te_s])
        s = sig(X[te_s])
        pred[te_s], lo[te_s], hi[te_s] = p, p - q * s, p + q * s
        cover.append(float(np.mean((y[te_s] >= lo[te_s]) & (y[te_s] <= hi[te_s]))))
    return pred, lo, hi, cover


def run(run_dir, force: bool = False) -> dict:
    data = run_dir / "data/train_v2.csv"
    st = Stage(run_dir, "s11_ml_eval", {"train_v2": data}, MODULES,
               params={"device": DEVICE, "folds": N_FOLDS, "alpha": CONFORMAL_ALPHA, "seed": SEED, "xgb": config.BG_PARAMS_LEGACY})
    if st.up_to_date() and not force:
        print("[s11] up to date, skipped")
        return {}

    df = pd.read_csv(data)
    feat_cache = run_dir / "features/train_v2_X.npz"
    if feat_cache.exists():
        X = np.load(feat_cache)["X"]
    else:
        X, failed = featurize_many(df.formula.tolist(), progress=True, n_jobs=10)
        feat_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feat_cache, X=X)
    ok = np.isfinite(X).all(axis=1)
    if (~ok).any():
        df.loc[~ok, ["formula"]].to_csv(st.path("data/featurize_failed.csv"), index=False)
        df, X = df[ok].reset_index(drop=True), X[ok]
    y_gap = df.band_gap.to_numpy(float)
    y_ef = df.formation_energy_per_atom.to_numpy(float)
    semi = df.is_semi.to_numpy(int)

    splits = make_splits(df)
    dump_json(st.path("splits/v2_splits.json"),
              {k: [{"train": tr.tolist(), "test": te.tolist()} for tr, te in v] for k, v in splits.items()})
    lofo_names = list(ANION_FAMILIES)

    results = []
    oof = {}
    for split_name, folds in splits.items():
        for task, y, mask in (("gap_semi", y_gap, semi == 1), ("gap_all", y_gap, None), ("ef", y_ef, None)):
            for mname, factory in _models().items():
                errs = []
                pred_all = np.full(len(y), np.nan)
                for fi, (tr, te) in enumerate(folds):
                    te_eval = te if mask is None else te[mask[te]]
                    if len(te_eval) == 0:
                        continue
                    p = _fit_predict(factory, X, y, tr, te_eval, mask_tr=mask)
                    pred_all[te_eval] = p
                    errs.append({"fold": lofo_names[fi] if split_name == "lofo_anion" else fi,
                                 "n_test": int(len(te_eval)), "mae": float(mean_absolute_error(y[te_eval], p))})
                ev = np.isfinite(pred_all)
                results.append({"split": split_name, "task": task, "model": mname,
                                "mae_pooled": float(mean_absolute_error(y[ev], pred_all[ev])),
                                "mae_fold_mean": float(np.mean([e["mae"] for e in errs])),
                                "mae_fold_sd": float(np.std([e["mae"] for e in errs])), "folds": errs})
                oof[(split_name, task, mname)] = pred_all
                print(f"[s11] {split_name:10s} {task:8s} {mname:5s} MAE={results[-1]['mae_pooled']:.4f}", flush=True)

        # gatekeeper classifier + cascade
        prob = np.full(len(y_gap), np.nan)
        for tr, te in folds:
            c = xgb.XGBClassifier(**config.CLS_PARAMS_LEGACY, device=DEVICE, n_jobs=-1, random_state=SEED).fit(X[tr], semi[tr])
            prob[te] = c.predict_proba(X[te])[:, 1]
        ev = np.isfinite(prob)
        spec = oof[(split_name, "gap_semi", "xgb")]
        # specialist predictions for metal rows were never made (they are masked); make them per fold
        spec_all = np.full(len(y_gap), np.nan)
        for tr, te in folds:
            spec_all[te] = _fit_predict(_models()["xgb"], X, y_gap, tr, te, mask_tr=semi == 1)
        casc = np.where(prob >= 0.5, np.clip(spec_all, 0, None), 0.0)
        results.append({"split": split_name, "task": "metal_cls", "model": "xgb_cls",
                        "roc_auc": float(roc_auc_score(semi[ev], prob[ev])),
                        "balanced_acc": float(balanced_accuracy_score(semi[ev], prob[ev] >= 0.5)),
                        "frac_semi": float(semi[ev].mean())})
        results.append({"split": split_name, "task": "gap_all", "model": "cascade_xgb",
                        "mae_pooled": float(mean_absolute_error(y_gap[ev], casc[ev]))})
        print(f"[s11] {split_name:10s} metal_cls AUC={results[-2]['roc_auc']:.4f}  cascade MAE={results[-1]['mae_pooled']:.4f}",
              flush=True)
        oof[(split_name, "metal_cls", "prob")] = prob
        oof[(split_name, "gap_all", "cascade_xgb")] = casc

    # conformal on the realistic split (chemsys) and the random split
    conf = {}
    for split_name in ("random", "chemsys"):
        p, lo, hi, cov = conformal(X, y_gap, semi, splits[split_name])
        ev = np.isfinite(lo)
        width = hi[ev] - lo[ev]
        conf[split_name] = {"alpha": CONFORMAL_ALPHA, "coverage_pooled": float(np.mean((y_gap[ev] >= lo[ev]) & (y_gap[ev] <= hi[ev]))),
                            "coverage_folds": cov, "width_median": float(np.median(width)),
                            "width_p90": float(np.quantile(width, 0.9))}
        oof[(split_name, "conformal", "lo")] = lo
        oof[(split_name, "conformal", "hi")] = hi
        print(f"[s11] conformal {split_name}: {conf[split_name]}", flush=True)

    pred_df = df[["row", "formula", "chemsys", "band_gap", "is_semi", "formation_energy_per_atom"]].copy()
    for (s, t, m), v in oof.items():
        pred_df[f"{s}__{t}__{m}"] = v
    pred_df.to_csv(st.path("predictions/oof_v2.csv"), index=False, float_format="%.5f")
    dump_json(st.path("metrics/ml_eval_v2.json"), {"results": results, "conformal": conf,
                                                   "n_rows": int(len(df)), "n_featurize_failed": int((~ok).sum()),
                                                   "feature_names": feature_names()})
    st.metrics = {"n_rows": int(len(df)), "conformal": conf}
    st.finish()
    return st.metrics
