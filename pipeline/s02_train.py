"""Stage 02: legacy split + XGBoost cascade training (reproduces notebook cell 2).

Legacy behaviour kept for the reproduction baseline:
  * one random 80/20 split by row (seed 42) - NOT grouped by formula (leaky; fixed in Phase C);
  * gatekeeper: the training set has no metals (all gaps 0.5-3.0 eV), so the notebook fell back
    to a classifier that always returns 1. We record that fact instead of hiding it;
  * bandgap regressor uses the frozen Optuna params from the April run.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from . import config
from .common import Stage, dump_json

MODULES = ("s02_train",)


class AlwaysSemiconductor:
    """Stand-in used when the training data contains a single class (legacy 'DummyClassifier')."""

    def predict(self, X):
        return np.ones(X.shape[0])


def legacy_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(config.SPLIT_SEED)
    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(config.TRAIN_FRACTION * n)
    return idx[:cut], idx[cut:]


def run(run_dir, force: bool = False) -> dict:
    x_path = run_dir / "features/train_X.npz"
    st = Stage(run_dir, "s02_train", {"training_table": config.TRAINING_TABLE, "train_X": x_path}, MODULES,
               params={"bg": config.BG_PARAMS_LEGACY, "fe": config.FE_PARAMS_LEGACY, "seed": config.MODEL_SEED})
    if st.up_to_date() and not force:
        print("[s02] up to date, skipped")
        return {}

    df = pd.read_csv(config.TRAINING_TABLE)
    X = np.load(x_path)["X"]
    y_bg = df["band_gap_eV"].to_numpy(dtype=np.float32).astype(float)
    y_fe = df["formation_energy_eV_atom"].to_numpy(dtype=np.float32).astype(float)
    is_semi = (y_bg > config.SEMI_THRESHOLD_EV).astype(int)

    tr, va = legacy_split(len(df))

    classes = np.unique(is_semi[tr])
    if len(classes) > 1:
        cls = xgb.XGBClassifier(**config.CLS_PARAMS_LEGACY, n_jobs=-1, random_state=config.MODEL_SEED)
        cls.fit(X[tr], is_semi[tr])
        gatekeeper = "xgb_classifier"
    else:
        cls = AlwaysSemiconductor()
        gatekeeper = "always_semiconductor (training set has one class)"

    semi_tr = is_semi[tr] == 1
    bg = xgb.XGBRegressor(**config.BG_PARAMS_LEGACY, n_jobs=-1, random_state=config.MODEL_SEED)
    bg.fit(X[tr][semi_tr], y_bg[tr][semi_tr])
    fe = xgb.XGBRegressor(**config.FE_PARAMS_LEGACY, n_jobs=-1, random_state=config.MODEL_SEED)
    fe.fit(X[tr], y_fe[tr])

    pred_bg = np.clip(bg.predict(X[va]) * cls.predict(X[va]), 0.0, None)
    pred_fe = fe.predict(X[va])
    bg_mae = float(mean_absolute_error(y_bg[va], pred_bg))
    fe_mae = float(mean_absolute_error(y_fe[va], pred_fe))

    train_formulas = set(df["formula"].to_numpy()[tr])
    val_formula_in_train = float(np.mean([f in train_formulas for f in df["formula"].to_numpy()[va]]))

    joblib.dump({"gatekeeper": cls, "bg_specialist": bg, "fe_specialist": fe, "model_mae": bg_mae,
                 "params": {"bg": config.BG_PARAMS_LEGACY, "fe": config.FE_PARAMS_LEGACY}},
                st.path("models/cascade_legacy.joblib"))
    dump_json(st.path("splits/legacy_random_seed42.json"), {"train": tr.tolist(), "val": va.tolist()})
    pd.DataFrame({"row": va, "formula": df["formula"].to_numpy()[va],
                  "bg_true": y_bg[va], "bg_pred": pred_bg, "fe_true": y_fe[va], "fe_pred": pred_fe}
                 ).to_csv(st.path("predictions/val_legacy.csv"), index=False, float_format="%.6f")

    st.metrics = {"n_train": int(len(tr)), "n_val": int(len(va)), "gatekeeper": gatekeeper,
                  "n_classes_in_train": int(len(classes)), "bg_mae_eV": bg_mae, "fe_mae_eV_atom": fe_mae,
                  "val_rows_formula_in_train": round(val_formula_in_train, 4)}
    dump_json(st.path("metrics/ml_legacy.json"), st.metrics)
    st.finish()
    print(f"[s02] {st.metrics}")
    return st.metrics
