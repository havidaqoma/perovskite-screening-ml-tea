"""Stage 11b: leakage audit on the ORIGINAL April training set (31,275 rows, 0.5-3.0 eV).

Same features, same XGB params, same 80/20 size as the April notebook; only the split changes:
  random_row  : April protocol (seed-42 shuffle)  -> formula leakage (43% of val formulas in train)
  formula     : GroupShuffleSplit by formula      -> no identical formula across the split
  chemsys     : GroupShuffleSplit by element set  -> whole chemical systems held out
Reports MAE for each, so the leakage inflation of 0.349 eV is measured, not argued.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .common import Stage, dump_json
from .s02_train import legacy_split
from .s11_ml_eval import _xgb_reg

MODULES = ("s11b_leakage_audit",)


def run(run_dir, force: bool = False, legacy_run: str = "legacy_repro") -> dict:
    X_path = config.RUNS_DIR / legacy_run / "features/train_X.npz"
    st = Stage(run_dir, "s11b_leakage_audit", {"training_table": config.TRAINING_TABLE, "X": X_path}, MODULES)
    if st.up_to_date() and not force:
        print("[s11b] up to date, skipped")
        return {}
    df = pd.read_csv(config.TRAINING_TABLE)
    X = np.load(X_path)["X"]
    y = df.band_gap_eV.to_numpy(np.float32).astype(float)
    chemsys = ["-".join(sorted(e.symbol for e in Composition(f).elements)) for f in df.formula]
    splits = {"random_row": legacy_split(len(df))}
    for name, groups in (("formula", df.formula.to_numpy()), ("chemsys", np.array(chemsys))):
        splits[name] = next(GroupShuffleSplit(1, test_size=0.2, random_state=config.SPLIT_SEED).split(X, groups=groups))
    out = {}
    for name, (tr, va) in splits.items():
        p = _xgb_reg().fit(X[tr], y[tr]).predict(X[va])
        leak = float(np.mean(np.isin(df.formula.to_numpy()[va], df.formula.to_numpy()[tr])))
        out[name] = {"n_train": int(len(tr)), "n_val": int(len(va)), "bg_mae_eV": float(mean_absolute_error(y[va], p)),
                     "val_formula_in_train": round(leak, 4)}
        print(f"[s11b] {name}: {out[name]}", flush=True)
    dump_json(st.path("metrics/leakage_audit.json"), out)
    st.metrics = out
    st.finish()
    return out
