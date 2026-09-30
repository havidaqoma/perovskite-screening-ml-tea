"""Stage 01: featurize the training table (31,275 MP entries) -> 139-dim matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .common import Stage, sha256_file
from .features import featurize_many, feature_names

MODULES = ("s01_featurize", "features")


def run(run_dir, force: bool = False) -> dict:
    if sha256_file(config.TRAINING_TABLE) is None:
        raise FileNotFoundError(config.TRAINING_TABLE)
    st = Stage(run_dir, "s01_featurize", {"training_table": config.TRAINING_TABLE}, MODULES)
    if st.up_to_date() and not force:
        print("[s01] up to date, skipped")
        return {}

    df = pd.read_csv(config.TRAINING_TABLE)
    if not (df["row"].to_numpy() == np.arange(len(df))).all():
        raise ValueError("training_table row index is not 0..N-1; split indices would be wrong")
    X, failed = featurize_many(df["formula"].tolist(), progress=True)
    if failed:
        # The legacy notebook silently dropped failures, which would shift every split index.
        # The April run printed Train 25020 | Val 6255 (= all 31,275 rows), so 0 failures is expected.
        raise RuntimeError(f"{len(failed)} formulas failed featurization, e.g. rows {failed[:5]}")

    np.savez_compressed(st.path("features/train_X.npz"), X=X)
    pd.Series(feature_names()).to_csv(st.path("features/feature_names.csv"), index=False, header=["name"])
    st.metrics = {"n_rows": int(len(df)), "n_features": int(X.shape[1]), "n_failed": 0,
                  "n_unique_formulas": int(df["formula"].nunique())}
    st.finish()
    print(f"[s01] {st.metrics}")
    return st.metrics
