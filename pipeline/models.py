"""Model loading shared by screening/TEA stages.

Two model sources:
  * "retrained"   : runs/<id>/models/cascade_legacy.joblib written by s02 (same data/split/params)
  * "april_joblib": the original xgboost_cascade_pipeline_2.joblib from the April 2026 notebook run
The April pickle references `__main__.DummyClassifier`; we register a compatible class there
before unpickling so the file loads without editing it.
"""
from __future__ import annotations

import __main__
from pathlib import Path

import joblib
import numpy as np

from . import config

APRIL_JOBLIB = config.ROOT / "xgboost_cascade_pipeline_2.joblib"


class DummyClassifier:  # name must match the April pickle
    def predict(self, X):
        return np.ones(X.shape[0])


def model_path(run_dir: Path, source: str) -> Path:
    if source == "retrained":
        return run_dir / "models/cascade_legacy.joblib"
    if source == "april_joblib":
        return APRIL_JOBLIB
    raise ValueError(f"unknown model source {source!r}")


def load_cascade(path: Path) -> dict:
    if not hasattr(__main__, "DummyClassifier"):
        __main__.DummyClassifier = DummyClassifier
    from .s02_train import AlwaysSemiconductor  # noqa: F401  (retrained pickle)
    pkg = joblib.load(path)
    for key in ("gatekeeper", "bg_specialist", "fe_specialist", "model_mae"):
        if key not in pkg:
            raise KeyError(f"{path.name} lacks '{key}'")
    return pkg
