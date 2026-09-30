"""Stage 04: ML screening of steric survivors (reproduces notebook cell 5, part 2).

Keeps candidates with 0.5 <= Eg <= 2.5 eV, Ef < 1.0 eV/atom and no toxic element.
Every survivor gets a row in screen/ml_predictions.csv (pass/fail + reason), not only the kept ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from . import config
from .common import Stage, dump_json
from .features import featurize_many
from .models import load_cascade, model_path

MODULES = ("s04_screen", "features", "models")


def run(run_dir, force: bool = False, model_source: str = "retrained") -> dict:
    surv = run_dir / "screen/steric_survivors.csv"
    model = model_path(run_dir, model_source)
    st = Stage(run_dir, "s04_screen", {"steric_survivors": surv, "model": model}, MODULES,
               params={"model_source": model_source, "eg": config.EG_WINDOW_EV, "ef_max": config.EF_MAX_EV_ATOM, "toxic": config.TOXIC})
    if st.up_to_date() and not force:
        print("[s04] up to date, skipped")
        return {}

    formulas = pd.read_csv(surv)["formula"].tolist()
    pkg = load_cascade(model)
    X, failed = featurize_many(formulas)
    ok = np.isfinite(X).all(axis=1)
    eg = np.full(len(formulas), np.nan)
    ef = np.full(len(formulas), np.nan)
    if ok.any():
        semi = pkg["gatekeeper"].predict(X[ok])
        eg[ok] = np.maximum(0.0, pkg["bg_specialist"].predict(X[ok]) * semi)
        ef[ok] = pkg["fe_specialist"].predict(X[ok])

    reasons = []
    for f, e_g, e_f, good in zip(formulas, eg, ef, ok):
        if not good:
            reasons.append("featurize_failed")
        elif not (config.EG_WINDOW_EV[0] <= e_g <= config.EG_WINDOW_EV[1]):
            reasons.append("eg_out_of_window")
        elif not (e_f < config.EF_MAX_EV_ATOM):
            reasons.append("ef_too_high")
        elif any(el.symbol in config.TOXIC for el in Composition(f).elements):
            reasons.append("toxic")
        else:
            reasons.append("pass")
    table = pd.DataFrame({"formula": formulas, "pred_Eg_eV": eg, "pred_Ef_eV_atom": ef, "result": reasons})
    table.to_csv(st.path("screen/ml_predictions.csv"), index=False, float_format="%.6f")
    cands = table[table.result == "pass"].reset_index(drop=True)
    cands.to_csv(st.path("screen/candidates.csv"), index=False, float_format="%.6f")

    st.metrics = {"model_source": model_source, "n_in": len(formulas), "n_candidates": int(len(cands)),
                  "rejections": table.result.value_counts().to_dict(), "model_mae_used": pkg["model_mae"]}
    dump_json(st.path("metrics/funnel_screen.json"), st.metrics)
    st.finish()
    print(f"[s04] {st.metrics}")
    return st.metrics
