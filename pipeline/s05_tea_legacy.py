"""Stage 05: LEGACY Monte Carlo TEA over the candidates (reproduces notebook cell 5, part 3).

Uses pipeline.tea_legacy (the April engine, flaws included) with the notebook's RNG protocol:
np.random.seed(42) once, then candidates in generation order, 50,000 draws each, Future-2030.
Output schema matches raw_data/01_original_top_discoveries.csv so the gate can diff them.
This stage exists for the reproduction baseline only; TEA v2 replaces it in Phase E.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .common import Stage, dump_json
from .models import load_cascade, model_path
from .tea_legacy import run_tea

MODULES = ("s05_tea_legacy", "tea_legacy", "models")


def run(run_dir, force: bool = False, model_source: str = "retrained") -> dict:
    cpath = run_dir / "screen/candidates.csv"
    model = model_path(run_dir, model_source)
    st = Stage(run_dir, "s05_tea_legacy", {"candidates": cpath, "model": model}, MODULES,
               params={"model_source": model_source, "seed": config.TEA_SEED, "iterations": config.TEA_ITERATIONS, "future": config.TEA_FUTURE})
    if st.up_to_date() and not force:
        print("[s05] up to date, skipped")
        return {}

    cands = pd.read_csv(cpath)
    mae = float(load_cascade(model)["model_mae"])
    np.random.seed(config.TEA_SEED)
    rows = []
    for f, eg, ef in zip(cands.formula, cands.pred_Eg_eV, cands.pred_Ef_eV_atom):
        lcoe, life, mat, pce = run_tea(f, eg, ef, mae, config.TEA_ITERATIONS, config.TEA_FUTURE)
        pce = pce * 100.0
        rows.append({"Formula": f, "Predicted_Bandgap_eV": eg, "Predicted_Ef_eV_atom": ef,
                     "Active_Material_Cost_m2": mat, "Panel_Lifetime_Years": life,
                     "PCE_Median": np.median(pce), "LCOE_Median": np.median(lcoe),
                     "LCOE_Q10_Best": np.percentile(lcoe, 10), "LCOE_Q90_Worst": np.percentile(lcoe, 90)})
    out = pd.DataFrame(rows).sort_values("LCOE_Median")
    out.to_csv(st.path("tea/legacy_fullstats.csv"), index=False, float_format="%.5f")

    top = out.head(10)
    st.metrics = {"n": int(len(out)), "mae_sigma_used": mae,
                  "top10_median_lcoe": float(top.LCOE_Median.median()),
                  "top1": top.Formula.iloc[0], "lifetime_unique_values": sorted(out.Panel_Lifetime_Years.unique().tolist())}
    dump_json(st.path("metrics/tea_legacy.json"), st.metrics)
    st.finish()
    print(f"[s05] {st.metrics}")
    return st.metrics
