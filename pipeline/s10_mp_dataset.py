"""Stage 10: fresh Materials Project training set v2 (replaces the April 0.5-3.0 eV-only set).

Why a new set:
  * the April set had no metals and no gaps < 0.5 eV, so the metal/semiconductor gatekeeper
    could never be trained (the cascade was a dummy);
  * it stored formulas only (no material_id, no energy_above_hull), so polymorphs were mixed
    and the "E_above_hull < 0.1" claim in the manuscript was untrue.

v2 query: summary.search(num_elements=(2,5), deprecated=False), radioactive elements excluded,
scalar fields only. Target = GROUND-STATE polymorph per reduced formula (lowest energy_above_hull),
which removes the polymorph noise from a composition-only target (one row per formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from . import config
from .common import Stage, dump_json
from .mp_client import cached_summary

MODULES = ("s10_mp_dataset", "mp_client")
RADIOACTIVE = ["Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"]
FIELDS = ["material_id", "formula_pretty", "nelements", "nsites", "band_gap", "is_gap_direct", "is_metal",
          "formation_energy_per_atom", "energy_above_hull", "theoretical"]
METAL_GAP_EV = 0.1          # gap below this = "metal / not a PV absorber" class


def run(run_dir, force: bool = False) -> dict:
    st = Stage(run_dir, "s10_mp_dataset", {}, MODULES, params={"fields": FIELDS, "exclude": RADIOACTIVE,
                                                               "metal_gap": METAL_GAP_EV})
    if st.up_to_date() and not force:
        print("[s10] up to date, skipped")
        return {}
    parts, metas = [], []
    for n in (2, 3, 4, 5):
        df, meta = cached_summary(f"summary_nel{n}", {"num_elements": [n, n], "deprecated": False,
                                                       "exclude_elements": RADIOACTIVE}, FIELDS)
        parts.append(df)
        metas.append({k: meta[k] for k in ("name", "mp_database_version", "retrieved_utc", "n_rows", "sha256")})
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.dropna(subset=["band_gap", "formation_energy_per_atom", "energy_above_hull"])
    raw["formula"] = [Composition(f).reduced_formula for f in raw.formula_pretty]
    raw = raw.sort_values(["formula", "energy_above_hull", "formation_energy_per_atom", "material_id"])
    n_poly = raw.groupby("formula").size()
    gs = raw.drop_duplicates("formula", keep="first").copy()
    gs["n_polymorphs"] = gs.formula.map(n_poly).astype(int)
    gs["chemsys"] = ["-".join(sorted(e.symbol for e in Composition(f).elements)) for f in gs.formula]
    gs["is_semi"] = (gs.band_gap >= METAL_GAP_EV).astype(int)
    gs = gs.reset_index(drop=True)
    gs.insert(0, "row", np.arange(len(gs)))
    cols = ["row", "formula", "material_id", "chemsys", "nelements", "nsites", "band_gap", "is_gap_direct",
            "is_semi", "formation_energy_per_atom", "energy_above_hull", "theoretical", "n_polymorphs"]
    gs[cols].to_csv(st.path("data/train_v2.csv"), index=False, float_format="%.6f")

    st.metrics = {"mp_queries": metas, "n_raw_entries": int(len(raw)), "n_formulas": int(len(gs)),
                  "n_chemsys": int(gs.chemsys.nunique()), "frac_semi": round(float(gs.is_semi.mean()), 4),
                  "frac_ehull_le_0.1": round(float((gs.energy_above_hull <= 0.1).mean()), 4),
                  "by_nelements": gs.nelements.value_counts().sort_index().to_dict()}
    dump_json(st.path("metrics/dataset_v2.json"), st.metrics)
    st.finish()
    print(f"[s10] {st.metrics}")
    return st.metrics
