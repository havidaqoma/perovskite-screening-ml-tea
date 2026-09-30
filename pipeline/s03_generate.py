"""Stage 03: combinatorial A2BB'X6 generation + physical moat (reproduces notebook cell 5, part 1).

Funnel with the pinned pymatgen 2026.3.23: 23,940 -> 6,080 (charge) -> 3,284 (steric).
The steric count depends on pymatgen's ionic-radius tables (2024.2.8 gives 3,615).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from . import config
from .common import Stage, dump_json
from .features import LEGACY_ERRORS, get_radius

MODULES = ("s03_generate", "features")


def generate() -> list[str]:
    out = []
    b_pairs = list(itertools.combinations(config.B_SITE, 2))
    x_pairs = list(itertools.combinations(config.X_SITE, 2))
    for a in config.A_SITE:
        for b1, b2 in b_pairs:
            for x in config.X_SITE:
                out.append(f"{a}2{b1}{b2}{x}6")
            for x1, x2 in x_pairs:
                out.append(f"{a}2{b1}{b2}{x1}3{x2}3")
    return out


def is_charge_balanced(formula: str) -> bool:
    try:
        comp = Composition(formula)
        amounts = list(comp.get_el_amt_dict().values())
        ox_states = [el.common_oxidation_states for el in comp.elements]
        if not all(ox_states):
            return False
        for combo in itertools.product(*ox_states):
            if abs(sum(c * a for c, a in zip(combo, amounts))) < 0.1:
                return True
        return False
    except LEGACY_ERRORS:
        return False


def steric_descriptors(formula: str) -> tuple[float, float] | None:
    try:
        elements = list(Composition(formula).get_el_amt_dict().keys())
        r_A = get_radius(elements[0])
        r_B = (get_radius(elements[1]) + get_radius(elements[2])) / 2.0
        r_X = get_radius(elements[3]) if len(elements) == 4 else (get_radius(elements[3]) + get_radius(elements[4])) / 2.0
        t = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))
        return float(t), float(r_B / r_X)
    except LEGACY_ERRORS:
        return None


def is_sterically_stable(formula: str) -> bool:
    d = steric_descriptors(formula)
    if d is None:
        return False
    t, mu = d
    return (config.TOL_RANGE[0] <= t <= config.TOL_RANGE[1]) and (config.MU_RANGE[0] <= mu <= config.MU_RANGE[1])


def run(run_dir, force: bool = False) -> dict:
    st = Stage(run_dir, "s03_generate", {}, MODULES,
               params={"A": config.A_SITE, "B": config.B_SITE, "X": config.X_SITE,
                       "t": config.TOL_RANGE, "mu": config.MU_RANGE})
    if st.up_to_date() and not force:
        print("[s03] up to date, skipped")
        return {}

    gen = generate()
    charge = [f for f in gen if is_charge_balanced(f)]
    rows = []
    for f in charge:
        d = steric_descriptors(f)
        t, mu = d if d else (np.nan, np.nan)
        rows.append({"formula": f, "t": t, "mu": mu, "steric_pass": is_sterically_stable(f)})
    table = pd.DataFrame(rows)
    table.to_csv(st.path("screen/charge_balanced.csv"), index=False, float_format="%.6f")
    steric = table.loc[table.steric_pass, "formula"].tolist()
    pd.Series(steric, name="formula").to_csv(st.path("screen/steric_survivors.csv"), index=False)

    st.metrics = {"n_generated": len(gen), "n_unique_generated": len(set(gen)),
                  "n_charge_balanced": len(charge), "n_steric": len(steric)}
    dump_json(st.path("metrics/funnel_generate.json"), st.metrics)
    st.finish()
    print(f"[s03] {st.metrics}")
    return st.metrics
