"""LEGACY techno-economic engine, verbatim from notebook cell 3 (April 2026).

This is the exact engine that produced raw_data/01_original_top_discoveries.csv, INCLUDING
the degenerate Ef->lifetime line that the Aug 27 review rejected. It exists only so the
regression gate can prove the pipeline reproduces the April numbers. The corrected TEA (v2)
is built in Phase E and must not import from this module.

Only change vs the notebook: the module-level print() was removed.
"""
import numpy as np
from pymatgen.core import Composition

ELEMENT_PRICES_KG = {
    "H": 1.39, "Li": 16.00, "O": 0.15, "F": 2.00, "Na": 3.00, "Mg": 2.32, "Al": 3.57, "S": 6.48, "Cl": 0.57, "K": 12.85,
    "Ca": 2.28, "Sc": 9350.00, "Ti": 7.59, "Cr": 8.52, "Mn": 1.94, "Fe": 0.25, "Co": 56.28, "Ni": 17.47, "Cu": 13.06,
    "Zn": 3.09, "Ga": 1331.58, "Ge": 5287.02, "Br": 3.78, "Rb": 10210.00, "Sr": 6.01, "Y": 33.00, "Zr": 23.14,
    "Mo": 74.84, "Ru": 16155.50, "Rh": 76840.00, "Pd": 42326.00, "Ag": 1506.00, "Cd": 2.36, "In": 787.93,
    "Sn": 34.13, "Sb": 51.80, "Te": 174.12, "I": 78.43, "Cs": 36994.83, "Ba": 0.26, "La": 4.00, "Ce": 4.36,
    "Pr": 148.80, "Nd": 139.40, "Gd": 55.00, "Tb": 2289.25, "Dy": 640.35, "Hf": 6961.10, "Ta": 209.00,
    "W": 32.07, "Re": 4012.15, "Pt": 47201.00, "Au": 96829.50, "Pb": 2.11, "Bi": 38.67
}


def calc_material_cost(formula, future=False):
    try:
        comp = Composition(formula)
        base_mass_g = 5.0
        utilization = 0.85 if future else 0.30
        required_mass_kg = (base_mass_g / utilization) / 1000.0
        raw_cost = sum([ELEMENT_PRICES_KG.get(el.symbol, 50.0) * comp.get_wt_fraction(el) * required_mass_kg
                        for el in comp.elements])
        processing_markup = 1.50 if future else 2.50
        return raw_cost * processing_markup
    except Exception:
        return 1.5


SQ_EG_POINTS = np.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.34, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.4, 2.6, 3.0])
SQ_EFF_POINTS = np.array([0.0, 0.0, 14.5, 24.5, 30.0, 32.8, 33.6, 33.7, 33.4, 32.1, 30.2, 28.1, 25.8, 23.5, 21.3, 19.1, 17.0, 13.0, 9.5, 4.0])


def sq_limit(bandgap_array):
    eff = np.interp(bandgap_array, SQ_EG_POINTS, SQ_EFF_POINTS, left=0.0, right=0.0)
    return eff / 100.0


def run_tea(formula, pred_Eg, pred_Ef, mae_error, iterations=50000, future=False):
    rate = np.random.uniform(0.03 if future else 0.04, 0.05 if future else 0.07, iterations)
    mat_cost = calc_material_cost(formula, future=future)

    if future:
        fixed_mod_cost = np.random.normal(31.0, 3.0, iterations)
        area_bos = np.random.normal(80.0, 5.0, iterations)
        inv_rate = 150.0
        om_rate = 10.0
        pce_max = 0.85
    else:
        fixed_mod_cost = np.random.normal(73.5, 5.0, iterations)
        area_bos = np.random.normal(136.0, 10.0, iterations)
        inv_rate = 300.0
        om_rate = 20.0
        pce_max = 0.70

    mod_cost = fixed_mod_cost + mat_cost
    if not future:
        if "Sn" in formula:
            mod_cost /= 0.85
        if pred_Ef > 0.5:
            mod_cost /= 0.80

    pce = np.clip(np.random.normal(sq_limit(np.random.normal(pred_Eg, mae_error, iterations)) * pce_max, 0.02,
                                   iterations), 0.01, 0.33)
    kw_per_m2 = 1.0 * pce
    capex_m2 = mod_cost + area_bos + (kw_per_m2 * inv_rate)

    life = max(5.0 if future else 1.0, min(30.0 if future else 20.0, (30.0 if future else 20.0) - (pred_Ef * 30.0)))

    energy_m2 = np.zeros(iterations)
    costs_m2 = capex_m2.copy()
    for yr in range(1, int(life) + 1):
        df = 1.0 / ((1.0 + rate) ** yr)
        deg_rate = np.clip(0.0074 + (pred_Ef * 0.02), 0.0074, 0.044)
        degradation = (1.0 - deg_rate) ** yr
        energy_m2 += (1471.0 * kw_per_m2 * degradation) * df
        costs_m2 += (om_rate * kw_per_m2) * df
        if yr == 15:
            costs_m2 += (kw_per_m2 * inv_rate) * df

    lcoe = costs_m2 / np.clip(energy_m2, 1e-5, None)
    return lcoe, life, mat_cost, pce
