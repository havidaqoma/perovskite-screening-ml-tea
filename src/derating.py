"""
Composition-aware PCE derating factors for perovskite PV.
Replaces blanket pce_max (0.85/0.70) with chemistry-specific derating.

Three physical penalties:
  1. Optical electronegativity mismatch (Δχ) — band-tailing from mixed anions
  2. Tolerance factor deviation from 1.0 — octahedral distortion → defects
  3. μ (B/X ionic radius ratio) — transport bottleneck when μ too high

Calibration anchors (soft targets ±0.05):
  CsPbI₃ (21.0% PCE, SQ≈30.2% → f≈0.70)
  Cs₂AgBiBr₆ (12.8% PCE, SQ≈17.0% → f≈0.75)
  CsSnI₃ (14.5% PCE, SQ≈33.4% → f≈0.43 — Sn²⁺ instability)

Usage:
    factor = derate_formula("Na2FeMnO3S3")
    pce_derated = sq_limit(eg) * factor   # instead of sq_limit(eg) * 0.85
"""

import numpy as np
from pymatgen.core import Composition, Element

# ---------------------------------------------------------------------------
# SITE CLASSIFICATION — explicit sets, NO electronegativity thresholds
# ---------------------------------------------------------------------------
A_SITES = {"Li", "Na", "K", "Rb", "Cs", "Ca", "Sr", "Ba"}
X_SITES = {"O", "S", "Se", "F", "Cl", "Br", "I"}
# B = everything else that has a Shannon radius (transition/post-transition metals)
B_SITES = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Gd", "Tb", "Dy",
    "Zr", "Hf", "Nb", "Ta", "Mo", "W", "Re",
    "Ga", "In", "Pb", "Sn", "Sb", "Bi", "Ge",
    "Ag", "Cd", "Pd", "Pt", "Au", "Tl",
}

# ---------------------------------------------------------------------------
# SHANNON IONIC RADII (Å) — common oxidation states for perovskites
# ---------------------------------------------------------------------------
SHANNON_RADII = {
    # A-site (12-coord for perovskite)
    "Li": 0.76, "Na": 1.39, "K": 1.64, "Rb": 1.72, "Cs": 1.88,
    "Ca": 1.34, "Sr": 1.44, "Ba": 1.61,
    # B-site (6-coord octahedral)
    "Ti": 0.605, "V": 0.54, "Cr": 0.615, "Mn": 0.645,
    "Fe": 0.645, "Co": 0.65, "Ni": 0.69,
    "Cu": 0.73, "Zn": 0.74, "Ga": 0.62, "Ge": 0.53,
    "Sn": 0.69, "Sb": 0.60, "Bi": 0.76, "In": 0.80,
    "Mo": 0.59, "W": 0.60, "Nb": 0.64, "Ta": 0.64,
    "Zr": 0.72, "Hf": 0.71, "Sc": 0.745, "Y": 0.90, "La": 1.032,
    "Ag": 0.95, "Cd": 0.95,
    "Ce": 1.01, "Pr": 0.99, "Nd": 0.983, "Gd": 0.938,
    "Tb": 0.923, "Dy": 0.912, "Re": 0.55,
    "Pb": 1.19,  # Pb2+ 6-coord (agy-flagged missing)
    "Pt": 0.625, "Pd": 0.64, "Au": 0.85, "Tl": 0.885,  # agy-flagged missing
    # X-site (6-coord)
    "O": 1.40, "F": 1.31, "Cl": 1.81, "Br": 1.96,
    "I": 2.20, "S": 1.84, "Se": 1.98,
}

# ---------------------------------------------------------------------------
# PAULING ELECTRONEGATIVITIES
# ---------------------------------------------------------------------------
CHI = {
    "H": 2.20, "Li": 0.98, "Na": 0.93, "K": 0.82, "Rb": 0.82, "Cs": 0.79,
    "Mg": 1.31, "Ca": 1.00, "Sr": 0.95, "Ba": 0.89,
    "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83,
    "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01,
    "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Bi": 2.02, "Mo": 2.16, "W": 2.36,
    "Nb": 1.60, "Ta": 1.50, "Zr": 1.33, "Hf": 1.3, "Y": 1.22, "La": 1.10,
    "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Ag": 1.93, "Sc": 1.36,
    "O": 3.44, "S": 2.58, "Se": 2.55,
    "F": 3.98, "Cl": 3.16, "Br": 2.96, "I": 2.66,
}

# ---------------------------------------------------------------------------
# DERATING CALIBRATION CONSTANTS
# ---------------------------------------------------------------------------
K_OPT = 0.10    # PCE loss per unit Δχ (linear)
K_DEF = 3.5     # PCE loss per unit |t-1|² (quadratic)
K_TRANS = 0.60  # PCE loss when μ exceeds optimal
MU_OPT = 0.52   # Optimal μ center (Goldschmidt stable range)
MU_SIGMA = 0.25 # Gaussian width for soft transport penalty
DERATE_MIN = 0.40
DERATE_MAX = 1.00


def classify_sites(formula: str):
    """
    Assign elements to A/B/X sites using explicit sets.
    Returns (a_els, b_els, x_els, amounts_dict).
    """
    comp = Composition(formula)
    el_dict = {str(el): float(amt) for el, amt in comp.items()}
    a_els, b_els, x_els = [], [], []
    for e in el_dict:
        if e in A_SITES:
            a_els.append(e)
        elif e in X_SITES:
            x_els.append(e)
        elif e in B_SITES:
            b_els.append(e)
        else:
            # Unknown element: default to B if it has a radius, else skip
            if e in SHANNON_RADII:
                b_els.append(e)
            else:
                x_els.append(e)  # conservative: treat as anion
    return a_els, b_els, x_els, el_dict


def calc_delta_chi(formula: str) -> float:
    """Pauling electronegativity mismatch among X-site anions only.
    Captures band-tailing from mixed anion sublattice, not general lattice ionicity.
    Returns 0.0 for single-anion compounds (no mixing penalty)."""
    _, _, x_els, _ = classify_sites(formula)
    if len(x_els) < 2:
        return 0.0
    chis = [CHI.get(e, 2.5) for e in x_els]
    return max(chis) - min(chis)


def calc_tolerance_factor(formula: str) -> float:
    """
    Goldschmidt t = (r_A + r_X) / (√2 × (r_B_avg + r_X_avg)).
    Returns default 1.0 on failure (no differentiation penalty).
    """
    a_els, b_els, x_els, el_dict = classify_sites(formula)
    if not a_els or not b_els or not x_els:
        return 1.0  # default: no distortion penalty

    # Use heaviest A-site cation
    a_el = max(a_els, key=lambda e: Element(e).Z)
    r_A = SHANNON_RADII.get(a_el, 0.0)
    if r_A == 0.0:
        return 1.0

    r_B_vals = [SHANNON_RADII.get(e, 0.65) for e in b_els]
    r_X_vals = [SHANNON_RADII.get(e, 1.80) for e in x_els]
    r_B = np.mean(r_B_vals)
    r_X = np.mean(r_X_vals)

    if r_B <= 0 or r_X <= 0:
        return 1.0

    t = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))
    return float(t)


def calc_mu(formula: str) -> float:
    """
    μ = r_B_avg / r_X_avg. Returns 0.52 (optimal) on failure.
    """
    _, b_els, x_els, _ = classify_sites(formula)
    if not b_els or not x_els:
        return MU_OPT  # default: no penalty

    r_B = np.mean([SHANNON_RADII.get(e, 0.65) for e in b_els])
    r_X = np.mean([SHANNON_RADII.get(e, 1.80) for e in x_els])
    if r_X <= 0:
        return MU_OPT
    return float(r_B / r_X)


def derate_formula(formula: str) -> float:
    """
    Composition-specific PCE derating factor ∈ [DERATE_MIN, DERATE_MAX].

    PCE_derated = SQ_limit(Eg) × derate_factor
    where derate_factor replaces the blanket pce_max (0.85/0.70).
    """
    dchi = calc_delta_chi(formula)
    t = calc_tolerance_factor(formula)
    mu = calc_mu(formula)

    # Optical penalty: higher Δχ → more band-tailing in mixed anions
    penalty_opt = K_OPT * dchi

    # Defect tolerance: deviation of t from 1.0
    penalty_def = K_DEF * (t - 1.0) ** 2

    # Transport penalty: μ beyond optimal range (Gaussian-like)
    penalty_trans = K_TRANS * ((mu - MU_OPT) / MU_SIGMA) ** 2 * 0.3

    f = 1.0 - (penalty_opt + penalty_def + penalty_trans)
    f = float(np.clip(f, DERATE_MIN, DERATE_MAX))
    # Cap at current technology baseline (0.85) so derating never INCREASES PCE
    f = min(f, 0.85)
    return f


def derate_batch(formulas: list) -> np.ndarray:
    """Derating factors for a list of formulas."""
    return np.array([derate_formula(f) for f in formulas])


# ---------------------------------------------------------------------------
# SELF-TEST & CALIBRATION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== DERATING CALIBRATION (inorganic anchors only) ===\n")

    test_cases = [
        # (formula, target_f_range_low, target_f_range_high, description)
        ("CsPbI3",         0.60, 0.80, "all-inorganic halide (anchor)"),
        ("Cs2AgBiBr6",     0.65, 0.85, "double perovskite (anchor)"),
        ("CsSnI3",         0.35, 0.55, "Sn-halide (Sn²⁺ instability)"),
        ("Na2FeMnO3S3",    None, None, "top-10 candidate (oxo-chalcogenide)"),
        ("Na2WSbS3Br3",    None, None, "top-1 candidate (chalcogenide-halide)"),
        ("Na2SbZnO3Br3",   None, None, "top-3 candidate (oxo-halide)"),
        ("K2TaMnO3F3",     None, None, "most stable (Ef=-2.96)"),
        ("Cs2TiBr6",       None, None, "Ti-halide vacancy-ordered"),
    ]

    for formula, f_lo, f_hi, desc in test_cases:
        f = derate_formula(formula)
        dchi = calc_delta_chi(formula)
        t = calc_tolerance_factor(formula)
        mu = calc_mu(formula)
        a, b, x, _ = classify_sites(formula)
        status = ""
        if f_lo is not None:
            in_range = f_lo <= f <= f_hi
            status = " ✓" if in_range else f" ✗ (expected {f_lo:.2f}-{f_hi:.2f})"
        print(f"  {formula:20s}  f={f:.3f}  Δχ={dchi:.2f}  t={t:.3f}  μ={mu:.3f}"
              f"  A={a} B={b} X={x}{status}  ({desc})")

    # Distribution over a small sample
    import os, sys
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Final_Top_Discoveries_FullStats.csv')
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        formulas = df['Formula'].tolist()
        factors = derate_batch(formulas)
        print(f"\n=== DISTRIBUTION OVER {len(formulas)} CANDIDATES ===")
        print(f"  min={factors.min():.4f}  median={np.median(factors):.4f}"
              f"  mean={factors.mean():.4f}  max={factors.max():.4f}"
              f"  std={factors.std():.4f}")
        print(f"  <0.50: {(factors<0.50).sum()}  |  0.50-0.70: {((factors>=0.50)&(factors<0.70)).sum()}"
              f"  |  0.70-0.85: {((factors>=0.70)&(factors<0.85)).sum()}"
              f"  |  ≥0.85: {(factors>=0.85).sum()}")
        # Per-chemistry-class means
        has_O = [any(c in f for c in ['O']) for f in formulas]
        has_h = [any(c in f for c in ['F','Cl','Br','I']) for f in formulas]
        has_s = [any(c in f for c in ['S','Se']) for f in formulas]
        if any(has_O): print(f"  Contains O:  mean f = {factors[np.array(has_O)].mean():.4f}")
        if any(has_h): print(f"  Contains halide: mean f = {factors[np.array(has_h)].mean():.4f}")
        if any(has_s): print(f"  Contains chalcogenide: mean f = {factors[np.array(has_s)].mean():.4f}")
        print(f"\n  Spread (std) = {factors.std():.4f}", end="")
        if factors.std() < 0.03:
            print("  ⚠ LOW SPREAD — derating may not meaningfully re-rank candidates.")
        else:
            print("  ✓ Good spread for re-ranking.")
