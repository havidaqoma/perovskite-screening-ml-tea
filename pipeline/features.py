"""139-dimensional composition features (verbatim logic from notebook cell 1 / legacy cell_00).

132 Magpie (matminer ElementProperty 'magpie') + 4 Mendeleev/Pettifor + Delta_chi + t + mu.
Do not "fix" anything here without a version bump: the legacy model depends on it bit-for-bit.
Known quirks kept on purpose (logged in IMPLEMENTATION_NOTES.md):
  * the steric t/mu features sort elements by electronegativity, so for mixed-anion or
    non-perovskite formulas "A", "B" and "X" are not the crystallographic sites;
  * the Mendeleev variance uses the plain list minus a scalar (works because numpy broadcasts).
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from pymatgen.core import Composition, Element
from pymatgen.core.units import UnitError  # subclasses BaseException; the notebook's bare except caught it

# Exceptions the legacy notebook swallowed with bare `except:` (UnitError is not an Exception subclass).
LEGACY_ERRORS = (Exception, UnitError)

warnings.filterwarnings("ignore")

MENDELEEV = {'He': 1, 'Ne': 2, 'Ar': 3, 'Kr': 4, 'Xe': 5, 'Rn': 6, 'F': 7, 'Cl': 8, 'Br': 9, 'I': 10,
             'O': 11, 'S': 12, 'Se': 13, 'Te': 14, 'N': 15, 'P': 16, 'As': 17, 'Sb': 18, 'Bi': 19,
             'C': 20, 'Si': 21, 'Ge': 22, 'Sn': 23, 'Pb': 24, 'B': 25, 'Al': 26, 'Ga': 27, 'In': 28,
             'Tl': 29, 'Zn': 30, 'Cd': 31, 'Hg': 32, 'Cu': 33, 'Ag': 34, 'Au': 35, 'Ni': 36, 'Pd': 37,
             'Pt': 38, 'Co': 39, 'Rh': 40, 'Ir': 41, 'Fe': 42, 'Ru': 43, 'Os': 44, 'Mn': 45, 'Tc': 46,
             'Re': 47, 'Cr': 48, 'Mo': 49, 'W': 50, 'V': 51, 'Nb': 52, 'Ta': 53, 'Ti': 54, 'Zr': 55,
             'Hf': 56, 'Sc': 57, 'Y': 58, 'Lu': 73, 'Li': 92, 'Na': 93, 'K': 94, 'Rb': 95, 'Cs': 96}

EXTRA_NAMES = ["Mendeleev_Mean", "Mendeleev_Max", "Mendeleev_Min", "Mendeleev_Var",
               "Delta_Electronegativity", "Goldschmidt_Tolerance", "Octahedral_Factor"]


@lru_cache(maxsize=1)
def _featurizer():
    from matminer.featurizers.composition import ElementProperty
    return ElementProperty.from_preset(preset_name="magpie")


def feature_names() -> list[str]:
    return list(_featurizer().feature_labels()) + EXTRA_NAMES


def get_radius(el_sym: str) -> float:
    el = Element(el_sym)
    return el.average_ionic_radius if el.average_ionic_radius else el.atomic_radius


def get_ultimate_features(formula: str) -> np.ndarray:
    comp = Composition(formula)
    elements = comp.elements
    fractions = [comp.get_atomic_fraction(el) for el in elements]

    magpie_feats = _featurizer().featurize(comp)

    mn = [float(MENDELEEV.get(el.symbol, 50.0)) for el in elements]
    mn_w = np.average(mn, weights=fractions)
    mn_feats = [mn_w, np.max(mn), np.min(mn), np.average((mn - mn_w) ** 2, weights=fractions)]

    X = [float(getattr(el, 'X', 0.0) or 0.0) for el in elements]
    delta_x = [np.max(X) - np.min(X)]

    try:
        sorted_els = sorted(elements, key=lambda e: getattr(e, 'X', 0.0) or 0.0)
        t_factor = (get_radius(sorted_els[0]) + get_radius(sorted_els[-1])) / (
            np.sqrt(2) * (get_radius(sorted_els[1]) + get_radius(sorted_els[-1])))
        mu_factor = get_radius(sorted_els[1]) / get_radius(sorted_els[-1])
        steric_feats = [t_factor, mu_factor]
    except LEGACY_ERRORS:
        steric_feats = [0.0, 0.0]

    return np.array(magpie_feats + mn_feats + delta_x + steric_feats, dtype=float)


def _featurize_chunk(formulas):
    warnings.filterwarnings("ignore")
    return featurize_many(formulas, progress=False, n_jobs=1)


def featurize_many(formulas, progress: bool = False, n_jobs: int = 1) -> tuple[np.ndarray, list[int]]:
    """Return (matrix, failed_row_indices). Failed rows are filled with NaN, never dropped,
    so row indices stay aligned with the input table. n_jobs>1 uses process chunks (same output)."""
    if n_jobs > 1 and len(formulas) > 2000:
        from joblib import Parallel, delayed
        formulas = list(formulas)
        size = int(np.ceil(len(formulas) / (n_jobs * 4)))
        chunks = [formulas[i:i + size] for i in range(0, len(formulas), size)]
        parts = Parallel(n_jobs=n_jobs, verbose=5 if progress else 0)(delayed(_featurize_chunk)(c) for c in chunks)
        mats, failed, offset = [], [], 0
        for (m, f), c in zip(parts, chunks):
            mats.append(m)
            failed.extend(offset + i for i in f)
            offset += len(c)
        return np.vstack(mats), failed
    it = formulas
    if progress:
        from tqdm import tqdm
        it = tqdm(formulas, desc="featurize", mininterval=5)
    rows, failed = [], []
    n = len(feature_names())
    for i, f in enumerate(it):
        try:
            rows.append(get_ultimate_features(f))
        except LEGACY_ERRORS:
            rows.append(np.full(n, np.nan))
            failed.append(i)
    return np.vstack(rows), failed
