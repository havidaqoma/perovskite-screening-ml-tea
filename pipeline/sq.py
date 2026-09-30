"""Detailed-balance (Shockley-Queisser) efficiency limit from the ASTM G173-03 AM1.5G spectrum.

Replaces the legacy hand-typed 20-point SQ table. Radiative limit only: step-function absorptance
(1 above Eg, 0 below), cell at 300 K, 1-sun AM1.5G (1000 W/m2), no concentration.
  J_sc(Eg) = q * integral_{E>Eg} photon flux(E) dE            (spectrum from pvlib, ASTM G173-03)
  J_0(Eg)  = q * (2*pi / (h^3 c^2)) * integral_{E>Eg} E^2 / (exp(E/kT) - 1) dE
  eta      = max_V  V * (J_sc - J_0 (exp(qV/kT) - 1)) / P_in
Expected: peak ~33.7% near 1.34 eV (Ruhle, Sol. Energy 2016, 10.1016/j.solener.2016.02.015).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

Q = 1.602176634e-19
H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
T_CELL = 300.0


@lru_cache(maxsize=1)
def _spectrum():
    from pvlib.spectrum import get_reference_spectra
    s = get_reference_spectra()                      # W m^-2 nm^-1, index = wavelength (nm)
    lam = s.index.to_numpy(float) * 1e-9             # m
    irr = s["global"].to_numpy(float) * 1e9          # W m^-2 m^-1
    p_in = float(np.trapezoid(s["global"].to_numpy(float), s.index.to_numpy(float)))
    E = H * C / lam                                  # J
    flux_per_lam = irr / E                           # photons s^-1 m^-2 m^-1
    # convert to per-energy: phi(E) dE = phi(lam) dlam, dlam/dE = lam^2/(hc)
    flux_per_E = flux_per_lam * lam ** 2 / (H * C)
    order = np.argsort(E)
    return E[order], flux_per_E[order], p_in


@lru_cache(maxsize=1)
def sq_table(eg_min: float = 0.3, eg_max: float = 4.0, n: int = 741) -> tuple[np.ndarray, np.ndarray]:
    E, phi, p_in = _spectrum()
    kT = KB * T_CELL
    Ebb = np.linspace(E.min(), 6.0 * Q, 20000)
    bb = 2 * np.pi / (H ** 3 * C ** 2) * Ebb ** 2 / np.expm1(Ebb / kT)       # photons s^-1 m^-2 J^-1
    egs = np.linspace(eg_min, eg_max, n)
    eff = np.zeros(n)
    # tail integrals accumulated FROM THE TOP (reverse cumsum). Computing total - cumsum cancels
    # catastrophically for the black-body tail (~exp(-Eg/kT)) and silently returns J0 = 0.
    def tail(y, x):
        seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
        return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])

    tail_sc = tail(phi, E)
    tail_bb = tail(bb, Ebb)
    for i, eg in enumerate(egs):
        jsc = Q * np.interp(eg * Q, E, tail_sc)
        j0 = Q * np.interp(eg * Q, Ebb, tail_bb)
        if jsc <= 0 or j0 <= 0:
            continue
        voc = kT / Q * np.log(jsc / j0 + 1.0)
        v = np.linspace(0.0, voc, 4000)
        p = v * (jsc - j0 * np.expm1(Q * v / kT))
        eff[i] = p.max() / p_in
    return egs, eff


def sq_limit(eg) -> np.ndarray:
    """SQ efficiency (fraction) for band gap(s) in eV; 0 outside the tabulated 0.3-4.0 eV range."""
    egs, eff = sq_table()
    return np.interp(np.asarray(eg, float), egs, eff, left=0.0, right=0.0)
