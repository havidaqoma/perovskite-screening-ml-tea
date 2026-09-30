"""Corrected perovskite geometry descriptors (Phase C/D fix for the legacy steric filter).

The legacy filter used pymatgen `average_ionic_radius`, which averages over ALL oxidation states
of an element (Br -> 0.88 A instead of 1.96 A for Br-), so it rejects Cs2AgBiBr6. Here:
  1. assign oxidation states with pymatgen's bond-valence-free `oxi_state_guesses` (first guess
     consistent with A+ / B,B' cations / X anions),
  2. take Shannon radii at that oxidation state: A in CN XII (fallback VIII -> VI), B/B' and X in CN VI,
  3. compute Goldschmidt t with the mean B radius, octahedral factor mu, and Bartel's tau
     (Bartel et al., Sci. Adv. 2019, 10.1126/sciadv.aav0693; perovskite if tau < 4.18).
Returns NaNs (never a pass) when no consistent oxidation-state assignment or radius exists.
"""
from __future__ import annotations

import math
from functools import lru_cache

from pymatgen.core import Composition, Species

from .features import LEGACY_ERRORS

ANIONS = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I"}
TAU_MAX = 4.18


@lru_cache(maxsize=None)
def shannon(el: str, ox: int, cns: tuple[str, ...]) -> float | None:
    sp = Species(el, ox)
    for cn in cns:
        try:
            return float(sp.get_shannon_radius(cn=cn, spin="High Spin" if el in ("Fe", "Mn", "Co", "Cr") else ""))
        except LEGACY_ERRORS:
            try:
                return float(sp.get_shannon_radius(cn=cn))
            except LEGACY_ERRORS:
                continue
    return None


def descriptors(formula: str, a_site: str) -> dict:
    """a_site = A element symbol (known from generation). B = other cations, X = anions."""
    nan = {"ox_ok": False, "rA": math.nan, "rB": math.nan, "rX": math.nan, "t": math.nan, "mu": math.nan,
           "tau": math.nan, "tau_perovskite": False}
    comp = Composition(formula)
    try:
        guesses = comp.oxi_state_guesses(max_sites=-1)
    except LEGACY_ERRORS:
        return nan
    for g in guesses:
        if g.get(a_site, 0) != 1 or any(g.get(x, 0) >= 0 for x in ANIONS if x in g):
            continue
        amt = comp.get_el_amt_dict()
        b_els = [e for e in amt if e != a_site and e not in ANIONS]
        x_els = [e for e in amt if e in ANIONS]
        rA = shannon(a_site, 1, ("XII", "VIII", "VI"))
        rBs = [shannon(e, int(round(g[e])), ("VI",)) for e in b_els]
        rXs = [shannon(e, int(round(g[e])), ("VI",)) for e in x_els]
        if rA is None or any(r is None for r in rBs + rXs) or any(float(g[e]) != round(g[e]) for e in b_els):
            continue
        rB = sum(rBs[i] * amt[b] for i, b in enumerate(b_els)) / sum(amt[b] for b in b_els)
        rX = sum(rXs[i] * amt[x] for i, x in enumerate(x_els)) / sum(amt[x] for x in x_els)
        nA = 1.0  # A-site oxidation state
        t = (rA + rX) / (math.sqrt(2) * (rB + rX))
        ratio = rA / rB
        tau = rX / rB - nA * (nA - ratio / math.log(ratio)) if ratio > 1 else math.inf
        return {"ox_ok": True, "rA": rA, "rB": rB, "rX": rX, "t": t, "mu": rB / rX, "tau": tau,
                "tau_perovskite": bool(tau < TAU_MAX), "ox_states": {k: float(v) for k, v in g.items()}}
    return nan
