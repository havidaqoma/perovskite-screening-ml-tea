"""Offline tests for the v2 components (geometry, conformal probability, stability math, A2BB'X6 parser)."""
import numpy as np
import pytest
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from pipeline.geometry import descriptors
from pipeline.s14_screen_v2 import gap_posterior_prob
from pipeline.s20_stability import hull_for_candidate, is_a2bbx6


def test_corrected_geometry_accepts_known_double_perovskites():
    for f in ("Cs2AgBiBr6", "Cs2AgBiCl6", "Cs2AgInCl6", "Cs2NaBiCl6"):
        d = descriptors(f, "Cs")
        assert d["ox_ok"] and d["tau_perovskite"], f
    d = descriptors("Cs2AgBiBr6", "Cs")
    assert d["rX"] == pytest.approx(1.96) and 0.85 < d["t"] < 0.95     # Br- CN VI Shannon radius


def test_geometry_fails_closed_without_oxidation_states():
    d = descriptors("Li2TiZrO6", "Li")                                   # no charge-balanced assignment
    assert not d["ox_ok"] and not d["tau_perovskite"]


def test_gap_posterior_prob_limits():
    signed = np.random.default_rng(0).normal(0, 1, 20000)
    p_centre = gap_posterior_prob(np.array([1.4]), np.array([0.01]), signed, 1.0, 1.8)[0]
    p_far = gap_posterior_prob(np.array([3.5]), np.array([0.01]), signed, 1.0, 1.8)[0]
    p_wide = gap_posterior_prob(np.array([1.4]), np.array([0.4]), signed, 1.0, 1.8)[0]
    assert p_centre == pytest.approx(1.0) and p_far == pytest.approx(0.0)
    assert 0.6 < p_wide < 0.7                                             # P(|z| <= 1) for sigma = 0.4



def test_hull_distance_and_probability():
    pd_ = PhaseDiagram([PDEntry(Composition("Na"), 0.0), PDEntry(Composition("Cl"), 0.0),
                        PDEntry(Composition("NaCl"), -4.0)])                # Ef(NaCl) = -2 eV/atom
    res = np.zeros(1000)                                                   # perfect model
    # Na2Cl (x_Na = 2/3) is not in the diagram; hull there = line Na(0) -- NaCl(-2 at x=0.5) = -4/3 eV/atom
    h = hull_for_candidate("Na2Cl", -4.0 / 3.0 + 0.045, res, pd_)   # 45 meV: off the float boundary
    assert h["hull_surface_ef"] == pytest.approx(-4.0 / 3.0, abs=1e-6)
    assert h["ehull_pred_meV"] == pytest.approx(45.0, abs=1e-3)
    assert h["p_ehull_le_35"] == 0.0 and h["p_ehull_le_50"] == 1.0
    assert "mp_ehull_meV" not in h


def test_a2bbx6_parser():
    assert is_a2bbx6("Cs2AgBiBr6") and is_a2bbx6("Na2FeMnO3S3")
    assert not is_a2bbx6("CsPbI3") and not is_a2bbx6("Cs2AgBiBr2I4") and not is_a2bbx6("Cs2SnI6")
