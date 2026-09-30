"""Offline unit tests for the scripted pipeline (no training, no network)."""
import json

import numpy as np
import pytest

from pipeline import config
from pipeline.common import Stage, sha256_file
from pipeline.features import feature_names, get_ultimate_features
from pipeline.s02_train import legacy_split
from pipeline.s03_generate import generate, is_charge_balanced, is_sterically_stable
from pipeline.tea_legacy import run_tea as legacy_run_tea


def test_feature_vector_is_139_dim():
    assert len(feature_names()) == 139
    v = get_ultimate_features("Cs2AgBiBr6")
    assert v.shape == (139,) and np.isfinite(v).all()


def test_generation_count_matches_notebook():
    gen = generate()
    assert len(gen) == 23940 == len(set(gen))


def test_charge_balance_examples():
    assert is_charge_balanced("Cs2AgBiBr6")        # Cs+ Ag+ Bi3+ Br-
    assert not is_charge_balanced("K2AgAgF6")       # Ag has no common state that balances F6


def test_legacy_steric_filter_rejects_cs2agbibr6_known_flaw():
    # KNOWN FLAW kept for reproduction: pymatgen average_ionic_radius averages ALL oxidation
    # states (Br -> 0.88 A instead of ~1.96 A for Br-), so mu = 1.20 and the best-known
    # lead-free double perovskite fails the legacy "physical moat". Fixed in Phase C/D.
    from pipeline.s03_generate import steric_descriptors
    t, mu = steric_descriptors("Cs2AgBiBr6")
    assert mu > 0.95 and not is_sterically_stable("Cs2AgBiBr6")


def test_legacy_split_is_deterministic_and_disjoint():
    tr1, va1 = legacy_split(31275)
    tr2, va2 = legacy_split(31275)
    assert (tr1 == tr2).all() and (va1 == va2).all()
    assert len(tr1) == 25020 and len(va1) == 6255
    assert not set(tr1) & set(va1)


def test_legacy_tea_keeps_degenerate_lifetime_on_purpose():
    # The April engine clamps lifetime to 30 yr for any negative Ef (the flaw the review found).
    np.random.seed(0)
    _, life, _, _ = legacy_run_tea("Na2FeMnO3S3", 1.40, -1.6, 0.35, iterations=200, future=True)
    assert life == 30.0


def test_stage_manifest_skip_and_invalidation(tmp_path, monkeypatch):
    inp = tmp_path / "in.txt"
    inp.write_text("a")
    st = Stage(tmp_path, "t", {"in": inp}, modules=("common",))
    st.path("out/x.txt").write_text("hello")
    st.finish()
    assert Stage(tmp_path, "t", {"in": inp}, modules=("common",)).up_to_date()
    inp.write_text("b")                               # input change -> must recompute
    assert not Stage(tmp_path, "t", {"in": inp}, modules=("common",)).up_to_date()
    inp.write_text("a")
    (tmp_path / "out/x.txt").write_text("tampered")   # output tamper -> must recompute
    assert not Stage(tmp_path, "t", {"in": inp}, modules=("common",)).up_to_date()


@pytest.mark.skipif(not config.TRAINING_TABLE.exists(), reason="training table not exported")
def test_training_table_shape():
    import pandas as pd
    df = pd.read_csv(config.TRAINING_TABLE)
    assert len(df) == 31275
    assert df.band_gap_eV.between(0.5, 3.0).all()      # documents the no-metals limitation
