import numpy as np

from src.tea_engine_optiona import run_tea


def test_five_candidate_tea_smoke_has_no_nan_lcoe():
    candidates = [
        ("Cs2AgBiBr6", 1.34, 0.05),
        ("Cs2SnI6", 1.00, 0.10),
        ("K2AgBiCl6", 1.60, 0.02),
        ("Na2AgBiBr6", 1.25, 0.08),
        ("Rb2AgBiBr6", 1.40, 0.06),
    ]

    for seed, (formula, bandgap, formation_energy) in enumerate(candidates):
        np.random.seed(100 + seed)
        lcoe, _, _, _ = run_tea(
            formula,
            bandgap,
            formation_energy,
            mae_error=0.0,
            iterations=1000,
            future=True,
        )
        if not np.isfinite(lcoe).all():
            raise AssertionError(f"NaN/inf LCOE for {formula}")
