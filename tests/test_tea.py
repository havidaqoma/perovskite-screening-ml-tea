import numpy as np

from src.tea_engine_optiona import run_tea, sq_limit


def test_run_tea_is_finite_and_explicit_lifetime_changes_lcoe():
    kwargs = dict(
        formula="Cs2AgBiBr6",
        pred_Eg=1.34,
        pred_Ef=0.05,
        mae_error=0.0,
        iterations=2000,
        future=False,
    )

    np.random.seed(7)
    short_lcoe, short_life, *_ = run_tea(**kwargs, lifetime_years=8.0)
    np.random.seed(7)
    long_lcoe, long_life, *_ = run_tea(**kwargs, lifetime_years=20.0)

    assert short_life == 8.0
    assert long_life == 20.0
    assert np.isfinite(short_lcoe).all()
    assert np.isfinite(long_lcoe).all()
    assert not np.isclose(np.median(short_lcoe), np.median(long_lcoe))


def test_sq_limit_is_higher_near_optimal_bandgap():
    assert sq_limit(1.34) > sq_limit(1.0)
