"""TEA v2 tests: hand-computed LCOE, area/power scaling, SQ limit, source-quote integrity."""
import re
from pathlib import Path

import numpy as np
import pytest

from pipeline.sq import sq_limit, sq_table
from pipeline.tea_params import P, price
from pipeline.tea_v2 import capex_kwdc, lcoe

ROOT = Path(__file__).resolve().parents[1]
BARE = dict(sbos_m2=0.0, fieldwork_m2=0.0, inverter_kwdc=0.0, ebos_kwdc=0.0, officework_kwdc=0.0, markup=1.0)


def test_lcoe_hand_calculation_no_discount():
    # 5 m2/kWdc x $50/m2 = $250/kWdc; 0.2 x 8760 = 1752 kWh/yr; 10 yr, r = 0 -> 250/17520
    v = lcoe(0.2, 50.0, **BARE, cf_ac=0.2, ilr=1.0, fom_kwac=0.0, r=0.0, deg=0.0, plant_life=10)
    assert float(v) == pytest.approx(250 / 17520, rel=1e-12)


def test_lcoe_hand_calculation_discount_and_replacement():
    ann = sum(1.05 ** -t for t in range(1, 11))                       # 7.72173
    v0 = lcoe(0.2, 50.0, **BARE, cf_ac=0.2, ilr=1.0, fom_kwac=0.0, r=0.05, deg=0.0, plant_life=10)
    assert float(v0) == pytest.approx(250 / (1752 * ann), rel=1e-10)
    # module life 5: one replacement at t = 5 (none at t = 10 = end of plant life)
    v1 = lcoe(0.2, 50.0, **BARE, cf_ac=0.2, ilr=1.0, fom_kwac=0.0, r=0.05, deg=0.0, plant_life=10, module_life=5)
    assert float(v1) == pytest.approx((250 + 250 * 1.05 ** -5) / (1752 * ann), rel=1e-10)


def test_degradation_resets_on_replacement():
    e = lcoe(0.2, 0.0, **{**BARE, "inverter_kwdc": 1.0}, cf_ac=0.2, ilr=1.0, fom_kwac=0.0, r=0.0, deg=0.1,
             plant_life=4, module_life=2, return_parts=True)[1]["pv_energy"]
    assert float(e) == pytest.approx(1752 * (1 + 0.9 + 1 + 0.9), rel=1e-12)


def test_area_costs_scale_inverse_with_efficiency_power_costs_do_not():
    a = capex_kwdc(0.10, 50, 20, 40, 35, 146, 63, 1.0)
    b = capex_kwdc(0.20, 50, 20, 40, 35, 146, 63, 1.0)
    assert float(a - b) == pytest.approx((50 + 20 + 40) * (10 - 5), rel=1e-12)


def test_zero_efficiency_is_infinite_cost_not_crash():
    v = lcoe(np.array([0.0, 0.2]), 50.0, **BARE, cf_ac=0.2, ilr=1.0, fom_kwac=0.0, r=0.05, deg=0.0, plant_life=10)
    assert np.isinf(v[0]) and np.isfinite(v[1])


def test_sq_limit_matches_published_values():
    egs, eff = sq_table()
    i = int(eff.argmax())
    assert 0.335 <= eff[i] <= 0.339 and 1.30 <= egs[i] <= 1.37           # Ruhle 2016: 33.7% near 1.34 eV
    assert sq_limit(1.1) == pytest.approx(0.329, abs=0.004)
    assert sq_limit(0.2) == 0.0 and sq_limit(5.0) == 0.0


def test_nrel_cost_structure_reproduced_from_components():
    from pipeline.s30_tea_v2 import e1_reproduction
    r = e1_reproduction()
    assert r["msp"]["abs_err"] < 2.0 and r["mmp"]["abs_err"] < 2.0     # $/kWdc, figure labels are integers
    for k, v in r["crosschecks"].items():
        assert abs(v["derived"] - v["printed"]) <= 1.0, k


def test_every_parameter_is_classified():
    for k, v in P.items():
        assert v["kind"] in {"direct", "figure", "derived", "scenario"}, k
        if v["kind"] == "derived":
            assert v.get("formula"), k


SOURCE_FILES = {"NREL23": "nrel_87303_q1_2023_benchmarks.txt", "ATB24": "nrel_atb2024_utility_pv.md",
                "EPFL25": "pvmag_epfl_perovskite_tea_2025.md", "ZHANG22": "zhang_natcommun2022_cs2agbibr6.md",
                "LAZ24 p35 + ATB24": "nrel_atb2024_utility_pv.md"}


def _norm(s):
    return re.sub(r"\s+", " ", s)


@pytest.mark.parametrize("key", [k for k, v in P.items() if v.get("quote")])
def test_direct_quotes_verbatim(key):
    v = P[key]
    src = SOURCE_FILES.get(v["src"]) or SOURCE_FILES.get(v["src"].split()[0])
    path = ROOT / "data" / "sources" / src
    if not path.exists():
        pytest.skip(f"local source copy {src} not present (licence-restricted or not fetched)")
    assert _norm(v["quote"]) in _norm(path.read_text(encoding="utf-8")), key


def test_element_prices_cover_candidate_space():
    from pipeline import config
    for el in set(config.A_SITE + config.B_SITE + config.X_SITE):
        pr, note = price(el)
        assert pr > 0 and note
