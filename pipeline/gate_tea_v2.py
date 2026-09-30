"""Gate for Phase E (TEA v2). Exit 1 on failure. Writes runs/<id>/gates/gate_tea_v2.json.

E-G1  NREL Q1-2023 utility cost structure reproduced from components (<= $2/kWdc) and every unit
      cross-check against values printed in OTHER figures agrees within 1 unit
E-G2  c-Si reference LCOE median lies inside Lazard LCOE+ 2024 unsubsidised utility range ($29-92/MWh)
E-G3  engine hand-calculation reproduced (independent closed form, r = 5%, one module replacement)
E-G4  internal consistency: perovskite with c-Si module $/m2 and c-Si durability breaks even at the
      c-Si efficiency (within 0.5 pp)
E-G5  efficiency dependence: LCOE strictly decreasing in efficiency for every module-cost scenario and the
      8% vs 18% ratio recorded (tests the legacy "thin-film decoupling" claim)
E-G6  provenance: every parameter classified; no formation energy anywhere in the TEA v2 code path;
      unsourced absorber prices shown to be immaterial (max effect < 5% of LCOE)
E-G7  Sobol sanity: sum S1 <= 1.05, every ST >= S1 - conf
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from .common import dump_json


def check(run_dir: Path) -> dict:
    res = {"checks": {}}

    def rec(name, ok, **info):
        res["checks"][name] = {"pass": bool(ok), **info}

    m = json.loads((run_dir / "metrics/tea_v2.json").read_text())
    e1 = m["e1_reproduction"]
    cc_ok = all(abs(v["derived"] - v["printed"]) <= 1.0 for v in e1["crosschecks"].values())
    rec("E-G1_nrel_reproduction", e1["msp"]["abs_err"] <= 2 and e1["mmp"]["abs_err"] <= 2 and cc_ok,
        msp=e1["msp"], mmp=e1["mmp"], crosschecks_ok=cc_ok)

    e2 = m["e2_csi_ref"]
    lo, hi = e2["lazard_utility_range_usd_mwh"]
    rec("E-G2_csi_benchmark", lo <= e2["msp_median_usd_mwh"] <= hi, **e2)

    from .tea_v2 import lcoe
    bare = dict(sbos_m2=0.0, fieldwork_m2=0.0, inverter_kwdc=0.0, ebos_kwdc=0.0, officework_kwdc=0.0, markup=1.0)
    v = float(lcoe(0.2, 50.0, **bare, cf_ac=0.2, ilr=1.0, fom_kwac=0.0, r=0.05, deg=0.0, plant_life=10, module_life=5))
    ann = sum(1.05 ** -t for t in range(1, 11))
    exact = (250 + 250 * 1.05 ** -5) / (1752 * ann)
    rec("E-G3_hand_calc", abs(v / exact - 1) < 1e-9, engine=v, closed_form=exact)

    c = m["e4_consistency"]
    rec("E-G4_csi_equivalence", abs(c["breakeven_eta_csi_equivalent"] - c["eta_csi"]) <= 0.005, **c)

    dec = m["e3_decoupling_test"]
    rec("E-G5_efficiency_dependence", all(v["monotonic_decreasing"] for v in dec.values()),
        ratio_lcoe_8_over_18={k: round(v["lcoe_8pct_over_18pct"], 3) for k, v in dec.items()},
        area_share_capex_10pct={k: round(v["area_share_of_capex_at_10pct"], 3) for k, v in dec.items()})

    from .tea_params import P
    classified = all(v["kind"] in {"direct", "figure", "derived", "scenario"} for v in P.values())
    root = Path(__file__).resolve().parent
    code = "".join((root / f).read_text(encoding="utf-8") for f in ("tea_v2.py", "s30_tea_v2.py", "tea_params.py"))
    ef_hits = re.findall(r"\bef_pred|\bpred_Ef\b|formation_energy", code)
    rec("E-G6_provenance", classified and not ef_hits and c["absorber_max_cost_effect_rel"] < 0.05,
        ef_references=ef_hits, absorber_max_effect=c["absorber_max_cost_effect_rel"],
        n_absorber_price_unsourced=m["e5_candidates"]["n_absorber_price_unsourced"])

    sob = pd.read_csv(run_dir / "tea_v2/sobol.csv")
    rec("E-G7_sobol_sanity", sob.S1.sum() <= 1.05 and (sob.ST >= sob.S1 - sob.S1_conf).all(),
        sum_S1=float(sob.S1.sum()))

    res["pass"] = all(v["pass"] for v in res["checks"].values())
    dump_json(run_dir / "gates/gate_tea_v2.json", res)
    return res


if __name__ == "__main__":
    out = check(Path(sys.argv[1]))
    for k, v in out["checks"].items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}  " + json.dumps({a: b for a, b in v.items() if a != 'pass'}, default=str)[:500])
    print("GATE:", "PASS" if out["pass"] else "FAIL")
    sys.exit(0 if out["pass"] else 1)
