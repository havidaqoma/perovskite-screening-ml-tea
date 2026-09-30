"""Stage 30: TEA v2 (Phase E). Outputs under runs/<id>/tea_v2/ and metrics/tea_v2.json.

E1  engine reproduction: NREL23 cost structure reproduced from components; units cross-checked
E2  c-Si reference LCOE distribution (same site/finance draws for every comparison: common random numbers)
E3  efficiency sweep 5-30% for perovskite module cost $50/$85/$335 per m2  -> tests "thin-film decoupling"
E4  break-even efficiency vs c-Si for each module-cost x lifetime x degradation scenario
E5  candidate LCOE: calibrated gap posterior + P(semiconductor) x f_sq scenarios -> P(LCOE <= c-Si)
E6  Sobol (SALib Saltelli) global sensitivity of perovskite LCOE
All randomness from np.random.default_rng(SEED); draws are shared across scenarios.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition

from .common import Stage, dump_json
from .sq import sq_limit
from .tea_params import P, price
from .tea_v2 import capex_kwdc, lcoe

MODULES = ("s30_tea_v2", "tea_v2", "tea_params", "sq")
SEED = 42
N_MC = 4000
BASIS = "msp"


def tag(f_sq: float, mod: str, life: int, deg: float) -> str:
    """Single source of truth for scenario column names."""
    return f"f{f_sq:.2f}_{mod}_L{life}_d{deg:g}"


def site_finance_draws(rng, n):
    return {"cf_ac": rng.uniform(P["cf_ac"]["low"], P["cf_ac"]["high"], n),
            "fom_kwac": rng.uniform(P["fom_kwac"]["low"], P["fom_kwac"]["high"], n),
            "r": rng.uniform(P["discount_real"]["low"], P["discount_real"]["high"], n)}


def plant_costs(basis=BASIS):
    return {"sbos_m2": P["sbos_m2"][basis], "fieldwork_m2": P["fieldwork_m2"][basis],
            "inverter_kwdc": P["inverter_kwdc"][basis], "ebos_kwdc": P["ebos_kwdc"][basis],
            "officework_kwdc": P["officework_kwdc"][basis], "markup": P["markup"][basis]}


def csi_module_m2(basis=BASIS):
    return P["module_csi_kwdc"][basis] * P["eta_csi"]["value"]


def lcoe_csi(d, basis=BASIS):
    return lcoe(P["eta_csi"]["value"], csi_module_m2(basis), **plant_costs(basis), cf_ac=d["cf_ac"],
                ilr=P["ilr"]["value"], fom_kwac=d["fom_kwac"], r=d["r"], deg=P["deg_csi"]["value"],
                plant_life=P["plant_life"]["value"])


def lcoe_pvk(eta, d, module_m2, module_life, deg, burnin=0.0, absorber_m2=0.0, basis=BASIS,
             replace_labor_frac=None):
    rl = P["replace_labor_frac"]["value"] if replace_labor_frac is None else replace_labor_frac
    return lcoe(eta, module_m2, **plant_costs(basis), cf_ac=d["cf_ac"], ilr=P["ilr"]["value"],
                fom_kwac=d["fom_kwac"], r=d["r"], deg=deg, plant_life=P["plant_life"]["value"],
                module_life=module_life, burnin=burnin, absorber_m2=absorber_m2, replace_labor_frac=rl)


def absorber_cost_m2(formula: str) -> tuple[float, bool]:
    """Elemental raw-material cost of the absorber per m2 module; flag if any element price is unsourced."""
    comp = Composition(formula)
    grams = (P["absorber_thickness_nm"]["value"] * 1e-7 * P["absorber_density_gcm3"]["value"] * 1e4
             / P["absorber_utilization"]["value"])                                     # g per m2
    cost, unsourced = 0.0, False
    for el in comp.elements:
        pr, note = price(el.symbol)
        unsourced |= note.startswith("legacy")
        cost += comp.get_wt_fraction(el) * grams / 1000.0 * pr
    return float(cost), unsourced


def e1_reproduction() -> dict:
    out = {}
    for b in ("msp", "mmp"):
        cap = float(capex_kwdc(P["eta_csi"]["value"], csi_module_m2(b), **plant_costs(b)))
        out[b] = {"capex_kwdc": cap, "nrel_total": P["total_kwdc"][b], "abs_err": abs(cap - P["total_kwdc"][b])}
    # unit cross-checks against values printed in other figures (non-circular)
    out["crosschecks"] = {
        "sbos_pvess_m2": {"derived": 138 * 0.205, "printed": 28},
        "fieldwork_pvess_m2": {"derived": 295 * 0.205, "printed": 61},
        "inverter_msp_kwdc": {"derived": 47 / 1.34, "printed": 35},
        "inverter_mmp_kwdc": {"derived": 65 / 1.34, "printed": 48},
        "ebos_msp_kwdc": {"derived": 196 / 1.34, "printed": 146},
        "ebos_mmp_kwdc": {"derived": 236 / 1.34, "printed": 176},
    }
    return out


def run(run_dir, force: bool = False) -> dict:
    scored = run_dir / "stability/v2_screen_scored.csv"
    model = run_dir / "models/v2_final.joblib"
    inputs = {"scored": scored, "model": model}
    if (run_dir / "stability/umlip.csv").exists():
        inputs["umlip"] = run_dir / "stability/umlip.csv"
    st = Stage(run_dir, "s30_tea_v2", inputs, MODULES,
               params={"seed": SEED, "n_mc": N_MC, "basis": BASIS})
    if st.up_to_date() and not force:
        print("[s30] up to date, skipped")
        return {}
    rng = np.random.default_rng(SEED)
    d = site_finance_draws(rng, N_MC)
    ref = lcoe_csi(d)                                     # $/kWh, per draw
    ref_mmp = lcoe_csi(d, "mmp")
    m = {"e1_reproduction": e1_reproduction(),
         "e2_csi_ref": {"msp_median_usd_mwh": float(np.median(ref) * 1000), "msp_p10": float(np.percentile(ref, 10) * 1000),
                        "msp_p90": float(np.percentile(ref, 90) * 1000), "mmp_median_usd_mwh": float(np.median(ref_mmp) * 1000),
                        "lazard_utility_range_usd_mwh": [29, 92], "irena_2023_global_avg_usd_mwh": 44}}

    # ---- E3 efficiency sweep (perovskite, c-Si-like stability to isolate the efficiency effect) ----
    etas = np.round(np.arange(0.05, 0.3001, 0.01), 3)
    rows = []
    for mod_name in ("low", "med", "high"):
        mod = P["module_pvk_m2"][mod_name]
        for e in etas:
            l = lcoe_pvk(e, d, mod, module_life=35, deg=0.007)
            rows.append({"module_cost": mod_name, "module_m2": mod, "eta": e, "lcoe_median": float(np.median(l) * 1000),
                         "lcoe_p10": float(np.percentile(l, 10) * 1000), "lcoe_p90": float(np.percentile(l, 90) * 1000),
                         "capex_kwdc": float(capex_kwdc(e, mod, **plant_costs())),
                         "p_below_csi": float(np.mean(l <= ref))})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(st.path("tea_v2/efficiency_sweep.csv"), index=False, float_format="%.5f")
    dec = {}
    for mod_name, g in sweep.groupby("module_cost"):
        g = g.set_index("eta")
        dec[mod_name] = {"lcoe_8pct_over_18pct": float(g.loc[0.08, "lcoe_median"] / g.loc[0.18, "lcoe_median"]),
                         "monotonic_decreasing": bool((np.diff(g.lcoe_median.to_numpy()) < 0).all()),
                         "area_share_of_capex_at_10pct": float(1 - (P["inverter_kwdc"][BASIS] + P["ebos_kwdc"][BASIS]
                                                                    + P["officework_kwdc"][BASIS]) * P["markup"][BASIS]
                                                               / g.loc[0.10, "capex_kwdc"])}
    m["e3_decoupling_test"] = dec

    # ---- E4 break-even efficiency ----
    be = []
    fine = np.round(np.arange(0.04, 0.4001, 0.0025), 4)
    for mod_name in ("low", "med", "high"):
        mod = P["module_pvk_m2"][mod_name]
        for life in P["module_life_pvk"]["grid"]:
            for deg in P["deg_pvk"]["grid"]:
                for burn in P["burnin_pvk"]["grid"]:
                    med = [np.median(lcoe_pvk(e, d, mod, life, deg, burn) / ref) for e in fine]
                    hit = [e for e, v in zip(fine, med) if v <= 1.0]
                    be.append({"module_cost": mod_name, "module_m2": mod, "module_life": life, "deg": deg,
                               "burnin": burn, "breakeven_eta": hit[0] if hit else np.nan,
                               "breakeven_reachable_by_sq": bool(hit and hit[0] <= float(sq_limit(1.34)))})
    be = pd.DataFrame(be)
    be.to_csv(st.path("tea_v2/breakeven_efficiency.csv"), index=False, float_format="%.4f")
    # consistency: perovskite with c-Si module $/m2 and c-Si durability must break even at the c-Si efficiency
    csi_m2 = csi_module_m2()
    med_c = [np.median(lcoe_pvk(e, d, csi_m2, 35, P["deg_csi"]["value"]) / ref) for e in fine]
    be_csi = [e for e, v in zip(fine, med_c) if v <= 1.0][0]
    # sensitivities of the base case (1.34 eV, f_sq 0.82, $50/m2, 35 y): absorber cost and replacement labour
    e134 = 0.82 * float(sq_limit(1.34))
    l0 = np.median(lcoe_pvk(e134, d, 50.0, 35, 0.007))
    l_abs = np.median(lcoe_pvk(e134, d, 50.0, 35, 0.007, absorber_m2=11.5))
    l10 = np.median(lcoe_pvk(e134, d, 50.0, 10, 0.02))
    l10_lab = np.median(lcoe_pvk(e134, d, 50.0, 10, 0.02, replace_labor_frac=P["replace_labor_frac"]["alt"]))
    m["e4_consistency"] = {"breakeven_eta_csi_equivalent": float(be_csi), "eta_csi": P["eta_csi"]["value"],
                           "base_lcoe_usd_mwh": float(l0 * 1000),
                           "absorber_max_cost_effect_rel": float(l_abs / l0 - 1),
                           "replace_labor_effect_rel_L10": float(l10_lab / l10 - 1)}
    m["e4_breakeven"] = {"n_scenarios": int(len(be)), "n_unreachable": int(be.breakeven_eta.isna().sum()),
                         "base_low_35y_deg0.7": float(be.query("module_cost=='low' and module_life==35 and deg==0.007 and burnin==0").breakeven_eta.iloc[0]),
                         "base_med_35y_deg0.7": float(be.query("module_cost=='med' and module_life==35 and deg==0.007 and burnin==0").breakeven_eta.iloc[0]),
                         "med_10y_deg2": float(be.query("module_cost=='med' and module_life==10 and deg==0.02 and burnin==0").breakeven_eta.iloc[0])}

    # ---- E5 candidate-level LCOE with calibrated gap posterior ----
    df = pd.read_csv(scored)
    mdl = joblib.load(model)
    signed = np.asarray(mdl["gap_cal_signed_scores"], float)
    n_c = 1000
    z = rng.choice(signed, size=(len(df), n_c), replace=True)                 # shared residual draws
    u_semi = rng.uniform(size=(len(df), n_c))
    idx = rng.integers(0, N_MC, size=n_c)                                     # site/finance draw per sample
    dd = {k: v[idx] for k, v in d.items()}
    ref_c = ref[idx]
    abs_cost = [absorber_cost_m2(f) for f in df.formula]
    df["absorber_cost_m2"] = [a for a, _ in abs_cost]
    df["absorber_price_unsourced"] = [u for _, u in abs_cost]
    scen = []
    # f_sq levels (review 2026-09-29): demonstrated cell (0.21), intermediate (0.50),
    # c-Si MODULE parity (eta_csi / SQ(1.12 eV), derived) and an idealised lab-CELL ceiling (0.82, unreachable
    # for a utility module: no cell-to-module loss). Durability: ideal (35 y, 0.7 %/yr, no replacement labour)
    # vs realistic (15 y, 2 %/yr, 50 % of installation labour repeated at each module swap).
    f_csi_mod = round(P["eta_csi"]["value"] / float(sq_limit(1.12)), 2)
    f_grid = sorted(set(P["f_sq"]["grid"]) | {f_csi_mod})
    durab = {"ideal": (35, 0.007, 0.0), "realistic": (15, 0.02, P["replace_labor_frac"]["alt"])}
    gap = df.gap_pred_eV.to_numpy()[:, None] + z * df.gap_sigma_eV.to_numpy()[:, None]
    semi = u_semi < df.p_semi.to_numpy()[:, None]
    sq_gap = sq_limit(gap)
    for f_sq in f_grid:
        eta = np.where(semi, f_sq * sq_gap, 0.0)
        for mod_name in ("low", "med"):
            for dname, (life, deg, lab) in durab.items():
                tg = tag(f_sq, mod_name, life, deg)
                l = lcoe_pvk(eta, dd, P["module_pvk_m2"][mod_name], life, deg,
                             absorber_m2=df.absorber_cost_m2.to_numpy()[:, None], replace_labor_frac=lab)
                df[f"p_lcoe_le_csi__{tg}"] = np.mean(l <= ref_c[None, :], axis=1)
                df[f"lcoe_median__{tg}"] = np.median(l, axis=1) * 1000
                scen.append(tg)
    REAL = f"p_lcoe_le_csi__{tag(f_csi_mod, 'med', 15, 0.02)}"          # headline: realistic
    BEST = f"p_lcoe_le_csi__{tag(f_csi_mod, 'low', 35, 0.007)}"         # optimistic but physical
    CEIL = f"p_lcoe_le_csi__{tag(0.82, 'low', 35, 0.007)}"              # idealised ceiling
    DEMO = f"p_lcoe_le_csi__{tag(0.21, 'low', 35, 0.007)}"
    for name, col in (("realistic", REAL), ("optimistic", BEST), ("ceiling", CEIL), ("demonstrated", DEMO)):
        df[f"p_competitive_x_stable__{name}"] = df[col] * df.p_stable50 * df.plausible
    # uMLIP veto (Stage 4): where MACE was run and puts the compound > 50 meV above the other phases,
    # the ML stability is overruled (MACE agreed with MP on 10/10 known controls). Reported alongside, not hidden.
    um = run_dir / "stability/umlip.csv"
    df["umlip_ehull_meV"] = np.nan
    if um.exists():
        u = pd.read_csv(um).query("status == 'ok'").set_index("formula")["umlip_ehull_vs_rest_meV"]
        df["umlip_ehull_meV"] = df.formula.map(u)
    df["umlip_veto"] = df.umlip_ehull_meV > 50
    for name in ("realistic", "optimistic", "ceiling", "demonstrated"):
        df[f"p_competitive_x_stable_umlip__{name}"] = df[f"p_competitive_x_stable__{name}"].where(~df.umlip_veto, 0.0)
    keep = ["formula", "status", "plausible", "p_semi", "gap_pred_eV", "gap_lo90_eV", "gap_hi90_eV", "p_stable50",
            "umlip_ehull_meV", "umlip_veto", "p_viable_plausible", "absorber_cost_m2", "absorber_price_unsourced"] + \
           [c for c in df.columns if c.startswith(("p_lcoe_le_csi__", "lcoe_median__", "p_competitive_x_stable"))]
    out = df[keep].sort_values(["p_competitive_x_stable_umlip__optimistic", "p_competitive_x_stable__ceiling"],
                               ascending=False)
    out.to_csv(st.path("tea_v2/candidate_lcoe.csv"), index=False, float_format="%.5f")
    from scipy.stats import spearmanr
    m["e5_candidates"] = {
        "n": int(len(out)),
        "f_sq_csi_module_parity": f_csi_mod,
        "scenario_definitions": {"realistic": REAL, "optimistic": BEST, "ceiling": CEIL, "demonstrated": DEMO},
        "expected_n_competitive_by_scenario": {t: float(out[f"p_lcoe_le_csi__{t}"].sum()) for t in scen},
        "expected_n_competitive_and_stable": {k: float(out[f"p_competitive_x_stable__{k}"].sum())
                                              for k in ("realistic", "optimistic", "ceiling", "demonstrated")},
        "expected_n_competitive_and_stable_umlip": {k: float(out[f"p_competitive_x_stable_umlip__{k}"].sum())
                                                    for k in ("realistic", "optimistic", "ceiling", "demonstrated")},
        "n_umlip_checked": int(out.umlip_ehull_meV.notna().sum()), "n_umlip_veto": int(out.umlip_veto.sum()),
        "n_candidates_p_ge_0.5": {k: int((out[f"p_competitive_x_stable__{k}"] >= 0.5).sum())
                                  for k in ("realistic", "optimistic", "ceiling", "demonstrated")},
        "max_p_competitive": {k: float(out[col].max()) for k, col in
                              (("realistic", REAL), ("optimistic", BEST), ("ceiling", CEIL), ("demonstrated", DEMO))},
        "rank_spearman_optimistic_vs_ceiling": float(spearmanr(out[BEST], out[CEIL])[0]),
        "absorber_cost_m2_median": float(out.absorber_cost_m2.median()),
        "absorber_cost_m2_max": float(out.absorber_cost_m2.max()),
        "n_absorber_price_unsourced": int(out.absorber_price_unsourced.sum()),
        "top10_optimistic": out.head(10)[["formula", "status", BEST, CEIL, "p_stable50", "umlip_ehull_meV",
                                          "p_competitive_x_stable_umlip__optimistic"]].round(3).to_dict("records"),
    }

    # ---- E6 Sobol sensitivity (perovskite LCOE, representative band gap 1.34 eV) ----
    from SALib.analyze import sobol
    from SALib.sample import sobol as sobol_sample
    prob = {"num_vars": 9,
            "names": ["f_sq", "module_m2", "cf_ac", "fom_kwac", "r", "deg", "module_life", "sbos_scale", "field_scale"],
            "bounds": [[0.21, 0.82], [50, 335], [0.15, 0.30], [11, 22], [0.04, 0.08], [0.007, 0.05], [5, 35],
                       [0.8, 1.2], [0.8, 1.2]]}
    X = sobol_sample.sample(prob, 1024, calc_second_order=False, seed=SEED)
    Yv = np.empty(len(X))
    base = plant_costs()
    sq134 = float(sq_limit(1.34))
    for life in range(5, 36):                     # module_life is integer-valued; evaluate grouped
        sel = np.round(X[:, 6]).astype(int) == life
        if not sel.any():
            continue
        Xs = X[sel]
        Yv[sel] = lcoe(Xs[:, 0] * sq134, Xs[:, 1], base["sbos_m2"] * Xs[:, 7], base["fieldwork_m2"] * Xs[:, 8],
                       base["inverter_kwdc"], base["ebos_kwdc"], base["officework_kwdc"], base["markup"],
                       cf_ac=Xs[:, 2], ilr=P["ilr"]["value"], fom_kwac=Xs[:, 3], r=Xs[:, 4], deg=Xs[:, 5],
                       plant_life=P["plant_life"]["value"], module_life=life)
    Si = sobol.analyze(prob, Yv, calc_second_order=False, seed=SEED)
    sob = pd.DataFrame({"param": prob["names"], "S1": Si["S1"], "S1_conf": Si["S1_conf"], "ST": Si["ST"],
                        "ST_conf": Si["ST_conf"]}).sort_values("ST", ascending=False)
    sob.to_csv(st.path("tea_v2/sobol.csv"), index=False, float_format="%.4f")
    m["e6_sobol"] = {"n_eval": int(len(X)), "sum_S1": float(Si["S1"].sum()),
                     "ranking_ST": sob[["param", "ST"]].round(3).to_dict("records")}

    dump_json(st.path("metrics/tea_v2.json"), m)
    st.metrics = {k: v for k, v in m.items() if k in ("e2_csi_ref", "e3_decoupling_test", "e4_breakeven")}
    st.finish()
    print(f"[s30] {st.metrics}")
    return m
