"""Stage 50: checks requested by the pre-submission review (2026-10-01). Read-only on earlier stages.

R1  Walterbos et al. (2026) HSE06 check. Every non-vacancy Cs2BB'X6 composition of their data set is passed
    through the SAME final model as s14 (classifier, gap regressor, difficulty model, conformal quantile).
    Our labels are Materials Project GGA/GGA+U gaps, theirs are spin-polarised HSE06 gaps (no SOC), so the
    test is about trends and classification, not absolute gaps. Compositions whose reduced formula is in
    our training table are reported separately (their predictions are in-sample).
R2  Stability-threshold sensitivity. Counts and expected values at every hull threshold s20 evaluated
    (0, 20, 35, 50, 100 meV/atom), recomputed from the s20/s22/s30 tables; known compounds use their MP hull
    distance and the MACE veto uses the same threshold.
R3  O&M and inverter-replacement sensitivity of the break-even efficiency, with the SAME site/finance draws
    as s30 (seed and draw order reproduced). Perovskite fixed O&M is scaled while silicon O&M is unchanged;
    inverter replacement at year 15 is added to both plants at the initial inverter price.
R4  Oxidation-state audit: every B-site oxidation state of +5 or higher among the v2 candidates, split by
    whether the anion set contains a reducing anion (S, Se, Te, I). The flag rule itself (s22) is unchanged.
R5  Energy-scheme consistency. The training formation energies (s10) are the MP summary defaults, which in
    recent database versions mix GGA, GGA+U and r2SCAN results; the hulls of s20 use GGA/GGA+U entries only.
    For every formula of the training table that occurs in an s20 chemical system, the formation energy of its
    lowest-energy GGA/GGA+U entry (corrected energies, elemental references of the same system) is compared
    with the training value; the known A2BB'X6 controls of s20 are reported separately.
R6  Module-price floor (review 2026-10-09). Break-even efficiency at module prices from zero up to the benchmark
    silicon module's own areal price, at silicon-grade durability (35 y, 0.7 %/yr, no burn-in), on the SAME draws
    as s30; it must reproduce the s30 break-even at $50 m-2 and at the silicon areal price, or the stage stops.
R7  Band-gap offset. The candidate Monte Carlo of s30 (same seed, draw order and residual draws) repeated with
    every gap draw shifted up by the median HSE06-minus-prediction offsets of R1, plus the PV-window probability
    and the expected number of viable absorbers at the same shifts; zero shift must reproduce s30 and s22 exactly.
R8  Stability-gate recall on the known candidates, the signed hull-distance error of the stable controls, the
    chain from the ML-probability count to the stability-gated pool, the MACE coverage of that pool, rank
    correlations of the scenario costs, and the viable-absorber count after the MACE veto.
"""
from __future__ import annotations

import gzip
import json

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import config
from .common import Stage, dump_json
from .features import featurize_many
from .s14_screen_v2 import PV_WINDOW_EV, gap_posterior_prob
from .s22_shortlist import OX_IMPLAUSIBLE
from .s30_tea_v2 import (N_MC, SEED, absorber_cost_m2, csi_module_m2, lcoe_csi, lcoe_pvk, plant_costs,
                         site_finance_draws)
from .sq import sq_limit
from .tea_params import P

MODULES = ("s50_review_checks",)
WALTERBOS = config.DATA_DIR / "external/walterbos2026/hdp_hse06_subset.csv"
THRESHOLDS = (0, 20, 35, 50, 100)
OM_SCALE = (1.0, 1.25, 1.5, 2.0)
INV_REPLACE_YEAR = 15
MODULE_PRICES = (0.0, 10.0, 25.0, 50.0)             # $/m2; the silicon areal price is added at run time
REDUCING = ("S", "Se", "Te", "I")
ENTRY_DIR = config.DATA_DIR / "mp_cache" / "entries"     # written by s20 (GGA/GGA+U entries per chemical system)
HALOGENS = {"F", "Cl", "Br", "I"}
NON_HALIDE_ANIONS = {"H", "B", "C", "N", "O", "Si", "P", "S", "Se", "Te"}   # any of these -> not a pure halide
CAND_ELEMENTS = set(config.A_SITE) | set(config.B_SITE) | HALOGENS          # elements of the screened space


def _key(f: str) -> str:
    return Composition(f).reduced_formula


def r1_walterbos(run_dir, st) -> dict:
    w = pd.read_csv(WALTERBOS)
    n_all = len(w)
    w = w[~w.comp_name_full.str.contains("Vac")].reset_index(drop=True)
    m = joblib.load(run_dir / "models/v2_final.joblib")
    X, _ = featurize_many(w.comp_name_full.tolist(), n_jobs=8)
    ok = np.isfinite(X).all(axis=1)
    # identical to s14_screen_v2 lines 63-71
    w["p_semi"] = np.nan
    w["gap_pred_eV"] = np.nan
    w["gap_sigma_eV"] = np.nan
    w.loc[ok, "p_semi"] = m["cls"].predict_proba(X[ok])[:, 1]
    w.loc[ok, "gap_pred_eV"] = m["spec"].predict(X[ok])
    w.loc[ok, "gap_sigma_eV"] = np.clip(m["diff"].predict(X[ok]), m["sigma_floor"], None)
    w["gap_lo90_eV"] = w.gap_pred_eV - m["q_gap"] * w.gap_sigma_eV
    w["gap_hi90_eV"] = w.gap_pred_eV + m["q_gap"] * w.gap_sigma_eV
    w["p_gap_pv"] = np.nan
    w.loc[ok, "p_gap_pv"] = gap_posterior_prob(w.gap_pred_eV[ok].to_numpy(), w.gap_sigma_eV[ok].to_numpy(),
                                               m["gap_cal_signed_scores"], *PV_WINDOW_EV)
    train = set(pd.read_csv(run_dir / "data/train_v2.csv", usecols=["formula"], keep_default_na=False).formula.map(_key))
    cands = pd.read_csv(run_dir / "stability/v2_screen_scored.csv", usecols=["formula", "status"])
    cands["key"] = cands.formula.map(_key)
    w["key"] = w.comp_name_full.map(_key)
    w["in_training"] = w.key.isin(train)
    w["in_v2_candidates"] = w.key.isin(set(cands.key))
    w["hse_nonmetal"] = w.cond_type.isin(["semiconductor", "insulator"])
    w["hse_in_pv_window"] = w.bandgap.between(*PV_WINDOW_EV)
    w.to_csv(st.path("review/walterbos_check.csv"), index=False, float_format="%.5f")

    def block(d: pd.DataFrame) -> dict:
        s = d[d.hse_nonmetal & d.p_semi.notna()]
        cover = (s.bandgap >= s.gap_lo90_eV) & (s.bandgap <= s.gap_hi90_eV)
        off = (s.bandgap - s.gap_pred_eV).median()
        return {"n": int(len(d)), "n_featurized": int(d.p_semi.notna().sum()),
                "n_metal_hse": int((~d.hse_nonmetal).sum()),
                "auc_semi_vs_hse_nonmetal": float(roc_auc_score(d.hse_nonmetal[d.p_semi.notna()],
                                                                d.p_semi.dropna())),
                "n_nonmetal": int(len(s)),
                "spearman_gap_pred_vs_hse": float(spearmanr(s.gap_pred_eV, s.bandgap)[0]),
                "median_hse_minus_pred_eV": float((s.bandgap - s.gap_pred_eV).median()),
                "mae_vs_hse_eV": float((s.bandgap - s.gap_pred_eV).abs().mean()),
                "coverage_hse_by_interval": float(cover.mean()),
                "frac_hse_above_interval": float((s.bandgap > s.gap_hi90_eV).mean()),
                "frac_hse_below_interval": float((s.bandgap < s.gap_lo90_eV).mean()),
                "coverage_after_median_shift": float(((s.bandgap - off >= s.gap_lo90_eV)
                                                      & (s.bandgap - off <= s.gap_hi90_eV)).mean()),
                "residual_mean_eV": float((s.bandgap - s.gap_pred_eV).mean()),
                "residual_sd_eV": float((s.bandgap - s.gap_pred_eV).std()),
                "n_metal_called_semi_p05": int((d[~d.hse_nonmetal].p_semi >= 0.5).sum()),
                "n_nonmetal_rejected_p05": int((s.p_semi < 0.5).sum()),
                "by_anion": {x: {"n": int(len(g)), "coverage": float(((g.bandgap >= g.gap_lo90_eV)
                                                                      & (g.bandgap <= g.gap_hi90_eV)).mean()),
                                 "median_hse_minus_pred_eV": float((g.bandgap - g.gap_pred_eV).median())}
                             for x, g in s.groupby("element.X")},
                "n_in_window_nonmetal": int(s.hse_in_pv_window.sum()),
                "n_hse_in_pv_window": int(s.hse_in_pv_window.sum()),
                "auc_p_gap_pv_vs_hse_window": float(roc_auc_score(s.hse_in_pv_window, s.p_gap_pv)),
                "frac_spin_forbidden_in_window": float(s[s.hse_in_pv_window].spin_forbidden.mean())}

    out = {"n_rows_source": int(n_all), "n_vacancy_dropped": int(n_all - len(w)),
           "all": block(w), "not_in_training": block(w[~w.in_training]), "in_training": block(w[w.in_training]),
           "n_in_training": int(w.in_training.sum()), "n_in_v2_candidates": int(w.in_v2_candidates.sum())}
    ov = w[w.in_v2_candidates & w.hse_nonmetal]
    out["v2_overlap"] = {"n": int(w.in_v2_candidates.sum()), "n_nonmetal": int(len(ov)),
                         "coverage_hse_by_interval": float(((ov.bandgap >= ov.gap_lo90_eV)
                                                            & (ov.bandgap <= ov.gap_hi90_eV)).mean()),
                         "median_hse_minus_pred_eV": float((ov.bandgap - ov.gap_pred_eV).median())}
    return out


def r2_thresholds(run_dir) -> dict:
    stab = pd.read_csv(run_dir / "stability/v2_screen_scored.csv")
    lc = pd.read_csv(run_dir / "tea_v2/candidate_lcoe.csv")
    defs = json.loads((run_dir / "metrics/tea_v2.json").read_text())["e5_candidates"]["scenario_definitions"]
    df = stab.merge(lc[["formula", "umlip_ehull_meV"] + list(defs.values())], on="formula", how="left")
    known = df.status.eq("known_mp")
    novel_ok = ~known & df.plausible
    out = {}
    for t in THRESHOLDS:
        p = df[f"p_ehull_le_{t}"].where(~known, (df.mp_ehull_meV <= t).astype(float))
        p = p.where(df.status.ne("unknown"), 0.0)
        p_pl = p.where(df.plausible, 0.0)
        veto = df.umlip_ehull_meV > t
        row = {"expected_viable_novel_plausible": float((df.p_semi * df.p_gap_pv * p_pl)[novel_ok].sum())}
        for name, col in defs.items():
            row[f"expected_competitive_stable_umlip__{name}"] = float((df[col] * p_pl).where(~veto, 0.0).sum())
        out[str(t)] = row
    # fail-closed: at the paper's 50 meV gate this recomputation must reproduce s22 and s30 exactly
    sl = json.loads((run_dir / "metrics/shortlist_v2.json").read_text())
    e5 = json.loads((run_dir / "metrics/tea_v2.json").read_text())["e5_candidates"]
    chk = [(out["50"]["expected_viable_novel_plausible"], sl["expected_viable_novel_plausible"])] + \
          [(out["50"][f"expected_competitive_stable_umlip__{k}"], e5["expected_n_competitive_and_stable_umlip"][k])
           for k in defs]
    if any(abs(a - b) > 1e-3 for a, b in chk):
        raise SystemExit(f"FAIL-CLOSED: R2 at 50 meV does not reproduce s22/s30: {chk}")
    return out


def r3_om(st) -> dict:
    rng = np.random.default_rng(SEED)
    d = site_finance_draws(rng, N_MC)          # first draws of s30: identical site/finance samples
    fine = np.round(np.arange(0.04, 0.4001, 0.0025), 4)
    inv = P["inverter_kwdc"]["msp"] * P["markup"]["msp"]
    rows = []
    for inv_on in (False, True):
        for k in OM_SCALE:
            dp = dict(d, fom_kwac=d["fom_kwac"] * k)
            ref, ref_parts = lcoe_csi(d), None
            if inv_on:
                from .s30_tea_v2 import csi_module_m2, plant_costs
                from .tea_v2 import lcoe
                _, rp = lcoe(P["eta_csi"]["value"], csi_module_m2(), **plant_costs(), cf_ac=d["cf_ac"],
                             ilr=P["ilr"]["value"], fom_kwac=d["fom_kwac"], r=d["r"], deg=P["deg_csi"]["value"],
                             plant_life=P["plant_life"]["value"], return_parts=True)
                add = inv * (1 + d["r"]) ** (-INV_REPLACE_YEAR)
                ref = (rp["capex"] + rp["pv_om"] + rp["pv_repl"] + add) / rp["pv_energy"]
            for mod_name in ("low", "med"):
                mod = P["module_pvk_m2"][mod_name]
                hit = np.nan
                for e in fine:
                    if inv_on:
                        _, pp = lcoe(e, mod, **plant_costs(), cf_ac=dp["cf_ac"], ilr=P["ilr"]["value"],
                                     fom_kwac=dp["fom_kwac"], r=dp["r"], deg=0.007,
                                     plant_life=P["plant_life"]["value"], module_life=35,
                                     replace_labor_frac=P["replace_labor_frac"]["value"], return_parts=True)
                        lp = (pp["capex"] + pp["pv_om"] + pp["pv_repl"] + add) / pp["pv_energy"]
                    else:
                        lp = lcoe_pvk(e, dp, mod, 35, 0.007)
                    if np.median(lp / ref) <= 1.0:
                        hit = float(e)
                        break
                rows.append({"inverter_replacement": inv_on, "om_scale_pvk": k, "module_cost": mod_name,
                             "module_m2": mod, "breakeven_eta": hit})
    be = pd.DataFrame(rows)
    be.to_csv(st.path("review/om_inverter_breakeven.csv"), index=False, float_format="%.4f")
    g = be.set_index(["inverter_replacement", "om_scale_pvk", "module_cost"]).breakeven_eta
    e4 = json.loads((st.run_dir / "metrics/tea_v2.json").read_text())["e4_breakeven"]
    if (g[(False, 1.0, "low")], g[(False, 1.0, "med")]) != (e4["base_low_35y_deg0.7"], e4["base_med_35y_deg0.7"]):
        raise SystemExit("FAIL-CLOSED: R3 base case does not reproduce the s30 break-even efficiencies")
    # size of the inverter-replacement term itself, so a null effect on break-even is not a silent no-op
    from .s30_tea_v2 import csi_module_m2, plant_costs
    from .tea_v2 import lcoe
    _, rp = lcoe(P["eta_csi"]["value"], csi_module_m2(), **plant_costs(), cf_ac=d["cf_ac"], ilr=P["ilr"]["value"],
                 fom_kwac=d["fom_kwac"], r=d["r"], deg=P["deg_csi"]["value"], plant_life=P["plant_life"]["value"],
                 return_parts=True)
    add = inv * (1 + d["r"]) ** (-INV_REPLACE_YEAR)
    base = (rp["capex"] + rp["pv_om"] + rp["pv_repl"]) / rp["pv_energy"]
    csi_effect = float(np.median((base + add / rp["pv_energy"]) / base) - 1)
    return {"inverter_replace_year": INV_REPLACE_YEAR, "inverter_replace_usd_kwdc": inv,
            "inverter_effect_on_csi_lcoe_rel": csi_effect, "eta_grid_step": 0.0025,
            "om_scale": list(OM_SCALE), "breakeven": {f"inv{int(i)}_om{k:g}_{mc}": v for (i, k, mc), v in g.items()}}


def r4_oxidation(run_dir, st) -> dict:
    s = pd.read_csv(run_dir / "stability/v2_screen_scored.csv")
    rows = []
    for r in s.itertuples():
        parts = [p for p in str(r.ox_states).split(";") if p]
        an = [p for p in parts if "-" in p]
        anions = {p.rstrip("+-0123456789") for p in an}
        a_el = r.a_site
        for p in parts:
            if "+" not in p:
                continue
            el, v = p.split("+")
            if el == a_el or int(v) < 5:
                continue
            rows.append({"ion": f"{el}+{v}", "reducing_anion": bool(anions & set(REDUCING)),
                         "flagged": p in OX_IMPLAUSIBLE, "status": r.status})
    t = pd.DataFrame(rows)
    tab = (t.groupby(["ion", "reducing_anion"]).size().unstack(fill_value=0)
           .rename(columns={False: "n_O_F_Cl_Br_only", True: "n_with_S_Se_Te_I"}).reset_index())
    tab["flagged"] = tab.ion.isin(OX_IMPLAUSIBLE)
    tab = tab.sort_values(["flagged", "n_with_S_Se_Te_I"], ascending=False)
    tab.to_csv(st.path("review/high_valent_b_site.csv"), index=False)

    # sensitivity: a stricter rule that also removes every B-site state >= +5 next to S, Se, Te or I
    def strict(ox: str) -> bool:
        p = [x for x in str(ox).split(";") if x]
        an = {x.rstrip("+-0123456789") for x in p if "-" in x}
        hi = [x for x in p[1:] if "+" in x and int(x.split("+")[1]) >= 5]
        return bool(hi) and bool(an & set(REDUCING))
    s["strict_flag"] = s.ox_states.map(strict) & s.plausible
    lc = pd.read_csv(run_dir / "tea_v2/candidate_lcoe.csv")
    d = s.merge(lc[["formula"] + [c for c in lc.columns if c.startswith("p_competitive_x_stable_umlip__")]],
                on="formula")
    nov = d.status.ne("known_mp")
    short = pd.read_csv(run_dir / "stability/shortlist_for_umlip.csv").formula
    um = lc[["formula", "umlip_ehull_meV"]].dropna()
    front = pd.read_csv(run_dir / "pareto/pareto.csv").query("front == 1").formula
    red = s.ox_states.fillna("").map(lambda o: bool({x.rstrip("+-0123456789") for x in o.split(";") if "-" in x}
                                                     & set(REDUCING)))
    out = {"rule": list(OX_IMPLAUSIBLE), "n_ions": int(len(tab)),
           "n_flagged": int(s.flag_ox_implausible.sum()),
           "n_flagged_with_reducing_anion": int((s.flag_ox_implausible & red).sum()),
           "n_high_valent_reducing_unflagged": int(d.strict_flag.sum()),
           "n_known_among_them": int((d.strict_flag & ~nov).sum()),
           "n_in_umlip_shortlist": int(short.isin(d.formula[d.strict_flag]).sum()),
           "n_umlip_shortlist": int(len(short)),
           "n_in_umlip_shortlist_within_50meV": int(um[um.formula.isin(d.formula[d.strict_flag])
                                                       & (um.umlip_ehull_meV <= 50)].shape[0]),
           "n_in_umlip_shortlist_with_mace": int(um.formula.isin(d.formula[d.strict_flag]).sum()),
           "n_on_pareto_front": int(front.isin(d.formula[d.strict_flag]).sum()),
           "expected_viable_novel": {"paper_rule": float(d.p_viable_plausible[nov].sum()),
                                     "strict_rule": float(d.p_viable_plausible[nov & ~d.strict_flag].sum())}}
    for k in ("optimistic", "ceiling", "realistic", "demonstrated"):
        c = f"p_competitive_x_stable_umlip__{k}"
        out[f"expected_competitive_stable_umlip__{k}"] = {"paper_rule": float(d[c].sum()),
                                                          "strict_rule": float(d[c][~d.strict_flag].sum())}
    return out


def r5_energy_scheme(run_dir, st) -> dict:
    files = sorted(ENTRY_DIR.glob("*.json.gz"))
    if not files:
        raise SystemExit(f"FAIL-CLOSED: no s20 entry cache in {ENTRY_DIR}; run s20 first")
    best: dict[str, float] = {}                       # reduced formula -> GGA/GGA+U formation energy (eV/atom)
    for fp in files:
        ents = json.loads(gzip.decompress(fp.read_bytes()))["entries"]
        comp = [Composition(e["composition"]) for e in ents]
        epa = [(e["energy"] + e.get("correction", 0.0)) / c.num_atoms for e, c in zip(ents, comp)]
        mu = {}
        for c, x in zip(comp, epa):
            if len(c.elements) == 1:
                el = c.elements[0].symbol
                mu[el] = min(mu.get(el, np.inf), x)
        low: dict[str, float] = {}
        for c, x in zip(comp, epa):
            rf = c.reduced_formula
            low[rf] = min(low.get(rf, np.inf), x)
        for rf, x in low.items():
            c = Composition(rf)
            if all(e.symbol in mu for e in c.elements):
                best[rf] = x - sum(c.get_atomic_fraction(e) * mu[e.symbol] for e in c.elements)
    tr = pd.read_csv(run_dir / "data/train_v2.csv", usecols=["formula", "formation_energy_per_atom"])
    d = tr[tr.formula.isin(best)].copy()
    d["ef_gga"] = d.formula.map(best)
    d["abs_diff_meV"] = 1000.0 * (d.formation_energy_per_atom - d.ef_gga).abs()

    def kind(f: str) -> str:
        el = {e.symbol for e in Composition(f).elements}
        if not el & HALOGENS or el & NON_HALIDE_ANIONS:
            return "other"
        return "candidate_element_halide" if el <= CAND_ELEMENTS else "halide_only"
    d["kind"] = d.formula.map(kind)
    d.to_csv(st.path("review/energy_scheme.csv"), index=False, float_format="%.6f")

    def summ(x: pd.Series) -> dict:
        return {"n": int(len(x)), "median_meV": float(x.median()), "n_gt_10meV": int((x > 10).sum()),
                "n_gt_50meV": int((x > 50).sum()), "frac_gt_10meV": float((x > 10).mean()),
                "frac_gt_50meV": float((x > 50).mean()), "max_meV": float(x.max())}
    ctrl = pd.read_csv(run_dir / "stability/control_known_a2bbx6.csv").merge(tr, on="formula")
    ctrl = ctrl.dropna(subset=["true_ef_pd"])
    cdiff = 1000.0 * (ctrl.formation_energy_per_atom - ctrl.true_ef_pd).abs()
    hal = d.kind != "other"                           # every pure halide (candidate-element ones included)
    cand = d[d.kind == "candidate_element_halide"].sort_values("abs_diff_meV", ascending=False)
    big = d.abs_diff_meV > 50
    return {"n_chemsys": len(files), "all": summ(d.abs_diff_meV),
            "pure_halide": summ(d.abs_diff_meV[hal]),
            "candidate_element_halide": summ(cand.abs_diff_meV),
            "other": summ(d.abs_diff_meV[~hal]),
            "n_gt50_pure_halide": int((big & hal).sum()), "n_gt50_other": int((big & ~hal).sum()),
            "candidate_element_halide_top": [{"formula": r.formula, "abs_diff_meV": float(r.abs_diff_meV)}
                                             for r in cand.head(5).itertuples()],
            "controls": summ(cdiff),
            "kind_rule": "pure halide = contains F/Cl/Br/I and none of " + ",".join(sorted(NON_HALIDE_ANIONS))
                         + "; candidate-element halide = pure halide whose elements are all in A_SITE, B_SITE "
                           "or the halogens"}


def _breakeven(d, ref, module_m2, fine):
    for e in fine:
        if np.median(lcoe_pvk(e, d, module_m2, 35, 0.007) / ref) <= 1.0:
            return float(e)
    return float("nan")


def _breakeven_exact(d, ref, module_m2, lo=0.04, hi=0.40, n_iter=50):
    """Exact efficiency at which the median LCOE ratio crosses 1 (bisection; the ratio falls with efficiency)."""
    f = lambda e: np.median(lcoe_pvk(e, d, module_m2, 35, 0.007) / ref) - 1.0
    if f(hi) > 0 or f(lo) <= 0:
        raise SystemExit(f"FAIL-CLOSED: R6 break-even not bracketed at module {module_m2} $/m2")
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(mid) <= 0 else (mid, hi)
    return float(hi)


def r6_module_floor(st) -> dict:
    rng = np.random.default_rng(SEED)
    d = site_finance_draws(rng, N_MC)          # first draws of s30: identical site/finance samples
    ref = lcoe_csi(d)
    fine = np.round(np.arange(0.04, 0.4001, 0.0025), 4)
    csi_m2 = float(csi_module_m2())
    rows = [{"module_m2": m, "breakeven_eta": _breakeven(d, ref, m, fine), "breakeven_eta_exact": _breakeven_exact(d, ref, m)}
            for m in MODULE_PRICES + (csi_m2,)]
    t = pd.DataFrame(rows)
    t.to_csv(st.path("review/module_price_floor.csv"), index=False, float_format="%.5f")
    # the grid value is the first 0.25-point step that breaks even, so the exact crossing lies just below it
    if not ((t.breakeven_eta - t.breakeven_eta_exact).between(-1e-6, 0.0025 + 1e-9)).all():
        raise SystemExit("FAIL-CLOSED: R6 exact break-even is not within one grid step below the grid value")
    tea = json.loads((st.run_dir / "metrics/tea_v2.json").read_text())
    be = dict(zip(t.module_m2, t.breakeven_eta))
    if (be[50.0], be[csi_m2]) != (tea["e4_breakeven"]["base_low_35y_deg0.7"],
                                  tea["e4_consistency"]["breakeven_eta_csi_equivalent"]):
        raise SystemExit("FAIL-CLOSED: R6 does not reproduce the s30 break-even at $50 m-2 or at the silicon price")
    pc = plant_costs()
    f_demo = min(P["f_sq"]["grid"])
    sq_peak = float(sq_limit(np.linspace(0.3, 4.0, 3701)).max())
    bx = dict(zip(t.module_m2, t.breakeven_eta_exact))
    return {"breakeven": {f"{m:g}": v for m, v in be.items()},
            "breakeven_exact": {f"{m:g}": v for m, v in bx.items()}, "csi_module_m2": csi_m2,
            "area_bos_m2": float(pc["sbos_m2"] + pc["fieldwork_m2"]), "markup": float(pc["markup"]),
            "f_sq_demonstrated": f_demo, "sq_peak": sq_peak, "eta_cap_demonstrated": f_demo * sq_peak,
            "conditions": "35 y module life, 0.7 %/yr, no burn-in, median of LCOE ratio over 4,000 draws"}


def r7_gap_offset(run_dir, st, r1) -> dict:
    df = pd.read_csv(run_dir / "stability/v2_screen_scored.csv")
    mdl = joblib.load(run_dir / "models/v2_final.joblib")
    signed = np.asarray(mdl["gap_cal_signed_scores"], float)
    lc = pd.read_csv(run_dir / "tea_v2/candidate_lcoe.csv").set_index("formula")
    tea = json.loads((run_dir / "metrics/tea_v2.json").read_text())["e5_candidates"]
    defs = tea["scenario_definitions"]
    # reproduce the s30 random stream: site/finance draws, then residual, semiconductor and draw-index samples
    rng = np.random.default_rng(SEED)
    d = site_finance_draws(rng, N_MC)
    ref = lcoe_csi(d)
    n_c = 1000
    z = rng.choice(signed, size=(len(df), n_c), replace=True)
    u_semi = rng.uniform(size=(len(df), n_c))
    idx = rng.integers(0, N_MC, size=n_c)
    dd = {k: v[idx] for k, v in d.items()}
    ref_c = ref[idx]
    semi = u_semi < df.p_semi.to_numpy()[:, None]
    absm = np.array([absorber_cost_m2(f)[0] for f in df.formula])[:, None]
    stab = (df.formula.map(lc.p_stable50) * df.formula.map(lc.plausible).astype(float)).to_numpy()
    veto = df.formula.map(lc.umlip_veto).fillna(False).astype(bool).to_numpy()
    nov = (df.status.ne("known_mp") & df.formula.map(lc.plausible).astype(bool)).to_numpy()
    shifts = {"none": 0.0, "unseen_median": float(r1["not_in_training"]["median_hse_minus_pred_eV"]),
              "in_training_median": float(r1["in_training"]["median_hse_minus_pred_eV"])}
    out = {"shifts_eV": shifts}
    for name, s in shifts.items():
        gap = df.gap_pred_eV.to_numpy()[:, None] + s + z * df.gap_sigma_eV.to_numpy()[:, None]
        sqg = sq_limit(gap)
        row = {}
        for scen, col in defs.items():
            f_sq, mod, life, deg = col.split("__")[1].split("_")
            f_sq, life, deg = float(f_sq[1:]), int(life[1:]), float(deg[1:])
            lab = P["replace_labor_frac"]["alt"] if life == 15 else 0.0
            eta = np.where(semi, f_sq * sqg, 0.0)
            l = lcoe_pvk(eta, dd, P["module_pvk_m2"][mod], life, deg, absorber_m2=absm, replace_labor_frac=lab)
            p = np.mean(l <= ref_c[None, :], axis=1)
            if s == 0.0 and not np.allclose(p, df.formula.map(lc[col]).to_numpy(), atol=1e-9):
                raise SystemExit(f"FAIL-CLOSED: R7 at zero shift does not reproduce s30 column {col}")
            row[f"expected_competitive_stable_umlip__{scen}"] = float(np.where(veto, 0.0, p * stab).sum())
            row[f"n_p_gt_0__{scen}"] = int((p > 0).sum())
        ok = df.gap_sigma_eV.notna().to_numpy()
        pgp = np.zeros(len(df))
        pgp[ok] = gap_posterior_prob(df.gap_pred_eV.to_numpy()[ok] + s, df.gap_sigma_eV.to_numpy()[ok], signed,
                                     *PV_WINDOW_EV)
        # the scored CSV stores rounded gaps and probabilities, so the reproduction tolerance is the CSV precision
        if s == 0.0 and not np.allclose(pgp[ok], df.p_gap_pv.to_numpy()[ok], atol=2e-4):
            raise SystemExit("FAIL-CLOSED: R7 at zero shift does not reproduce the s14 window probability")
        va = df.p_semi.to_numpy() * pgp * stab
        row["expected_viable_novel_plausible"] = float(va[nov].sum())
        row["expected_viable_novel_plausible_after_veto"] = float(np.where(veto, 0.0, va)[nov].sum())
        out[name] = row
    e5 = tea["expected_n_competitive_and_stable_umlip"]
    if any(abs(out["none"][f"expected_competitive_stable_umlip__{k}"] - e5[k]) > 1e-6 for k in defs):
        raise SystemExit("FAIL-CLOSED: R7 at zero shift does not reproduce the s30 expected counts")
    s22 = json.loads((run_dir / "metrics/shortlist_v2.json").read_text())["expected_viable_novel_plausible"]
    if abs(out["none"]["expected_viable_novel_plausible"] - s22) > 0.01:
        raise SystemExit("FAIL-CLOSED: R7 at zero shift does not reproduce the s22 viable-absorber count")
    return out


def r8_gate_recall(run_dir) -> dict:
    s = pd.read_csv(run_dir / "stability/v2_screen_scored.csv")
    pa = pd.read_csv(run_dir / "pareto/pareto.csv")
    lc = pd.read_csv(run_dir / "tea_v2/candidate_lcoe.csv")
    kn = s[s.status.eq("known_mp")]
    mp_ok = kn.mp_ehull_meV <= 50
    ml_ok = kn.p_ehull_le_50 >= 0.5
    cs = kn.set_index("formula").loc["Cs2BiAgBr6"]
    c = pd.read_csv(run_dir / "stability/control_known_a2bbx6.csv").dropna(subset=["true_ehull_vs_rest_meV",
                                                                                "ehull_pred_meV"])
    err = c.ehull_pred_meV - c.true_ehull_vs_rest_meV
    cst = c.true_ehull_vs_rest_meV <= 50
    # chain from the ML probability count to the stability-gated pool
    m = pa.merge(s[["formula", "p_ehull_le_50"]], on="formula", how="left")
    a = int((m.p_ehull_le_50 >= 0.5).sum())
    b = int((m.p_stable50 >= 0.5).sum())
    cc = int(((m.p_stable50 >= 0.5) & m.plausible.astype(bool)).sum())
    pool = m[m.p_stable_final >= 0.5]
    stb = json.loads((run_dir / "metrics/stability_v2.json").read_text())
    par = json.loads((run_dir / "metrics/pareto.json").read_text())
    if a != stb["n_p50_ge_0.5_by_thr"]["50"] or len(pool) != par["n_stable_gated_pool"]:
        raise SystemExit("FAIL-CLOSED: R8 pool chain does not reproduce s20/s40 counts")
    um = pd.read_csv(run_dir / "stability/umlip.csv").query("status == 'ok'")
    nv = um[um.role == "novel"]
    from scipy.stats import spearmanr as _sp
    tea = json.loads((run_dir / "metrics/tea_v2.json").read_text())["e5_candidates"]["scenario_definitions"]
    po, pc = lc[tea["optimistic"]], lc[tea["ceiling"]]
    lo, lce = (lc[tea[k].replace("p_lcoe_le_csi", "lcoe_median")] for k in ("optimistic", "ceiling"))
    nvp = pa[pa.status.ne("known_mp") & pa.plausible.astype(bool)]
    return {"known_n": int(len(kn)), "known_mp_stable": int(mp_ok.sum()),
            "known_mp_stable_ml_pass": int((mp_ok & ml_ok).sum()), "known_mp_stable_ml_reject": int((mp_ok & ~ml_ok).sum()),
            "cs2agbibr6": {"p_ml": float(cs.p_ehull_le_50), "ehull_ml_meV": float(cs.ehull_pred_meV),
                           "ehull_mp_meV": float(cs.mp_ehull_meV)},
            "controls_signed_err_median_meV": float(err.median()), "controls_stable_n": int(cst.sum()),
            "controls_stable_signed_err_median_meV": float(err[cst].median()),
            "controls_unstable_signed_err_median_meV": float(err[~cst].median()),
            "chain": {"ml_p_ge_0.5": a, "known_by_mp": b, "plausible": cc, "after_mace_veto": int(len(pool))},
            "pool_n": int(len(pool)), "pool_known": int(pool.status.eq("known_mp").sum()),
            "pool_mace_checked": int(pool.umlip_ehull_meV.notna().sum()),
            "pool_novel_mace_checked": int((pool.umlip_ehull_meV.notna() & pool.status.ne("known_mp")).sum()),
            "pool_novel_unchecked": int((pool.umlip_ehull_meV.isna() & pool.status.ne("known_mp")).sum()),
            "shortlist_novel_n": int(len(nv)), "shortlist_novel_pass": int((nv.umlip_ehull_vs_rest_meV <= 50).sum()),
            "spearman_p_opt_ceil": float(_sp(po, pc)[0]), "spearman_lcoe_opt_ceil": float(_sp(lo, lce)[0]),
            "n_p_gt_0_optimistic": int((po > 0).sum()), "n_p_gt_0_ceiling": int((pc > 0).sum()),
            "n_candidates": int(len(lc)),
            "viable_novel_plausible_before_veto": float((nvp.p_semi * nvp.p_gap_pv * nvp.p_stable50).sum()),
            "viable_novel_plausible_after_veto": float((nvp.p_semi * nvp.p_gap_pv * nvp.p_stable_final).sum())}


def run(run_dir, force: bool = False) -> dict:
    inputs = {"walterbos": WALTERBOS, "model": run_dir / "models/v2_final.joblib",
              "scored": run_dir / "stability/v2_screen_scored.csv", "cand_lcoe": run_dir / "tea_v2/candidate_lcoe.csv",
              "train": run_dir / "data/train_v2.csv", "shortlist": run_dir / "stability/shortlist_for_umlip.csv",
              "pareto": run_dir / "pareto/pareto.csv", "s22_metrics": run_dir / "metrics/shortlist_v2.json",
              "s30_metrics": run_dir / "metrics/tea_v2.json",
              "controls": run_dir / "stability/control_known_a2bbx6.csv",
              "umlip": run_dir / "stability/umlip.csv", "stability_metrics": run_dir / "metrics/stability_v2.json",
              "pareto_metrics": run_dir / "metrics/pareto.json"}
    st = Stage(run_dir, "s50_review_checks", inputs, MODULES,
               params={"thresholds": THRESHOLDS, "om_scale": OM_SCALE, "inv_year": INV_REPLACE_YEAR,
                       "module_prices": MODULE_PRICES,
                       "seed": SEED, "n_mc": N_MC})
    if st.up_to_date() and not force:
        print("[s50] up to date, skipped")
        return {}
    r1 = r1_walterbos(run_dir, st)
    st.metrics = {"r1_walterbos": r1, "r2_thresholds": r2_thresholds(run_dir),
                  "r3_om_inverter": r3_om(st), "r4_oxidation": r4_oxidation(run_dir, st),
                  "r5_energy_scheme": r5_energy_scheme(run_dir, st), "r6_module_floor": r6_module_floor(st),
                  "r7_gap_offset": r7_gap_offset(run_dir, st, r1), "r8_gate_recall": r8_gate_recall(run_dir)}
    dump_json(st.path("metrics/review_checks.json"), st.metrics)
    st.finish()
    print("[s50] done")
    return st.metrics


if __name__ == "__main__":
    import sys
    m = run(config.RUNS_DIR / "v2", force="--force" in sys.argv)
    print(json.dumps(m, indent=1, default=float)[:6000])
