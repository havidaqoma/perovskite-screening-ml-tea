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
from .s30_tea_v2 import N_MC, SEED, lcoe_csi, lcoe_pvk, site_finance_draws
from .tea_params import P

MODULES = ("s50_review_checks",)
WALTERBOS = config.DATA_DIR / "external/walterbos2026/hdp_hse06_subset.csv"
THRESHOLDS = (0, 20, 35, 50, 100)
OM_SCALE = (1.0, 1.25, 1.5, 2.0)
INV_REPLACE_YEAR = 15
REDUCING = ("S", "Se", "Te", "I")
ENTRY_DIR = config.DATA_DIR / "mp_cache" / "entries"     # written by s20 (GGA/GGA+U entries per chemical system)
HALOGENS = {"F", "Cl", "Br", "I"}
CHALC_PNICT_O = {"O", "S", "Se", "Te", "N", "P", "As", "Sb"}


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
        return "halide_only" if el & HALOGENS and not el & CHALC_PNICT_O else "other"
    d["kind"] = d.formula.map(kind)
    d.to_csv(st.path("review/energy_scheme.csv"), index=False, float_format="%.6f")

    def summ(x: pd.Series) -> dict:
        return {"n": int(len(x)), "median_meV": float(x.median()), "frac_gt_10meV": float((x > 10).mean()),
                "frac_gt_50meV": float((x > 50).mean()), "max_meV": float(x.max())}
    ctrl = pd.read_csv(run_dir / "stability/control_known_a2bbx6.csv").merge(tr, on="formula")
    ctrl = ctrl.dropna(subset=["true_ef_pd"])
    cdiff = 1000.0 * (ctrl.formation_energy_per_atom - ctrl.true_ef_pd).abs()
    return {"n_chemsys": len(files), "all": summ(d.abs_diff_meV),
            "halide_only": summ(d.abs_diff_meV[d.kind == "halide_only"]),
            "other": summ(d.abs_diff_meV[d.kind == "other"]), "controls": summ(cdiff)}


def run(run_dir, force: bool = False) -> dict:
    inputs = {"walterbos": WALTERBOS, "model": run_dir / "models/v2_final.joblib",
              "scored": run_dir / "stability/v2_screen_scored.csv", "cand_lcoe": run_dir / "tea_v2/candidate_lcoe.csv",
              "train": run_dir / "data/train_v2.csv", "shortlist": run_dir / "stability/shortlist_for_umlip.csv",
              "pareto": run_dir / "pareto/pareto.csv", "s22_metrics": run_dir / "metrics/shortlist_v2.json",
              "s30_metrics": run_dir / "metrics/tea_v2.json",
              "controls": run_dir / "stability/control_known_a2bbx6.csv"}
    st = Stage(run_dir, "s50_review_checks", inputs, MODULES,
               params={"thresholds": THRESHOLDS, "om_scale": OM_SCALE, "inv_year": INV_REPLACE_YEAR,
                       "seed": SEED, "n_mc": N_MC})
    if st.up_to_date() and not force:
        print("[s50] up to date, skipped")
        return {}
    st.metrics = {"r1_walterbos": r1_walterbos(run_dir, st), "r2_thresholds": r2_thresholds(run_dir),
                  "r3_om_inverter": r3_om(st), "r4_oxidation": r4_oxidation(run_dir, st),
                  "r5_energy_scheme": r5_energy_scheme(run_dir, st)}
    dump_json(st.path("metrics/review_checks.json"), st.metrics)
    st.finish()
    print("[s50] done")
    return st.metrics


if __name__ == "__main__":
    import sys
    m = run(config.RUNS_DIR / "v2", force="--force" in sys.argv)
    print(json.dumps(m, indent=1, default=float)[:6000])
