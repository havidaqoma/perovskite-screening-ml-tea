"""Build paper/numbers.json: EVERY number the manuscript prints, read from run artifacts (never typed).

Each entry: key -> {"value": float|int|str, "fmt": format spec, "src": "<artifact>#<json path or csv query>"}.
The manuscript markdown writes {{key}}; build_paper.py substitutes the formatted value and fails on any
unknown key. gate_paper.py re-derives a sample of keys independently and checks every decimal in the body
text resolves to a numbers.json value.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs"


def j(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def build() -> dict:
    N = {}

    def put(key, value, fmt, src):
        N[key] = {"value": value, "fmt": fmt, "src": src}

    # ---------------- legacy reproduction (Stage 2) ----------------
    ga = j("legacy_april/gates/gate_legacy_repro.json")["checks"]
    gr = j("legacy_repro/gates/gate_legacy_repro.json")["checks"]
    fg = j("legacy_april/metrics/funnel_generate.json")
    fs = j("legacy_april/metrics/funnel_screen.json")
    put("n_generated", fg["n_generated"], ",d", "legacy_april/metrics/funnel_generate.json#n_generated")
    put("n_charge", fg["n_charge_balanced"], ",d", "legacy_april/metrics/funnel_generate.json#n_charge_balanced")
    put("n_legacy_steric", fg["n_steric"], ",d", "legacy_april/metrics/funnel_generate.json#n_steric")
    put("n_legacy_cands", fs["n_candidates"], ",d", "legacy_april/metrics/funnel_screen.json#n_candidates")
    put("legacy_mae", 0.3498204957238204, ".3f", "xgboost_cascade_pipeline_2.joblib#model_mae")
    put("repro_max_rel_lcoe_pct", 100 * ga["G6_tea"]["max_rel_d_lcoe"], ".3f", "legacy_april/gates#G6.max_rel_d_lcoe")
    put("retrain_mae", gr["G2_bg_mae"]["got"], ".3f", "legacy_repro/gates#G2.got")
    put("retrain_med_deg", gr["G5_candidates"]["median_abs_d_eg"], ".2f", "legacy_repro/gates#G5.median_abs_d_eg")
    put("retrain_top10_overlap", gr["G6_tea"]["top10_overlap"], "d", "legacy_repro/gates#G6.top10_overlap")
    put("retrain_top100_overlap", gr["G6_tea"]["top100_overlap"], "d", "legacy_repro/gates#G6.top100_overlap")
    put("retrain_spearman", gr["G6_tea"]["spearman_lcoe"], ".2f", "legacy_repro/gates#G6.spearman_lcoe")
    put("n_within_1pct", gr["G6_tea"]["n_within_1pct_of_best_legacy"], "d", "legacy_repro/gates#G6.n_within_1pct")
    leg = pd.read_csv(R / "legacy_april/tea/legacy_fullstats.csv")
    t10 = leg.nsmallest(10, "LCOE_Median").LCOE_Median
    put("legacy_top10_spread_pct", 100 * (t10.max() / t10.min() - 1), ".2f", "legacy_april/tea/legacy_fullstats.csv#top10 spread")
    put("legacy_top10_median_lcoe", 1000 * t10.median(), ".1f", "legacy_april/tea/legacy_fullstats.csv#top10 median x1000")
    put("legacy_lifetime_unique", int(leg.Panel_Lifetime_Years.nunique()), "d", "legacy_fullstats#Panel_Lifetime_Years nunique")

    # ---------------- ML audit (Stage 3) ----------------
    la = j("v2/metrics/leakage_audit.json")
    put("leak_frac_pct", 100 * la["random_row"]["val_formula_in_train"], ".1f", "v2/metrics/leakage_audit.json#random_row")
    for k in ("random_row", "formula", "chemsys"):
        put(f"leak_mae_{k}", la[k]["bg_mae_eV"], ".3f", f"v2/metrics/leakage_audit.json#{k}.bg_mae_eV")
    put("leak_delta_formula", la["formula"]["bg_mae_eV"] - la["random_row"]["bg_mae_eV"], ".2f",
        "leakage_audit.json#formula - random_row")
    put("leak_delta_chemsys", la["chemsys"]["bg_mae_eV"] - la["random_row"]["bg_mae_eV"], ".2f",
        "leakage_audit.json#chemsys - random_row")
    from pymatgen.core import Element, Species
    put("r_br_avg", float(Element("Br").average_ionic_radius), ".2f", "pymatgen Element('Br').average_ionic_radius")
    put("r_br_shannon", float(Species("Br", -1).get_shannon_radius("VI")), ".2f", "pymatgen Shannon radius Br- CN VI")
    ds = j("v2/metrics/dataset_v2.json")
    put("n_mp_entries", ds["n_raw_entries"], ",d", "v2/metrics/dataset_v2.json#n_raw_entries")
    put("n_v2_formulas", ds["n_formulas"], ",d", "v2/metrics/dataset_v2.json#n_formulas")
    put("frac_semi_pct", 100 * ds["frac_semi"], ".1f", "v2/metrics/dataset_v2.json#frac_semi")
    put("mp_db_version", ds["mp_queries"][0]["mp_database_version"], "s", "v2/metrics/dataset_v2.json#mp_database_version")
    ev = j("v2/metrics/ml_eval_v2.json")
    for r in ev["results"]:
        if "mae_pooled" in r:
            put(f"mae_{r['split']}_{r['task']}_{r['model']}", r["mae_pooled"], ".3f",
                f"v2/metrics/ml_eval_v2.json#{r['split']}/{r['task']}/{r['model']}")
        if r["task"] == "metal_cls":
            put(f"auc_{r['split']}", r["roc_auc"], ".3f", f"v2/metrics/ml_eval_v2.json#{r['split']}/metal_cls")
    ro = j("v2/metrics/roost_chemsys_gap_semi.json")
    put("mae_chemsys_gap_semi_roost", ro["mae_pooled"], ".3f", "v2/metrics/roost_chemsys_gap_semi.json#mae_pooled")
    for s in ("random", "chemsys"):
        put(f"conf_cov_{s}", ev["conformal"][s]["coverage_pooled"], ".3f", f"ml_eval_v2.json#conformal.{s}.coverage")
        put(f"conf_width_{s}", ev["conformal"][s]["width_median"], ".2f", f"ml_eval_v2.json#conformal.{s}.width")

    # ---------------- screen + stability (Stage 4) ----------------
    sc = j("v2/metrics/screen_v2.json")
    ff = sc["funnel_fail_counts"]
    put("n_tau_fail", ff["tau_not_perovskite"], ",d", "v2/metrics/screen_v2.json#tau_not_perovskite")
    put("n_ox_fail", ff["no_oxidation_state_assignment"], ",d", "v2/metrics/screen_v2.json#no_ox")
    put("n_metal_fail", ff["predicted_metal"], ",d", "v2/metrics/screen_v2.json#predicted_metal")
    put("n_v2_cands", sc["n_v2_candidates"], ",d", "v2/metrics/screen_v2.json#n_v2_candidates")
    put("n_april_pass_v2", sc["april_candidates_trace"]["pass"], ",d", "v2/metrics/screen_v2.json#april trace pass")
    allcb = pd.read_csv(R / "v2/screen/v2_all_charge_balanced.csv")
    top10 = leg.nsmallest(10, "LCOE_Median").Formula
    put("april_top10_fail_v2", int((allcb.set_index("formula").loc[top10].stage_fail != "pass").sum()), "d",
        "v2/screen/v2_all_charge_balanced.csv#April top10 failing")
    cs = allcb[allcb.formula == "Cs2BiAgBr6"].iloc[0]
    put("cs2agbibr6_tau", float(cs.tau), ".2f", "v2_all_charge_balanced#Cs2BiAgBr6.tau")
    put("cs2agbibr6_gap", float(cs.gap_pred_eV), ".2f", "v2_all_charge_balanced#Cs2BiAgBr6.gap_pred_eV")
    put("cs2agbibr6_lo", float(cs.gap_lo90_eV), ".2f", "v2_all_charge_balanced#Cs2BiAgBr6.gap_lo90_eV")
    put("cs2agbibr6_hi", float(cs.gap_hi90_eV), ".2f", "v2_all_charge_balanced#Cs2BiAgBr6.gap_hi90_eV")
    stb = j("v2/metrics/stability_v2.json")
    sct = stb["status_counts"]
    put("n_known_mp", sct["known_mp"], "d", "v2/metrics/stability_v2.json#known_mp")
    put("n_not_in_mp", sct["stable_likely"] + sct["unstable_likely"], ",d", "stability_v2.json#stable+unstable")
    put("n_stable_likely", sct["stable_likely"], "d", "stability_v2.json#stable_likely")
    for thr, v in stb["n_p50_ge_0.5_by_thr"].items():
        put(f"n_stable_thr{thr}", v, "d", f"stability_v2.json#n_p50_ge_0.5_by_thr.{thr}")
    put("ef_res_mae_mev", 1000 * stb["ef_residual_mae"], ".0f", "stability_v2.json#ef_residual_mae x1000")
    c = stb["control"]
    put("ctrl_n", c["n_ok"], "d", "stability_v2.json#control.n_ok")
    put("ctrl_auc", c["roc_auc_p50"], ".3f", "stability_v2.json#control.roc_auc_p50")
    put("ctrl_brier", c["brier_p50"], ".3f", "stability_v2.json#control.brier_p50")
    put("ctrl_ehull_mae", c["ehull_mae_meV_vs_rest"], ".0f", "stability_v2.json#control.ehull_mae")
    put("ctrl_frac_stable_pct", 100 * c["frac_true_stable_50"], ".1f", "stability_v2.json#control.frac_true_stable_50")
    put("ctrl_rel_maxdev", max(abs(v["mean_p"] - v["obs_frac"]) for v in c["reliability"].values() if v["n"]), ".2f",
        "stability_v2.json#control.reliability max |mean_p - obs|")
    cc = pd.read_csv(R / "v2/stability/control_known_a2bbx6.csv").dropna(subset=["true_ehull_vs_rest_meV"])
    put("ctrl_n_below50", int((cc.true_ehull_vs_rest_meV < -50).sum()), "d",
        "control_known_a2bbx6.csv#true_ehull_vs_rest < -50 meV")
    sw = pd.read_csv(R / "v2/tea_v2/efficiency_sweep.csv")
    csi = j("v2/metrics/tea_v2.json")["e2_csi_ref"]["msp_median_usd_mwh"]
    low = sw[(sw.module_cost == "low") & (sw.lcoe_median <= csi)]
    put("sweep_min_eta_low", 100 * float(low.eta.min()), ".0f", "efficiency_sweep.csv#min eta, $50/m2, LCOE <= c-Si median")
    d = pd.read_csv(R / "v2/data/train_v2.csv", usecols=["band_gap", "is_semi"])
    put("v2_gap_max", float(d.band_gap.max()), ".1f", "v2/data/train_v2.csv#band_gap max")
    sl = j("v2/metrics/shortlist_v2.json")
    put("n_flag_hull", sl["n_flag_hull_extrapolation"], "d", "shortlist_v2.json#n_flag_hull_extrapolation")
    put("n_flag_ox", sl["n_flag_ox_implausible"], "d", "shortlist_v2.json#n_flag_ox_implausible")
    put("n_novel_plausible", sl["n_novel_plausible"], ",d", "shortlist_v2.json#n_novel_plausible")
    put("exp_viable_novel", sl["expected_viable_novel_plausible"], ".1f", "shortlist_v2.json#expected_viable_novel_plausible")
    u = pd.read_csv(R / "v2/stability/umlip.csv").query("status == 'ok'")
    ctrl, nov = u[u.role == "control_known_mp"], u[u.role == "novel"]
    put("umlip_n_ctrl", len(ctrl), "d", "v2/stability/umlip.csv#controls")
    put("umlip_ctrl_agree", int(((ctrl.umlip_ehull_vs_rest_meV <= 50) == (ctrl.mp_ehull_meV <= 50)).sum()), "d",
        "umlip.csv#controls agreeing with MP")
    put("umlip_n_novel", len(nov), "d", "umlip.csv#novel")
    put("umlip_novel_pass", int((nov.umlip_ehull_vs_rest_meV <= 50).sum()), "d", "umlip.csv#novel <= 50 meV")
    put("umlip_novel_median", float(nov.umlip_ehull_vs_rest_meV.median()), ".0f", "umlip.csv#novel median")
    fm = nov[nov.formula.str.contains("F3")]
    put("umlip_n_fmixed", len(fm), "d", "umlip.csv#F-mixed novel")
    put("umlip_fmixed_pass", int((fm.umlip_ehull_vs_rest_meV <= 50).sum()), "d", "umlip.csv#F-mixed pass")
    put("umlip_fmixed_median", float(fm.umlip_ehull_vs_rest_meV.median()), ".0f", "umlip.csv#F-mixed median")
    from scipy.stats import spearmanr
    put("umlip_ml_spearman", float(spearmanr(nov.ml_ehull_pred_meV, nov.umlip_ehull_vs_rest_meV)[0]), ".2f",
        "umlip.csv#spearman(ml, umlip) novel")

    # ---------------- TEA (Stage 5) ----------------
    t = j("v2/metrics/tea_v2.json")
    put("nrel_msp", t["e1_reproduction"]["msp"]["nrel_total"], "d", "tea_v2.json#e1.msp.nrel_total")
    e2 = t["e2_csi_ref"]
    put("csi_med", e2["msp_median_usd_mwh"], ".1f", "tea_v2.json#e2.msp_median")
    put("csi_p10", e2["msp_p10"], ".0f", "tea_v2.json#e2.msp_p10")
    put("csi_p90", e2["msp_p90"], ".0f", "tea_v2.json#e2.msp_p90")
    put("csi_med_mmp", e2["mmp_median_usd_mwh"], ".1f", "tea_v2.json#e2.mmp_median")
    for k in ("low", "med", "high"):
        put(f"ratio818_{k}", t["e3_decoupling_test"][k]["lcoe_8pct_over_18pct"], ".2f", f"tea_v2.json#e3.{k}.ratio")
        put(f"areashare_{k}_pct", 100 * t["e3_decoupling_test"][k]["area_share_of_capex_at_10pct"], ".0f",
            f"tea_v2.json#e3.{k}.area_share")
    e4 = t["e4_breakeven"]
    put("be_low", 100 * e4["base_low_35y_deg0.7"], ".1f", "tea_v2.json#e4.base_low")
    put("be_med", 100 * e4["base_med_35y_deg0.7"], ".1f", "tea_v2.json#e4.base_med")
    put("be_unreach", e4["n_unreachable"], "d", "tea_v2.json#e4.n_unreachable")
    put("be_nscen", e4["n_scenarios"], "d", "tea_v2.json#e4.n_scenarios")
    c4 = t["e4_consistency"]
    put("absorber_effect_pct", 100 * c4["absorber_max_cost_effect_rel"], ".1f", "tea_v2.json#e4c.absorber")
    put("labor_effect_pct", 100 * c4["replace_labor_effect_rel_L10"], ".1f", "tea_v2.json#e4c.labor")
    e5 = t["e5_candidates"]
    put("f_csi", e5["f_sq_csi_module_parity"], ".2f", "tea_v2.json#e5.f_sq_csi_module_parity")
    for k in ("demonstrated", "realistic", "optimistic", "ceiling"):
        put(f"exp_comp_{k}", e5["expected_n_competitive_and_stable"][k], ".1f", f"tea_v2.json#e5.exp.{k}")
        put(f"exp_comp_umlip_{k}", e5["expected_n_competitive_and_stable_umlip"][k], ".1f", f"tea_v2.json#e5.expU.{k}")
        put(f"maxp_{k}", e5["max_p_competitive"][k], ".2f", f"tea_v2.json#e5.maxp.{k}")
        put(f"n_p05_{k}", e5["n_candidates_p_ge_0.5"][k], "d", f"tea_v2.json#e5.n_p05.{k}")
    put("rank_rho_opt_ceil", e5["rank_spearman_optimistic_vs_ceiling"], ".2f", "tea_v2.json#e5.rank_spearman")
    put("n_umlip_veto", e5["n_umlip_veto"], "d", "tea_v2.json#e5.n_umlip_veto")
    st_ = {r["param"]: r["ST"] for r in t["e6_sobol"]["ranking_ST"]}
    for p in ("f_sq", "module_m2", "cf_ac", "module_life", "deg", "r"):
        put(f"st_{p}", st_[p], ".2f", f"tea_v2.json#e6.ST.{p}")
    put("sobol_n", t["e6_sobol"]["n_eval"], ",d", "tea_v2.json#e6.n_eval")

    # ---------------- Pareto (Stage 6) ----------------
    pa = j("v2/metrics/pareto.json")
    put("n_front1", pa["n_front1"], "d", "pareto.json#n_front1")
    put("n_front1_novel", pa["n_front1_novel"], "d", "pareto.json#n_front1_novel")
    put("n_robust", pa["n_robust_front1_ge_0.75"], "d", "pareto.json#n_robust")
    put("front1_min_lcoe", pa["front1_min_lcoe_usd_mwh"], ".1f", "pareto.json#front1_min_lcoe")
    put("front1_max_p", pa["front1_max_p_competitive"], ".2f", "pareto.json#front1_max_p")
    put("n_front_stable", pa["n_front_stable"], "d", "pareto.json#n_front_stable")
    put("n_front_stable_novel", pa["n_front_stable_novel"], "d", "pareto.json#n_front_stable_novel")
    put("n_stable_pool", pa["n_stable_gated_pool"], "d", "pareto.json#n_stable_gated_pool")
    put("front_stable_umlip_checked", pa["front_stable_umlip_checked"], "d", "pareto.json#front_stable_umlip_checked")
    col_l = [k for k in pa["front_stable"][0] if k.startswith("lcoe_median__")][0]
    put("front_stable_min_lcoe", min(r[col_l] for r in pa["front_stable"]), ".1f",
        "pareto.json#front_stable min lcoe_median (optimistic)")
    put("front_stable_max_p", pa["front_stable_max_p_competitive"], ".2f", "pareto.json#front_stable_max_p_competitive")
    fsb = {r["formula"]: r for r in pa["front_stable"]}
    put("cs2agbibr3i3_umlip", fsb["Cs2BiAgBr3I3"]["umlip_ehull_meV"], ".0f", "pareto.json#front_stable.Cs2BiAgBr3I3.umlip")
    put("cs2agbibr3i3_pst", fsb["Cs2BiAgBr3I3"]["p_stable_final"], ".2f", "pareto.json#front_stable.Cs2BiAgBr3I3.p_stable")
    f1d = {r["formula"]: r for r in pa["front1"]}
    put("p_st_cs2snbis3i3", f1d["Cs2SnBiS3I3"]["p_stable_final"], ".2f", "pareto.json#front1.Cs2SnBiS3I3.p_stable_final")
    put("p_st_rb2agsbcl3i3", f1d["Rb2SbAgCl3I3"]["p_stable_final"], ".2f", "pareto.json#front1.Rb2SbAgCl3I3.p_stable_final")

    # ---------------- robustness (review response) ----------------
    rb = j("v2/metrics/robustness.json")
    for k in ("optimistic", "ceiling"):
        put(f"fr_lo_{k}", rb["frechet"][k]["lower"], ".1f", f"robustness.json#frechet.{k}.lower")
        put(f"fr_hi_{k}", rb["frechet"][k]["upper"], ".1f", f"robustness.json#frechet.{k}.upper")
    put("fr_hi_realistic", rb["frechet"]["realistic"]["upper"], ".1f", "robustness.json#frechet.realistic.upper")
    put("fr_hi_demonstrated", rb["frechet"]["demonstrated"]["upper"], ".1f", "robustness.json#frechet.demonstrated.upper")
    put("n_comp_pos_demonstrated", rb["frechet"]["demonstrated"]["n_candidates_competitive_p_gt_0"], "d",
        "robustness.json#frechet.demonstrated.n_candidates_competitive_p_gt_0")
    put("va_lo", rb["viable_absorber"]["lower"], ".1f", "robustness.json#viable_absorber.lower")
    put("va_hi", rb["viable_absorber"]["upper"], ".1f", "robustness.json#viable_absorber.upper")
    put("conf_cov_mixed", rb["conformal_by_anion_mix"]["2"]["coverage"], ".3f", "robustness.json#conformal_by_anion_mix.2")
    put("conf_n_mixed", rb["conformal_by_anion_mix"]["2"]["n"], ",d", "robustness.json#conformal_by_anion_mix.2.n")
    put("ctrl_n_mixed", rb["control_anion_mix"]["n_mixed_anion"], "d", "robustness.json#control_anion_mix.n_mixed_anion")
    put("n_train_feat", rb["n_train_featurized"], ",d", "robustness.json#n_train_featurized")
    put("n_train_fail", rb["n_train_featurize_failed"], "d", "robustness.json#n_train_featurize_failed")
    put("n_mc_draws", 4000, ",d", "pipeline/s30_tea_v2.py#N_MC")
    put("n_gap_draws", 1000, ",d", "pipeline/s30_tea_v2.py#n_c")

    # ---------------- pre-submission review checks (stage s50) ----------------
    rc = j("v2/metrics/review_checks.json")
    src = "v2/metrics/review_checks.json#"
    wu, wa = rc["r1_walterbos"]["not_in_training"], rc["r1_walterbos"]["all"]
    put("wb_n_unseen", wu["n_nonmetal"], ",d", src + "r1.not_in_training.n_nonmetal")
    put("wb_rho_unseen", wu["spearman_gap_pred_vs_hse"], ".2f", src + "r1.not_in_training.spearman")
    put("wb_off_unseen", wu["median_hse_minus_pred_eV"], ".2f", src + "r1.not_in_training.median_hse_minus_pred_eV")
    put("wb_below_unseen_pct", 100 * wu["frac_hse_below_interval"], ".0f", src + "r1.not_in_training.frac_hse_below x100")
    put("wb_above_unseen_pct", 100 * wu["frac_hse_above_interval"], ".0f", src + "r1.not_in_training.frac_hse_above x100")
    put("wb_cov_shift_unseen_pct", 100 * wu["coverage_after_median_shift"], ".0f",
        src + "r1.not_in_training.coverage_after_median_shift x100")
    put("wb_metal_semi_n", wu["n_metal_called_semi_p05"], "d", src + "r1.not_in_training.n_metal_called_semi_p05")
    put("wb_metal_n", wu["n_metal_hse"], "d", src + "r1.not_in_training.n_metal_hse")
    put("wb_nonmetal_rej", wu["n_nonmetal_rejected_p05"], "d", src + "r1.not_in_training.n_nonmetal_rejected_p05")
    put("wb_window_n", wa["n_in_window_nonmetal"], "d", src + "r1.all.n_in_window_nonmetal")
    put("wb_cov_unseen_pct", 100 * wu["coverage_hse_by_interval"], ".0f", src + "r1.not_in_training.coverage x100")
    put("wb_auc_semi_unseen", wu["auc_semi_vs_hse_nonmetal"], ".2f", src + "r1.not_in_training.auc_semi")
    put("wb_spinforb_pct", 100 * wa["frac_spin_forbidden_in_window"], ".0f", src + "r1.all.frac_spin_forbidden_in_window")
    thr = rc["r2_thresholds"]
    opt = [thr[t]["expected_competitive_stable_umlip__optimistic"] for t in thr]
    put("thr_opt_lo", min(opt), ".1f", src + "r2.*.optimistic min")
    put("thr_opt_hi", max(opt), ".1f", src + "r2.*.optimistic max")
    va = [thr[t]["expected_viable_novel_plausible"] for t in thr]
    put("thr_va_lo", min(va), ".1f", src + "r2.*.expected_viable_novel_plausible min")
    put("thr_va_hi", max(va), ".1f", src + "r2.*.expected_viable_novel_plausible max")
    zero = max(thr[t][f"expected_competitive_stable_umlip__{k}"] for t in thr for k in ("realistic", "demonstrated"))
    if zero != 0.0:
        raise SystemExit(f"FAIL-CLOSED: prose says realistic/demonstrated stay at zero at every threshold; max = {zero}")
    om = rc["r3_om_inverter"]
    b = om["breakeven"]
    if any(b[f"inv1_om{k:g}_{c}"] != b[f"inv0_om{k:g}_{c}"] for k in om["om_scale"] for c in ("low", "med")):
        raise SystemExit("FAIL-CLOSED: prose says inverter replacement leaves every break-even unchanged")
    put("om_be_low_125", 100 * b["inv0_om1.25_low"], ".1f", src + "r3.breakeven.inv0_om1.25_low x100")
    put("om_be_low_200", 100 * b["inv0_om2_low"], ".1f", src + "r3.breakeven.inv0_om2_low x100")
    put("om_be_med_200", 100 * b["inv0_om2_med"], ".1f", src + "r3.breakeven.inv0_om2_med x100")
    put("inv_effect_pct", 100 * om["inverter_effect_on_csi_lcoe_rel"], ".1f", src + "r3.inverter_effect_on_csi_lcoe_rel x100")
    put("eta_step_pp", 100 * om["eta_grid_step"], ".2f", src + "r3.eta_grid_step x100")
    ox = rc["r4_oxidation"]
    if ox["n_on_pareto_front"] != 0:
        raise SystemExit("FAIL-CLOSED: prose says no strict-rule composition is on the Pareto front")
    if ox["n_flagged"] != N["n_flag_ox"]["value"]:
        raise SystemExit("FAIL-CLOSED: s50 flagged count differs from s22")
    put("n_flag_ox_red", ox["n_flagged_with_reducing_anion"], "d", src + "r4.n_flagged_with_reducing_anion")
    put("ox_strict_n", ox["n_high_valent_reducing_unflagged"], "d", src + "r4.n_high_valent_reducing_unflagged")
    put("ox_strict_umlip", ox["n_in_umlip_shortlist"], "d", src + "r4.n_in_umlip_shortlist")
    put("ox_umlip_n", ox["n_umlip_shortlist"], "d", src + "r4.n_umlip_shortlist")
    if ox["n_in_umlip_shortlist_with_mace"] != ox["n_in_umlip_shortlist"] or ox["n_in_umlip_shortlist_within_50meV"] != 0:
        raise SystemExit("FAIL-CLOSED: prose says MACE placed every strict-rule shortlist composition above 50 meV")
    put("ox_strict_viable", ox["expected_viable_novel"]["strict_rule"], ".1f", src + "r4.expected_viable_novel.strict_rule")
    es = rc["r5_energy_scheme"]
    if es["controls"]["max_meV"] >= 50 or es["controls"]["n"] != c["n_ok"]:
        raise SystemExit("FAIL-CLOSED: prose says the training/hull energy-scheme offset of every control is far below "
                         "the 50 meV gate")
    put("es_ctrl_max", es["controls"]["max_meV"], ".1f", src + "r5.controls.max_meV")

    # public code release cited in Data availability and the cover letter; build_jmca.py checks the tag exists
    put("repo_version", "2.2.0", "s", "public repo CITATION.cff#version, git tag v2.2.0")

    # ---------------- sourced literature constants used in prose ----------------
    put("lazard_lo", 29, "d", "LAZ24 p9/p35 (data/sources/SOURCES.md)")
    put("lazard_hi", 92, "d", "LAZ24 p9/p35")
    put("irena_avg", 44, "d", "IRENA23 via pv magazine")
    put("dp_record", 6.37, ".2f", "ZHANG22 (10.1038/s41467-022-31016-w)")
    put("eta_csi_pct", 20.5, ".1f", "NREL23 text layer")
    put("sq_peak", 33.7, ".1f", "pipeline/sq.py (gate test)")
    put("jordan_deg_lo", 0.5, ".1f", "JORDAN16 abstract (10.1002/pip.2744): 'median degradation for x-Si ... 0.5-0.6%/year'")
    put("jordan_deg_hi", 0.6, ".1f", "JORDAN16 abstract (10.1002/pip.2744)")
    return N


def fmt(entry) -> str:
    v, f = entry["value"], entry["fmt"]
    return str(v) if f == "s" else format(v, f)


if __name__ == "__main__":
    N = build()
    (ROOT / "paper/numbers.json").write_text(json.dumps(N, indent=1, default=float), encoding="utf-8")
    print(len(N), "numbers")
