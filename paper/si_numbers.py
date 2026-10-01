"""Build paper/si_numbers.json: every number the ESI prose prints that is not already in numbers.json.

Same contract as paper_numbers.py: key -> {"value", "fmt", "src"}; the ESI markdown writes {{key}} and build_si.py
substitutes the formatted value (fails on an unknown key). Values come from run artifacts, or, for method settings,
from the pipeline constant itself (imported, or read from the source line), so a changed setting changes the ESI.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs"
sys.path.insert(0, str(ROOT))


def j(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def src_const(module: str, pattern: str) -> str:
    """Read one literal from a pipeline source line (for settings defined inside functions)."""
    text = (ROOT / "pipeline" / f"{module}.py").read_text(encoding="utf-8")
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"FAIL-CLOSED: pattern {pattern!r} not found in pipeline/{module}.py")
    return m.group(1)


def build() -> dict:
    N = {}

    def put(key, value, fmt, src):
        N[key] = {"value": value, "fmt": fmt, "src": src}

    from pipeline import config, geometry, s11_ml_eval, s14_screen_v2, s20_stability, s21_umlip, s22_shortlist, \
        s30_tea_v2, s40_pareto
    from pipeline.s10_mp_dataset import METAL_GAP_EV
    from pipeline.sq import T_CELL, sq_table
    from pipeline.tea_params import P

    # ---------------- Note S1: training data ----------------
    ds = j("v2/metrics/dataset_v2.json")
    put("si_n_chemsys", ds["n_chemsys"], ",d", "v2/metrics/dataset_v2.json#n_chemsys")
    put("si_frac_ehull01_pct", 100 * ds["frac_ehull_le_0.1"], ".1f", "dataset_v2.json#frac_ehull_le_0.1")
    put("si_metal_gap", METAL_GAP_EV, ".1f", "pipeline/s10_mp_dataset.py#METAL_GAP_EV")
    put("si_n_queries", len(ds["mp_queries"]), "d", "dataset_v2.json#mp_queries")
    fm = j("v2/metrics/final_models_v2.json")
    put("si_n_semi", fm["n_semi"], ",d", "v2/metrics/final_models_v2.json#n_semi")

    # ---------------- Note S2-S3: features, splits, models ----------------
    ev = j("v2/metrics/ml_eval_v2.json")
    put("si_n_features", len(ev["feature_names"]), "d", "ml_eval_v2.json#feature_names")
    put("si_n_folds", s11_ml_eval.N_FOLDS, "d", "pipeline/s11_ml_eval.py#N_FOLDS")
    put("si_seed", s11_ml_eval.SEED, "d", "pipeline/s11_ml_eval.py#SEED")
    put("si_n_anion_fam", len(s11_ml_eval.ANION_FAMILIES), "d", "pipeline/s11_ml_eval.py#ANION_FAMILIES")
    put("si_ridge_alpha", float(src_const("s11_ml_eval", r"Ridge\(alpha=([\d.]+)\)")), ".1f", "s11_ml_eval.py#Ridge alpha")
    put("si_rf_trees", int(src_const("s11_ml_eval", r"RandomForestRegressor\(n_estimators=(\d+)")), "d", "s11_ml_eval.py#rf")
    ro = j("v2/metrics/roost_chemsys_gap_semi.json")
    put("si_roost_epochs", ro["epochs"], "d", "roost_chemsys_gap_semi.json#epochs")
    put("si_roost_lr", src_const("s12_roost", r"lr=([\de.-]+)"), "s", "pipeline/s12_roost.py#AdamW lr")
    put("si_roost_wd", src_const("s12_roost", r"weight_decay=([\de.-]+)"), "s", "pipeline/s12_roost.py#weight_decay")
    put("si_roost_batch", int(src_const("s12_roost", r"batch_size=(\d+)")), "d", "pipeline/s12_roost.py#batch_size")
    put("si_roost_emb", src_const("s12_roost", r'elem_embedding="(\w+)"'), "s", "pipeline/s12_roost.py#elem_embedding")
    roost_folds = [f["mae"] for f in ro["folds"]]
    put("si_roost_fold_min", min(roost_folds), ".3f", "roost_chemsys_gap_semi.json#folds.mae min")
    put("si_roost_fold_max", max(roost_folds), ".3f", "roost_chemsys_gap_semi.json#folds.mae max")

    # ---------------- Note S4: conformal ----------------
    put("si_alpha", s11_ml_eval.CONFORMAL_ALPHA, ".2f", "pipeline/s11_ml_eval.py#CONFORMAL_ALPHA")
    put("si_sigma_floor", float(src_const("s13_final_models", r'"sigma_floor": ([\d.]+)')), ".2f",
        "pipeline/s13_final_models.py#sigma_floor")
    put("si_cal_frac", float(src_const("s11_ml_eval", r"int\(([\d.]+) \* len\(perm\)\)")), ".1f",
        "pipeline/s11_ml_eval.py#calibration fraction")
    put("si_cal_min", int(src_const("s11_ml_eval", r"n_cal = max\((\d+),")), "d", "pipeline/s11_ml_eval.py#n_cal floor")
    put("si_final_cal_frac", float(src_const("s13_final_models", r"test_size=([\d.]+)")), ".1f",
        "pipeline/s13_final_models.py#GroupShuffleSplit test_size")
    put("si_q_gap", fm["q_gap"], ".2f", "final_models_v2.json#q_gap")
    put("si_n_cal", fm["n_cal"], ",d", "final_models_v2.json#n_cal")
    for s in ("random", "chemsys"):
        c = ev["conformal"][s]
        put(f"si_cov_{s}_min", min(c["coverage_folds"]), ".3f", f"ml_eval_v2.json#conformal.{s}.coverage_folds min")
        put(f"si_cov_{s}_max", max(c["coverage_folds"]), ".3f", f"ml_eval_v2.json#conformal.{s}.coverage_folds max")
        put(f"si_width_p90_{s}", c["width_p90"], ".2f", f"ml_eval_v2.json#conformal.{s}.width_p90")
    rb = j("v2/metrics/robustness.json")
    put("si_cov_single", rb["conformal_by_anion_mix"]["1"]["coverage"], ".3f", "robustness.json#conformal_by_anion_mix.1")
    put("si_n_single", rb["conformal_by_anion_mix"]["1"]["n"], ",d", "robustness.json#conformal_by_anion_mix.1.n")
    put("si_cov_mix_alk", rb["conformal_mixed_anion_alkali"]["coverage"], ".3f", "robustness.json#conformal_mixed_anion_alkali")
    put("si_n_mix_alk", rb["conformal_mixed_anion_alkali"]["n"], ",d", "robustness.json#conformal_mixed_anion_alkali.n")

    # ---------------- Note S5: geometry and screening gates ----------------
    put("si_tau_max", geometry.TAU_MAX, ".2f", "pipeline/geometry.py#TAU_MAX")
    put("si_pv_lo", s14_screen_v2.PV_WINDOW_EV[0], ".1f", "pipeline/s14_screen_v2.py#PV_WINDOW_EV")
    put("si_pv_hi", s14_screen_v2.PV_WINDOW_EV[1], ".1f", "pipeline/s14_screen_v2.py#PV_WINDOW_EV")
    put("si_p_semi_min", s14_screen_v2.P_SEMI_MIN, ".1f", "pipeline/s14_screen_v2.py#P_SEMI_MIN")
    put("si_toxic", ", ".join(config.TOXIC), "s", "pipeline/config.py#TOXIC")
    put("si_n_a", len(config.A_SITE), "d", "pipeline/config.py#A_SITE")
    put("si_n_b", len(config.B_SITE), "d", "pipeline/config.py#B_SITE")
    put("si_n_x", len(config.X_SITE), "d", "pipeline/config.py#X_SITE")
    from math import comb
    put("si_n_bpairs", comb(len(config.B_SITE), 2), "d", "C(len(config.B_SITE), 2), as in s03_generate.generate")
    put("si_n_xcomb", len(config.X_SITE) + comb(len(config.X_SITE), 2), "d",
        "len(X_SITE) + C(len(X_SITE), 2), as in s03_generate.generate")
    put("si_n_xpairs", comb(len(config.X_SITE), 2), "d", "C(len(config.X_SITE), 2)")
    sc = j("v2/metrics/screen_v2.json")
    put("si_n_pv_absorber_05", sc["p_pv_absorber_ge_0.5"], "d", "screen_v2.json#p_pv_absorber_ge_0.5")
    tr = sc["april_candidates_trace"]
    put("si_april_tau_fail", tr.get("tau_not_perovskite", 0), ",d", "screen_v2.json#april_candidates_trace.tau_not_perovskite")
    put("si_april_ox_fail", tr.get("no_oxidation_state_assignment", 0), "d", "screen_v2.json#april_candidates_trace.no_ox")
    put("si_april_metal_fail", tr.get("predicted_metal", 0), "d", "screen_v2.json#april_candidates_trace.predicted_metal")

    # ---------------- Note S6: stability ----------------
    put("si_gate_mev", s20_stability.GATE_MEV, ".0f", "pipeline/s20_stability.py#GATE_MEV")
    put("si_p_stable_min", s20_stability.P_STABLE_MIN, ".1f", "pipeline/s20_stability.py#P_STABLE_MIN")
    put("si_control_n", s20_stability.CONTROL_N, "d", "pipeline/s20_stability.py#CONTROL_N")
    put("si_ef_res_n", fm["ef_residual_n"], ",d", "final_models_v2.json#ef_residual_n")
    put("si_ef_res_bias_mev", 1000 * fm["ef_residual_bias"], ".1f", "final_models_v2.json#ef_residual_bias x1000")
    ctrl = j("v2/metrics/stability_v2.json")["control"]
    put("si_ctrl_prec", ctrl["gate_precision"], ".2f", "stability_v2.json#control.gate_precision")
    put("si_ctrl_rec", ctrl["gate_recall"], ".2f", "stability_v2.json#control.gate_recall")
    put("si_ctrl_failed", ctrl["n"] - ctrl["n_ok"], "d", "stability_v2.json#control.n - n_ok")
    put("si_hull_extrap", s22_shortlist.HULL_EXTRAP_MEV, ".0f", "pipeline/s22_shortlist.py#HULL_EXTRAP_MEV")
    put("si_ox_implausible", " and ".join(o.replace("+", "") + "+" for o in s22_shortlist.OX_IMPLAUSIBLE), "s",
        "pipeline/s22_shortlist.py#OX_IMPLAUSIBLE")
    put("si_top_n_novel", s22_shortlist.TOP_N_NOVEL, "d", "pipeline/s22_shortlist.py#TOP_N_NOVEL")
    put("si_n_control_umlip", s22_shortlist.N_CONTROL, "d", "pipeline/s22_shortlist.py#N_CONTROL")

    # ---------------- Note S7: MACE ----------------
    put("si_fmax", s21_umlip.FMAX, ".2f", "pipeline/s21_umlip.py#FMAX")
    put("si_steps", s21_umlip.STEPS, "d", "pipeline/s21_umlip.py#STEPS")
    u = pd.read_csv(R / "v2/stability/umlip.csv")
    ok = u[u.status == "ok"]
    put("si_umlip_ok", len(ok), "d", "v2/stability/umlip.csv#status==ok")
    put("si_umlip_conv", int(ok.converged.astype(bool).sum()), "d", "umlip.csv#converged")
    put("si_umlip_nref_med", float(ok.n_ref_phases.median()), ".0f", "umlip.csv#n_ref_phases median")
    put("si_umlip_nref_max", int(ok.n_ref_phases.max()), "d", "umlip.csv#n_ref_phases max")
    put("si_umlip_sec_med", float(ok.seconds.median()), ".0f", "umlip.csv#seconds median")
    cu = ok[ok.role == "control_known_mp"].umlip_ehull_vs_rest_meV
    put("si_umlip_ctrl_min", float(cu.min()), ".0f", "umlip.csv#control min")
    put("si_umlip_ctrl_max", float(cu.max()), ".0f", "umlip.csv#control max")
    nv = ok[ok.role == "novel"].umlip_ehull_vs_rest_meV
    put("si_umlip_novel_min", float(nv.min()), ".0f", "umlip.csv#novel min")
    put("si_umlip_novel_max", float(nv.max()), ".0f", "umlip.csv#novel max")

    # ---------------- Note S8: detailed balance ----------------
    egs, eff = sq_table()
    put("si_sq_t", T_CELL, ".0f", "pipeline/sq.py#T_CELL")
    put("si_sq_peak_eg", float(egs[eff.argmax()]), ".2f", "pipeline/sq.py#sq_table argmax")
    put("si_sq_peak", 100 * float(eff.max()), ".2f", "pipeline/sq.py#sq_table max")
    put("si_sq_n", len(egs), "d", "pipeline/sq.py#sq_table n")
    put("si_sq_egmin", float(egs.min()), ".1f", "pipeline/sq.py#sq_table eg_min")
    put("si_sq_egmax", float(egs.max()), ".1f", "pipeline/sq.py#sq_table eg_max")
    t = j("v2/metrics/tea_v2.json")
    put("si_f_csi", t["e5_candidates"]["f_sq_csi_module_parity"], ".2f", "tea_v2.json#e5_candidates.f_sq_csi_module_parity")
    put("si_f_mid", sorted(P["f_sq"]["grid"])[1], ".2f", "pipeline/tea_params.py#f_sq grid")

    # ---------------- Note S9-S10: TEA ----------------
    e1 = t["e1_reproduction"]
    put("si_e1_msp", e1["msp"]["capex_kwdc"], ".1f", "tea_v2.json#e1_reproduction.msp.capex_kwdc")
    put("si_e1_mmp", e1["mmp"]["capex_kwdc"], ".1f", "tea_v2.json#e1_reproduction.mmp.capex_kwdc")
    put("si_e1_mmp_ref", e1["mmp"]["nrel_total"], ",d", "tea_v2.json#e1_reproduction.mmp.nrel_total")
    put("si_be_csi", 100 * t["e4_consistency"]["breakeven_eta_csi_equivalent"], ".1f", "tea_v2.json#e4_consistency")
    put("si_base_lcoe", t["e4_consistency"]["base_lcoe_usd_mwh"], ".1f", "tea_v2.json#e4_consistency.base_lcoe_usd_mwh")
    put("si_n_mc", s30_tea_v2.N_MC, ",d", "pipeline/s30_tea_v2.py#N_MC")
    put("si_tea_seed", s30_tea_v2.SEED, "d", "pipeline/s30_tea_v2.py#SEED")
    put("si_sobol_base", int(src_const("s30_tea_v2", r"sobol_sample\.sample\(prob, (\d+)")), ",d", "s30_tea_v2.py#Saltelli N")
    put("si_sobol_k", len(t["e6_sobol"]["ranking_ST"]), "d", "tea_v2.json#e6_sobol.ranking_ST")
    put("si_sobol_n", t["e6_sobol"]["n_eval"], ",d", "tea_v2.json#e6_sobol.n_eval")
    put("si_sobol_sum_s1", t["e6_sobol"]["sum_S1"], ".2f", "tea_v2.json#e6_sobol.sum_S1")
    put("si_sobol_inter_pct", 100 * (1 - round(t["e6_sobol"]["sum_S1"], 2)), ".0f", "1 - tea_v2.json#e6_sobol.sum_S1 (as printed)")
    put("si_abs_thick", P["absorber_thickness_nm"]["value"], "d", "tea_params.py#absorber_thickness_nm")
    put("si_abs_rho", P["absorber_density_gcm3"]["value"], ".1f", "tea_params.py#absorber_density_gcm3")
    put("si_abs_util", P["absorber_utilization"]["value"], ".2f", "tea_params.py#absorber_utilization")
    e5 = t["e5_candidates"]
    put("si_abs_med", e5["absorber_cost_m2_median"], ".2f", "tea_v2.json#e5_candidates.absorber_cost_m2_median")
    put("si_abs_max", e5["absorber_cost_m2_max"], ".2f", "tea_v2.json#e5_candidates.absorber_cost_m2_max")
    put("si_abs_unsourced", e5["n_absorber_price_unsourced"], ",d", "tea_v2.json#e5_candidates.n_absorber_price_unsourced")
    put("si_plant_life", P["plant_life"]["value"], "d", "tea_params.py#plant_life")
    put("si_ilr", P["ilr"]["value"], ".2f", "tea_params.py#ilr")
    put("si_repl_alt", P["replace_labor_frac"]["alt"], ".1f", "tea_params.py#replace_labor_frac.alt")
    put("si_repl_base", P["replace_labor_frac"]["value"], ".0f", "tea_params.py#replace_labor_frac.value")
    be = pd.read_csv(R / "v2/tea_v2/breakeven_efficiency.csv")
    put("si_be_reachable", int(be.breakeven_reachable_by_sq.sum()), "d", "breakeven_efficiency.csv#breakeven_reachable_by_sq")
    put("si_be_above_sq", int((be.breakeven_eta.notna() & ~be.breakeven_reachable_by_sq.astype(bool)).sum()), "d",
        "breakeven_efficiency.csv#breakeven_eta notna & not reachable_by_sq")
    put("si_be_eta_lo", 100 * float(be.breakeven_eta.dropna().min()), ".1f", "breakeven_efficiency.csv#min")
    grid = re.search(r"fine = np\.round\(np\.arange\(([\d.]+), ([\d.]+), ([\d.]+)\)", (ROOT / "pipeline/s30_tea_v2.py").read_text())
    put("si_be_grid_lo", 100 * float(grid.group(1)), ".0f", "pipeline/s30_tea_v2.py#break-even grid start")
    put("si_be_grid_hi", 100 * round(float(grid.group(2)), 2), ".0f", "pipeline/s30_tea_v2.py#break-even grid end")
    put("si_be_grid_step", 100 * float(grid.group(3)), ".2f", "pipeline/s30_tea_v2.py#break-even grid step")
    put("si_n_csi_draws", s30_tea_v2.N_MC, ",d", "pipeline/s30_tea_v2.py#N_MC")
    put("si_roost_val_frac", float(src_const("s12_roost", r"int\(([\d.]+) \* len\(perm\)\)")), ".1f",
        "pipeline/s12_roost.py#validation fraction")
    put("si_roost_val_min", int(src_const("s12_roost", r"nv = max\((\d+),")), "d", "pipeline/s12_roost.py#validation floor")
    put("si_csi_gap", float(src_const("s30_tea_v2", r'P\["eta_csi"\]\["value"\] / float\(sq_limit\(([\d.]+)\)\)')), ".2f",
        "pipeline/s30_tea_v2.py#c-Si gap for f_SQ parity")
    put("si_reach_gap", float(src_const("s30_tea_v2", r"hit\[0\] <= float\(sq_limit\(([\d.]+)\)\)")), ".2f",
        "pipeline/s30_tea_v2.py#gap used for the reachability test")
    put("si_sobol_gap", float(src_const("s30_tea_v2", r"sq134 = float\(sq_limit\(([\d.]+)\)\)")), ".2f",
        "pipeline/s30_tea_v2.py#gap used in the Sobol model")
    put("si_umlip_unconv", int((~ok.converged.astype(bool)).sum()), "d", "umlip.csv#not converged")
    put("si_umlip_unconv_min", float(ok[~ok.converged.astype(bool)].umlip_ehull_vs_rest_meV.min()), ".0f",
        "umlip.csv#min hull distance among unconverged runs")
    from pipeline.s10_mp_dataset import RADIOACTIVE
    put("si_radioactive", ", ".join(RADIOACTIVE), "s", "pipeline/s10_mp_dataset.py#RADIOACTIVE")
    put("si_a_list", ", ".join(config.A_SITE), "s", "pipeline/config.py#A_SITE")
    put("si_b_list", ", ".join(config.B_SITE), "s", "pipeline/config.py#B_SITE")
    put("si_x_list", ", ".join(config.X_SITE), "s", "pipeline/config.py#X_SITE")
    s1 = pd.read_csv(ROOT / "paper/figures/si_data/figS1_conformal.csv")
    s1b = s1[s1.panel == "b"].reset_index(drop=True)
    put("si_cov_bin_lo", float(s1b.coverage.iloc[0]), ".3f", "figures/si_data/figS1_conformal.csv#panel b first bin")
    put("si_cov_bin_lo_label", str(s1b.gap_bin_eV.iloc[0]).replace("\u2013", " to "), "s",
        "figures/si_data/figS1_conformal.csv#panel b first bin label")
    put("si_cov_bin_rest_min", float(s1b.coverage.iloc[1:].min()), ".3f", "figS1_conformal.csv#panel b other bins min")
    put("si_cov_bin_rest_max", float(s1b.coverage.iloc[1:].max()), ".3f", "figS1_conformal.csv#panel b other bins max")

    # ---------------- Note S11: Pareto ----------------
    put("si_n_scen", len(s40_pareto.SCENARIOS), "d", "pipeline/s40_pareto.py#SCENARIOS")
    put("si_n_variants", 2 * len(s40_pareto.SCENARIOS), "d", "pipeline/s40_pareto.py#variants")
    put("si_robust_thr", float(src_const("s40_pareto", r"front1_frequency >= ([\d.]+)")), ".2f", "s40_pareto.py#robust thr")
    put("si_spearman", e5["rank_spearman_optimistic_vs_ceiling"], ".2f", "tea_v2.json#rank_spearman_optimistic_vs_ceiling")
    va = rb["viable_absorber"]
    put("si_va_ind", va["independent"], ".1f", "robustness.json#viable_absorber.independent")

    # ---------------- Note S12: hybrid-functional check (stage s50, R1) ----------------
    from pipeline import s50_review_checks
    rc = j("v2/metrics/review_checks.json")
    r1 = rc["r1_walterbos"]
    S = "v2/metrics/review_checks.json#r1."
    put("si_wb_n_src", r1["n_rows_source"], ",d", S + "n_rows_source")
    put("si_wb_n_vac", r1["n_vacancy_dropped"], "d", S + "n_vacancy_dropped")
    put("si_wb_n", r1["all"]["n"], ",d", S + "all.n")
    put("si_wb_n_metal", r1["all"]["n_metal_hse"], "d", S + "all.n_metal_hse")
    put("si_wb_n_train", r1["n_in_training"], "d", S + "n_in_training")
    put("si_wb_n_cands", r1["n_in_v2_candidates"], "d", S + "n_in_v2_candidates")
    for blk, tag in (("all", "all"), ("not_in_training", "unseen"), ("in_training", "train")):
        b = r1[blk]
        put(f"si_wb_rho_{tag}", b["spearman_gap_pred_vs_hse"], ".2f", S + f"{blk}.spearman_gap_pred_vs_hse")
        put(f"si_wb_off_{tag}", b["median_hse_minus_pred_eV"], ".2f", S + f"{blk}.median_hse_minus_pred_eV")
        put(f"si_wb_cov_{tag}_pct", 100 * b["coverage_hse_by_interval"], ".0f", S + f"{blk}.coverage x100")
        put(f"si_wb_above_{tag}_pct", 100 * b["frac_hse_above_interval"], ".0f", S + f"{blk}.frac_hse_above x100")
        put(f"si_wb_auc_semi_{tag}", b["auc_semi_vs_hse_nonmetal"], ".2f", S + f"{blk}.auc_semi_vs_hse_nonmetal")
        put(f"si_wb_auc_pv_{tag}", b["auc_p_gap_pv_vs_hse_window"], ".2f", S + f"{blk}.auc_p_gap_pv_vs_hse_window")
    put("si_wb_mae_unseen", r1["not_in_training"]["mae_vs_hse_eV"], ".2f", S + "not_in_training.mae_vs_hse_eV")
    u = r1["not_in_training"]
    put("si_wb_below_unseen_pct", 100 * u["frac_hse_below_interval"], ".0f", S + "not_in_training.frac_hse_below x100")
    put("si_wb_shift_cov_unseen_pct", 100 * u["coverage_after_median_shift"], ".0f",
        S + "not_in_training.coverage_after_median_shift x100")
    put("si_wb_res_mean", u["residual_mean_eV"], ".2f", S + "not_in_training.residual_mean_eV")
    put("si_wb_res_sd", u["residual_sd_eV"], ".2f", S + "not_in_training.residual_sd_eV")
    put("si_wb_res_mean_train", r1["in_training"]["residual_mean_eV"], ".2f", S + "in_training.residual_mean_eV")
    put("si_wb_below_train_pct", 100 * r1["in_training"]["frac_hse_below_interval"], ".0f",
        S + "in_training.frac_hse_below x100")
    put("si_wb_metal_semi", u["n_metal_called_semi_p05"], "d", S + "not_in_training.n_metal_called_semi_p05")
    put("si_wb_n_metal_unseen", u["n_metal_hse"], "d", S + "not_in_training.n_metal_hse")
    put("si_wb_nonmetal_rej", u["n_nonmetal_rejected_p05"], ",d", S + "not_in_training.n_nonmetal_rejected_p05")
    ba = u["by_anion"]
    if min(ba, key=lambda x: ba[x]["coverage"]) != "I" or max(ba, key=lambda x: ba[x]["coverage"]) != "F":
        raise SystemExit("FAIL-CLOSED: prose says iodides have the lowest and fluorides the highest coverage")
    for x in ("I", "F"):
        put(f"si_wb_cov_{x}_pct", 100 * ba[x]["coverage"], ".0f", S + f"not_in_training.by_anion.{x}.coverage x100")
        put(f"si_wb_n_{x}", ba[x]["n"], "d", S + f"not_in_training.by_anion.{x}.n")
    put("si_wb_npv_unseen", r1["not_in_training"]["n_hse_in_pv_window"], "d", S + "not_in_training.n_hse_in_pv_window")
    put("si_wb_n_train_nonmetal", r1["in_training"]["n_nonmetal"], "d", S + "in_training.n_nonmetal")
    put("si_wb_ov_cov_pct", 100 * r1["v2_overlap"]["coverage_hse_by_interval"], ".0f", S + "v2_overlap.coverage x100")
    put("si_wb_ov_off", r1["v2_overlap"]["median_hse_minus_pred_eV"], ".2f", S + "v2_overlap.median_hse_minus_pred_eV")

    # ---------------- Note S13: sensitivity checks (stage s50, R2-R4) ----------------
    om = rc["r3_om_inverter"]
    put("si_inv_year", om["inverter_replace_year"], "d", "pipeline/s50_review_checks.py#INV_REPLACE_YEAR")
    put("si_inv_usd", om["inverter_replace_usd_kwdc"], ".1f", "review_checks.json#r3.inverter_replace_usd_kwdc")
    put("si_om_max_pct", 100 * (max(s50_review_checks.OM_SCALE) - 1), ".0f", "pipeline/s50_review_checks.py#OM_SCALE")
    ox = rc["r4_oxidation"]
    put("si_ox_viable_paper", ox["expected_viable_novel"]["paper_rule"], ".1f", "review_checks.json#r4.expected_viable_novel.paper_rule")
    put("si_ox_opt_paper", ox["expected_competitive_stable_umlip__optimistic"]["paper_rule"], ".1f",
        "review_checks.json#r4.optimistic.paper_rule")
    put("si_ox_opt_strict", ox["expected_competitive_stable_umlip__optimistic"]["strict_rule"], ".1f",
        "review_checks.json#r4.optimistic.strict_rule")
    put("si_ox_ceil_paper", ox["expected_competitive_stable_umlip__ceiling"]["paper_rule"], ".1f",
        "review_checks.json#r4.ceiling.paper_rule")
    put("si_ox_ceil_strict", ox["expected_competitive_stable_umlip__ceiling"]["strict_rule"], ".1f",
        "review_checks.json#r4.ceiling.strict_rule")
    return N


if __name__ == "__main__":
    N = build()
    (ROOT / "paper/si_numbers.json").write_text(json.dumps(N, indent=1, default=float), encoding="utf-8")
    print(len(N), "SI numbers")
