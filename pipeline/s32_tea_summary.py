"""Stage 32: Phase E summary markdown, generated from runs/<id>/metrics/tea_v2.json + gates only."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def build(run_dir: Path) -> str:
    m = json.loads((run_dir / "metrics/tea_v2.json").read_text())
    g = json.loads((run_dir / "gates/gate_tea_v2.json").read_text())
    e1, e2, e3, e4, c, e5, e6 = (m[k] for k in ("e1_reproduction", "e2_csi_ref", "e3_decoupling_test", "e4_breakeven",
                                                 "e4_consistency", "e5_candidates", "e6_sobol"))
    L = ["# Stage 5 summary: TEA v2 (auto-generated from run artifacts)", "",
         "Basis: NREL Q1-2023 utility PV benchmark (100 MWdc single-axis tracker, 2022 USD, minimum sustainable "
         "price), simple real LCOE, 35-year plant. Area-scaled costs (module, structural BOS, installation labour) "
         "are divided by efficiency; power-scaled costs (inverter, electrical BOS, office work) are not.", "",
         "## Engine checks",
         f"- NREL cost structure rebuilt from its components: ${e1['msp']['capex_kwdc']:.0f}/kWdc vs printed "
         f"${e1['msp']['nrel_total']}/kWdc (MSP); ${e1['mmp']['capex_kwdc']:.0f} vs ${e1['mmp']['nrel_total']} (MMP).",
         f"- c-Si reference LCOE: median **${e2['msp_median_usd_mwh']:.1f}/MWh** (P10–P90 "
         f"${e2['msp_p10']:.0f}–{e2['msp_p90']:.0f}); Lazard 2024 utility range ${e2['lazard_utility_range_usd_mwh'][0]}–"
         f"{e2['lazard_utility_range_usd_mwh'][1]}/MWh; IRENA 2023 global average ${e2['irena_2023_global_avg_usd_mwh']}/MWh.",
         f"- Consistency: a module priced like c-Si with c-Si durability breaks even at "
         f"{100 * c['breakeven_eta_csi_equivalent']:.2f}% (c-Si module: {100 * c['eta_csi']:.1f}%).", "",
         "## The April \"thin-film economic decoupling\" claim",
         "| Perovskite module cost | LCOE at 8% ÷ LCOE at 18% | Area-scaled share of capex at 10% |", "|---|---:|---:|"]
    for k, lab in (("low", "$50/m²"), ("med", "$85/m²"), ("high", "$335/m²")):
        L.append(f"| {lab} | {e3[k]['lcoe_8pct_over_18pct']:.2f}× | {e3[k]['area_share_of_capex_at_10pct']:.0%} |")
    L += ["", "LCOE falls strictly with efficiency in every case. **The decoupling claim does not survive an area-aware "
          "cost model.**", "",
          "## Break-even module efficiency (to match the c-Si median)",
          f"- $50/m², 35-year life, 0.7 %/yr: **{100 * e4['base_low_35y_deg0.7']:.1f}%**",
          f"- $85/m², 35-year life, 0.7 %/yr: **{100 * e4['base_med_35y_deg0.7']:.1f}%**",
          f"- {e4['n_unreachable']} of {e4['n_scenarios']} durability/cost scenarios never break even at any "
          "efficiency ≤ 40% (the single-junction SQ limit is 33.7%).", "",
          "## Candidates (1,256 v2 screen survivors, calibrated band-gap posterior, P(semiconductor), stability)",
          f"Efficiency = f_SQ × SQ(gap). f_SQ scenarios: 0.21 = demonstrated Cs₂AgBiBr₆ cell; 0.50 = intermediate; "
          f"**{e5['f_sq_csi_module_parity']:.2f} = today's c-Si module relative to its own SQ limit** (the best a mature "
          "module technology achieves); 0.82 = lead-halide lab-cell ceiling (no cell-to-module loss; not a module value).", "",
          "| Scenario | Definition | Expected # competitive AND stable | … after uMLIP veto | # with P ≥ 0.5 | max P(LCOE ≤ c-Si) |",
          "|---|---|---:|---:|---:|---:|"]
    defs = {"demonstrated": "f 0.21, $50/m², 35 y, 0.7 %/yr", "realistic": f"f {e5['f_sq_csi_module_parity']:.2f}, $85/m², 15 y, 2 %/yr",
            "optimistic": f"f {e5['f_sq_csi_module_parity']:.2f}, $50/m², 35 y, 0.7 %/yr", "ceiling": "f 0.82, $50/m², 35 y, 0.7 %/yr"}
    for k in ("demonstrated", "realistic", "optimistic", "ceiling"):
        L.append(f"| {k} | {defs[k]} | {e5['expected_n_competitive_and_stable'][k]:.1f} | "
                 f"{e5['expected_n_competitive_and_stable_umlip'][k]:.1f} | {e5['n_candidates_p_ge_0.5'][k]} | "
                 f"{e5['max_p_competitive'][k]:.2f} |")
    L += ["", f"uMLIP veto: {e5['n_umlip_veto']} of {e5['n_umlip_checked']} MACE-checked compounds are > 50 meV above "
          "the other phases and are zeroed.", "",
          "| Top candidates, optimistic scenario | MP status | P(LCOE ≤ c-Si) optimistic | … ceiling | P(stable) | MACE E_hull (meV) |",
          "|---|---|---:|---:|---:|---:|"]
    best, ceil = e5["scenario_definitions"]["optimistic"], e5["scenario_definitions"]["ceiling"]
    for r in e5["top10_optimistic"]:
        u = r["umlip_ehull_meV"]
        L.append(f"| {r['formula']} | {r['status']} | {r[best]:.2f} | {r[ceil]:.2f} | {r['p_stable50']:.2f} | "
                 f"{'–' if u is None or u != u else f'{u:.0f}'} |")
    L += ["", f"Rank agreement between the optimistic and ceiling scenarios: Spearman ρ = "
          f"{e5['rank_spearman_optimistic_vs_ceiling']:.2f}. **The candidate ranking depends on the efficiency assumption.**",
          "",
          "Caveat on the ordering: K-site compounds dominate the top of the list partly because Cs and Rb have no "
          "market price (USGS). Reagent-grade prices are used for them, which adds about $4–7/m² of absorber cost "
          "and roughly 2–3% of LCOE. At probabilities this low, that is enough to reorder the list, so the order "
          "carries no weight.", "",
          "## What drives LCOE (Sobol total-order indices, band gap 1.34 eV)", "| Parameter | S_T |", "|---|---:|"]
    for r in e6["ranking_ST"]:
        L.append(f"| {r['param']} | {r['ST']:.3f} |")
    L += ["", f"Absorber raw-material cost is immaterial: the most expensive composition (${e5['absorber_cost_m2_max']:.1f}/m²) "
          f"changes LCOE by {100 * c['absorber_max_cost_effect_rel']:.1f}%. Repeating half the installation labour at each "
          f"module replacement raises the 10-year-module LCOE by {100 * c['replace_labor_effect_rel_L10']:.1f}%.", "",
          "## Gate", f"Phase E: **{'PASS' if g['pass'] else 'FAIL'}** "
          f"({sum(v['pass'] for v in g['checks'].values())}/{len(g['checks'])})", ""]
    return "\n".join(L)


if __name__ == "__main__":
    rd = Path(sys.argv[1])
    txt = build(rd)
    (rd / "SUMMARY_stage5.md").write_text(txt, encoding="utf-8")
    print(txt)
