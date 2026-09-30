"""Stage 23: human-readable summary of Stages 3-4 (Phase C + D) from run artifacts only.

Every number in the output is read from runs/<id>/metrics/*.json, gates/*.json or stability/*.csv.
Nothing is typed by hand. Output: runs/<id>/SUMMARY_stage3_4.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def build(run_dir: Path) -> str:
    ev = _j(run_dir / "metrics/ml_eval_v2.json")
    leak = _j(run_dir / "metrics/leakage_audit.json")
    ds = _j(run_dir / "metrics/dataset_v2.json")
    scr = _j(run_dir / "metrics/screen_v2.json")
    stab = _j(run_dir / "metrics/stability_v2.json")
    sl = _j(run_dir / "metrics/shortlist_v2.json")
    g1 = _j(run_dir / "gates/gate_ml_audit.json")
    g2 = _j(run_dir / "gates/gate_stability.json")
    roost = {p.stem.replace("roost_", ""): _j(p)["mae_pooled"] for p in (run_dir / "metrics").glob("roost_*.json")}
    L = ["# Stages 3–4 summary (auto-generated from run artifacts)", ""]

    L += ["## Data", f"- MP database {ds['mp_queries'][0]['mp_database_version']}: {ds['n_raw_entries']:,} entries → "
          f"**{ds['n_formulas']:,} formulas** (ground-state polymorph each), {ds['frac_semi']:.1%} semiconductors (gap ≥ 0.1 eV).", ""]

    L += ["## Leakage audit (April data, April model, only the split changes)", "| Split | Bandgap MAE (eV) | Val formulas also in train |", "|---|---:|---:|"]
    for k, lab in (("random_row", "April protocol (random rows)"), ("formula", "Grouped by formula"), ("chemsys", "Grouped by chemical system")):
        L.append(f"| {lab} | {leak[k]['bg_mae_eV']:.3f} | {leak[k]['val_formula_in_train']:.1%} |")
    L.append("")

    mae = {(r["split"], r["task"], r["model"]): r.get("mae_pooled") for r in ev["results"]}
    L += ["## Baselines on v2 (pooled out-of-fold MAE, eV; Ef in eV/atom)",
          "| Split | Task | Mean | Ridge | RF | XGB | Roost |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in ("random", "chemsys", "lofo_anion"):
        for t in ("gap_semi", "gap_all", "ef"):
            ro = roost.get(f"{s}_{t}")
            L.append(f"| {s} | {t} | " + " | ".join(f"{mae[(s, t, m)]:.3f}" for m in ("mean", "ridge", "rf", "xgb"))
                     + f" | {ro:.3f} |" if ro else f"| {s} | {t} | " + " | ".join(f"{mae[(s, t, m)]:.3f}" for m in ("mean", "ridge", "rf", "xgb")) + " | – |")
    cls = {r["split"]: r for r in ev["results"] if r["task"] == "metal_cls"}
    L += ["", "Gatekeeper ROC-AUC: " + ", ".join(f"{k} {v['roc_auc']:.3f}" for k, v in cls.items()), ""]
    c = ev["conformal"]
    L += [f"Conformal 90% gap intervals: coverage {c['random']['coverage_pooled']:.3f} (random), "
          f"{c['chemsys']['coverage_pooled']:.3f} (chemsys); median width {c['chemsys']['width_median']:.2f} eV.", ""]

    ft = scr["funnel_fail_counts"]
    L += ["## Screening funnel (v2)", f"- Charge-balanced: {scr['n_charge_balanced']:,}",
          f"- Rejected: τ not perovskite {ft.get('tau_not_perovskite', 0):,}; no oxidation-state assignment "
          f"{ft.get('no_oxidation_state_assignment', 0):,}; predicted metal {ft.get('predicted_metal', 0):,}; toxic {ft.get('toxic', 0):,}",
          f"- **Pass: {scr['n_v2_candidates']:,}** (April: {scr['n_legacy_steric_pass']:,} passed the legacy steric filter; "
          f"of the April 3,280 candidates, {scr['april_candidates_trace'].get('pass', 0):,} pass v2 geometry+gatekeeper)", ""]

    ct = stab["control"]
    L += ["## Stability (ML hull distance vs MP competing phases)", f"- Status: {stab['status_counts']}",
          f"- Candidates with P(E_hull ≤ thr) ≥ 0.5: {stab['n_p50_ge_0.5_by_thr']}",
          f"- **Control (380 known MP A₂BB′X₆, hull rebuilt without them):** ROC-AUC {ct['roc_auc_p50']:.3f}, "
          f"Brier {ct['brier_p50']:.3f}, gate precision {ct['gate_precision']:.2f}, recall {ct['gate_recall']:.2f}, "
          f"E_hull MAE {ct['ehull_mae_meV_vs_rest']:.0f} meV", "", "| P(stable) bin | n | mean predicted | observed |", "|---|---:|---:|---:|"]
    for k, v in ct["reliability"].items():
        L.append(f"| {k} | {v['n']} | {v['mean_p']:.2f} | {v['obs_frac']:.2f} |")
    L.append("")

    L += ["## Joint viability  P(semi) × P(gap 1.0–1.8 eV) × P(E_hull ≤ 50 meV)",
          f"- Novel (not in MP): {sl['n_novel']:,}; flagged hull extrapolation {sl['n_flag_hull_extrapolation']}, "
          f"implausible oxidation state {sl['n_flag_ox_implausible']}",
          f"- Expected number of viable novel compounds (sum of probabilities, plausible only): **{sl['expected_viable_novel_plausible']:.1f}**",
          f"- Plausible novel with p ≥ 0.1/0.2/0.3/0.5: {sl['n_novel_plausible_p_viable_ge']}", "",
          "| Top novel candidates | Oxidation states | P(semi) | P(gap PV) | P(stable) | P(viable) |", "|---|---|---:|---:|---:|---:|"]
    for r in sl["top10_novel"]:
        L.append(f"| {r['formula']} | {r['ox_states']} | {r['p_semi']:.2f} | {r['p_gap_pv']:.2f} | {r['p_stable50']:.2f} | {r['p_viable_plausible']:.2f} |")
    L.append("")

    u = run_dir / "stability/umlip.csv"
    if u.exists():
        ud = pd.read_csv(u)
        ok = ud[ud.status == "ok"]
        L += ["## uMLIP check (MACE-MP-0, rock-salt ordered prototype, hull vs other phases re-relaxed with MACE)",
              f"- Finished {len(ok)}/{len(ud)}"]
        for role in ("control_known_mp", "novel"):
            sub = ok[ok.role == role]
            if len(sub):
                L.append(f"- {role}: n={len(sub)}, uMLIP E_hull ≤ 50 meV for {(sub.umlip_ehull_vs_rest_meV <= 50).mean():.0%}; "
                         f"median {sub.umlip_ehull_vs_rest_meV.median():.0f} meV")
        ctrl = ok[ok.role == "control_known_mp"]
        if len(ctrl):
            agree = ((ctrl.umlip_ehull_vs_rest_meV <= 50) == (ctrl.mp_ehull_meV <= 50)).mean()
            L.append(f"- Agreement with MP on known controls (stable ≤ 50 meV yes/no): {agree:.0%}")
        nov = ok[ok.role == "novel"]
        if len(nov):
            both = ((nov.umlip_ehull_vs_rest_meV <= 50) & (nov.ml_ehull_pred_meV <= 50)).mean()
            L.append(f"- Novel: ML and uMLIP both ≤ 50 meV for {both:.0%}")
        L.append("")

    L += ["## Gates", f"- Phase C (ML audit): **{'PASS' if g1 and g1['pass'] else 'FAIL/absent'}**",
          f"- Phase D (stability): **{'PASS' if g2 and g2['pass'] else 'FAIL/absent'}**", ""]
    return "\n".join(L)


if __name__ == "__main__":
    rd = Path(sys.argv[1])
    txt = build(rd)
    (rd / "SUMMARY_stage3_4.md").write_text(txt, encoding="utf-8")
    print(txt)
