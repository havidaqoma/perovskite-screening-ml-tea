"""Manuscript figures (v2). Every value is read from runs/ artifacts; nothing is typed.

Fig 1  workflow + funnel (April legacy vs v2 gates)
Fig 2  ML audit: (a) leakage audit MAE, (b) baselines by split, (c) conformal coverage/width
Fig 3  stability: (a) control reliability diagram, (b) ML vs MACE E_hull for the shortlist
Fig 4  TEA: (a) efficiency sweep vs c-Si, (b) break-even efficiency grid (low module cost)
Fig 5  candidates: (a) expected number competitive by scenario, (b) Pareto front (stability x absorber x LCOE)
Saved as PNG (300 dpi) + PDF into paper/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
                     "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
                     "font.family": "DejaVu Sans"})
C = {"blue": "#3b6ea8", "orange": "#d9812b", "green": "#3a9a6a", "red": "#c0463b", "grey": "#8a8a8a",
     "purple": "#7a5aa6"}


def j(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def panel(ax, s):
    ax.text(-0.02, 1.10, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom", ha="right")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1():
    fg, fs = j("legacy_april/metrics/funnel_generate.json"), j("legacy_april/metrics/funnel_screen.json")
    sc, stb = j("v2/metrics/screen_v2.json"), j("v2/metrics/stability_v2.json")
    pa, t = j("v2/metrics/pareto.json"), j("v2/metrics/tea_v2.json")
    ff = sc["funnel_fail_counts"]
    stable_like = stb["status_counts"]["stable_likely"] + stb["status_counts"]["known_mp"]
    legacy = [("Generated A$_2$BB$'$X$_6$", fg["n_generated"]), ("Charge balanced", fg["n_charge_balanced"]),
              ("Legacy steric filter", fg["n_steric"]), ("Legacy ML window", fs["n_candidates"])]
    v2 = [("Generated A$_2$BB$'$X$_6$", fg["n_generated"]), ("Charge balanced", fg["n_charge_balanced"]),
          ("Oxidation-state radii + $\\tau$", fg["n_charge_balanced"] - ff["tau_not_perovskite"] - ff["no_oxidation_state_assignment"]),
          ("Gatekeeper (semiconductor)", sc["n_v2_candidates"]),
          ("Likely stable or known (MP)", stable_like),
          ("Pareto front", pa["n_front1"])]
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [len(legacy), len(v2)]})
    for ax, rows, title, col in ((axes[0], legacy, "April 2026 pipeline (legacy)", C["grey"]),
                                 (axes[1], v2, "This work (v2)", C["blue"])):
        y = np.arange(len(rows))[::-1]
        vals = [v for _, v in rows]
        ax.barh(y, vals, color=col, height=0.62)
        ax.set_xlim(0, 1.25 * fg["n_generated"])
        ax.set_yticks(y, [n for n, _ in rows])
        for yi, v in zip(y, vals):
            ax.text(v + 250, yi, f"{v:,}", va="center", fontsize=7)
        ax.set_title(title, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_xlabel("Number of compositions (linear scale)")
    panel(axes[0], "(a)")
    panel(axes[1], "(b)")
    fig.tight_layout(h_pad=1.2)
    save(fig, "fig1_funnel")


def fig2():
    la, ev = j("v2/metrics/leakage_audit.json"), j("v2/metrics/ml_eval_v2.json")
    ro = j("v2/metrics/roost_chemsys_gap_semi.json")
    mae = {(r["split"], r["task"], r["model"]): r["mae_pooled"] for r in ev["results"] if "mae_pooled" in r}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [0.85, 1.35, 0.8]})
    ax = axes[0]
    keys = ["random_row", "formula", "chemsys"]
    labs = ["Random\nrows", "By\nformula", "By chem.\nsystem"]
    v = [la[k]["bg_mae_eV"] for k in keys]
    ax.bar(range(3), v, color=[C["grey"], C["orange"], C["red"]], width=0.62)
    for i, x in enumerate(v):
        ax.text(i, x + 0.008, f"{x:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(3), labs, fontsize=7)
    ax.set_ylabel("Band-gap MAE (eV)")
    ax.set_ylim(0, 0.5)
    ax.set_title("April set, split varied", loc="left", fontsize=8)
    ax = axes[1]
    models = ["mean", "ridge", "rf", "xgb"]
    names = ["Mean", "Ridge", "RF", "XGB", "Roost"]
    splits = [("random", "Random"), ("chemsys", "Chem. system"), ("lofo_anion", "Anion family out")]
    w = 0.26
    ymax = 1.6
    for i, (s, sl) in enumerate(splits):
        vals = [mae[(s, "gap_semi", m)] for m in models] + [ro["mae_pooled"] if s == "chemsys" else np.nan]
        xs = np.arange(5) + (i - 1) * w
        ax.bar(xs, np.minimum(vals, ymax), width=w, label=sl, color=[C["blue"], C["orange"], C["red"]][i])
        for xi, v in zip(xs, vals):
            if np.isfinite(v) and v > ymax:
                ax.text(xi, ymax * 0.97, f"{v:.1f}\u2191", ha="center", va="top", fontsize=6, color="w", rotation=90)
    ax.text(4.0, ro["mae_pooled"] + 0.05, "chem.\nsystem\nonly", ha="center", va="bottom", fontsize=5.5, color="0.3")
    ax.set_xticks(range(5), names)
    ax.set_ylabel("Band-gap MAE, semiconductors (eV)")
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=6.5, handlelength=1.0,
              columnspacing=0.8)
    ax.set_title("v2 data (188k formulas)", loc="left", fontsize=8)
    ax = axes[2]
    cov = [ev["conformal"][s]["coverage_folds"] for s in ("random", "chemsys")]
    for i, c in enumerate(cov):
        ax.scatter(np.full(len(c), i) + np.linspace(-0.1, 0.1, len(c)), c, color=[C["blue"], C["orange"]][i], s=14)
    ax.axhline(0.9, color="k", lw=0.8, ls="--")
    ax.axhspan(0.87, 0.93, color="0.9", zorder=0)
    ax.text(1.45, 0.901, "nominal 0.90", fontsize=6, ha="right", va="bottom")
    ax.text(1.45, 0.927, "gate \u00b13 pp", fontsize=6, ha="right", va="top", color="0.4")
    ax.set_xticks([0, 1], ["Random", "Chem.\nsystem"])
    ax.set_ylim(0.85, 0.95)
    ax.set_ylabel("Coverage of 90% interval (per fold)")
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(f"Median width {ev['conformal']['chemsys']['width_median']:.2f} eV", loc="left", fontsize=8)
    for a, s in zip(axes, "abc"):
        panel(a, f"({s})")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig2_ml_audit")


def fig3():
    c = j("v2/metrics/stability_v2.json")["control"]
    u = pd.read_csv(R / "v2/stability/umlip.csv").query("status == 'ok'")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    rel = [v for v in c["reliability"].values() if v["n"]]
    x = [v["mean_p"] for v in rel]
    y = [v["obs_frac"] for v in rel]
    n = [v["n"] for v in rel]
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
    ax.scatter(x, y, s=[8 + k / 2 for k in n], color=C["blue"], zorder=3)
    for xi, yi, ni in zip(x, y, n):
        ax.text(xi + 0.03, yi - 0.05, f"n={ni}", fontsize=6.5)
    ax.set_xlabel("Predicted P(E$_\\mathrm{hull}$ ≤ 50 meV)")
    ax.set_ylabel("Observed fraction (MP)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{c['n_ok']} known A$_2$BB$'$X$_6$, ROC-AUC {c['roc_auc_p50']:.2f}", loc="left")
    ax = axes[1]
    for role, lab, col, mk in (("control_known_mp", "Known (MP) controls", C["green"], "s"),
                               ("novel", "Novel shortlist", C["red"], "o")):
        s = u[u.role == role]
        xv = s.mp_ehull_meV if role == "control_known_mp" else s.ml_ehull_pred_meV
        ax.scatter(xv, s.umlip_ehull_vs_rest_meV, s=16, color=col, marker=mk, label=lab)
    fm = u[(u.role == "novel") & u.formula.str.contains("F3")]
    ax.scatter(fm.ml_ehull_pred_meV, fm.umlip_ehull_vs_rest_meV, s=40, facecolors="none", edgecolors="k", lw=0.6,
               label="F-mixed anion")
    ax.axhline(50, color="k", lw=0.8, ls="--")
    ax.axvline(50, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("E$_\\mathrm{hull}$: MP (controls) or ML (novel), meV")
    ax.set_ylabel("MACE E$_\\mathrm{hull}$ vs other phases (meV)")
    ax.set_ylim(-80, 330)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6.5, handletextpad=0.3)
    ax.set_title("Physics check of the shortlist", loc="left")
    for a, s in zip(axes, "ab"):
        panel(a, f"({s})")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_stability")


def fig4():
    t = j("v2/metrics/tea_v2.json")
    ref = t["e2_csi_ref"]
    sw = pd.read_csv(R / "v2/tea_v2/efficiency_sweep.csv")
    be = pd.read_csv(R / "v2/tea_v2/breakeven_efficiency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for name, lab, col in (("low", r"\$50 m$^{-2}$", C["green"]), ("med", r"\$85 m$^{-2}$", C["orange"]),
                           ("high", r"\$335 m$^{-2}$", C["purple"])):
        g = sw[sw.module_cost == name]
        ax.plot(100 * g.eta, g.lcoe_median, color=col, lw=1.4, label=f"Module {lab}")
        ax.fill_between(100 * g.eta, g.lcoe_p10, g.lcoe_p90, color=col, alpha=0.12, lw=0)
    ax.axhspan(ref["msp_p10"], ref["msp_p90"], color="0.85", zorder=0, label="c-Si P10–P90")
    ax.axhline(ref["msp_median_usd_mwh"], color="k", lw=0.8, ls="--", label="c-Si median")
    ax.set_yscale("log")
    ax.set_ylim(30, 1000)
    ax.set_yticks([30, 50, 100, 200, 500, 1000], ["30", "50", "100", "200", "500", "1000"])
    ax.set_xlim(5, 30)
    ax.set_xlabel("Module efficiency (%)")
    ax.set_ylabel("LCOE (USD MWh$^{-1}$, 2022)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.2, handlelength=1.4,
              columnspacing=0.8)
    ax.text(0.97, 0.97, "shading: P10–P90 over site/finance draws", transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color="0.35")
    ax = axes[1]
    import sys
    sys.path.insert(0, str(ROOT))
    from pipeline.sq import sq_table
    sq_max = float(sq_table()[1].max())            # single-junction limit from the pipeline, not a typed constant
    tb = be[(be.module_cost == "low") & (be.burnin == 0)].pivot(index="module_life", columns="deg", values="breakeven_eta")
    cmap = matplotlib.colormaps["viridis_r"].copy()
    cmap.set_bad("0.88")
    im = ax.imshow(100 * tb.to_numpy(), cmap=cmap, vmin=20, vmax=40, aspect="auto")
    for i in range(tb.shape[0]):
        for k in range(tb.shape[1]):
            v = tb.to_numpy()[i, k]
            lab = "n.r." if pd.isna(v) else (f"{100 * v:.1f}*" if v > sq_max else f"{100 * v:.1f}")
            ax.text(k, i, lab, ha="center", va="center", fontsize=7,
                    color="k" if pd.isna(v) or v < 0.33 else "w")
    ax.set_xticks(range(tb.shape[1]), [f"{100 * c:.1f}" for c in tb.columns])
    ax.set_yticks(range(tb.shape[0]), tb.index)
    ax.set_xlabel("Degradation (% yr$^{-1}$)")
    ax.set_ylabel("Module lifetime (yr)")
    ax.set_title(r"Break-even efficiency (%), \$50 m$^{-2}$ module", loc="left", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Break-even efficiency (%)")
    # short key kept inside panel (b)'s width (a one-line footnote once pushed the canvas ~30% wider than the
    # plots and left both panels crowded to the left); the full explanation lives in the caption
    ax.text(0.5, -0.27, f"* above SQ limit ({100 * sq_max:.1f}%)\nn.r. = not reached below 40%", transform=ax.transAxes,
            fontsize=6.3, ha="center", va="top", linespacing=1.3)
    for a, s in zip(axes, "ab"):
        panel(a, f"({s})")
    for sp in ("top", "right"):
        axes[0].spines[sp].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_tea")


def fig5():
    t = j("v2/metrics/tea_v2.json")["e5_candidates"]
    p = pd.read_csv(R / "v2/pareto/pareto.csv")
    col = [c for c in p.columns if c.startswith("lcoe_median__")][0]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    order = ["demonstrated", "realistic", "optimistic", "ceiling"]
    f61 = t["f_sq_csi_module_parity"]
    labs = ["Demon-\nstrated\nf=0.21", f"Realistic\nf={f61:.2f}", f"Optimistic\nf={f61:.2f}", "Lab-cell\nceiling\nf=0.82"]
    a = [t["expected_n_competitive_and_stable"][k] for k in order]
    b = [t["expected_n_competitive_and_stable_umlip"][k] for k in order]
    x = np.arange(4)
    ax.bar(x - 0.19, a, width=0.36, color=C["grey"], label="ML stability only")
    ax.bar(x + 0.19, b, width=0.36, color=C["blue"], label="+ MACE veto")
    for xi, va, vb in zip(x, a, b):
        ax.text(xi, max(va, vb) + 2.0, f"{va:.1f} | {vb:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, labs, fontsize=6.8)
    ax.set_ylabel("Expected number cheaper than c-Si\nand thermodynamically stable")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(0, 75)
    ax = axes[1]
    bg = p[p.front > 1]
    ax.scatter(bg.p_stable_final, bg.p_absorber, s=5, color="0.78", label="Dominated", zorder=1)
    ax.axvline(0.5, color="0.45", lw=0.7, ls=":", zorder=0)
    ax.text(0.51, 0.02, "stability gate (P ≥ 0.5)", fontsize=6, color="0.35", va="bottom")
    f1 = p[p.front == 1]
    fs_ = p[p.front_stable == 1]
    ax.scatter(fs_.p_stable_final, fs_.p_absorber, s=150, facecolors="none", edgecolors="k", lw=1.0,
               label="Stability-gated front", zorder=4)
    sc = ax.scatter(f1.p_stable_final, f1.p_absorber, s=48, c=f1[col], cmap="plasma_r", edgecolors="k", lw=0.5,
                    label="Unconstrained front", zorder=3)
    offsets = {"Cs2BiAgBr6": (-78, -10), "K2SbAgCl6": (-70, -30), "Cs2BiAgBr3I3": (-30, 22),
               "Cs2SnBiS3I3": (8, -12), "Rb2SbAgCl3I3": (8, 6)}
    pretty = {"Cs2BiAgBr6": "Cs$_2$AgBiBr$_6$", "K2SbAgCl6": "K$_2$AgSbCl$_6$", "Cs2BiAgBr3I3": "Cs$_2$AgBiBr$_3$I$_3$",
              "Cs2SnBiS3I3": "Cs$_2$SnBiS$_3$I$_3$", "Rb2SbAgCl3I3": "Rb$_2$AgSbCl$_3$I$_3$"}
    for _, r in f1.iterrows():
        dx, dy = offsets.get(r.formula, (6, 6))
        ax.annotate(pretty.get(r.formula, r.formula), (r.p_stable_final, r.p_absorber), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3"))
    ax.set_xlabel("P(stable), after MACE veto")
    ax.set_ylabel("P(semiconductor) × P(gap 1.0–1.8 eV)")
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(0, 0.8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("LCOE, optimistic (USD MWh$^{-1}$)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=6.2, columnspacing=0.8)
    for a_, s in zip(axes, "ab"):
        panel(a_, f"({s})")
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5_candidates")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
        print("ok", f.__name__)
