"""Supplementary figures (ESI). Every value is read from runs/ artifacts or computed by pipeline code; nothing typed.

Fig S1  conformal band-gap intervals: (a) coverage per fold, (b) coverage by DFT-gap bin, (c) interval widths
Fig S2  geometry: (a) Bartel tau of the charge-balanced space, (b) fate of the original candidates under the v2 gates
Fig S3  stability model on known A2BB'X6 controls: (a) ML vs MP hull distance, (b) gate-threshold sensitivity
Fig S4  detailed-balance limit and the f_SQ efficiency scenarios
Fig S5  break-even module efficiency for every module cost and burn-in setting
Fig S6  candidate-level LCOE: (a) median LCOE vs predicted gap, (b) P(LCOE <= c-Si) under two scenarios
Fig S7  Sobol first-order and total indices of perovskite LCOE
Fig S8  predicted band gap against the Walterbos et al. HSE06 gap (stage s50, check R1)
(Numbered in the order the ESI cites them.) Fig S1 needs runs/v2/predictions/oof_v2.csv, a regenerable s11 output
that the public repository does not ship; without it Fig S1 is skipped and its committed PNG/PDF/CSV are kept.
PNG (300 dpi) + PDF -> paper/figures/figS*.{png,pdf}; plotted data -> paper/figures/si_data/figS*.csv
(the ESI workbook carries these CSVs, so every SI figure has its raw data).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from make_figures_v2 import C, OUT, R, j, panel, save  # noqa: E402  (same style, colours and save helper)

DATA = OUT / "si_data"
DATA.mkdir(parents=True, exist_ok=True)


def dump(df: pd.DataFrame, name: str) -> None:
    df.to_csv(DATA / f"{name}.csv", index=False, float_format="%.6g")


def despine(*axes):
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)


def figS1():
    ev = j("v2/metrics/ml_eval_v2.json")["conformal"]
    rb = j("v2/metrics/robustness.json")
    o = pd.read_csv(R / "v2/predictions/oof_v2.csv",
                    usecols=["band_gap", "is_semi", "chemsys__conformal__lo", "chemsys__conformal__hi"])
    o = o[(o.is_semi == 1) & o.chemsys__conformal__lo.notna()].copy()
    o["covered"] = (o.band_gap >= o.chemsys__conformal__lo) & (o.band_gap <= o.chemsys__conformal__hi)
    o["width"] = o.chemsys__conformal__hi - o.chemsys__conformal__lo
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [0.8, 1.2, 1.0]})
    ax = axes[0]
    rows = []
    for i, (s, col) in enumerate((("random", C["blue"]), ("chemsys", C["orange"]))):
        cov = ev[s]["coverage_folds"]
        ax.scatter(np.full(len(cov), i) + np.linspace(-0.12, 0.12, len(cov)), cov, s=16, color=col)
        rows += [{"panel": "a", "split": s, "fold": k, "coverage": c} for k, c in enumerate(cov)]
    ax.axhline(1 - ev["chemsys"]["alpha"], color="k", lw=0.8, ls="--")
    ax.set_xticks([0, 1], ["Random", "Chemical\nsystem"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.85, 0.95)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Per fold (nominal 0.90)", loc="left", fontsize=8)
    ax = axes[1]
    bins = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 12.0]
    o["bin"] = pd.cut(o.band_gap, bins, right=False)
    g = o.groupby("bin", observed=True).agg(n=("covered", "size"), coverage=("covered", "mean"),
                                            width_median=("width", "median")).reset_index()
    labels = [f"{b.left:g}–{b.right:g}" for b in g["bin"]]
    ax.bar(range(len(g)), g.coverage, color=C["orange"], width=0.65)
    ax.axhline(0.9, color="k", lw=0.8, ls="--")
    for k, (c, n) in enumerate(zip(g.coverage, g.n)):
        ax.text(k, 0.04, f"n={n:,}", ha="center", fontsize=5.6, rotation=90, va="bottom", color="w")
    ax.set_xticks(range(len(g)), labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("DFT band gap (eV)")
    ax.set_ylabel("Coverage, chemical-system folds")
    ax.set_title("By gap range (n inside bars)", loc="left", fontsize=8)
    rows += [{"panel": "b", "gap_bin_eV": lab, "n": int(n), "coverage": c, "width_median_eV": w}
             for lab, n, c, w in zip(labels, g.n, g.coverage, g.width_median)]
    ax = axes[2]
    ax.hist(o.width, bins=np.arange(0, 8.01, 0.25), color=C["grey"])
    ax.axvline(ev["chemsys"]["width_median"], color=C["red"], lw=1.0, label="median")
    ax.axvline(ev["chemsys"]["width_p90"], color=C["red"], lw=1.0, ls=":", label="90th percentile")
    ax.set_xlabel("90% interval width (eV)")
    ax.set_ylabel("Compounds")
    ax.legend(loc="upper right", fontsize=6.2)
    ax.set_title("Interval width", loc="left", fontsize=8)
    cnt, edges = np.histogram(o.width, bins=np.arange(0, 8.01, 0.25))
    rows += [{"panel": "c", "width_lo_eV": a, "width_hi_eV": b, "n": int(c)} for a, b, c in zip(edges[:-1], edges[1:], cnt)]
    rows += [{"panel": "note", "split": f"anion_families={k}", "n": v["n"], "coverage": v["coverage"]}
             for k, v in rb["conformal_by_anion_mix"].items()]
    for a, s in zip(axes, "abc"):
        panel(a, f"({s})")
    despine(*axes)
    fig.tight_layout(w_pad=1.6)
    save(fig, "figS1_conformal")
    dump(pd.DataFrame(rows), "figS1_conformal")


def figS2():
    s = pd.read_csv(R / "v2/screen/v2_all_charge_balanced.csv")
    sc = j("v2/metrics/screen_v2.json")
    tau = s[np.isfinite(s.tau)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    edges = np.arange(3.0, 8.01, 0.1)
    for flag, lab, col in ((True, "accepted by legacy steric filter", C["blue"]),
                           (False, "rejected by legacy steric filter", C["grey"])):
        ax.hist(tau[tau.legacy_steric_pass == flag].tau.clip(upper=8.0), bins=edges, color=col, alpha=0.65, label=lab)
    ax.axvline(4.18, color="k", lw=0.9, ls="--")
    ax.text(4.12, ax.get_ylim()[1] * 0.97, "τ = 4.18", ha="right", va="top", fontsize=6.5)
    cs = s[s.formula == "Cs2BiAgBr6"].iloc[0]
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.25)                     # headroom so the control label sits above the bars
    ax.annotate(f"Cs$_2$AgBiBr$_6$ (τ = {cs.tau:.2f})", (cs.tau, ymax * 0.02), xytext=(3.1, ymax * 1.12),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("Bartel tolerance factor τ (values above 8 pooled)")
    ax.set_ylabel("Charge-balanced compositions")
    ax.legend(loc="upper right", fontsize=6.2)
    ax = axes[1]
    tr = sc["april_candidates_trace"]
    order = [("pass", "passes all v2 gates"), ("tau_not_perovskite", "τ ≥ 4.18"),
             ("no_oxidation_state_assignment", "no oxidation-state\nassignment"), ("predicted_metal", "predicted metal")]
    vals = [tr.get(k, 0) for k, _ in order]
    ax.barh(range(len(order)), vals, color=[C["green"], C["red"], C["orange"], C["purple"]])
    for k, v in enumerate(vals):
        ax.text(v + 30, k, f"{v:,}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(order)), [lab for _, lab in order], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("Original candidates")
    ax.set_title("Original candidates under the v2 gates", loc="left", fontsize=8)
    for a, s_ in zip(axes, "ab"):
        panel(a, f"({s_})")
    despine(*axes)
    fig.tight_layout(w_pad=2.0)
    save(fig, "figS2_geometry")
    cnt = [{"panel": "a", "legacy_steric_pass": flag, "tau_lo": a, "tau_hi": b, "n": int(c)}
           for flag in (True, False)
           for a, b, c in zip(edges[:-1], edges[1:], np.histogram(tau[tau.legacy_steric_pass == flag].tau.clip(upper=8.0),
                                                                  bins=edges)[0])]
    cnt += [{"panel": "b", "stage_fail": k, "n": v} for k, v in zip([k for k, _ in order], vals)]
    dump(pd.DataFrame(cnt), "figS2_geometry")


def figS3():
    c = pd.read_csv(R / "v2/stability/control_known_a2bbx6.csv").dropna(subset=["true_ehull_vs_rest_meV"])
    st = j("v2/metrics/stability_v2.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    lim = (-150, 400)
    stable = c.true_ehull_meV <= 50
    ax.scatter(c.true_ehull_vs_rest_meV[~stable], c.ehull_pred_meV[~stable], s=7, color=C["grey"],
               label="MP E$_\\mathrm{hull}$ > 50 meV")
    ax.scatter(c.true_ehull_vs_rest_meV[stable], c.ehull_pred_meV[stable], s=7, color=C["blue"],
               label="MP E$_\\mathrm{hull}$ ≤ 50 meV")
    ax.plot(lim, lim, color="k", lw=0.8, ls="--")
    ax.axhline(50, color="0.5", lw=0.6, ls=":")
    ax.axvline(50, color="0.5", lw=0.6, ls=":")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("MP hull distance vs other phases (meV atom$^{-1}$)")
    ax.set_ylabel("ML hull distance (meV atom$^{-1}$)")
    ax.legend(loc="upper left", fontsize=6.2)
    n_out = int(((c.ehull_pred_meV.abs() > 400) | (c.true_ehull_vs_rest_meV.abs() > 400)).sum())
    ax.set_title(f"{len(c)} controls, {n_out} outside the axes", loc="left", fontsize=8)
    ax = axes[1]
    thr = st["n_p50_ge_0.5_by_thr"]
    ks = sorted(thr, key=lambda k: int(k))
    ax.bar(range(len(ks)), [thr[k] for k in ks], color=[C["orange"] if k == "50" else C["grey"] for k in ks])
    for k_, k in enumerate(ks):
        ax.text(k_, thr[k] + 6, str(thr[k]), ha="center", fontsize=6.5)
    ax.set_xticks(range(len(ks)), ks)
    ax.set_xlabel("Hull-distance threshold (meV atom$^{-1}$)")
    ax.set_ylabel("Candidates with P(stable) ≥ 0.5")
    ax.set_ylim(0, max(thr.values()) * 1.15)
    ax.set_title(f"All {st['n']:,} v2 candidates", loc="left", fontsize=8)
    for a, s in zip(axes, "ab"):
        panel(a, f"({s})")
    despine(*axes)
    fig.tight_layout(w_pad=2.0)
    save(fig, "figS3_stability")
    d = c[["formula", "material_id", "ehull_pred_meV", "true_ehull_vs_rest_meV", "true_ehull_meV", "p_ehull_le_50"]].assign(panel="a")
    d = pd.concat([d, pd.DataFrame([{"panel": "b", "threshold_meV": int(k), "n_p_ge_0.5": thr[k]} for k in ks])])
    dump(d, "figS3_stability")


def figS4():
    from pipeline.sq import sq_limit, sq_table
    from pipeline.tea_params import P
    egs, eff = sq_table()
    f_csi = round(P["eta_csi"]["value"] / float(sq_limit(1.12)), 2)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.axvspan(1.0, 1.8, color="0.92", zorder=0, label="PV window 1.0–1.8 eV")
    ax.plot(egs, 100 * eff, color="k", lw=1.4, label="Detailed-balance limit")
    rows = [{"eg_eV": e, "sq_pct": 100 * v} for e, v in zip(egs, eff)]
    for f, col in ((0.82, C["purple"]), (f_csi, C["blue"]), (0.21, C["orange"])):
        ax.plot(egs, 100 * f * eff, color=col, lw=1.1, ls="--", label=f"f$_\\mathrm{{SQ}}$ = {f:.2f}")
    ax.scatter([1.12], [100 * P["eta_csi"]["value"]], color=C["red"], s=22, zorder=4, label="c-Si module (benchmark)")
    ax.set_xlim(0.5, 3.0)
    ax.set_ylim(0, 36)
    ax.set_xlabel("Band gap (eV)")
    ax.set_ylabel("Efficiency (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.2)
    despine(ax)
    fig.tight_layout()
    save(fig, "figS4_sq")
    dump(pd.DataFrame(rows), "figS4_sq")


def figS_sobol():
    s = pd.read_csv(R / "v2/tea_v2/sobol.csv")
    pretty = {"f_sq": "f$_\\mathrm{SQ}$", "module_m2": "Module cost", "cf_ac": "Capacity factor",
              "module_life": "Module lifetime", "deg": "Degradation", "r": "Discount rate", "field_scale": "Field labour",
              "fom_kwac": "Fixed O&M", "sbos_scale": "Structural BOS"}
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    y = np.arange(len(s))
    ax.barh(y + 0.18, s.ST, height=0.34, xerr=s.ST_conf, color=C["blue"], label="Total, S$_T$", error_kw=dict(lw=0.6))
    ax.barh(y - 0.18, s.S1, height=0.34, xerr=s.S1_conf, color=C["orange"], label="First order, S$_1$", error_kw=dict(lw=0.6))
    ax.set_yticks(y, [pretty.get(p, p) for p in s.param])
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index (bars: 95% bootstrap interval)")
    ax.legend(loc="lower right", fontsize=6.5)
    despine(ax)
    fig.tight_layout()
    save(fig, "figS7_sobol")
    dump(s, "figS7_sobol")


def figS_breakeven():
    from pipeline.sq import sq_table
    sq_max = float(sq_table()[1].max())
    be = pd.read_csv(R / "v2/tea_v2/breakeven_efficiency.csv")
    costs = [("low", "50"), ("med", "85"), ("high", "335")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    cmap = matplotlib.colormaps["viridis_r"].copy()
    cmap.set_bad("0.88")
    for r_, burn in enumerate(sorted(be.burnin.unique())):
        for c_, (mc, usd) in enumerate(costs):
            ax = axes[r_, c_]
            tb = be[(be.module_cost == mc) & (be.burnin == burn)].pivot(index="module_life", columns="deg",
                                                                          values="breakeven_eta")
            im = ax.imshow(100 * tb.to_numpy(), cmap=cmap, vmin=20, vmax=40, aspect="auto")
            for i in range(tb.shape[0]):
                for k in range(tb.shape[1]):
                    v = tb.to_numpy()[i, k]
                    lab = "n.r." if pd.isna(v) else (f"{100 * v:.1f}*" if v > sq_max else f"{100 * v:.1f}")
                    ax.text(k, i, lab, ha="center", va="center", fontsize=6.3,
                            color="k" if pd.isna(v) or v < 0.33 else "w")
            ax.set_xticks(range(tb.shape[1]), [f"{100 * d:.1f}" for d in tb.columns])
            ax.set_yticks(range(tb.shape[0]), tb.index)
            ax.set_title(f"USD {usd} m$^{{-2}}$, burn-in {100 * burn:.0f}%", fontsize=7.5, loc="left")
            if r_ == 1:
                ax.set_xlabel("Degradation (% yr$^{-1}$)")
            if c_ == 0:
                ax.set_ylabel("Module lifetime (yr)")
    for a, s in zip(axes.flat, "abcdef"):
        panel(a, f"({s})")
    fig.tight_layout(h_pad=1.8, w_pad=1.2)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Break-even efficiency (%)")
    save(fig, "figS5_breakeven")
    dump(be, "figS5_breakeven")


def figS_candidates():
    c = pd.read_csv(R / "v2/tea_v2/candidate_lcoe.csv")
    sc = pd.read_csv(R / "v2/stability/v2_screen_scored.csv")[["formula", "gap_pred_eV"]]
    c = c.drop(columns=["gap_pred_eV"]).merge(sc, on="formula", how="left")
    ref = j("v2/metrics/tea_v2.json")["e2_csi_ref"]["msp_median_usd_mwh"]
    opt, ceil = "f0.61_low_L35_d0.007", "f0.82_low_L35_d0.007"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    l = c[f"lcoe_median__{opt}"].replace([np.inf], np.nan)
    shown = l.notna() & (l <= 1000)
    sca = ax.scatter(c.gap_pred_eV[shown], l[shown], c=c.p_semi[shown], cmap="viridis", s=5, vmin=0.5, vmax=1)
    ax.axhline(ref, color="k", lw=0.8, ls="--", label="c-Si median")
    ax.axvspan(1.0, 1.8, color="0.93", zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("Predicted band gap (eV)")
    ax.set_ylabel("Median LCOE, f$_\\mathrm{SQ}$ = 0.61 (USD MWh$^{-1}$)")
    ax.legend(loc="lower right", fontsize=6.2)
    ax.set_title(f"{int(shown.sum()):,} of {len(c):,} candidates below 1000 USD MWh$^{{-1}}$", loc="left", fontsize=7.5)
    cb = fig.colorbar(sca, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("P(semiconductor)")
    ax = axes[1]
    edges = np.linspace(0, 1, 21)
    for col_, lab, colr in ((opt, "f$_\\mathrm{SQ}$ = 0.61", C["blue"]), (ceil, "f$_\\mathrm{SQ}$ = 0.82 (ceiling)", C["purple"])):
        ax.hist(c[f"p_lcoe_le_csi__{col_}"], bins=edges, color=colr, alpha=0.6, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("P(LCOE ≤ c-Si)")
    ax.set_ylabel("Candidates")
    ax.legend(loc="upper right", fontsize=6.2)
    ax.set_title("USD 50 m$^{-2}$, 35-yr module", loc="left", fontsize=7.5)
    for a, s in zip(axes, "ab"):
        panel(a, f"({s})")
    despine(*axes)
    fig.tight_layout(w_pad=1.8)
    save(fig, "figS6_candidates")
    dump(c[["formula", "status", "gap_pred_eV", "p_semi", f"lcoe_median__{opt}", f"p_lcoe_le_csi__{opt}",
            f"p_lcoe_le_csi__{ceil}"]], "figS6_candidates")


def figS_walterbos():
    w = pd.read_csv(R / "v2/review/walterbos_check.csv")
    w = w[w.hse_nonmetal].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True)
    lim = (-0.5, 10.0)
    for ax, (flag, title) in zip(axes, ((False, "not in the training set"), (True, "in the training set"))):
        d = w[w.in_training == flag]
        inside = (d.bandgap >= d.gap_lo90_eV) & (d.bandgap <= d.gap_hi90_eV)
        ax.scatter(d.gap_pred_eV[~inside], d.bandgap[~inside], s=4, color=C["orange"], alpha=0.6,
                   label="HSE06 gap outside the 90% interval", rasterized=True)
        ax.scatter(d.gap_pred_eV[inside], d.bandgap[inside], s=4, color=C["blue"], alpha=0.6,
                   label="HSE06 gap inside the 90% interval", rasterized=True)
        ax.plot(lim, lim, color="k", lw=0.8, ls="--")
        ax.axhspan(1.0, 1.8, color="0.92", zorder=0)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("Predicted band gap, this work (eV)")
        ax.set_title(f"{title}, n = {len(d):,}", loc="left", fontsize=7.5)
    axes[0].set_ylabel("HSE06 band gap, Walterbos et al. (eV)")
    axes[1].legend(loc="lower right", fontsize=6.2, markerscale=2.5)
    for a, s_ in zip(axes, "ab"):
        panel(a, f"({s_})")
    despine(*axes)
    fig.tight_layout(w_pad=1.5)
    save(fig, "figS8_hse06")
    dump(w[["comp_name_full", "in_training", "in_v2_candidates", "gap_pred_eV", "gap_lo90_eV", "gap_hi90_eV",
            "p_semi", "bandgap", "cond_type", "spin_forbidden"]], "figS8_hse06")


FIGS = (figS1, figS2, figS3, figS4, figS_breakeven, figS_candidates, figS_sobol, figS_walterbos)

if __name__ == "__main__":
    for f in FIGS:
        if f is figS_walterbos and not (R / "v2/review/walterbos_check.csv").exists():
            print("SKIP figS8: runs/v2/review/walterbos_check.csv not present (run stage s50); committed figure kept")
            continue
        if f is figS1 and not (R / "v2/predictions/oof_v2.csv").exists():
            print("SKIP figS1: runs/v2/predictions/oof_v2.csv not present (re-run stage s11); committed figure kept")
            continue
        f()
        print("ok", f.__name__)
