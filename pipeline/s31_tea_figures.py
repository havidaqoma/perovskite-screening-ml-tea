"""Stage 31: Phase E figures (from runs/<id>/tea_v2/*.csv and metrics only; no hand-typed numbers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
                     "legend.frameon": False, "savefig.dpi": 300})


def main(run_dir: Path) -> list[Path]:
    out = run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((run_dir / "metrics/tea_v2.json").read_text())
    ref = m["e2_csi_ref"]
    paths = []

    # Fig E1: efficiency sweep vs c-Si reference band
    sw = pd.read_csv(run_dir / "tea_v2/efficiency_sweep.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for name, lab, col in (("low", "$50/m²", "#1b9e77"), ("med", "$85/m²", "#d95f02"), ("high", "$335/m²", "#7570b3")):
        g = sw[sw.module_cost == name]
        ax.plot(100 * g.eta, g.lcoe_median, color=col, lw=1.4, label=f"perovskite module {lab}")
        ax.fill_between(100 * g.eta, g.lcoe_p10, g.lcoe_p90, color=col, alpha=0.12, lw=0)
    ax.axhspan(ref["msp_p10"], ref["msp_p90"], color="0.8", alpha=0.6, lw=0, label="c-Si reference P10–P90")
    ax.axhline(ref["msp_median_usd_mwh"], color="k", lw=0.8, ls="--", label="c-Si median")
    ax.set_xlabel("Module efficiency (%)")
    ax.set_ylabel("LCOE ($/MWh, 2022 USD)")
    ax.set_yscale("log")
    ax.set_ylim(30, 1000)
    ax.set_xlim(5, 30)
    ax.set_yticks([30, 50, 100, 200, 500, 1000], ["30", "50", "100", "200", "500", "1000"])
    ax.legend(fontsize=6.5, loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.8")
    fig.tight_layout()
    p = out / "figE1_efficiency_sweep.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # Fig E2: break-even efficiency heat table (low module cost, burn-in 0)
    be = pd.read_csv(run_dir / "tea_v2/breakeven_efficiency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    fig.subplots_adjust(bottom=0.30, wspace=0.08)
    for ax, mod in zip(axes, ("low", "med")):
        t = be[(be.module_cost == mod) & (be.burnin == 0)].pivot(index="module_life", columns="deg", values="breakeven_eta")
        cmap = matplotlib.colormaps["viridis_r"].copy()
        cmap.set_bad("0.85")
        im = ax.imshow(100 * t.to_numpy(), cmap=cmap, vmin=20, vmax=40, aspect="auto")
        ax.set_xticks([x - 0.5 for x in range(1, t.shape[1])], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, t.shape[0])], minor=True)
        ax.grid(which="minor", color="w", lw=1.2)
        ax.tick_params(which="minor", length=0)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.to_numpy()[i, j]
                label = "not reached" if pd.isna(v) else f"{100 * v:.1f}"
                ax.text(j, i, label.replace(" ", "\n"), ha="center", va="center",
                        fontsize=6 if pd.isna(v) else 7, color="0.1" if pd.isna(v) else ("w" if v > 0.30 else "k"))
        ax.set_xticks(range(t.shape[1]), [f"{100 * c:.1f}" for c in t.columns])
        ax.set_yticks(range(t.shape[0]), t.index)
        ax.set_xlabel("Degradation (%/yr)")
        ax.set_title(f"module ${50 if mod == 'low' else 85}/m²", fontsize=8)
    axes[0].set_ylabel("Module lifetime (yr)")
    fig.colorbar(im, ax=axes, label="Break-even efficiency (%)", shrink=0.9)
    fig.text(0.08, 0.03, "Grey = not reached: no efficiency up to 40% matches the c-Si median LCOE. "
             "Detailed-balance (SQ) limit at 1.34 eV = 33.7%.", fontsize=6.5, va="bottom", wrap=True)
    p = out / "figE2_breakeven.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # Fig E3: Sobol total-order indices
    sob = pd.read_csv(run_dir / "tea_v2/sobol.csv").sort_values("ST")
    labels = {"f_sq": "fraction of SQ realised", "module_m2": "module cost ($/m²)", "cf_ac": "capacity factor",
              "module_life": "module lifetime", "deg": "degradation rate", "r": "discount rate",
              "field_scale": "installation labour", "fom_kwac": "fixed O&M", "sbos_scale": "structural BOS"}
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.barh([labels[x] for x in sob.param], sob.ST, xerr=sob.ST_conf, color="#4c72b0", height=0.6)
    ax.set_xlabel("Sobol total-order index S_T")
    fig.tight_layout()
    p = out / "figE3_sobol.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


if __name__ == "__main__":
    for p in main(Path(sys.argv[1])):
        print(p)
