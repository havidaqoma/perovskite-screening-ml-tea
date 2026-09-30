"""Stage 22: joint screening score, chemical-plausibility flags, and shortlist for the uMLIP check.

p_viable = P(semiconductor) * P(gap in 1.0-1.8 eV) * P(E_hull <= 50 meV)
All three probabilities are calibrated on held-out chemical systems (s11/s13/s20-control).
Known-MP compounds use MP's own E_hull (P = 1 if <= 50 meV else 0).

Plausibility flags (novel candidates only; calibration on KNOWN compounds cannot catch these):
  flag_hull_extrapolation : predicted E_hull < -50 meV, i.e. far BELOW the known hull. In the control
                            set of 380 real A2BB'X6 compounds only 0.5% lie < -50 meV and 0% < -100 meV
                            relative to the other phases, so such predictions are ML extrapolation.
  flag_ox_implausible     : requires Mn7+ or Cr6+ on the octahedral B site. Both are essentially
                            only tetrahedral (MnO4-, CrO4 2-), and are reduced by S2-/Se2-/I-.
Flagged rows keep their scores (reported) but p_viable_plausible = 0 and they are not shortlisted.

Outputs:
  stability/v2_screen_scored.csv     every v2 candidate, all components + flags
  stability/shortlist_for_umlip.csv  top-N plausible NOVEL + N_CONTROL known-MP compounds (method control)
"""
from __future__ import annotations

import pandas as pd

from .common import Stage, dump_json
from .s14_screen_v2 import _a_site

MODULES = ("s22_shortlist",)
TOP_N_NOVEL = 30
N_CONTROL = 10
HULL_EXTRAP_MEV = -50.0
OX_IMPLAUSIBLE = ("Mn+7", "Cr+6")


def run(run_dir, force: bool = False) -> dict:
    stab = run_dir / "stability/v2_stability.csv"
    st = Stage(run_dir, "s22_shortlist", {"stability": stab}, MODULES,
               params={"top_n_novel": TOP_N_NOVEL, "n_control": N_CONTROL, "hull_extrap_meV": HULL_EXTRAP_MEV,
                       "ox_implausible": OX_IMPLAUSIBLE})
    if st.up_to_date() and not force:
        print("[s22] up to date, skipped")
        return {}
    df = pd.read_csv(stab)
    p_st = df["p_ehull_le_50"].copy()
    known = df.status.eq("known_mp")
    p_st[known] = (df.loc[known, "mp_ehull_meV"] <= 50).astype(float)
    p_st[df.status.eq("unknown")] = 0.0
    df["p_stable50"] = p_st
    df["p_viable"] = df.p_semi * df.p_gap_pv * df.p_stable50
    df["flag_hull_extrapolation"] = ~known & (df.ehull_pred_meV < HULL_EXTRAP_MEV)
    df["flag_ox_implausible"] = df.ox_states.fillna("").apply(lambda s: any(o in s.split(";") for o in OX_IMPLAUSIBLE))
    df["plausible"] = ~(df.flag_hull_extrapolation | df.flag_ox_implausible)
    df["p_viable_plausible"] = df.p_viable.where(df.plausible, 0.0)
    df["a_site"] = [_a_site(f) for f in df.formula]
    df = df.sort_values("p_viable_plausible", ascending=False).reset_index(drop=True)
    known = df.status.eq("known_mp")        # recompute AFTER the re-sort (the old mask is misaligned)
    df.to_csv(st.path("stability/v2_screen_scored.csv"), index=False, float_format="%.5f")

    novel = df[~known & df.plausible].head(TOP_N_NOVEL).assign(role="novel")
    ctrl = df[known].head(N_CONTROL).assign(role="control_known_mp")
    short = pd.concat([novel, ctrl], ignore_index=True)
    short.to_csv(st.path("stability/shortlist_for_umlip.csv"), index=False, float_format="%.5f")

    nv = df[~known]
    st.metrics = {"n": int(len(df)), "n_known_mp": int(known.sum()), "n_novel": int(len(nv)),
                  "n_flag_hull_extrapolation": int(nv.flag_hull_extrapolation.sum()),
                  "n_flag_ox_implausible": int(nv.flag_ox_implausible.sum()),
                  "n_novel_plausible": int(nv.plausible.sum()),
                  "expected_viable_novel_all": float(nv.p_viable.sum()),
                  "expected_viable_novel_plausible": float(nv.p_viable_plausible.sum()),
                  "n_novel_plausible_p_viable_ge": {t: int((nv.p_viable_plausible >= t).sum()) for t in (0.1, 0.2, 0.3, 0.5)},
                  "shortlist_novel_min_p": float(novel.p_viable_plausible.min()),
                  "top10_novel": novel.head(10)[["formula", "ox_states", "p_semi", "p_gap_pv", "p_stable50",
                                                 "p_viable_plausible"]].round(3).to_dict("records")}
    dump_json(st.path("metrics/shortlist_v2.json"), st.metrics)
    st.finish()
    print(f"[s22] {st.metrics}")
    return st.metrics
