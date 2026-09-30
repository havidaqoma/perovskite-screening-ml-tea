"""Stage 20: stability of v2 candidates against Materials Project competing phases (Phase D, part 1).

For each candidate, in its own chemical system:
  * Build the MP GGA/GGA+U convex hull from cached entries (pymatgen PhaseDiagram).
  * If MP already contains the exact reduced formula, report MP's own E_hull (status "known_mp").
  * Otherwise insert a pseudo-entry whose formation energy is the ML prediction and compute
    E_hull_pred = max(0, Ef_pred - E_hull_surface(composition)) via get_decomp_and_e_above_hull.
    The ML error is propagated with the empirical out-of-fold Ef residuals of 4-5 element
    compounds on held-out chemical systems:  P(E_hull_true <= thr) = P(Ef_true <= surface + thr),
    Ef_true = Ef_pred - residual.
  * Status: stable_likely  if P(E_hull <= 50 meV) >= 0.5
            unstable_likely otherwise
            unknown         if the hull could not be built (never passes).
Entry data is cached per element set under data/mp_cache/entries/<elements>.json.gz.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from . import config
from .common import Stage, dump_json

MODULES = ("s20_stability",)
ENTRY_DIR = config.DATA_DIR / "mp_cache" / "entries"
THRESHOLDS_MEV = (0, 20, 35, 50, 100)
GATE_MEV = 50.0
P_STABLE_MIN = 0.5


def _entries_for(elements: tuple[str, ...], mpr, attempts: int = 4) -> list[ComputedEntry]:
    ENTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = ENTRY_DIR / ("-".join(elements) + ".json.gz")
    if path.exists():
        raw = json.loads(gzip.decompress(path.read_bytes()))
        return [ComputedEntry.from_dict(d) for d in raw["entries"]]
    import time
    for k in range(attempts):
        try:
            ents = mpr.get_entries_in_chemsys(list(elements), additional_criteria={"thermo_types": ["GGA_GGA+U"]},
                                              compatible_only=True, property_data=None, inc_structure=False)
            break
        except Exception:
            if k == attempts - 1:
                raise                      # still fails closed (row -> "unknown"); never a silent pass
            time.sleep(15 * 2 ** k)        # transient network errors: back off 15/30/60 s
    if not ents:
        raise RuntimeError(f"MP returned no entries for {elements}")
    payload = {"elements": list(elements), "n": len(ents), "entries": [e.as_dict() for e in ents]}
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    return ents


def hull_for_candidate(formula: str, ef_pred: float, residuals: np.ndarray, pd_: PhaseDiagram,
                       exclude_self: bool = False) -> dict:
    comp = Composition(formula)
    if exclude_self:
        # control mode: truth = the compound's own GGA/GGA+U E_hull; the ML hull is built WITHOUT it
        known = [e for e in pd_.all_entries if e.composition.reduced_formula == comp.reduced_formula]
        if not known:
            raise ValueError(f"control compound {formula} not in MP entries")
        best = min(known, key=lambda e: e.energy_per_atom)
        truth = 1000.0 * pd_.get_e_above_hull(best)
        truth_ef = pd_.get_form_energy_per_atom(best)
        pd_ = PhaseDiagram([e for e in pd_.all_entries if e.composition.reduced_formula != comp.reduced_formula])
        out = hull_for_candidate(formula, ef_pred, residuals, pd_)
        out.update(true_ehull_meV=truth, true_ef_pd=truth_ef,
                   true_ehull_vs_rest_meV=1000.0 * (truth_ef - out["hull_surface_ef"]))
        return out
    # E_hull surface energy at this composition, as formation energy per atom
    known = [e for e in pd_.all_entries if e.composition.reduced_formula == comp.reduced_formula]
    out = {}
    if known:
        best = min(known, key=lambda e: e.energy_per_atom)
        out["mp_ehull_meV"] = 1000.0 * pd_.get_e_above_hull(best)
        out["mp_entry_id"] = str(best.entry_id)
    # formation energy of the hull at comp: decompose a dummy entry with Ef = +huge
    el_refs = pd_.el_refs
    e_ref = sum(comp.get_atomic_fraction(el) * el_refs[el].energy_per_atom for el in comp.elements)
    probe = PDEntry(comp, (e_ref + 10.0) * comp.num_atoms)          # far above hull
    _, e_above_probe = pd_.get_decomp_and_e_above_hull(probe, allow_negative=True)
    surface_ef = 10.0 - e_above_probe                                 # hull formation energy (eV/atom)
    out["hull_surface_ef"] = surface_ef
    out["ehull_pred_meV"] = 1000.0 * (ef_pred - surface_ef)          # may be negative (= predicted new hull phase)
    for thr in THRESHOLDS_MEV:
        # true Ef = ef_pred - r ; stable if true Ef - surface <= thr
        out[f"p_ehull_le_{thr}"] = float(np.mean(ef_pred - residuals - surface_ef <= thr / 1000.0))
    return out


CONTROL_N = 400
A_ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
X_ANION = {"O", "S", "Se", "F", "Cl", "Br", "I"}


def is_a2bbx6(formula: str) -> bool:
    a = Composition(formula).get_el_amt_dict()
    alk = [e for e in a if e in A_ALKALI]
    an = [e for e in a if e in X_ANION]
    bs = [e for e in a if e not in A_ALKALI and e not in X_ANION]
    if len(alk) != 1 or a[alk[0]] != 2 or len(bs) != 2 or any(a[b] != 1 for b in bs):
        return False
    return (len(an) == 1 and a[an[0]] == 6) or (len(an) == 2 and all(a[x] == 3 for x in an))


def control(run_dir, mpr, residuals) -> tuple[pd.DataFrame, dict]:
    """Known MP A2BB'X6 compounds, ML Ef from OUT-OF-FOLD chemsys predictions (never trained on them)."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    d = pd.read_csv(run_dir / "data/train_v2.csv")
    o = pd.read_csv(run_dir / "predictions/oof_v2.csv", usecols=["row", "chemsys__ef__xgb"])
    d = d.merge(o, on="row")
    d = d[d.nelements.isin([4, 5]) & d.chemsys__ef__xgb.notna()]
    d = d[[is_a2bbx6(f) for f in d.formula]]
    d = d.sample(n=min(CONTROL_N, len(d)), random_state=42).reset_index(drop=True)
    rows = []
    for i, r in enumerate(d.itertuples()):
        els = tuple(sorted(e.symbol for e in Composition(r.formula).elements))
        try:
            h = hull_for_candidate(r.formula, r.chemsys__ef__xgb, residuals, PhaseDiagram(_entries_for(els, mpr)),
                                   exclude_self=True)
            rows.append({"formula": r.formula, "material_id": r.material_id, "ef_oof": r.chemsys__ef__xgb, **h})
        except Exception as exc:
            rows.append({"formula": r.formula, "error": f"{type(exc).__name__}: {exc}"[:200]})
        if i % 50 == 0:
            print(f"[s20-control] {i}/{len(d)}", flush=True)
    c = pd.DataFrame(rows)
    ok = c.true_ehull_meV.notna() if "true_ehull_meV" in c else pd.Series(False, index=c.index)
    c_ok = c[ok]
    y = (c_ok.true_ehull_meV <= GATE_MEV).astype(int)
    p = c_ok[f"p_ehull_le_{int(GATE_MEV)}"]
    bins = pd.cut(p, [0, .2, .4, .6, .8, 1.0], include_lowest=True)
    m = {"n": int(len(c)), "n_ok": int(len(c_ok)), "frac_true_stable_50": float(y.mean()),
         "roc_auc_p50": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
         "brier_p50": float(brier_score_loss(y, p)),
         "gate_precision": float(y[p >= P_STABLE_MIN].mean()) if (p >= P_STABLE_MIN).any() else None,
         "gate_recall": float((p[y == 1] >= P_STABLE_MIN).mean()) if (y == 1).any() else None,
         "ehull_mae_meV_vs_rest": float(np.mean(np.abs(c_ok.ehull_pred_meV - c_ok.true_ehull_vs_rest_meV))),
         "reliability": {str(k): {"n": int(v.size), "mean_p": float(p[v.index].mean()) if v.size else None,
                                  "obs_frac": float(y[v.index].mean()) if v.size else None}
                         for k, v in p.groupby(bins, observed=False)}}
    return c, m


def run(run_dir, force: bool = False) -> dict:
    cands = run_dir / "screen/v2_candidates.csv"
    model = run_dir / "models/v2_final.joblib"
    st = Stage(run_dir, "s20_stability", {"candidates": cands, "model": model,
                                          "oof": run_dir / "predictions/oof_v2.csv"}, MODULES,
               params={"thresholds": THRESHOLDS_MEV, "gate_meV": GATE_MEV, "p_min": P_STABLE_MIN})
    if st.up_to_date() and not force:
        print("[s20] up to date, skipped")
        return {}
    df = pd.read_csv(cands)
    res = joblib.load(model)["ef_residuals"]
    from .mp_client import _rester

    rows, cache_pd = [], {}
    with _rester() as mpr:
        db_version = mpr.get_database_version()
        for i, r in enumerate(df.itertuples()):
            els = tuple(sorted(e.symbol for e in Composition(r.formula).elements))
            try:
                if els not in cache_pd:
                    cache_pd[els] = PhaseDiagram(_entries_for(els, mpr))
                h = hull_for_candidate(r.formula, r.ef_pred_eV_atom, res, cache_pd[els])
                h["status"] = ("known_mp" if "mp_ehull_meV" in h else
                               "stable_likely" if h[f"p_ehull_le_{int(GATE_MEV)}"] >= P_STABLE_MIN else
                               "unstable_likely")
            except Exception as exc:  # never a pass
                h = {"status": "unknown", "error": f"{type(exc).__name__}: {exc}"[:200]}
            rows.append({"formula": r.formula, **h})
            if i % 100 == 0:
                print(f"[s20] {i}/{len(df)} {r.formula} {h.get('status')}", flush=True)
            if len(cache_pd) > 64:
                cache_pd.clear()
        ctrl, ctrl_m = control(run_dir, mpr, res)
    out = df.merge(pd.DataFrame(rows), on="formula", how="left")
    out.to_csv(st.path("stability/v2_stability.csv"), index=False, float_format="%.5f")
    ctrl.to_csv(st.path("stability/control_known_a2bbx6.csv"), index=False, float_format="%.5f")

    st.metrics = {"mp_database_version": db_version, "n": int(len(out)),
                  "status_counts": out.status.value_counts().to_dict(),
                  "n_p50_ge_0.5_by_thr": {thr: int((out[f"p_ehull_le_{thr}"] >= 0.5).sum()) for thr in THRESHOLDS_MEV
                                          if f"p_ehull_le_{thr}" in out},
                  "ef_residual_mae": float(np.mean(np.abs(res))), "ef_residual_bias": float(np.mean(res)),
                  "control": ctrl_m}
    dump_json(st.path("metrics/stability_v2.json"), st.metrics)
    st.finish()
    print(f"[s20] {st.metrics}")
    return st.metrics
