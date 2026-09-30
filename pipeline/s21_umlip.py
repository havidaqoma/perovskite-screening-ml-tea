"""Stage 21 (WSL/GPU): uMLIP structural check for the shortlist (Phase D, part 2).

For each shortlisted A2BB'X6 formula:
  1. Build the rock-salt-ordered double perovskite (Fm-3m, Cs2AgBiBr6 prototype, 40-atom cubic cell)
     with a lattice guess from Shannon radii; for X3X'3 compositions, anions are ordered in a
     fac arrangement per octahedron is NOT attempted - they are distributed in the lowest-symmetry
     way available in the 40-atom cell (layered X / X' along c), documented as a limitation.
  2. Relax cell + positions with MACE-MP-0 (medium) + FrechetCellFilter, fmax 0.05 eV/A, <= 400 steps.
  3. Energy -> E_hull against a MACE-computed hull of the MP competing phases in the same chemical
     system (MP structures re-relaxed with the SAME potential, so errors partly cancel).
This is a physics sanity check of the ML hull distance, not DFT. Output: runs/<id>/stability/umlip.csv

Run:  ~/venvs/mlperov-gpu/bin/python -m pipeline.s21_umlip runs/v2 --top 150
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

ANIONS = ("O", "S", "Se", "Te", "F", "Cl", "Br", "I")
FMAX, STEPS = 0.05, 400
COLUMNS = ["formula", "role", "status", "converged", "e_per_atom", "a_relaxed", "umlip_ehull_vs_rest_meV",
           "n_ref_phases", "ml_ehull_pred_meV", "mp_ehull_meV", "error", "seconds"]


def build_double_perovskite(formula: str, a_site: str, rB: float, rX: float):
    """40-atom conventional Fm-3m cell: 8 A, 4 B, 4 B', 24 X."""
    from pymatgen.core import Composition, Lattice, Structure
    comp = Composition(formula)
    amt = comp.get_el_amt_dict()
    b = [e for e in amt if e != a_site and e not in ANIONS]
    x = [e for e in amt if e in ANIONS]
    if len(b) == 1:                       # B = B' (e.g. A2B2X6 written as ABX3 double cell)
        b = [b[0], b[0]]
    a = 4.0 * (rB + rX)                  # conventional Fm-3m cell = 2 x (B-X-B') repeat = 4(rB+rX)
    lat = Lattice.cubic(a)
    species, coords = [], []
    # B on (0,0,0)+fcc, B' on (1/2,0,0)+fcc  (rock-salt order)
    fcc = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    for f in fcc:
        species.append(b[0]); coords.append(f)
        species.append(b[1]); coords.append(((f[0] + .5) % 1, f[1], f[2]))
    # A on (1/4,1/4,1/4)+ (3/4,..) fcc
    for f in fcc:
        for s in ((.25, .25, .25), (.75, .75, .75)):
            species.append(a_site); coords.append(tuple((f[i] + s[i]) % 1 for i in range(3)))
    # X: between B and B' along each axis, at (1/4,0,0)-type positions +fcc
    xpos = []
    for f in fcc:
        for d in ((.25, 0, 0), (.75, 0, 0), (0, .25, 0), (0, .75, 0), (0, 0, .25), (0, 0, .75)):
            xpos.append(tuple((f[i] + d[i]) % 1 for i in range(3)))
    if len(x) == 1:
        xs = [x[0]] * 24
    else:                                   # layered: first half (lowest z) X, second X'
        order = np.argsort([p[2] + 1e-3 * p[0] + 1e-6 * p[1] for p in xpos])
        xs = [None] * 24
        for k, i in enumerate(order):
            xs[i] = x[0] if k < 12 else x[1]
    species += xs; coords += xpos
    s = Structure(lat, species, coords)
    if s.composition.reduced_formula != comp.reduced_formula:
        raise ValueError(f"prototype composition {s.composition.reduced_formula} != {comp.reduced_formula}")
    return s


def relax(atoms, calc):
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    atoms = atoms.copy()
    atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = opt.run(fmax=FMAX, steps=STEPS)
    return atoms, bool(converged), float(atoms.get_potential_energy()) / len(atoms)


def mp_structures(elements, cache_dir: Path, mpr):
    """Hull-defining (E_hull <= 1 meV) MP structures in the chemical system and all its subsystems."""
    from pymatgen.core import Structure
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / ("-".join(sorted(elements)) + ".json.gz")
    if path.exists():
        raw = json.loads(gzip.decompress(path.read_bytes()))
        return [(d["id"], Structure.from_dict(d["s"])) for d in raw]
    import itertools
    systems = ["-".join(sorted(c)) for n in range(1, len(elements) + 1) for c in itertools.combinations(elements, n)]
    docs = mpr.materials.summary.search(chemsys=systems, energy_above_hull=(0, 0.001), deprecated=False,
                                        fields=["material_id", "structure", "energy_above_hull", "nsites"])
    keep = [d for d in docs if d.nsites <= 60]
    # Always include every pure element (its lowest-E_hull polymorph, smallest cell), even if the
    # hull-defining elemental cell is large: without terminal entries the hull cannot be built.
    have = {str(d.structure.composition.elements[0]) for d in keep if len(d.structure.composition.elements) == 1}
    for el in sorted(set(elements) - have):
        ed = mpr.materials.summary.search(chemsys=el, deprecated=False,
                                          fields=["material_id", "structure", "energy_above_hull", "nsites"])
        if not ed:
            raise RuntimeError(f"MP has no elemental entry for {el}")
        keep.append(min(ed, key=lambda d: (round(d.energy_above_hull, 3), d.nsites)))
    path.write_bytes(gzip.compress(json.dumps([{"id": str(d.material_id), "s": d.structure.as_dict()} for d in keep]).encode()))
    return [(str(d.material_id), d.structure) for d in keep]


def main(run_dir: Path, top: int, mp_key: str | None):
    """top = max rows of the shortlist to process (shortlist = plausible novel + known-MP controls)."""
    import torch
    from mace.calculators import mace_mp
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.io.ase import AseAtomsAdaptor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="medium", device=dev, default_dtype="float64")
    shortlist = pd.read_csv(run_dir / "stability/shortlist_for_umlip.csv").head(top)
    out_csv = run_dir / "stability/umlip.csv"
    done = set(pd.read_csv(out_csv).query("status == 'ok'").formula) if out_csv.exists() else set()
    ref_cache = run_dir / "stability/umlip_ref_energies.json"
    ref_e = json.loads(ref_cache.read_text()) if ref_cache.exists() else {}
    from mp_api.client import MPRester
    mpr = MPRester(mp_key, mute_progress_bars=True)
    ad = AseAtomsAdaptor()
    for r in shortlist.itertuples():
        if r.formula in done:
            continue
        t0 = time.time()
        row = {"formula": r.formula, "role": r.role}
        try:
            s0 = build_double_perovskite(r.formula, r.a_site, r.rB, r.rX)
            at, conv, e_pa = relax(ad.get_atoms(s0), calc)
            row.update(converged=conv, e_per_atom=e_pa, a_relaxed=float(np.cbrt(at.get_volume())))
            els = sorted(e.symbol for e in Composition(r.formula).elements)
            entries = []
            self_red = Composition(r.formula).reduced_formula
            for mid, s in mp_structures(els, run_dir.parent.parent / "data/mp_cache/umlip_structs", mpr):
                if s.composition.reduced_formula == self_red:
                    continue                       # hull of the OTHER phases only (fair for known controls)
                if mid not in ref_e:
                    _, c2, e2 = relax(ad.get_atoms(s), calc)
                    ref_e[mid] = {"e_per_atom": e2, "formula": s.composition.formula, "converged": c2}
                    ref_cache.write_text(json.dumps(ref_e))
                entries.append(PDEntry(Composition(ref_e[mid]["formula"]),
                                       ref_e[mid]["e_per_atom"] * Composition(ref_e[mid]["formula"]).num_atoms, name=mid))
            pd_ = PhaseDiagram(entries)
            cand = PDEntry(Composition(r.formula), e_pa * Composition(r.formula).num_atoms, name="cand")
            _, eh = pd_.get_decomp_and_e_above_hull(cand, allow_negative=True)
            row.update(umlip_ehull_vs_rest_meV=1000 * eh, n_ref_phases=len(entries),
                       ml_ehull_pred_meV=getattr(r, "ehull_pred_meV", float("nan")),
                       mp_ehull_meV=getattr(r, "mp_ehull_meV", float("nan")), status="ok")
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:200])
        row["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(row, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--top", type=int, default=150)
    a = ap.parse_args()
    import os, re
    key = os.environ.get("MP_API_KEY") or re.search(r"^MP_API_KEY=(.*)$", (Path(".env")).read_text(), re.M).group(1).strip()
    main(a.run_dir, a.top, key)
