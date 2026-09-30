"""One-time export: perovskite_dataset_9D_Charge.pt -> data/training_table.csv.

Needs torch + torch_geometric (run in the original WSL env `perovskite_ai`).
The pipeline itself only reads the CSV, so torch is not a pipeline dependency.

Provenance of the .pt (from the author's earlier MEGNet notebook MEGNET-TEA_ultraheavy_v15, cell 2; not shipped):
    mpr.materials.summary.search(band_gap=(0.5, 3.0), num_elements=(3, 4),
        fields=["structure","band_gap","formation_energy_per_atom","formula_pretty"])
    -> 33,073 docs (April 2026); entries containing Pb/U/Th/Pu/Tc dropped; CrystalNN failures dropped.
    y[0,0] = band_gap (eV, PBE, MP), y[0,1] = formation_energy_per_atom (eV/atom).
Row order is preserved exactly: the original random split (np.random.seed(42) shuffle) indexes rows.

Usage (WSL):  python pipeline/export_training_table.py perovskite_dataset_9D_Charge.pt data/training_table.csv
"""
import csv
import hashlib
import sys

import torch


def main(src: str, dst: str) -> None:
    ds = torch.load(src, weights_only=False)
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "formula", "band_gap_eV", "formation_energy_eV_atom", "n_sites"])
        for i, d in enumerate(ds):
            w.writerow([i, d.formula, f"{d.y[0, 0].item():.6f}", f"{d.y[0, 1].item():.6f}", int(d.x.shape[0])])
    h = hashlib.sha256(open(src, "rb").read()).hexdigest()
    print(f"rows={len(ds)} src_sha256={h}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
