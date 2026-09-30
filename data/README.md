# data/

| File | Rows | What it is | Provenance |
|---|---:|---|---|
| `training_table.csv` | 31,275 | Training set for the XGBoost cascade: `row, formula, band_gap_eV, formation_energy_eV_atom, n_sites` | Exported from `perovskite_dataset_9D_Charge.pt` (not redistributed; sha256 `3f5e0903d402feebf395d746e5fde68d6bf9824d3102a21264a69e2faed28c71`) by `pipeline/export_training_table.py`, row order preserved. |

## How the source dataset was built (recovered 2026-09-29)

From the author's earlier MEGNet notebook `MEGNET-TEA_ultraheavy_v15` (study archive, not redistributed), cell 2, run in April 2026:

```python
mpr.materials.summary.search(band_gap=(0.5, 3.0), num_elements=(3, 4),
    fields=["structure", "band_gap", "formation_energy_per_atom", "formula_pretty"])
# -> 33,073 docs. Dropped: structures containing Pb/U/Th/Pu/Tc, and CrystalNN failures -> 31,275.
```

- Units: `band_gap_eV` = MP PBE(+U) gap (eV). `formation_energy_eV_atom` = MP formation energy per atom (eV/atom).
- **No `energy_above_hull` filter was applied.** The v1 manuscript's statement "E_above_hull < 0.1 eV/atom" is wrong.
- **No metals or gaps < 0.5 eV**, because of the query window. So the metal/semiconductor gatekeeper could never be trained.
- 20,699 unique formulas. 4,044 formulas have more than one polymorph (median gap spread 0.335 eV).
- The MP database version was not recorded (April 2026). The `.pt` file is the frozen snapshot.
