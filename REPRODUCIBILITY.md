# Reproducibility

## Random Seeds

All ML models use `random_state=42` throughout:
- XGBoost bandgap regressor: `random_state=42`
- XGBoost formation energy regressor: `random_state=42`
- Train/test split: `random_state=42` via `train_test_split`
- TEA Monte Carlo: uses `numpy.random` (unseeded for stochastic variation across runs)

## Environment

```bash
pip install -r requirements.txt
```

Key dependency versions (verified working):
- Python 3.11.15
- pandas >= 2.0 (tested 3.0.5)
- scikit-learn >= 1.3
- xgboost >= 2.0
- pymatgen >= 2023.0
- openpyxl >= 3.1 (tested 3.1.5)
- python-docx >= 1.0 (tested 1.2.0)
- matplotlib >= 3.7 (tested 3.11.1)

## Data Provenance

| File | Source | Description |
|------|--------|-------------|
| `Final_Top_Discoveries_FullStats.csv` | XGB-TEA pipeline (TMF → XGBoost → TEA) | 3,280 double perovskite candidates with full Monte Carlo statistics |
| `Final_Top_Discoveries_OptionA.csv` | Option A: composition-aware PCE derating | Same candidates with derate_factor, Δχ, t, μ_r columns + re-ranked |
| `Final_Top_Discoveries_OptionA_LifeCorr.csv` | Post-processing: Ef-dependent lifetime model | Adds Lifetime_Ef_current and Lifetime_Ef_future columns (illustrative) |
| `01_original_top_discoveries.csv` | XGB cascade predictions | Original 100 top PCE candidates |
| `02_xgb_discoveries.csv` | XGB cascade | Top candidates with XGB-predicted bandgap and Ef |
| `03_xgb_discoveries_fullstats.csv` | XGB + MC-TEA | Monte Carlo statistics for the top candidates |
| `04_magpie_discoveries.csv` | Magpie features | Magpie-derived feature vectors for candidates |
| `05_contour_lcoe_penalty.csv` | Penalty analysis | LCOE contour plots with PCE penalty |
| `06_multi_dim_risk.csv` | Multi-dimensional risk | Risk assessment across PCE, LCOE, cost dimensions |
| `07_parallel_risk_matrix.csv` | Parallel risk matrix | Parallel coordinate risk visualization data |
| `08_risk_topology.csv` | Risk topology | Topological risk mapping data |

## Model Training

The XGBoost models were trained on Materials Project data (31,275 inorganic compounds) with 132 Magpie + composition features. Hyperparameters were tuned via Optuna (100 trials). The ML → TEA → derating chain is deterministic given the same input CSVs.

## v2 audit pipeline (manuscript)

The sections above describe the ORIGINAL study. The v2 audit (`pipeline/`, `runs/`, `paper/`) is run through `python -m pipeline.run` (see README, "v2 audit pipeline (manuscript)").

- **Environment.** Exact pins: `requirements-v2.lock.txt` (Python 3.12). CI uses the light set in `requirements-ci.txt` (no torch). The optional GPU stages s12 (Roost) and s21 (MACE-MP-0) run in a separate environment built by `env/setup_wsl_gpu.sh`, pinned in `env/freeze-wsl-gpu.txt`.
- **Seeds.** 42 everywhere in the v2 stages (`pipeline/config.py`, `pipeline/s11_ml_eval.py`). XGBoost device (CPU or CUDA) is recorded in each stage manifest.
- **Manifests.** Every stage writes `runs/<run-id>/stage_<name>.json` with the sha256 of each input and output, parameters, metrics, package versions and git commit. The `.gitattributes` rule `runs/** -text` keeps committed bytes identical to the hashed bytes.
- **Materials Project.** Stages s10, s20 and s21 need your own `MP_API_KEY` (environment or local `.env`). The run in `runs/v2` used MP database version 2026.04.13 (`runs/v2/metrics/dataset_v2.json`). A newer MP release will move the numbers slightly.
- **Not shipped (regenerable).** `data/mp_cache/`, `runs/v2/data/`, `runs/v2/splits/`, `runs/v2/predictions/`, `runs/v2/models/`, `runs/v2/features/`. Their hashes are in the committed stage manifests.
- **Stage s50 (pre-submission review checks).** `python -m pipeline.run --v2 --run-id v2 --only s50` compares the final model with the HSE06 gaps of Walterbos et al. (shipped subset, hash-checked by `scripts/fetch_walterbos2026.py`) and runs the hull-threshold, O&M/inverter and oxidation-state sensitivity checks. It needs the s13 model and `runs/v2/data/train_v2.csv` (both not redistributed), and stops if its 50 meV and base-case rows do not reproduce s22 and s30 exactly.

| Shipped file | Source | Licence |
|---|---|---|
| `data/training_table.csv` | Materials Project, April 2026 export (see `data/README.md`) | CC BY 4.0 |
| `runs/v2/**` | This pipeline on Materials Project database 2026.04.13 | CC BY 4.0 derivative / MIT |
| `data/external/walterbos2026/hdp_hse06_subset.csv` | Walterbos et al. 2026 (arXiv:2606.11928), Zenodo 10.5281/zenodo.20598121, 8 columns unchanged | CC BY 4.0 |
| `data/sources/usgs_mcs2024_*.txt` | USGS Mineral Commodity Summaries 2024 | Public domain |
| `data/sources/nrel_*` | NREL Q1 2023 cost benchmark; NREL ATB 2024 | Public domain |
| `data/sources/zhang_natcommun2022_cs2agbibr6.md` | Zhang et al., Nat. Commun. 2022 | CC BY 4.0 |
| `paper/numbers.json` | `paper/paper_numbers.py` over `runs/` | MIT |
| `paper/si_numbers.json`, `paper/figures/si_data/*.csv`, `paper/figures/figS*` | `paper/si_numbers.py`, `paper/make_si_figures.py` over `runs/` and `pipeline/` | MIT |

## License

MIT License — see LICENSE file.
