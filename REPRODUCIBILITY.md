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

## License

MIT License — see LICENSE file.
