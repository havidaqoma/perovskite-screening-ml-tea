# Raw Data Manifest

All files in this directory are CSVs exported from the ML → TEA screening pipeline for lead-free A₂BB′X₆ double perovskites.

## Files

| # | Filename | Rows | Description | Source Step |
|---|----------|------|-------------|-------------|
| 1 | `01_original_top_discoveries.csv` | 100 | Original top PCE candidates from first-pass screening | TMF + XGBoost cascade |
| 2 | `02_xgb_discoveries.csv` | 100 | XGB-predicted bandgap + Ef for top candidates | XGB cascade |
| 3 | `03_xgb_discoveries_fullstats.csv` | 3,280 | Full Monte Carlo TEA statistics (10,000 iterations each) | XGB → MC-TEA |
| 4 | `04_magpie_discoveries.csv` | 3,280 | Magpie feature vectors for all candidates | Feature engineering |
| 5 | `05_contour_lcoe_penalty.csv` | ~1,000 | LCOE contour with PCE penalty overlay | TEA sensitivity |
| 6 | `06_multi_dim_risk.csv` | 3,280 | Multi-dimensional risk (PCE, LCOE, cost) | Risk analysis |
| 7 | `07_parallel_risk_matrix.csv` | 3,280 | Parallel coordinate risk matrix | Risk analysis |
| 8 | `08_risk_topology.csv` | 3,280 | Topological risk mapping | Risk analysis |
| 9 | `09_optiona_derated_rankings.csv` | 3,280 | Option A: composition-aware derated rankings | Option A pipeline |

## Provenance

- **ML models**: XGBoost regressors (random_state=42) trained on 31,275 Materials Project entries with 132 Magpie + composition features
- **TEA engine**: Monte Carlo simulation (10,000 iterations) with triangular distributions for module cost, BOS, discount rate, degradation
- **Option A derating**: Δχ (electronegativity), t² (tolerance factor), μ_r (mu-ratio) with K_OPT=0.10, K_DEF=3.5, K_TRANS=0.60
- **License**: MIT (see root LICENSE)

## Key Numbers

- Total candidates screened: 3,280 unique A₂BB′X₆ compositions
- Bandgap range: ~0.5–3.0 eV (XGB-predicted)
- Formation energy range: Ef from ~-2.5 to +1.5 eV/atom
- Best LCOE candidate: Na₂FeMnO₃S₃ (LCOE_Median ≈ $0.04/kWh)
- Flagship finding: Na₂WSbS₃Br₃ demoted from Rank 1 → Rank 349 after composition-aware derating (Δχ = 0.38)
