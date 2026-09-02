# Raw Data Package

Unmodified source CSVs for the ML-perovskite TEA workflow.

The "Original filename" column names the files in the study archive (the working
repo where they were produced); each ships here under its numbered name.

| # | File | Description | Original filename (study archive) |
|---|------|-------------|-------------|
| 01 | 01_original_top_discoveries.csv | Original top-1000 candidates (pre-ML re-rank, full stats) | Final_Top_Discoveries_FullStats.csv |
| 02 | 02_xgb_discoveries.csv | XGBoost-discovered candidates (post-ML screening) | xgboost_perovskite_discoveries.csv |
| 03 | 03_xgb_discoveries_fullstats.csv | XGBoost discoveries with full statistical columns | xgboost_perovskite_discoveries_FullStats.csv |
| 04 | 04_magpie_discoveries.csv | Magpie-discovered candidates (alternative feature set) | xgboost_perovskite_Magpie_Discoveries.csv |
| 05 | 05_contour_lcoe_penalty.csv | LCOE penalty contour matrix (Eg vs MAE sensitivity) | Contour_Matrix_LCOE_Penalty.csv |
| 06 | 06_multi_dim_risk.csv | Multi-dimensional financial risk matrix | Multi_Dimensional_Financial_Risk.csv |
| 07 | 07_parallel_risk_matrix.csv | Parallel financial risk matrix (Monte Carlo outputs) | Parallel_Financial_Risk_Matrix.csv |
| 08 | 08_risk_topology.csv | Risk topology map for Eg ~1.35 eV target | Risk_Topology_Target_1.35eV.csv |
| 09 | 09_optiona_derated_rankings.csv | Option A composition-aware PCE derating results (3,280 candidates, 50k MC iterations) | results_optionA/Final_Top_Discoveries_OptionA.csv |
