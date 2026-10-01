# Walterbos et al. (2026) HSE06 halide double perovskite gaps (external check data)

`hdp_hse06_subset.csv` holds 8 columns of the table `AnalysisResults/HDP_CombinedInfo_260510.csv` (full table not redistributed here) from:

- L. Walterbos, A. McEwan, R. Shinde and J. George, *Spin-Polarized Electronic Structure and Chemical
  Bonding Data for 2,500+ Halide Double Perovskites*, arXiv:2606.11928 (2026),
  doi:10.48550/arxiv.2606.11928.
- Archive: Zenodo record 20598121, doi:10.5281/zenodo.20598121 ("Luccerboi/HDP_WorkFLow_Analysis v0.1.0").
- Licence: **CC BY 4.0**. Values are copied unchanged; only columns were selected.

The gaps are spin-polarised HSE06 gaps on PBEsol-relaxed Cs2BB'X6 structures, without spin-orbit coupling.
`cond_type` is the authors' classification (metallic, half-metal, semiconductor, insulator).

Regenerate with `python scripts/fetch_walterbos2026.py`. `SOURCE_SHA256` records the sha256 of the source
CSV inside the Zenodo archive; the script fails if the upstream file changes.

Used by `pipeline/s50_review_checks.py` (check R1) to compare our composition-only predictions, trained on
Materials Project GGA/GGA+U gaps, with an independent hybrid-functional data set.
