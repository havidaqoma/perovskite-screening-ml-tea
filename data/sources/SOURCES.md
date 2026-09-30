# Source registry for TEA v2 (Phase E)

Retrieved 2026-09-29. Every TEA parameter in `pipeline/tea_params.py` points to one ID below. Each parameter has a `kind`:

- **direct**: the value is printed in the source text. The verbatim quote is checked by `tests/test_tea_v2.py::test_direct_quotes_verbatim` whenever the local text copy is present.
- **figure**: the value is printed as a data label in a figure (so there is no text layer). The page and image are recorded here, and the value is cross-checked by gate E-G1.
- **derived**: arithmetic on direct/figure values. The formula is stored with the parameter.
- **scenario**: a declared assumption with no single source. It is always swept or reported as a range and is never presented as sourced.

| ID | Source | URL / DOI | Local copy (gitignored unless noted) | Licence note |
|---|---|---|---|---|
| NREL23 | Ramasamy et al., *U.S. Solar Photovoltaic System and Energy Storage Cost Benchmarks, With Minimum Sustainable Price Analysis: Q1 2023*, NREL/TP-7A40-87303 (Sept 2023), 2022 USD | https://docs.nlr.gov/docs/fy23osti/87303.pdf (direct download blocked; identical 52-page PDF from a mirror, title page verified) | `sources/nrel_87303_q1_2023_benchmarks.txt` (tracked), `sources/pdf/mirror_87303.pdf`, figure renders `pdf/nrel87303_p39..41.png` | US Government work (text copy tracked; PDF and figure renders not redistributed) |
| ATB24 | NREL Annual Technology Baseline 2024, Utility-Scale PV | https://atb.nlr.gov/electricity/2024/utility-scale_pv | `sources/nrel_atb2024_utility_pv.md` (tracked) | US Government work |
| LAZ24 | Lazard, *Levelized Cost of Energy+ (June 2024)*, LCOE v17.0 | https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf | `sources/lazard_lcoeplus_2024.txt`, `pdf/lazard_*.png` | **No redistribution**: numbers cited only; the file is never committed (local copy not redistributed) |
| EPFL25 | *Techno-economic analysis framework for perovskite solar module production at various manufacturing capacities*, Renewable Energy 256 (2026), via pv magazine 2025-07-03 | 10.1016/j.renene.2025.123752 (Crossref-verified); https://www.pv-magazine.com/2025/07/03/bottom-up-cost-model-for-perovskite-solar-module-manufacturing/ | `sources/pvmag_epfl_perovskite_tea_2025.md` | Press article, not committed (local copy not redistributed) |
| IRENA23 | IRENA, *Renewable power generation costs in 2023*, via pv magazine 2024-09-27 | https://www.pv-magazine.com/2024/09/27/global-average-solar-lcoe-stood-at-0-044-kwh-in-2023-says-irena/ | `sources/pvmag_irena_lcoe_2023.md` | Press article, not committed (local copy not redistributed) |
| ZHANG22 | Zhang et al., *Hydrogenated Cs2AgBiBr6 for significantly improved efficiency of lead-free inorganic double perovskite solar cell*, Nat. Commun. 13, 3397 (2022) | 10.1038/s41467-022-31016-w (Crossref-verified) | `sources/zhang_natcommun2022_cs2agbibr6.md` (tracked, CC BY 4.0) | CC BY 4.0 |
| USGS24 | USGS *Mineral Commodity Summaries 2024* (individual commodity sheets; 2023 estimates) | https://pubs.usgs.gov/periodicals/mcs2024/mcs2024-<commodity>.pdf | `sources/usgs_mcs2024_*.txt` (tracked) | US Government work |
| RUHLE16 | Rühle, *Tabulated values of the Shockley–Queisser limit for single junction solar cells*, Sol. Energy (2016). Used only as a check on `pipeline/sq.py` (peak 33.7% near 1.34 eV) | 10.1016/j.solener.2016.02.015 (Crossref-verified) | none | citation only |

## Figure-read values (NREL23, 2022 USD, 100-MWdc single-axis tracker, 20.5%-efficient 2.57 m² modules, ILR 1.34)

Read from the printed data labels on pages 39–41 (Figures 13–15), with no pixel estimation:

| Figure | Item | MSP | MMP |
|---|---|---:|---:|
| 13 (PV only, $/kWdc) | Module / Inverter / SBOS / EBOS / Fieldwork / Officework / Other / **Total** | 253 / 35 / 129 / 146 / 236 / 63 / 111 / **973** | 372 / 48 / 128 / 176 / 236 / 66 / 134 / **1,161** |
| 13 (PV+ESS, $/kWdc) | SBOS / Fieldwork | 138 / 295 | 137 / 295 |
| 14 | Inverter $/kWac; SBOS $/m² (PV+ESS) | 47; 28 | 65; 30 (28 with 45X credit) |
| 15 | EBOS $/kWac; Fieldwork $/m² (PV+ESS) | 196; 61 | 236; 61 |

Cross-checks that confirm the units and the area-scaling (gate E-G1):
- $/kWdc × 0.205 kWdc/m² reproduces the printed $/m²: SBOS 138 → 28.3 (printed 28); fieldwork 295 → 60.5 (printed 61).
- $/kWac ÷ 1.34 reproduces the printed $/kWdc: inverter 47 → 35.1 (35), 65 → 48.5 (48); EBOS 196 → 146.3 (146), 236 → 176.1 (176).
- The PV-only MSP components sum to 973. The text layer states "$0.97/Wdc" (MSP) and "$1.16/Wdc" (MMP).
