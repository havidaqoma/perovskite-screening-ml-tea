# Stages 3–4 summary (auto-generated from run artifacts)

## Data
- MP database 2026.04.13: 235,270 entries → **188,037 formulas** (ground-state polymorph each), 32.4% semiconductors (gap ≥ 0.1 eV).

## Leakage audit (April data, April model, only the split changes)
| Split | Bandgap MAE (eV) | Val formulas also in train |
|---|---:|---:|
| April protocol (random rows) | 0.350 | 43.2% |
| Grouped by formula | 0.394 | 0.0% |
| Grouped by chemical system | 0.415 | 0.0% |

## Baselines on v2 (pooled out-of-fold MAE, eV; Ef in eV/atom)
| Split | Task | Mean | Ridge | RF | XGB | Roost |
|---|---|---:|---:|---:|---:|---:|
| random | gap_semi | 1.256 | 0.944 | 0.602 | 0.479 | – |
| random | gap_all | 0.973 | 0.627 | 0.325 | 0.257 | – |
| random | ef | 0.770 | 0.260 | 0.110 | 0.090 | – |
| chemsys | gap_semi | 1.256 | 0.946 | 0.616 | 0.491 | 0.488 |
| chemsys | gap_all | 0.973 | 0.627 | 0.333 | 0.264 | – |
| chemsys | ef | 0.770 | 0.260 | 0.114 | 0.093 | – |
| lofo_anion | gap_semi | 1.271 | 9.355 | 0.997 | 1.013 | – |
| lofo_anion | gap_all | 1.318 | 1.162 | 1.051 | 0.993 | – |
| lofo_anion | ef | 0.948 | 0.620 | 0.536 | 0.549 | – |

Gatekeeper ROC-AUC: random 0.968, chemsys 0.967, lofo_anion 0.755

Conformal 90% gap intervals: coverage 0.900 (random), 0.895 (chemsys); median width 1.98 eV.

## Screening funnel (v2)
- Charge-balanced: 6,080
- Rejected: τ not perovskite 4,386; no oxidation-state assignment 420; predicted metal 18; toxic 0
- **Pass: 1,256** (April: 3,284 passed the legacy steric filter; of the April 3,280 candidates, 645 pass v2 geometry+gatekeeper)

## Stability (ML hull distance vs MP competing phases)
- Status: {'unstable_likely': 940, 'stable_likely': 277, 'known_mp': 39}
- Candidates with P(E_hull ≤ thr) ≥ 0.5: {'0': 172, '20': 212, '35': 243, '50': 285, '100': 440}
- **Control (380 known MP A₂BB′X₆, hull rebuilt without them):** ROC-AUC 0.835, Brier 0.151, gate precision 0.66, recall 0.58, E_hull MAE 43 meV

| P(stable) bin | n | mean predicted | observed |
|---|---:|---:|---:|
| (-0.001, 0.2] | 156 | 0.09 | 0.07 |
| (0.2, 0.4] | 89 | 0.29 | 0.29 |
| (0.4, 0.6] | 58 | 0.50 | 0.50 |
| (0.6, 0.8] | 44 | 0.69 | 0.68 |
| (0.8, 1.0] | 33 | 0.87 | 0.76 |

## Joint viability  P(semi) × P(gap 1.0–1.8 eV) × P(E_hull ≤ 50 meV)
- Novel (not in MP): 1,217; flagged hull extrapolation 63, implausible oxidation state 49
- Expected number of viable novel compounds (sum of probabilities, plausible only): **42.0**
- Plausible novel with p ≥ 0.1/0.2/0.3/0.5: {'0.1': 128, '0.2': 51, '0.3': 15, '0.5': 0}

| Top novel candidates | Oxidation states | P(semi) | P(gap PV) | P(stable) | P(viable) |
|---|---|---:|---:|---:|---:|
| Cs2BiAgBr3I3 | Cs+1;Bi+3;Ag+1;Br-1;I-1 | 0.89 | 0.76 | 0.71 | 0.48 |
| Rb2GeMnCl6 | Rb+1;Ge+2;Mn+2;Cl-1 | 0.86 | 0.53 | 0.87 | 0.40 |
| K2GeMnCl6 | K+1;Ge+2;Mn+2;Cl-1 | 0.84 | 0.54 | 0.87 | 0.39 |
| Cs2GeMnCl6 | Cs+1;Ge+2;Mn+2;Cl-1 | 0.85 | 0.50 | 0.88 | 0.37 |
| Cs2SbAgF3Cl3 | Cs+1;Sb+3;Ag+1;F-1;Cl-1 | 0.92 | 0.46 | 0.83 | 0.35 |
| Rb2SbAgF3Cl3 | Rb+1;Sb+3;Ag+1;F-1;Cl-1 | 0.92 | 0.45 | 0.84 | 0.35 |
| Cs2MnCuCl3Br3 | Cs+1;Mn+2;Cu+2;Cl-1;Br-1 | 0.67 | 0.62 | 0.83 | 0.34 |
| K2GeCuF3Cl3 | K+1;Ge+2;Cu+2;F-1;Cl-1 | 0.68 | 0.55 | 0.91 | 0.34 |
| Rb2GeFeCl6 | Rb+1;Ge+2;Fe+2;Cl-1 | 0.85 | 0.48 | 0.81 | 0.33 |
| Cs2GeMnCl3Br3 | Cs+1;Ge+2;Mn+2;Cl-1;Br-1 | 0.86 | 0.53 | 0.72 | 0.33 |

## uMLIP check (MACE-MP-0, rock-salt ordered prototype, hull vs other phases re-relaxed with MACE)
- Finished 40/40
- control_known_mp: n=10, uMLIP E_hull ≤ 50 meV for 100%; median -4 meV
- novel: n=30, uMLIP E_hull ≤ 50 meV for 27%; median 88 meV
- Agreement with MP on known controls (stable ≤ 50 meV yes/no): 100%
- Novel: ML and uMLIP both ≤ 50 meV for 27%

## Gates
- Phase C (ML audit): **PASS**
- Phase D (stability): **PASS**
