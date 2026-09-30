# Stage 5 summary: TEA v2 (auto-generated from run artifacts)

Basis: NREL Q1-2023 utility PV benchmark (100 MWdc single-axis tracker, 2022 USD, minimum sustainable price), simple real LCOE, 35-year plant. Area-scaled costs (module, structural BOS, installation labour) are divided by efficiency; power-scaled costs (inverter, electrical BOS, office work) are not.

## Engine checks
- NREL cost structure rebuilt from its components: $973/kWdc vs printed $973/kWdc (MSP); $1160 vs $1161 (MMP).
- c-Si reference LCOE: median **$58.7/MWh** (P10–P90 $44–82); Lazard 2024 utility range $29–92/MWh; IRENA 2023 global average $44/MWh.
- Consistency: a module priced like c-Si with c-Si durability breaks even at 20.50% (c-Si module: 20.5%).

## The April "thin-film economic decoupling" claim
| Perovskite module cost | LCOE at 8% ÷ LCOE at 18% | Area-scaled share of capex at 10% |
|---|---:|---:|
| $50/m² | 1.79× | 84% |
| $85/m² | 1.86× | 87% |
| $335/m² | 2.06× | 94% |

LCOE falls strictly with efficiency in every case. **The decoupling claim does not survive an area-aware cost model.**

## Break-even module efficiency (to match the c-Si median)
- $50/m², 35-year life, 0.7 %/yr: **20.2%**
- $85/m², 35-year life, 0.7 %/yr: **26.0%**
- 53 of 90 durability/cost scenarios never break even at any efficiency ≤ 40% (the single-junction SQ limit is 33.7%).

## Candidates (1,256 v2 screen survivors, calibrated band-gap posterior, P(semiconductor), stability)
Efficiency = f_SQ × SQ(gap). f_SQ scenarios: 0.21 = demonstrated Cs₂AgBiBr₆ cell; 0.50 = intermediate; **0.61 = today's c-Si module relative to its own SQ limit** (the best a mature module technology achieves); 0.82 = lead-halide lab-cell ceiling (no cell-to-module loss; not a module value).

| Scenario | Definition | Expected # competitive AND stable | … after uMLIP veto | # with P ≥ 0.5 | max P(LCOE ≤ c-Si) |
|---|---|---:|---:|---:|---:|
| demonstrated | f 0.21, $50/m², 35 y, 0.7 %/yr | 0.0 | 0.0 | 0 | 0.00 |
| realistic | f 0.61, $85/m², 15 y, 2 %/yr | 0.0 | 0.0 | 0 | 0.00 |
| optimistic | f 0.61, $50/m², 35 y, 0.7 %/yr | 4.5 | 3.8 | 0 | 0.18 |
| ceiling | f 0.82, $50/m², 35 y, 0.7 %/yr | 64.0 | 55.9 | 9 | 0.82 |

uMLIP veto: 22 of 40 MACE-checked compounds are > 50 meV above the other phases and are zeroed.

| Top candidates, optimistic scenario | MP status | P(LCOE ≤ c-Si) optimistic | … ceiling | P(stable) | MACE E_hull (meV) |
|---|---|---:|---:|---:|---:|
| K2SbAgCl6 | known_mp | 0.16 | 0.77 | 1.00 | 32 |
| K2AgInCl6 | known_mp | 0.12 | 0.53 | 1.00 | 6 |
| K2GeMnCl6 | stable_likely | 0.12 | 0.57 | 0.87 | 29 |
| K2MnZnCl6 | stable_likely | 0.11 | 0.45 | 0.90 | 49 |
| K2SnFeS3F3 | stable_likely | 0.09 | 0.37 | 0.83 | – |
| K2AgInF6 | known_mp | 0.07 | 0.44 | 1.00 | – |
| K2MoFeO3F3 | stable_likely | 0.07 | 0.33 | 0.88 | – |
| K2FeSbS3F3 | stable_likely | 0.08 | 0.36 | 0.84 | – |
| K2NbTaO6 | known_mp | 0.06 | 0.39 | 1.00 | – |
| K2CrMnO3F3 | stable_likely | 0.07 | 0.37 | 0.89 | – |

Rank agreement between the optimistic and ceiling scenarios: Spearman ρ = 0.16. **The candidate ranking depends on the efficiency assumption.**

Caveat on the ordering: K-site compounds dominate the top of the list partly because Cs and Rb have no market price (USGS). Reagent-grade prices are used for them, which adds about $4–7/m² of absorber cost and roughly 2–3% of LCOE. At probabilities this low, that is enough to reorder the list, so the order carries no weight.

## What drives LCOE (Sobol total-order indices, band gap 1.34 eV)
| Parameter | S_T |
|---|---:|
| f_sq | 0.467 |
| module_m2 | 0.365 |
| cf_ac | 0.158 |
| module_life | 0.119 |
| deg | 0.022 |
| r | 0.020 |
| field_scale | 0.001 |
| fom_kwac | 0.000 |
| sbos_scale | 0.000 |

Absorber raw-material cost is immaterial: the most expensive composition ($11.5/m²) changes LCOE by 4.9%. Repeating half the installation labour at each module replacement raises the 10-year-module LCOE by 8.8%.

## Gate
Phase E: **PASS** (7/7)
