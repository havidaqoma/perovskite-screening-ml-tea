# Lead-Free Double Perovskite Screening: ML + Monte Carlo TEA

[![CI](https://github.com/havidaqoma/perovskite-screening-ml-tea/actions/workflows/ci.yml/badge.svg)](https://github.com/havidaqoma/perovskite-screening-ml-tea/actions/workflows/ci.yml)

A screening study that chains XGBoost property prediction with a 50,000-iteration
Monte Carlo techno-economic analysis (TEA) to find economically viable, lead-free
double perovskites for solar cells. 3,280 candidates screened, scored on levelized
cost of electricity (LCOE) under two manufacturing scenarios, and re-ranked with a
composition-aware efficiency derating model.

Plain version: we train a fast model to guess two properties that decide whether a
perovskite can work (bandgap and formation energy), then simulate the factory, the
module lifetime, and the electricity market 50,000 times per candidate to ask which
materials stay cheap when things go wrong.

This repository holds two things:

1. The **original study** (notebook pipeline in `src/`, `scripts/`, `raw_data/`,
   `results_optionA/`). Its headline table is below.
2. The **v2 audit pipeline** (`pipeline/`, `runs/`, `paper/`) that revisits the
   original study and backs the manuscript. See
   [v2 audit pipeline (manuscript)](#v2-audit-pipeline-manuscript).

## Headline results of the original study

These are the ORIGINAL study's numbers. The v2 audit revisits them: the original
random train/validation split leaks (43.2% of validation rows share a formula with
a training row), and the band-gap MAE of 0.350 eV on the random split rises to
0.415 eV when whole chemical systems are held out
(`runs/v2/metrics/leakage_audit.json`).

| Metric | Value |
|---|---|
| Bandgap prediction MAE (test set) | 0.3494 eV |
| Formation energy MAE (test set) | 0.0763 eV/atom |
| Candidates screened | 3,280 |
| Monte Carlo iterations per candidate | 50,000 |
| Manufacturing scenarios | Current (rigid FTO glass), Future 2030 (roll-to-roll on PET) |

Key insight: thin-film economic decoupling. Under roll-to-roll manufacturing,
area-dependent balance-of-system costs dominate over efficiency penalties, so LCOE
is less sensitive to a moderate efficiency derating than the conventional wisdom
assumes.

> The v2 audit revisits this insight and does not support it for a U.S. utility
> plant: with real area-scaled balance-of-system costs, a module must reach about
> 20% efficiency (20.25% in the low-cost, 35-year, 0.7%/yr scenario) to match
> silicon (`runs/v2/metrics/tea_v2.json`, `e4_breakeven`).

## v2 audit pipeline (manuscript)

The manuscript "Band-gap uncertainty, thermodynamic stability and area-scaled plant
costs limit the techno-economic viability of lead-free double perovskite
photovoltaics" is under submission. arXiv: to be added.

The v2 pipeline re-runs the screen with evaluation and cost models that a reviewer
can check stage by stage:

- **Grouped evaluation.** Models are scored on random, chemical-system and
  leave-one-anion-family-out splits, so a formula (or a whole chemical system)
  never sits on both sides of a split.
- **Conformal band-gap intervals.** Split-conformal intervals give each predicted
  gap a calibrated range, and screening uses P(semiconductor) and P(gap in the
  PV window) instead of a point estimate.
- **Calibrated stability.** An ML hull distance against Materials Project competing
  phases, with the formation-energy error propagated into P(stable). It is checked
  on 400 known A2BB'X6 compounds and, for the shortlist, with MACE-MP-0 relaxations.
- **Area-resolved U.S. utility TEA.** A real-dollar LCOE engine built from the NREL
  Q1 2023 cost benchmark: area-scaled costs are divided by efficiency, power-scaled
  costs are not, and module replacement resets degradation. Every parameter carries
  its source ID (`pipeline/tea_params.py`, `data/sources/SOURCES.md`).
- **Stability-gated Pareto front.** Candidates are ranked on P(absorber),
  P(stable) and LCOE, and only stability-gated candidates enter the front.

Numbers from this run (read from the committed artifacts): 1,256 v2 candidates
pass the corrected geometry screen (`runs/v2/metrics/screen_v2.json`); 202 enter
the stability-gated pool and 3 sit on the stability-gated Pareto front, 1 of them
not in Materials Project (`runs/v2/metrics/pareto.json`). All four v2 gates pass
(`runs/v2/gates/`).

### Stage order

The single entry point is `pipeline/run.py`:

```bash
# v2: s10 -> s11 -> s11b -> s13 -> s14 -> s20 -> s22 -> s30 -> s40 -> s41 -> s50, then the gates
python -m pipeline.run --v2 --run-id v2 --gate
# one stage only (repeatable), e.g. the Pareto front and robustness numbers
python -m pipeline.run --v2 --run-id v2 --only s40 --only s41
# legacy reproduction of the original study
python -m pipeline.run --run-id legacy_repro --gate
```

| Stage | Module | Needs |
|---|---|---|
| s10 | `s10_mp_dataset.py`: fresh Materials Project training set | `MP_API_KEY` (first run; cached after) |
| s11, s11b | `s11_ml_eval.py`, `s11b_leakage_audit.py`: grouped evaluation, conformal intervals, leakage audit | s10 output |
| s12 | `s12_roost.py`: Roost baseline (optional, GPU, torch + aviary) | s11 folds |
| s13 | `s13_final_models.py`: final models + calibration | s10 output |
| s14 | `s14_screen_v2.py`: corrected geometry + probabilistic screen | s13 models |
| s20 | `s20_stability.py`: ML hull distance, 400-compound control | `MP_API_KEY` (competing phases; cached after) |
| s21 | `s21_umlip.py`: MACE-MP-0 check (optional, GPU, torch + mace-torch + ase) | `MP_API_KEY`, shortlist |
| s22 | `s22_shortlist.py`: joint P(viable), plausibility flags | s14, s20 |
| s30, s31, s32 | `s30_tea_v2.py`, `s31_tea_figures.py`, `s32_tea_summary.py`: TEA v2 | s30 needs the s13 model (`runs/v2/models/v2_final.joblib`, not redistributed); s31/s32 read committed files |
| s40, s41 | `s40_pareto.py`, `s41_robustness.py`: Pareto front, robustness | s40 reads committed files; s41 also needs `runs/v2/predictions/oof_v2.csv` from s11 (not redistributed) |
| s50 | `s50_review_checks.py`: HSE06 comparison with Walterbos et al. (2026), hull-threshold, O&M/inverter and oxidation-state sensitivity | `data/external/walterbos2026/` (shipped; `scripts/fetch_walterbos2026.py` re-downloads and checks its hash), the s13 model and the s10 training table (not redistributed); stops if its 50 meV / base-case rows do not reproduce s22 and s30 |

Offline with the shipped files alone: the test suite, s31, s32, s40,
`paper/make_figures_v2.py` (it rebuilds byte-identical PNGs) and 224 of the 225
keys in `paper/paper_numbers.py` (the missing one, `v2_gap_max`, reads the s10 training table `runs/v2/data/train_v2.csv`, not redistributed). Two gates pass directly (`python -m
pipeline.gate_tea_v2 runs/v2`, `python -m pipeline.gate_stability runs/v2`).
`pipeline.gate_pareto` checks that the figures are newer than their inputs by file
time, which a fresh clone does not preserve: run `python paper/make_figures_v2.py`
first and it passes. `pipeline.gate_ml_audit` check C-G6 needs the regenerated
`runs/v2/splits/` and `runs/v2/predictions/` (not redistributed), so it fails
until s11 has been re-run. Running a gate rewrites its report in
`runs/v2/gates/`; the committed reports are from the full run. Stages s10, s20
and s21 query Materials Project and need your own free key
(https://next-gen.materialsproject.org/api) in `MP_API_KEY` or in a local `.env`
(gitignored). Our run used MP database version 2026.04.13
(`runs/v2/metrics/dataset_v2.json`). The two GPU stages run in a separate
environment built by `env/setup_wsl_gpu.sh` (pins in `env/freeze-wsl-gpu.txt`).
Exact pins for the main environment are in `requirements-v2.lock.txt`.

### What is not shipped, and why

- `data/mp_cache/` (Materials Project query cache, about 359 MB, MP data under
  CC BY 4.0): rebuilt by s10/s20 with your own key.
- `runs/v2/data/`, `runs/v2/splits/`, `runs/v2/predictions/`, `runs/v2/models/`,
  `runs/v2/features/`: heavy, regenerable intermediates (training table, folds,
  out-of-fold predictions, `*.joblib` models, feature matrices). Re-run the stages
  above to rebuild them. Each committed `runs/*/stage_*.json` manifest records the
  sha256 of every input and output, so a rebuilt file can be checked against the
  one we used.
- Licence-restricted source copies (Lazard, press articles): cited by number only.
  The verbatim-quote test skips when they are absent.
- Manuscript text and review material.

### Number-to-artifact map

`paper/numbers.json` maps every number printed in the manuscript to the artifact
it was read from (`value`, `fmt`, `src` per key). `paper/paper_numbers.py`
rebuilds it from `runs/`; `paper/make_figures_v2.py` rebuilds `paper/figures/`.
A few keys read `runs/v2/data/train_v2.csv` (not redistributed) or the original model
file, so a full rebuild needs the s10 stage first.

### Electronic Supplementary Information (ESI)

The ESI of the manuscript (Supplementary Notes S1 to S14, Figures S1 to S8,
Tables S1 to S12) is built from the same artifacts. The code and data behind it
ship here; the ESI text itself is part of the manuscript and is not.

| File | What it holds |
|---|---|
| `paper/make_si_figures.py` | Builds Figures S1 to S8 (`paper/figures/figS*.{png,pdf}`) and the data plotted in each (`paper/figures/si_data/figS*.csv`) |
| `paper/si_numbers.py` | Builds `paper/si_numbers.json`: the 180 numbers and method settings quoted in the ESI, each with its source. Settings are read from the pipeline code itself (imported, or matched on the source line), and the script stops if a pattern is not found |
| `paper/si_refs_verified.json` | The two ESI-only references (ASE, FIRE), checked against OpenAlex |
| `paper/make_si_excel.py` | Builds the supplementary workbook `paper/build/SI_data_v2.xlsx`: the data behind every main and ESI figure, every number with its source, every TEA assumption, and the references |

```bash
python paper/make_si_figures.py   # Figures S2-S8 offline; S1 needs runs/v2/predictions/ (re-run s11) and is skipped otherwise
python paper/si_numbers.py        # offline
python paper/make_si_excel.py     # offline
```

The numbering follows the order in which the ESI cites the figures: S5 is the
break-even map for every module cost and burn-in setting, S6 the candidate-level
LCOE, S7 the Sobol indices and S8 the comparison with HSE06 band gaps.

## Repository layout

```
src/                     reusable modules (novelty check, stability check, derating, TEA engine)
scripts/baseline_gate.py   baseline acceptance gate (offline)
scripts/run_optionA_batch.py  composition-aware derating batch run
scripts/_cells/            notebook cells as importable scripts (16 cells)
XGBoost_Mendeleev_MAgpie_v23C.ipynb  full training notebook (outputs cleared)
tests/                     pytest suite (offline; no network, no API key)
raw_data/                  9 CSVs, the full screening data package (see raw_data/README.md)
results_optionA/           Option A derated rankings + lifetime-corrected variant
pipeline/                  v2 audit pipeline (stages s01-s50, gates, TEA v2, MP client)
runs/                      committed v2 and legacy-reproduction artifacts (metrics, gates, manifests, CSVs)
paper/                     numbers.json and si_numbers.json maps, figure and SI scripts, main and ESI figures
data/                      training_table.csv (legacy training set), sources/ (redistributable TEA source texts)
env/                       optional GPU environment for the Roost and MACE stages
examples/ outputs/         quickstart script; runtime output dir
REPRODUCIBILITY.md         seeds, environment, data provenance
DATA_PROVENANCE.md         licenses and attribution for every input
```

## Quick start (no API key, no internet needed)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python examples/quickstart_tea.py                    # re-runs the TEA on the top 5 candidates
python -m pytest -q                                  # offline test suite
```

## Retraining the model (needs a free Materials Project key)

The notebook `XGBoost_Mendeleev_MAgpie_v23C.ipynb` trains the XGBoost cascade from
Materials Project data. Set your key via the environment (never hardcode it):

```bash
export MP_API_KEY=***   # free at https://next-gen.materialsproject.org/api
```

or copy `.env.example` to `.env`. The key is read at runtime; nothing in this
repository contains one.

## Limitations, stated honestly

- Formation-energy MAE of 0.0763 eV/atom is good for a descriptor-based model but
  is not density functional theory accuracy; borderline candidates need DFT.
- The TEA is a cost model, not a fab measurement. Its scenario parameters
  (deposition yield, module lifetime, financing) carry wide Monte Carlo
  distributions on purpose.
- Novelty checks run against Materials Project; a zero match supports novelty but
  does not prove a compound has never been made.
- No device (solar cell) was built or tested in this study. This is screening plus
  technico-economics.

## License and citation

Code and data generated in this study: MIT (see `LICENSE`). Third-party computed
data remains under its own terms; see `DATA_PROVENANCE.md` (Materials Project data
is CC BY 4.0 and requires attribution). If you reuse this work, cite the
`CITATION.cff` entry.
