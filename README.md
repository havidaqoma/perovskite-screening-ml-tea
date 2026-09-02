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

## Headline results

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
data/ examples/ outputs/   runtime dirs (empty; created by code or left for your caches)
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
