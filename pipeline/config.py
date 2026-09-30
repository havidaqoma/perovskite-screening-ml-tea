"""Frozen configuration for the screening + TEA pipeline.

Every constant here was copied verbatim from the April 2026 notebook
(XGBoost_Mendeleev_MAgpie_v23C.ipynb, extracted to legacy/notebook_cells/).
Changing any value changes the science: record why in IMPLEMENTATION_NOTES.md.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
LEGACY_DIR = ROOT / "raw_data"          # legacy (April 2026) exported outputs

TRAINING_TABLE = DATA_DIR / "training_table.csv"
# sha256 of the source graph dataset the table was exported from (see export_training_table.py)
TRAINING_SOURCE_SHA256 = "3f5e0903d402feebf395d746e5fde68d6bf9824d3102a21264a69e2faed28c71"
LEGACY_FULLSTATS = LEGACY_DIR / "01_original_top_discoveries.csv"   # == notebook cell 7 output

# ---- split (cell_00): np.random.seed(42); shuffle(arange(N)); first 80% train ----
SPLIT_SEED = 42
TRAIN_FRACTION = 0.8
SEMI_THRESHOLD_EV = 0.01            # is_semi = band_gap > 0.01 eV

# ---- models (cell_01) ----
MODEL_SEED = 42
# Optuna best params recorded in xgboost_cascade_pipeline_2.joblib (30 unseeded trials, Apr 27 2026).
# Frozen so the legacy model is reproducible; re-tuning is opt-in (--tune) with a seeded sampler.
BG_PARAMS_LEGACY = {
    "n_estimators": 1239,
    "max_depth": 9,
    "learning_rate": 0.041692845270894896,
    "subsample": 0.9270593186624888,
    "colsample_bytree": 0.6766397361548855,
    "min_child_weight": 2,
}
FE_PARAMS_LEGACY = {
    "n_estimators": 800,
    "max_depth": 9,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
CLS_PARAMS_LEGACY = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8}

# ---- combinatorial generation + physical moat (cell_07) ----
A_SITE = ["K", "Rb", "Cs", "Na", "Li"]
B_SITE = ["Sn", "Ge", "Ti", "Zr", "V", "Nb", "Ta", "Cr", "Mo", "W",
          "Fe", "Mn", "Bi", "Sb", "Cu", "Ag", "Zn", "In", "Ga"]
X_SITE = ["O", "S", "Se", "F", "Cl", "Br", "I"]
TOL_RANGE = (0.75, 1.15)
MU_RANGE = (0.35, 0.95)

# ---- ML screening filter (cell_07) ----
EG_WINDOW_EV = (0.5, 2.5)
EF_MAX_EV_ATOM = 1.0
TOXIC = ("Cd", "Hg", "As", "Tl", "Pb", "U", "Th")

# ---- legacy TEA run (cell_07) ----
TEA_SEED = 42
TEA_ITERATIONS = 50_000
TEA_FUTURE = True

# ---- expected legacy numbers (notebook stdout, Apr 2026) used by the regression gate ----
EXPECTED_LEGACY = {
    "n_train": 25020,
    "n_val": 6255,
    "bg_mae_eV": 0.3498204957238204,
    "fe_mae_eV_atom": 0.0763,          # printed to 4 dp
    "n_generated": 23940,
    "n_steric": 3284,
    "n_candidates": 3280,
}
