"""
Baseline reproduction gate: verify tea_engine_optiona.py (comp_factor=None)
reproduces the original CSV results within MC noise.

Strategy: pick 5 candidates from the CSV, re-run run_tea with identical params
(seed 42, future=True, 50k iterations, MAE=0.3494), compare PCE_Median/LCOE_Median.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from tea_engine_optiona import run_tea

# --- Load original CSV for ground truth ---
csv_path = os.path.join(os.path.dirname(__file__), '..', 'Final_Top_Discoveries_FullStats.csv')
df = pd.read_csv(csv_path)

# Pick 5 candidates spanning the ranking (top-1, top-5, mid, low, bottom)
indices = [0, 4, 100, 500, 2000]
test_candidates = df.iloc[indices][['Formula', 'Predicted_Bandgap_eV', 'Predicted_Ef_eV_atom',
                                     'PCE_Median', 'LCOE_Median', 'LCOE_Q90_Worst']].copy()

# Ground truth values from CSV
gt_pce = test_candidates['PCE_Median'].values
gt_lcoe = test_candidates['LCOE_Median'].values

# --- Re-run with comp_factor=None (legacy behavior) ---
MAE_ERROR = 0.3494  # from notebook: model_mae
N_ITER = 50000
SEED = 42

results = []
for i, row in test_candidates.iterrows():
    formula = row['Formula']
    eg = row['Predicted_Bandgap_eV']
    ef = row['Predicted_Ef_eV_atom']

    # Run with seed
    np.random.seed(SEED)
    lcoe_dist, life, mat_cost, pce_dist = run_tea(
        formula, eg, ef, MAE_ERROR, iterations=N_ITER, future=True, comp_factor=None
    )
    pce_dist_pct = pce_dist * 100.0

    pce_median = np.median(pce_dist_pct)
    lcoe_median = np.median(lcoe_dist)

    results.append({
        'Formula': formula,
        'GT_PCE': row['PCE_Median'],
        'RE_PCE': pce_median,
        'PCE_Delta': pce_median - row['PCE_Median'],
        'GT_LCOE': row['LCOE_Median'],
        'RE_LCOE': lcoe_median,
        'LCOE_Delta': lcoe_median - row['LCOE_Median'],
    })

res = pd.DataFrame(results)
print("=== BASELINE REPRODUCTION GATE ===\n")
print(res.to_string(index=False, float_format='%.4f'))

# Acceptance criteria
pce_max_abs_delta = res['PCE_Delta'].abs().max()
lcoe_max_pct_delta = (res['LCOE_Delta'].abs() / res['GT_LCOE'].abs().clip(lower=1e-10)).max() * 100

print(f"\nPCE max |delta|: {pce_max_abs_delta:.4f} pp")
print(f"LCOE max |% delta|: {lcoe_max_pct_delta:.2f}%")

# Tight gate: PCE within 0.5pp, LCOE within 5% (MC noise)
PCE_GATE = 0.5  # percentage points
LCOE_GATE = 5.0  # percent

pce_pass = pce_max_abs_delta < PCE_GATE
lcoe_pass = lcoe_max_pct_delta < LCOE_GATE

print(f"\nPCE gate (< {PCE_GATE}pp): {'PASS ✓' if pce_pass else 'FAIL ✗'}")
print(f"LCOE gate (< {LCOE_GATE}%): {'PASS ✓' if lcoe_pass else 'FAIL ✗'}")
print(f"\nOVERALL: {'PASS ✓' if pce_pass and lcoe_pass else 'FAIL ✗'}")
