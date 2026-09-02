"""
Option A batch runner: rerun run_tea for all 3280 candidates with composition-specific
derating factors. Output: results_optionA/Final_Top_Discoveries_OptionA.csv

Usage:
    python scripts/run_optionA_batch.py
    python scripts/run_optionA_batch.py --iterations 10000  # faster pilot
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from tqdm import tqdm
from tea_engine_optiona import run_tea
from derating import derate_formula, calc_delta_chi, calc_tolerance_factor, calc_mu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pilot', type=int, default=0, help='Run only N candidates (0=all)')
    args = parser.parse_args()

    N_ITER = args.iterations
    SEED = args.seed
    MAE_ERROR = 0.3494

    # Load original CSV (single source of truth)
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Final_Top_Discoveries_FullStats.csv')
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} candidates from {csv_path}")

    if args.pilot > 0:
        df = df.head(args.pilot)
        print(f"PILOT MODE: running first {args.pilot} candidates")

    # Output directory
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results_optionA')
    os.makedirs(out_dir, exist_ok=True)

    # --- Compute derating factors ---
    print("Computing composition-specific derating factors...")
    formulas = df['Formula'].tolist()
    derate_factors = []
    delta_chis = []
    tolerance_factors = []
    mu_ratios = []

    for f in tqdm(formulas, desc="Derating"):
        derate_factors.append(derate_formula(f))
        delta_chis.append(calc_delta_chi(f))
        tolerance_factors.append(calc_tolerance_factor(f))
        mu_ratios.append(calc_mu(f))

    derate_factors = np.array(derate_factors)
    print(f"Derating stats: min={derate_factors.min():.4f} median={np.median(derate_factors):.4f} "
          f"max={derate_factors.max():.4f} std={derate_factors.std():.4f}")

    # --- Run TEA with derating ---
    np.random.seed(SEED)
    results = []

    t0 = time.time()
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Monte Carlo TEA"):
        formula = row['Formula']
        eg = row['Predicted_Bandgap_eV']
        ef = row['Predicted_Ef_eV_atom']
        f_derate = derate_factors[i]

        # Reset seed per candidate for reproducibility
        np.random.seed(SEED)
        lcoe_dist, life, mat_cost, pce_dist = run_tea(
            formula, eg, ef, MAE_ERROR, iterations=N_ITER, future=True, comp_factor=f_derate
        )
        pce_dist_pct = pce_dist * 100.0

        results.append({
            'Formula': formula,
            'Predicted_Bandgap_eV': eg,
            'Predicted_Ef_eV_atom': ef,
            'Active_Material_Cost_m2': mat_cost,
            'Panel_Lifetime_Years': life,
            'Derate_Factor': f_derate,
            'Delta_Chi': delta_chis[i],
            'Tolerance_Factor': tolerance_factors[i],
            'Mu_Ratio': mu_ratios[i],
            'PCE_Median': np.median(pce_dist_pct),
            'LCOE_Median': np.median(lcoe_dist),
            'LCOE_Q10_Best': np.percentile(lcoe_dist, 10),
            'LCOE_Q90_Worst': np.percentile(lcoe_dist, 90),
            'PCE_Old_Median': row.get('PCE_Median', np.nan),
            'LCOE_Old_Median': row.get('LCOE_Median', np.nan),
        })

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save
    result_df = pd.DataFrame(results)
    # Compute rank deltas
    result_df['Rank_Old'] = result_df['LCOE_Old_Median'].rank(method='min')
    result_df['Rank_New'] = result_df['LCOE_Median'].rank(method='min')
    result_df['Rank_Delta'] = result_df['Rank_Old'] - result_df['Rank_New']

    out_path = os.path.join(out_dir, 'Final_Top_Discoveries_OptionA.csv')
    result_df.to_csv(out_path, index=False, float_format='%.5f')
    print(f"Saved to {out_path} ({len(result_df)} rows)")

    # --- Summary ---
    print("\n=== TOP 10 BY NEW LCOE (DERATED) ===")
    top10 = result_df.nsmallest(10, 'LCOE_Median')
    display_cols = ['Formula', 'Derate_Factor', 'PCE_Median', 'LCOE_Median', 'Rank_Old', 'Rank_New', 'Rank_Delta']
    print(top10[display_cols].to_string(index=False, float_format='%.4f'))

    print("\n=== TOP 10 BY OLD LCOE (ORIGINAL) ===")
    top10_old = result_df.nsmallest(10, 'LCOE_Old_Median')
    print(top10_old[display_cols].to_string(index=False, float_format='%.4f'))

    # Derating distribution
    print(f"\n=== DERATING DISTRIBUTION ===")
    print(f"  min={derate_factors.min():.4f}  median={np.median(derate_factors):.4f}  "
          f"mean={derate_factors.mean():.4f}  max={derate_factors.max():.4f}  std={derate_factors.std():.4f}")

    # PCE conservation check
    pce_old = result_df['PCE_Old_Median'].values
    pce_new = result_df['PCE_Median'].values
    valid = ~np.isnan(pce_old)
    if valid.any():
        ratio = pce_new[valid] / np.clip(pce_old[valid], 1e-10, None)
        print(f"  PCE_new/PCE_old: min={ratio.min():.4f} median={np.median(ratio):.4f} max={ratio.max():.4f}")
        print(f"  PCE_new <= PCE_old everywhere: {(pce_new[valid] <= pce_old[valid] * 1.01).all()}")

    print("\n=== DONE ===")


if __name__ == '__main__':
    main()
