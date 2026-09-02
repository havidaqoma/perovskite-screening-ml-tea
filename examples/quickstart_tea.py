"""Fast offline demo: run the shipped TEA engine on the top screened candidates.

Reads the real screening output (raw_data/09_optiona_derated_rankings.csv) and
re-runs the Monte Carlo technico-economic analysis for the top candidates under
both manufacturing scenarios. No API key, no internet, no model files needed.

Usage:  python examples/quickstart_tea.py
Takes a few seconds (5,000 MC iterations per candidate here vs 50,000 in the study).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tea_engine_optiona import run_tea, sq_limit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKINGS = os.path.join(ROOT, "raw_data", "09_optiona_derated_rankings.csv")


def main() -> None:
    df = pd.read_csv(RANKINGS).sort_values("Rank_New").head(5)
    print(f"{'formula':>12} {'Eg[eV]':>7} {'SQlimit[%]':>10} "
          f"{'LCOE now':>9} {'LCOE 2030':>10} {'PCE med[%]':>10}")
    for _, row in df.iterrows():
        formula = str(row["Formula"])
        eg = float(row["Predicted_Bandgap_eV"])
        ef = float(row["Predicted_Ef_eV_atom"])
        lifetime = float(row.get("Panel_Lifetime_Years", 20.0))
        kwargs = dict(formula=formula, pred_Eg=eg, pred_Ef=ef,
                      mae_error=0.0, iterations=5000, lifetime_years=lifetime)

        np.random.seed(42)  # same seed for both scenarios: only manufacturing differs
        cur_lcoe, cur_life, _mat, cur_pce = run_tea(future=False, **kwargs)
        np.random.seed(42)
        fut_lcoe, _life, _mat, fut_pce = run_tea(future=True, **kwargs)

        print(f"{formula:>12} {eg:7.3f} {100.0 * sq_limit(eg):10.1f} "
              f"{np.median(cur_lcoe):9.3f} {np.median(fut_lcoe):10.3f} "
              f"{100.0 * float(np.median(cur_pce)):10.1f}")
    print("\nLCOE in $/kWh (median of the Monte Carlo distribution); PCE median from the")
    print("derated engine. Same seed both scenarios, so the columns differ only through")
    print("the manufacturing model. Full-study settings: scripts/run_optionA_batch.py.")


if __name__ == "__main__":
    main()
