# CELL 8: MATERIALS PROJECT RETROSPECTIVE VALIDATION
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.check_novelty import validate_novelty

warnings.filterwarnings("ignore")

# An absent key intentionally selects the offline cache path.
MP_API_KEY = os.environ.get("MP_API_KEY")

# --- 1. LOAD OUR AI DISCOVERIES ---
df_ai = pd.read_csv("xgboost_perovskite_discoveries_FullStats.csv")
ai_formulas = df_ai["Formula"].tolist()

print(f"Loaded {len(df_ai)} AI-generated candidates.")
print("Validating candidate formulas against the Materials Project cache/API.\n")

# --- 2. FAIL-CLOSED NOVELTY VALIDATION ---
report_df, exit_code = validate_novelty(
    ai_formulas,
    mp_api_key=MP_API_KEY,
    online=bool(MP_API_KEY),
)
if exit_code != 0:
    unknown_count = int((report_df["status"] == "unknown").sum())
    print(f"Novelty validation failed closed: {unknown_count} unresolved candidate(s).")
    sys.exit(exit_code)

# --- 3. MERGE AND ANALYZE RESOLVED MATCHES ---
known = report_df[report_df["status"] == "known"].copy()
novel_count = int((report_df["status"] == "novel").sum())

if not known.empty:
    # Only merge resolved MP matches. Unresolved rows cannot be called novel.
    df_validation = pd.merge(df_ai, known, on="Formula", how="inner")
    synthesized_count = int(df_validation["is_synthesized"].fillna(False).sum())

    print("\n" + "=" * 50)
    print("      RETROSPECTIVE GROUND TRUTH RESULTS")
    print("=" * 50)
    print(f"Total AI Candidates Generated : {len(df_ai)}")
    print(f"Completely Novel Materials    : {novel_count} (Resolved no match)")
    print(f"Matches found in MP Database  : {len(df_validation)}")
    print(f"Physically Synthesized in Lab : {synthesized_count} materials")
    print("=" * 50)

    # The novelty report intentionally contains no MP bandgap measurements.
    # Preserve the plot path for enriched reports, but guard it so a plain
    # novelty validation cannot fabricate a blind bandgap comparison.
    plot_columns = {"MP_DFT_Bandgap_eV", "Predicted_Bandgap_eV"}
    if len(df_validation) > 0 and plot_columns.issubset(df_validation.columns):
        blind_mae = np.mean(
            np.abs(
                df_validation["Predicted_Bandgap_eV"]
                - df_validation["MP_DFT_Bandgap_eV"]
            )
        )
        print(f"\n>>> BLIND VALIDATION MAE against MP DFT: {blind_mae:.4f} eV <<<")

        plt.figure(figsize=(7, 7))
        plt.scatter(
            df_validation["MP_DFT_Bandgap_eV"],
            df_validation["Predicted_Bandgap_eV"],
            c=df_validation["is_synthesized"],
            cmap="bwr",
            edgecolor="k",
            s=80,
            alpha=0.8,
        )

        min_v = 0.0
        max_v = max(
            df_validation["MP_DFT_Bandgap_eV"].max(),
            df_validation["Predicted_Bandgap_eV"].max(),
        ) + 0.5
        plt.plot([min_v, max_v], [min_v, max_v], "k--", lw=2)
        plt.title(
            "Blind Retrospective Validation against Materials Project",
            fontweight="bold",
        )
        plt.xlabel("True MP DFT Bandgap (eV)")
        plt.ylabel("XGBoost Predicted Bandgap (eV)")
        plt.text(
            0.5,
            max_v - 0.5,
            f"Blind MAE: {blind_mae:.3f} eV",
            bbox=dict(facecolor="white", alpha=0.8),
        )
        plt.grid(True, alpha=0.3)
        plt.show()

        df_validation.to_csv("Validated_MP_Matches.csv", index=False)
        print("Saved detailed matches to 'Validated_MP_Matches.csv'")
else:
    print(
        f"No resolved Materials Project matches; {novel_count} candidate(s) were "
        "explicitly resolved as novel."
    )
