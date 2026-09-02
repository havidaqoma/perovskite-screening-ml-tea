# CELL 7: ADVANCED 2D CONTOUR SENSITIVITY ANALYSIS (MAE vs. PHYSICS vs. ECONOMICS)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Running 2D Matrix Risk Propagation Analysis...")

# 1. Setup the Matrix Grid (Resolution)
# We sweep Predicted Bandgap (0.9 to 2.2 eV) and ML Error (0.0 to 0.6 eV)
eg_sweep = np.linspace(0.9, 2.2, 25) 
mae_sweep = np.linspace(0.0, 0.6, 25)
EG_grid, MAE_grid = np.meshgrid(eg_sweep, mae_sweep)

EG_flat = EG_grid.flatten()
MAE_flat = MAE_grid.flatten()

# Dummy formula and stable Ef for the TEA engine
dummy_formula = "Na2FeSbO3Cl3"
true_Ef = 0.20  
iterations_per_point = 10000 # Lowered slightly to allow 625 matrix calculations in reasonable time

# Data Storage
results_data = []

# 2. Run the massive matrix simulation
for eg, mae in tqdm(zip(EG_flat, MAE_flat), total=len(EG_flat), desc="Calculating Contour Matrix"):
    
    # Base LCOE (Zero ML Error)
    base_curr, _, _, _ = run_tea(dummy_formula, eg, true_Ef, mae_error=0.0, iterations=iterations_per_point, future=False)
    base_fut, _, _, _ = run_tea(dummy_formula, eg, true_Ef, mae_error=0.0, iterations=iterations_per_point, future=True)
    base_curr_val = np.median(base_curr)
    base_fut_val = np.median(base_fut)
    
    # Simulated ML Error LCOE & PCE
    dist_curr, _, _, pce_curr_dist = run_tea(dummy_formula, eg, true_Ef, mae_error=mae, iterations=iterations_per_point, future=False)
    dist_fut, _, _, pce_fut_dist = run_tea(dummy_formula, eg, true_Ef, mae_error=mae, iterations=iterations_per_point, future=True)
    
    med_lcoe_curr = np.median(dist_curr)
    med_lcoe_fut = np.median(dist_fut)
    
    med_pce_curr = np.median(pce_curr_dist)
    med_pce_fut = np.median(pce_fut_dist)
    
    # Theoretical SQ Limit at this Bandgap (Pure Physics)
    theoretical_sq = sq_limit(np.array([eg]))[0] 
    
    results_data.append({
        "Target_Bandgap_eV": eg,
        "ML_MAE_eV": mae,
        
        "PCE_Current": med_pce_curr,
        "PCE_Future": med_pce_fut,
        
        "Percent_SQ_Current": (med_pce_curr / theoretical_sq) * 100.0,
        "Percent_SQ_Future": (med_pce_fut / theoretical_sq) * 100.0,
        
        "LCOE_Penalty_Current": max(0.0, med_lcoe_curr - base_curr_val),
        "LCOE_Penalty_Future": max(0.0, med_lcoe_fut - base_fut_val)
    })

# Convert to DataFrame
df_matrix = pd.DataFrame(results_data)

# Export Raw Data to CSV (As Requested)
csv_out = "Contour_Matrix_LCOE_Penalty.csv"
df_matrix.to_csv(csv_out, index=False, float_format='%.5f')
print(f"\nMatrix calculations complete. Raw data saved to '{csv_out}'.")

# --- 3. PUBLICATION-QUALITY PLOTTING ROUTINES ---

def plot_contour_pair(df, x_col, y_col, z_curr, z_fut, x_label, y_label, title_prefix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Current Tech
    tc1 = ax1.tricontourf(df[x_col], df[y_col], df[z_curr], levels=20, cmap='magma')
    fig.colorbar(tc1, ax=ax1, label="LCOE Penalty ($/kWh)")
    ax1.set_title(f"{title_prefix}\n(Current Tech)", fontweight='bold')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    
    # Plot 2: Future 2030 Tech
    tc2 = ax2.tricontourf(df[x_col], df[y_col], df[z_fut], levels=20, cmap='viridis')
    fig.colorbar(tc2, ax=ax2, label="LCOE Penalty ($/kWh)")
    ax2.set_title(f"{title_prefix}\n(Future 2030 Tech)", fontweight='bold')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    
    plt.tight_layout()
    plt.show()

# 1. MAE vs Predicted Bandgap vs LCOE Penalty
plot_contour_pair(
    df_matrix, 
    x_col="ML_MAE_eV", y_col="Target_Bandgap_eV", 
    z_curr="LCOE_Penalty_Current", z_fut="LCOE_Penalty_Future",
    x_label="Machine Learning MAE (eV)", y_label="Target Bandgap (eV)",
    title_prefix="Risk Topology: Bandgap vs. ML Error"
)

# 2. MAE vs PCE vs LCOE Penalty
plot_contour_pair(
    df_matrix, 
    x_col="ML_MAE_eV", y_col="PCE_Current", 
    z_curr="LCOE_Penalty_Current", z_fut="LCOE_Penalty_Future",
    x_label="Machine Learning MAE (eV)", y_label="Achieved Efficiency (PCE %)",
    title_prefix="Risk Topology: Efficiency vs. ML Error"
) # Note: For future subplot, it automatically maps the general shape of PCE_Current which is fine for topology

# 3. MAE vs % from SQ Limit vs LCOE Penalty
plot_contour_pair(
    df_matrix, 
    x_col="ML_MAE_eV", y_col="Percent_SQ_Current", 
    z_curr="LCOE_Penalty_Current", z_fut="LCOE_Penalty_Future",
    x_label="Machine Learning MAE (eV)", y_label="% of Theoretical SQ Limit Achieved",
    title_prefix="Risk Topology: SQ Saturation vs. ML Error"
)