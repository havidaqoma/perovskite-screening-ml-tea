# CELL 7: THEORETICAL SENSITIVITY ANALYSIS (BANDGAP MAE vs. LCOE ERROR)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Running Theoretical Risk Propagation Analysis...")

# 1. Define a hypothetical "Perfect" Perovskite as our mathematical baseline
# We center it at the exact peak of the Shockley-Queisser limit (1.34 eV)
# and give it a high stability (Formation Energy = 0.2 eV/atom)
dummy_formula = "Na2FeSbO3Cl3" 
true_Eg = 1.33  
true_Ef = 0.20  

# 2. Sweep the Machine Learning MAE from 0.0 (Perfect) to 1.0 eV (Terrible)
mae_sweep = np.linspace(0.0, 1.0, 40)
lcoe_penalty_current = []
lcoe_penalty_future = []

# 3. Calculate the "Zero-Risk" Baseline LCOE (MAE = 0.0)
baseline_curr, _, _, _ = run_tea(dummy_formula, true_Eg, true_Ef, mae_error=0.0, iterations=50000, future=False)
baseline_curr_val = np.median(baseline_curr)

baseline_fut, _, _, _ = run_tea(dummy_formula, true_Eg, true_Ef, mae_error=0.0, iterations=50000, future=True)
baseline_fut_val = np.median(baseline_fut)

# 4. Run the Monte Carlo Engine for every level of MAE
for simulated_mae in tqdm(mae_sweep, desc="Simulating Economic Risk"):
    # Current Technology
    dist_curr, _, _, _ = run_tea(dummy_formula, true_Eg, true_Ef, mae_error=simulated_mae, iterations=50000, future=False)
    # The "Error" is how much the uncertainty drove the expected LCOE up from the perfect baseline
    lcoe_penalty_current.append(np.median(dist_curr) - baseline_curr_val)
    
    # Future 2030 Technology
    dist_fut, _, _, _ = run_tea(dummy_formula, true_Eg, true_Ef, mae_error=simulated_mae, iterations=50000, future=True)
    lcoe_penalty_future.append(np.median(dist_fut) - baseline_fut_val)

# --- 5. PUBLICATION-QUALITY SENSITIVITY PLOT ---
plt.figure(figsize=(9, 6))

# Plot Current Tech
plt.plot(mae_sweep, lcoe_penalty_current, color='crimson', lw=3, label='Current Glass-Based Tech')
plt.fill_between(mae_sweep, 0, lcoe_penalty_current, color='crimson', alpha=0.1)

# Plot Future Tech
plt.plot(mae_sweep, lcoe_penalty_future, color='teal', lw=3, label='Future 2030 Flexible R2R Tech')
plt.fill_between(mae_sweep, 0, lcoe_penalty_future, color='teal', alpha=0.1)

# Highlight our specific model's performance on the curve
plt.axvline(x=model_mae, color='k', linestyle='--', lw=2, label=f"Our XGBoost Model (MAE = {model_mae:.2f} eV)")
current_penalty_at_model = np.interp(model_mae, mae_sweep, lcoe_penalty_current)
plt.scatter([model_mae], [current_penalty_at_model], color='black', s=80, zorder=5)

plt.title("Economic Sensitivity to Machine Learning Error", fontsize=15, fontweight='bold')
plt.xlabel("Machine Learning Bandgap MAE (eV)", fontsize=13)
plt.ylabel("Induced LCOE Penalty ($ / kWh)", fontsize=13)

# Add text explaining the physics
plt.text(0.05, 0.85, "Non-linear penalty driven by\nShockley-Queisser limits", 
         transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.xlim(0, 0.5)
plt.ylim(0, 0.05)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()