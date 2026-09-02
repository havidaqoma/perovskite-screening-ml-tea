# CELL 7F: HIGH-SPEED PARALLEL DEVELOPMENTAL RISK TOPOLOGY (LITERATURE ALIGNED)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

# ==========================================
TARGET_BANDGAP = 1.35
MC_SAMPLES = 10000    
N_THREADS = 12       
GRID_RES = 100       
# ==========================================

print(f"Initializing Parallel Risk Engine on {N_THREADS} threads...")

mae_sweep = np.linspace(0.01, 0.60, GRID_RES)
maturity_sweep = np.linspace(20.0, 100.0, GRID_RES)

# 3. Vectorized Physics & Economic Function (NOW ALIGNED WITH CELL 3)
def calculate_lcoe_vector(eg_array, maturity_pct):
    # Physics
    sq_raw = 33.0 - 15.0 * (eg_array - 1.34)**2
    sq_raw = np.clip(sq_raw, 0.5, 33.0)
    sq_raw[(eg_array < 0.5) | (eg_array > 2.8)] = 0.5
    pce = (sq_raw * (maturity_pct / 100.0)) / 100.0
    
    # ALIGNED Economics (Future 2030)
    kw = 1.0 * pce
    # CAPEX: $31/m2 (Module) + $80/m2 (BOS) = $111/m2 Area Cost
    capex = 31.0 + 80.0 + (150.0 * kw)
    # Energy: 1471 kWh/yr * 30 years discounted at 4% w/ degradation (~17.292 multiplier)
    energy = 1471.0 * kw * 17.292 
    costs = capex + (10.0 * kw * 17.292) + (150.0 * kw * 0.555)
    return costs / (energy + 1e-9)

def compute_grid_point(curr_mae, curr_mat):
    # Baseline LCOE
    baseline_lcoe = calculate_lcoe_vector(np.array([TARGET_BANDGAP]), curr_mat)[0]
    
    # Monte Carlo Spread
    eg_dist = np.random.normal(TARGET_BANDGAP, curr_mae, MC_SAMPLES)
    lcoe_dist = calculate_lcoe_vector(eg_dist, curr_mat)
    
    # Return LCOE MAE Risk
    return np.mean(np.abs(lcoe_dist - baseline_lcoe))

start_time = time.time()
tasks = [(m, mat) for mat in maturity_sweep for m in mae_sweep]

results = Parallel(n_jobs=N_THREADS)(
    delayed(compute_grid_point)(m, mat) for m, mat in tasks
)

z_risk_matrix = np.array(results).reshape(GRID_RES, GRID_RES)
print(f"Calculation Complete in {time.time() - start_time:.2f} seconds.")

# --- PUBLICATION-QUALITY PLOTTING ---
MAE_grid, MATURITY_grid = np.meshgrid(mae_sweep, maturity_sweep)
plt.figure(figsize=(11, 8))

levels = np.linspace(0, 0.1, 50)
cp = plt.contourf(MAE_grid, MATURITY_grid, z_risk_matrix, levels=levels, cmap='YlOrRd', extend='both')
cbar = plt.colorbar(cp)
cbar.set_label('Financial Risk (LCOE MAE in $/kWh)', fontsize=12, fontweight='bold')

plt.title(f"Financial Volatility Map: AI Error vs. Material Maturity\n(Target Eg = {TARGET_BANDGAP} eV | Future 2030 Tech)", 
          fontweight='bold', fontsize=14)
plt.xlabel("Machine Learning Prediction Error (MAE in eV)", fontsize=12)
plt.ylabel("Material Maturity (% of SQ Limit achieved)", fontsize=12)

plt.axvline(x=model_mae, color='black', linestyle=':', lw=2)
plt.text(model_mae + 0.01, 25, f"Our XGBoost Model\n(MAE={model_mae:.2f} eV)", 
         fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.grid(True, alpha=0.15, linestyle='--')
plt.tight_layout()
plt.show()
# %%