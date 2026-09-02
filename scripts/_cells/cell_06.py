# CELL 4B: FORMATION ENERGY & LCOE PARITY PLOTS
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

# --- 1. FORMATION ENERGY PARITY PLOT ---
ef_mae = mean_absolute_error(y_fe[val_idx], val_pred_fe)
print(f"\n>>> XGBOOST FORMATION ENERGY MAE: {ef_mae:.4f} eV/atom <<<")

plt.figure(figsize=(8, 7))
hb = plt.hexbin(y_fe[val_idx], val_pred_fe, gridsize=50, cmap='plasma', norm=LogNorm(), mincnt=1)
cb = plt.colorbar(hb, label='Log10(Density of Materials)')

min_ef, max_ef = np.min(y_fe[val_idx]), np.max(y_fe[val_idx])
plt.plot([min_ef, max_ef], [min_ef, max_ef], 'w--', lw=2.5, label="Perfect Physics")

plt.title("XGBoost Formation Energy Parity", fontsize=15, fontweight='bold')
plt.xlabel("True DFT Formation Energy ($E_f$ in eV/atom)", fontsize=13)
plt.ylabel("Predicted Formation Energy ($E_f$ in eV/atom)", fontsize=13)
plt.text(min_ef + 0.1*(max_ef-min_ef), max_ef - 0.1*(max_ef-min_ef), 
         f"MAE = {ef_mae:.4f} eV/atom", fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='k'))
plt.legend(loc='lower right')
plt.grid(True, alpha=0.2)
plt.show()


# --- 2. TECHNO-ECONOMIC ANALYSIS (LCOE) ---
lcoe_dft_curr, lcoe_ai_curr = [], []
lcoe_dft_fut, lcoe_ai_fut = [], []

print("\nRunning LCOE Parity Analysis on Validation Set...")

for i in tqdm(range(len(val_idx)), desc="Monte Carlo TEA"):
    formula = formulas[val_idx][i]
    pred_Eg = val_pred_bg_final[i]
    pred_Ef = val_pred_fe[i]
    true_Eg = y_bg[val_idx][i]
    true_Ef = y_fe[val_idx][i]
    
    # Current Tech
    ai_c, _, _, _ = run_tea(formula, pred_Eg, pred_Ef, model_mae, 1000, False)
    dft_c, _, _, _ = run_tea(formula, true_Eg, true_Ef, 0.0, 1000, False)
    lcoe_ai_curr.append(np.median(ai_c))
    lcoe_dft_curr.append(np.median(dft_c))
    
    # Future Tech
    ai_f, _, _, _ = run_tea(formula, pred_Eg, pred_Ef, model_mae, 1000, True)
    dft_f, _, _, _ = run_tea(formula, true_Eg, true_Ef, 0.0, 1000, True)
    lcoe_ai_fut.append(np.median(ai_f))
    lcoe_dft_fut.append(np.median(dft_f))


# --- 3. LCOE PARITY PLOTS ---

# Plot Current Tech
curr_mae = mean_absolute_error(lcoe_dft_curr, lcoe_ai_curr)
plt.figure(figsize=(7, 7))
plt.scatter(lcoe_dft_curr, lcoe_ai_curr, alpha=0.3, color='crimson', edgecolors='k')
plt.plot([0, 1.0], [0, 1.0], 'k--', lw=2.5)
plt.title("XGBoost LCOE Parity (Current Tech)", fontsize=14, fontweight='bold')
plt.xlabel("LCOE from True DFT ($/kWh)", fontsize=12)
plt.ylabel("LCOE from XGBoost ($/kWh)", fontsize=12)
plt.text(0.05, 0.9, f"MAE = ${curr_mae:.4f}/kWh", fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='k'))
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.show()

# Plot Future Tech
fut_mae = mean_absolute_error(lcoe_dft_fut, lcoe_ai_fut)
plt.figure(figsize=(7, 7))
plt.scatter(lcoe_dft_fut, lcoe_ai_fut, alpha=0.3, color='teal', edgecolors='k')
plt.plot([0, 0.5], [0, 0.5], 'k--', lw=2.5)
plt.title("XGBoost LCOE Parity (Future 2030 Tech)", fontsize=14, fontweight='bold')
plt.xlabel("LCOE from True DFT ($/kWh)", fontsize=12)
plt.ylabel("LCOE from XGBoost ($/kWh)", fontsize=12)
plt.text(0.025, 0.45, f"MAE = ${fut_mae:.4f}/kWh", fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='k'))
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)
plt.grid(True, alpha=0.3)
plt.show()