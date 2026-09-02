# CELL 4A: PLOTTING LCOE PARITY & MAE FOR CURRENT AND FUTURE TECH

# --- PLOT CURRENT ---
curr_mae = mean_absolute_error(lcoe_dft_curr, lcoe_ai_curr)
plt.figure(figsize=(7, 7))
plt.scatter(lcoe_dft_curr, lcoe_ai_curr, alpha=0.3, color='blue', edgecolors='k')
plt.plot([0, 1.0], [0, 1.0], 'r--', lw=2.5)
plt.title("XGBoost LCOE Parity (Current Tech)", fontweight='bold')
plt.xlabel("LCOE from DFT ($/kWh)"); plt.ylabel("LCOE from XGBoost ($/kWh)")
plt.text(0.05, 0.9, f"MAE = ${curr_mae:.4f}/kWh", bbox=dict(facecolor='white', alpha=0.8))
plt.xlim(0, 1.0); plt.ylim(0, 1.0); plt.grid(True, alpha=0.3); plt.show()

# --- PLOT FUTURE ---
fut_mae = mean_absolute_error(lcoe_dft_fut, lcoe_ai_fut)
plt.figure(figsize=(7, 7))
plt.scatter(lcoe_dft_fut, lcoe_ai_fut, alpha=0.3, color='orange', edgecolors='k')
plt.plot([0, 1.0], [0, 1.0], 'r--', lw=2.5)
plt.title("XGBoost LCOE Parity (Future 2030 Tech)", fontweight='bold')
plt.xlabel("LCOE from DFT ($/kWh)"); plt.ylabel("LCOE from XGBoost ($/kWh)")
plt.text(0.025, 0.45, f"MAE = ${fut_mae:.4f}/kWh", bbox=dict(facecolor='white', alpha=0.8))
plt.xlim(0, 1.0); plt.ylim(0, 1.0); plt.grid(True, alpha=0.3); plt.show()