# CELL 6: PHYSICAL ERROR VS ECONOMIC RISK PLOT
bg_errors = np.abs(y_bg[val_idx] - val_pred_bg_final)
lcoe_errors_curr = np.abs(np.array(lcoe_dft_curr) - np.array(lcoe_ai_curr))
lcoe_errors_fut = np.abs(np.array(lcoe_dft_fut) - np.array(lcoe_ai_fut))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(bg_errors, lcoe_errors_curr, alpha=0.4, color='crimson')
ax1.set_title("Physical Error vs Economic Risk (Current Tech)", fontweight='bold')
ax1.set_xlabel("XGBoost Bandgap Prediction Error (eV)")
ax1.set_ylabel("LCOE Error ($/kWh)")
ax1.set_ylim(-0.05, 1.0); ax1.grid(True, alpha=0.3)

ax2.scatter(bg_errors, lcoe_errors_fut, alpha=0.4, color='teal')
ax2.set_title("Physical Error vs Economic Risk (Future 2030)", fontweight='bold')
ax2.set_xlabel("XGBoost Bandgap Prediction Error (eV)")
ax2.set_ylabel("LCOE Error ($/kWh)")
ax2.set_ylim(-0.01, 0.2); ax2.grid(True, alpha=0.3)
plt.show()

# Export Results
df_htvs.to_csv("xgboost_perovskite_discoveries.csv", index=False, float_format='%.4f')
print("\n--- PIPELINE COMPLETE. Data saved to xgboost_perovskite_discoveries.csv ---")