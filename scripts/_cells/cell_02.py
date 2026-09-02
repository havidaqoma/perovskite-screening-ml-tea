# CELL 2B: INSTANT MODEL LOADING & VALIDATION PLOTTING
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error
import shap

# FIX: We must define the DummyClassifier structure so joblib knows how to rebuild it from the save file.
class DummyClassifier:
    def predict(self, X): return np.ones(X.shape[0])

print("Loading pre-trained XGBoost Cascade...")
pipeline = joblib.load("xgboost_cascade_pipeline_2.joblib")

xgb_cls = pipeline['gatekeeper']
xgb_bg_specialist = pipeline['bg_specialist']
xgb_fe = pipeline['fe_specialist']
model_mae = pipeline['model_mae']

print(f"Models loaded! Saved Training MAE: {model_mae:.4f} eV")

# --- 1. RUN INFERENCE ON VALIDATION SET ---
val_pred_cls = xgb_cls.predict(X_features[val_idx])
val_pred_bg_raw = xgb_bg_specialist.predict(X_features[val_idx])
val_pred_bg_final = np.clip(val_pred_bg_raw * val_pred_cls, 0.0, None)
val_pred_fe = xgb_fe.predict(X_features[val_idx])

# --- 2. PLOT FORMATION ENERGY PARITY ---
ef_mae = mean_absolute_error(y_fe[val_idx], val_pred_fe)
plt.figure(figsize=(8, 7))
hb = plt.hexbin(y_fe[val_idx], val_pred_fe, gridsize=50, cmap='plasma', norm=LogNorm(), mincnt=1)
plt.colorbar(hb, label='Log10(Density of Materials)')
min_ef, max_ef = np.min(y_fe[val_idx]), np.max(y_fe[val_idx])
plt.plot([min_ef, max_ef], [min_ef, max_ef], 'w--', lw=2.5, label="Perfect Physics")
plt.title("XGBoost Formation Energy Parity", fontsize=15, fontweight='bold')
plt.xlabel("True DFT Formation Energy ($E_f$ in eV/atom)", fontsize=13)
plt.ylabel("Predicted Formation Energy ($E_f$ in eV/atom)", fontsize=13)
plt.text(min_ef + 0.1, max_ef - 0.5, f"MAE = {ef_mae:.4f} eV/atom", fontsize=12, bbox=dict(facecolor='white', alpha=0.9))
plt.legend(loc='lower right'); plt.grid(True, alpha=0.2); plt.show()

# --- 3. PLOT BANDGAP PARITY ---
bg_mae = mean_absolute_error(y_bg[val_idx], val_pred_bg_final)
plt.figure(figsize=(8, 7))
hb = plt.hexbin(y_bg[val_idx], val_pred_bg_final, gridsize=50, cmap='inferno', norm=LogNorm(), mincnt=1)
plt.colorbar(hb, label='Log10(Density of Materials)')
min_bg, max_bg = 0.0, 4.0
plt.plot([min_bg, max_bg], [min_bg, max_bg], 'w--', lw=2.5, label="Perfect Physics")
plt.title("XGBoost Cascade Bandgap Parity", fontsize=15, fontweight='bold')
plt.xlabel("True DFT Bandgap (eV)", fontsize=13)
plt.ylabel("Predicted Bandgap (eV)", fontsize=13)
plt.text(0.2, 3.5, f"MAE = {bg_mae:.4f} eV", fontsize=12, bbox=dict(facecolor='white', alpha=0.9))
plt.xlim(0, 4); plt.ylim(0, 4); plt.legend(loc='lower right'); plt.grid(True, alpha=0.2); plt.show()

# --- 4. SHAP EXPLAINER ---
print("\nGenerating SHAP Explainer...")
explainer_bg = shap.TreeExplainer(xgb_bg_specialist)
shap_values_bg = explainer_bg.shap_values(X_features[val_idx])
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_bg, X_features[val_idx], feature_names=final_feature_names, max_display=15, show=False)
plt.title("SHAP Feature Importance: Top 15 Driving Physics", fontweight='bold')
plt.tight_layout()
plt.show()
# %%