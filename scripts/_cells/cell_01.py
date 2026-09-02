# CELL 2: BAYESIAN OPTIMIZATION, XGBOOST CASCADE & MODEL SAVING
import shap
import optuna
import joblib
import os
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Silence the hyper-verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------------------------------------------------------------
# STAGE 1: DYNAMIC BINARY GATEKEEPER
# -------------------------------------------------------------------------------------
unique_classes = np.unique(is_semi_y[train_idx])

if len(unique_classes) > 1:
    print("--- STAGE 1: Training Binary Gatekeeper ---")
    xgb_cls = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, n_jobs=-1, random_state=42)
    xgb_cls.fit(X_features[train_idx], is_semi_y[train_idx])
else:
    print("--- STAGE 1 SKIPPED: Dataset contains only Semiconductors (Class 1). ---")
    class DummyClassifier:
        def predict(self, X): return np.ones(X.shape[0])
    xgb_cls = DummyClassifier()

# -------------------------------------------------------------------------------------
# STAGE 2: BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA)
# -------------------------------------------------------------------------------------
print("\n--- STAGE 2: Running Bayesian Optimization (Optuna) ---")
semi_mask_train = (is_semi_y[train_idx] == 1)

# We split the training data internally so Optuna has a "blind" validation set to test against.
X_tune_bg, X_test_bg, y_tune_bg, y_test_bg = train_test_split(
    X_features[train_idx][semi_mask_train], y_bg[train_idx][semi_mask_train], test_size=0.2, random_state=42)

def bg_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 600, 1500), 
        'max_depth': trial.suggest_int('max_depth', 6, 12),           
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True), 
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),      
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5) 
    }
    model = xgb.XGBRegressor(**param, n_jobs=-1, random_state=42)
    model.fit(X_tune_bg, y_tune_bg, eval_set=[(X_test_bg, y_test_bg)], verbose=False)
    
    preds = model.predict(X_test_bg)
    return mean_absolute_error(y_test_bg, preds)

print("Tuning Bandgap Regressor (Running 30 Mathematical Trials)...")
bg_study = optuna.create_study(direction='minimize')
bg_study.optimize(bg_objective, n_trials=30) 
print(f"Best Optuna Parameters Found: {bg_study.best_params}")

# -------------------------------------------------------------------------------------
# STAGE 3: TRAINING THE FINAL SPECIALIST REGRESSORS
# -------------------------------------------------------------------------------------
print("\n--- STAGE 3: Training Final Models ---")
xgb_bg_specialist = xgb.XGBRegressor(**bg_study.best_params, n_jobs=-1, random_state=42)
xgb_bg_specialist.fit(X_features[train_idx][semi_mask_train], y_bg[train_idx][semi_mask_train])

xgb_fe = xgb.XGBRegressor(n_estimators=800, max_depth=9, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
xgb_fe.fit(X_features[train_idx], y_fe[train_idx])

# -------------------------------------------------------------------------------------
# STAGE 4: INFERENCE AND THE CASCADE MULTIPLICATION
# -------------------------------------------------------------------------------------
print("\n--- INFERENCE ON GLOBAL VALIDATION SET ---")
val_pred_cls = xgb_cls.predict(X_features[val_idx])
val_pred_bg_raw = xgb_bg_specialist.predict(X_features[val_idx])

val_pred_bg_final = np.clip(val_pred_bg_raw * val_pred_cls, a_min=0.0, a_max=None)
val_pred_fe = xgb_fe.predict(X_features[val_idx])

# -------------------------------------------------------------------------------------
# STAGE 5: EVALUATION & PUBLICATION PLOTTING
# -------------------------------------------------------------------------------------
final_mae = mean_absolute_error(y_bg[val_idx], val_pred_bg_final)
print(f"\n>>> FINAL OPTIMIZED XGBOOST MAE: {final_mae:.4f} eV <<<")

plt.figure(figsize=(8, 7))
hb = plt.hexbin(y_bg[val_idx], val_pred_bg_final, gridsize=50, cmap='inferno', norm=LogNorm(), mincnt=1)
cb = plt.colorbar(hb, label='Log10(Density of Materials)')
min_val, max_val = plt.xlim()
plt.plot([min_val, max_val], [min_val, max_val], 'w--', lw=2.5, label="Perfect Physics")

plt.title("Optuna-Optimized XGBoost Cascade: Bandgap Parity", fontsize=15, fontweight='bold')
plt.xlabel("True DFT Bandgap (eV)", fontsize=13)
plt.ylabel("Cascade Predicted Bandgap (eV)", fontsize=13)
plt.text(min_val+0.2, max_val-0.5, f"MAE = {final_mae:.4f} eV", fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='k'))
plt.legend(loc='lower right')
plt.grid(True, alpha=0.2)
plt.show()

# -------------------------------------------------------------------------------------
# STAGE 6: SAVE THE PIPELINE TO DISK
# -------------------------------------------------------------------------------------
model_mae = final_mae

print("\n--- SAVING AI PIPELINE TO DISK ---")
pipeline_package = {
    'gatekeeper': xgb_cls,
    'bg_specialist': xgb_bg_specialist,
    'fe_specialist': xgb_fe,
    'model_mae': model_mae,
    'optuna_params': bg_study.best_params
}

save_path = "xgboost_cascade_pipeline_2.joblib"
joblib.dump(pipeline_package, save_path)
print(f"Successfully saved all models and parameters to '{save_path}'!")

# -------------------------------------------------------------------------------------
# STAGE 7: SHAP EXPLAINER
# -------------------------------------------------------------------------------------
print("\n--- RUNNING SHAP EXPLAINER ---")
explainer_bg = shap.TreeExplainer(xgb_bg_specialist)
shap_values_bg = explainer_bg.shap_values(X_features[val_idx])

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_bg, X_features[val_idx], feature_names=final_feature_names, max_display=15, show=False)
plt.title("SHAP Feature Importance: Top 15 Driving Physics", fontweight='bold')
plt.tight_layout()
plt.show()