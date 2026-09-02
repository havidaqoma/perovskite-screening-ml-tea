# CELL 4: LCOE PARITY PLOTS (CURRENT & FUTURE 2030)
from tqdm import tqdm

lcoe_dft_curr, lcoe_ai_curr = [], []
lcoe_dft_fut, lcoe_ai_fut =[],[]

print("Running LCOE Parity Analysis on Validation Set...")

for i in tqdm(range(len(val_idx))):
    formula = formulas[val_idx][i]
    pred_Eg = val_pred_bg_final[i]
    pred_Ef = val_pred_fe[i]
    true_Eg = y_bg[val_idx][i]
    true_Ef = y_fe[val_idx][i]
    
    # Current Tech
    ai_c, _, _, _ = run_tea(formula, pred_Eg, pred_Ef, model_mae, 1000, False)
    dft_c, _, _, _ = run_tea(formula, true_Eg, true_Ef, 0.0, 1000, False)
    lcoe_ai_curr.append(np.median(ai_c)); lcoe_dft_curr.append(np.median(dft_c))
    
    # Future Tech
    ai_f, _, _, _ = run_tea(formula, pred_Eg, pred_Ef, model_mae, 1000, True)
    dft_f, _, _, _ = run_tea(formula, true_Eg, true_Ef, 0.0, 1000, True)
    lcoe_ai_fut.append(np.median(ai_f)); lcoe_dft_fut.append(np.median(dft_f))
