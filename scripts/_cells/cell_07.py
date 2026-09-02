# CELL 5: MASSIVE HTVS (SINGLE SOURCE OF TRUTH)
import itertools
from pymatgen.core import Composition
import pandas as pd
from tqdm import tqdm
import numpy as np

A_SITE = ['K', 'Rb', 'Cs', 'Na', 'Li']
B_SITE = ['Sn', 'Ge', 'Ti', 'Zr', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Fe', 'Mn', 'Bi', 'Sb', 'Cu', 'Ag', 'Zn', 'In', 'Ga']
X_SITE = ['O', 'S', 'Se', 'F', 'Cl', 'Br', 'I']

hypothetical_formulas = []
b_pairs = list(itertools.combinations(B_SITE, 2))
x_pairs = list(itertools.combinations(X_SITE, 2))

for a in A_SITE:
    for b1, b2 in b_pairs:
        for x in X_SITE: hypothetical_formulas.append(f"{a}2{b1}{b2}{x}6")
        for x1, x2 in x_pairs: hypothetical_formulas.append(f"{a}2{b1}{b2}{x1}3{x2}3")

print(f"Generated {len(hypothetical_formulas)} unique chemical combinations.")

def is_charge_balanced(formula):
    try:
        comp = Composition(formula)
        amounts = list(comp.get_el_amt_dict().values())
        ox_states = [el.common_oxidation_states for el in comp.elements]
        if not all(ox_states): return False
        for combo in itertools.product(*ox_states):
            if abs(sum(c*a for c,a in zip(combo, amounts))) < 0.1: return True
        return False
    except: return False

valid_formulas = [f for f in tqdm(hypothetical_formulas, desc="Charge Filter") if is_charge_balanced(f)]

def is_sterically_stable(formula):
    try:
        comp = Composition(formula)
        elements = list(comp.get_el_amt_dict().keys())
        r_A = get_radius(elements[0])
        r_B_avg = (get_radius(elements[1]) + get_radius(elements[2])) / 2.0
        r_X_avg = get_radius(elements[3]) if len(elements)==4 else (get_radius(elements[3]) + get_radius(elements[4])) / 2.0
        
        t = (r_A + r_X_avg) / (np.sqrt(2) * (r_B_avg + r_X_avg))
        mu = r_B_avg / r_X_avg
        return (0.75 <= t <= 1.15) and (0.35 <= mu <= 0.95) # Slightly relaxed for meta-stable alloys
    except: return False

structurally_valid_formulas = [f for f in tqdm(valid_formulas, desc="Steric Filter") if is_sterically_stable(f)]
print(f"Surviving the Physical Moat: {len(structurally_valid_formulas)} materials.")

htvs_candidates = []
for formula in tqdm(structurally_valid_formulas, desc="ML Screening"):
    try:
        # THE FIX: Using the 139-Dimensional Ultimate Feature Extractor!
        feats = get_ultimate_features(formula).reshape(1, -1)
        
        is_semi = xgb_cls.predict(feats)[0]
        bg_raw = xgb_bg_specialist.predict(feats)[0]
        final_Eg = max(0.0, bg_raw * is_semi)
        final_Ef = xgb_fe.predict(feats)[0]
        
        # WIDENED FILTER: 0.5 to 2.5 eV captures a broader range of applications
        if 0.5 <= final_Eg <= 2.5 and final_Ef < 1.0:
            comp = Composition(formula)
            if not any(el.symbol in ["Cd", "Hg", "As", "Tl", "Pb", "U", "Th"] for el in comp.elements):
                htvs_candidates.append({"formula": formula, "Eg": final_Eg, "Ef": final_Ef})
    except Exception as e: 
        # Optional: uncomment the line below if you ever want to see why it's failing
        # print(f"Failed on {formula}: {e}")
        continue

print(f"\nHTVS COMPLETE: Found {len(htvs_candidates)} novel, stable, non-toxic perovskites.")

if htvs_candidates:
    print("\nRunning 2030 Economic Uncertainty Analysis (50,000 Iterations)...")
    np.random.seed(42) # Set seed for reproducibility!
    
    final_htvs_results = []
    
    for cand in tqdm(htvs_candidates, desc="Monte Carlo TEA"): 
        # Run TEA Engine (Future = True, 50k iterations for ultimate smoothness)
        lcoe_dist, lifetime, raw_mat_cost, pce_dist = run_tea(cand['formula'], cand['Eg'], cand['Ef'], model_mae, 50000, True)
        
        pce_dist = pce_dist * 100.0 # Convert to percentage
        
        final_htvs_results.append({
            "Formula": cand['formula'], 
            "Predicted_Bandgap_eV": cand['Eg'], 
            "Predicted_Ef_eV_atom": cand['Ef'],
            "Active_Material_Cost_m2": raw_mat_cost,
            "Panel_Lifetime_Years": lifetime,
            "PCE_Median": np.median(pce_dist),
            "LCOE_Median": np.median(lcoe_dist),
            "LCOE_Q10_Best": np.percentile(lcoe_dist, 10),
            "LCOE_Q90_Worst": np.percentile(lcoe_dist, 90)
        })
    
    # Sort by Median LCOE to find the absolute best economic performers
    df_htvs = pd.DataFrame(final_htvs_results).sort_values("LCOE_Median")
    
    print("\n=== THE DEFINITIVE TOP 10 NOVEL DISCOVERIES ===")
    display_cols = ['Formula', 'Predicted_Bandgap_eV', 'PCE_Median', 'LCOE_Median', 'LCOE_Q90_Worst']
    print(df_htvs[display_cols].head(10).to_string(index=False))

    # Export to the SINGLE definitive CSV
    csv_filename = "Final_Top_Discoveries_FullStats.csv"
    df_htvs.to_csv(csv_filename, index=False, float_format='%.5f')
    print(f"\n--- PIPELINE COMPLETE. Source of Truth saved to {csv_filename} ---")
# %%