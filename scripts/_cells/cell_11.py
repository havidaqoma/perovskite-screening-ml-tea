# CELL 7D: SINGLE-TARGET RISK TOPOLOGY (MAE vs. PCE vs. LCOE PENALTY) - FIXED
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# USER INPUTS
# ==========================================
TARGET_BANDGAP = 1.35  # Manually input your target bandgap here!
N_SAMPLES = 100000     # Increased to 100k for ultra-smooth high-res hexbin mapping
# ==========================================

print(f"Generating Economic Risk Contours for Target Bandgap: {TARGET_BANDGAP} eV...")

# 1. Generate X-Axis: A uniform spread of Machine Learning MAE
maes = np.random.uniform(0.0, 0.60, N_SAMPLES)

# 2. Simulate the AI's predictions based on that MAE
sampled_egs = np.random.normal(TARGET_BANDGAP, maes)

# 3. Physics Engine: Calculate resulting Power Conversion Efficiency (PCE)
def fast_physics_pce(eg_array):
    eff = 33.0 - 15.0 * (eg_array - 1.34)**2
    eff = np.clip(eff, 0.0, 33.0)
    eff[(eg_array < 0.9) | (eg_array > 2.5)] = 0.0 
    return (eff * 0.85) / 100.0 # Return as fractional efficiency

pces_frac = fast_physics_pce(sampled_egs)

# Prevent absolute 0% efficiency to avoid infinity in LCOE math
pces_frac = np.clip(pces_frac, 0.005, None) 

# Calculate the perfect baseline PCE for this specific bandgap
base_pce_frac = fast_physics_pce(np.array([TARGET_BANDGAP]))[0]

# 4. Y-Axis Data
y_pce_percent = pces_frac * 100.0
y_sq_saturation = (pces_frac / base_pce_frac) * 100.0

# 5. Economic Engine: Calculate LCOE Penalty (Z-Axis)
def vectorized_lcoe(pce_array, future=False):
    kw = 1.0 * pce_array
    if future:
        capex = 25.5 + 0.1 + (150.0 * kw)
        energy = 1471.0 * kw * 17.292 # 30 years @ 4% discount
        costs = capex + (10.0 * kw * 17.292) + (150.0 * kw * 0.555) 
    else:
        capex = 272.1 + 0.1 + (300.0 * kw)
        energy = 1471.0 * kw * 15.622 # 25 years @ 4% discount
        costs = capex + (20.0 * kw * 15.622) + (300.0 * kw * 0.555)
    return costs / energy

lcoe_curr = vectorized_lcoe(pces_frac, future=False)
lcoe_fut = vectorized_lcoe(pces_frac, future=True)

base_lcoe_curr = vectorized_lcoe(np.array([base_pce_frac]), future=False)[0]
base_lcoe_fut = vectorized_lcoe(np.array([base_pce_frac]), future=True)[0]

z_penalty_curr = np.clip(lcoe_curr - base_lcoe_curr, 0.0, None)
z_penalty_fut = np.clip(lcoe_fut - base_lcoe_fut, 0.0, None)

# --- SAVE RAW DATA TO CSV ---
df_plot = pd.DataFrame({
    "ML_MAE_eV": maes,
    "Achieved_PCE_Percent": y_pce_percent,
    "SQ_Saturation_Percent": y_sq_saturation,
    "LCOE_Penalty_Current": z_penalty_curr,
    "LCOE_Penalty_Future": z_penalty_fut
})
csv_filename = f"Risk_Topology_Target_{TARGET_BANDGAP}eV.csv"
df_plot.to_csv(csv_filename, index=False, float_format="%.5f")
print(f"Data saved to {csv_filename}")

# --- PUBLICATION-QUALITY PLOTTING (HEXBIN C-MAPPING) ---

def plot_specific_hexbin(x, y, z_curr, z_fut, y_label, title_prefix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Current Tech
    # We use C=z_curr to color the hexes by LCOE Penalty. mincnt=1 removes the hallucinated empty space.
    hb1 = ax1.hexbin(x, y, C=z_curr, gridsize=60, cmap='magma', vmin=0, vmax=0.30, mincnt=1)
    cb1 = fig.colorbar(hb1, ax=ax1, label="LCOE Penalty ($/kWh)")
    ax1.set_title(f"{title_prefix}\n(Current Tech)", fontweight='bold')
    ax1.set_xlabel("Machine Learning MAE (eV)")
    ax1.set_ylabel(y_label)
    
    # Plot 2: Future Tech
    hb2 = ax2.hexbin(x, y, C=z_fut, gridsize=60, cmap='viridis', vmin=0, vmax=0.08, mincnt=1)
    cb2 = fig.colorbar(hb2, ax=ax2, label="LCOE Penalty ($/kWh)")
    ax2.set_title(f"{title_prefix}\n(Future 2030 Tech)", fontweight='bold')
    ax2.set_xlabel("Machine Learning MAE (eV)")
    ax2.set_ylabel(y_label)
    
    # Add our model marker
    for ax in [ax1, ax2]:
        ax.axvline(x=model_mae, color='white', linestyle='--', lw=2.5)
        # Bounding box ensures text is readable over both dark purple and bright yellow
        ax.text(model_mae + 0.02, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.15, 
                f"Our XGBoost\n(MAE = {model_mae:.2f} eV)", color='black', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.show()

# 1. MAE vs PCE vs LCOE Penalty
plot_specific_hexbin(
    maes, y_pce_percent, z_penalty_curr, z_penalty_fut, 
    y_label="Achieved Efficiency (PCE %)", 
    title_prefix=f"Risk Topology: Efficiency vs. LCOE Penalty\nTarget Eg = {TARGET_BANDGAP} eV"
)

# 2. MAE vs SQ Saturation vs LCOE Penalty
plot_specific_hexbin(
    maes, y_sq_saturation, z_penalty_curr, z_penalty_fut, 
    y_label="% of Theoretical SQ Limit Achieved", 
    title_prefix=f"Risk Topology: SQ Saturation vs. LCOE Penalty\nTarget Eg = {TARGET_BANDGAP} eV"
)
# --- PUBLICATION-QUALITY PLOTTING (ZOOMED FOR CLARITY) ---

def plot_zoomed_hexbin(x, y, z_curr, z_fut, y_label, title_prefix, y_min, y_max):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Current Tech
    hb1 = ax1.hexbin(x, y, C=z_curr, gridsize=200, cmap='viridis', vmin=0, vmax=0.05, mincnt=1)
    cb1 = fig.colorbar(hb1, ax=ax1, label="LCOE Financial Penalty ($/kWh)")
    ax1.set_title(f"{title_prefix}\n(Current Tech)", fontweight='bold')
    ax1.set_xlabel("Machine Learning MAE (eV)")
    ax1.set_ylabel(y_label)
    ax1.set_ylim(y_min, y_max) # ZOOM IN
    
    # Plot 2: Future Tech 
    hb2 = ax2.hexbin(x, y, C=z_fut, gridsize=200, cmap='viridis', vmin=0, vmax=0.05, mincnt=1)
    cb2 = fig.colorbar(hb2, ax=ax2, label="LCOE Financial Penalty ($/kWh)")
    ax2.set_title(f"{title_prefix}\n(Future 2030 Tech)", fontweight='bold')
    ax2.set_xlabel("Machine Learning MAE (eV)")
    ax2.set_ylabel(y_label)
    ax2.set_ylim(y_min, y_max) # ZOOM IN
    
    # Add our model marker and an academic note about the zoomed axis
    for ax in [ax1, ax2]:
        ax.axvline(x=model_mae, color='white', linestyle='--', lw=2.5)
        ax.text(model_mae + 0.02, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.15, 
                f"Our XGBoost\n(MAE = {model_mae:.2f} eV)", color='black', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round,pad=0.3'))
        
        # Add academic transparency note
        ax.text(0.02, 0.03, "*Catastrophic Phase-Shift failures (<10% PCE)\noccur but are cropped for visual clarity.", 
                transform=ax.transAxes, fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    plt.show()

# 1. MAE vs PCE vs LCOE Penalty (Zoomed to 10% - 30% Efficiency)
y_max_limit_pce = (base_pce_frac * 100) + 1.0
plot_zoomed_hexbin(
    maes, y_pce_percent, z_penalty_curr, z_penalty_fut, 
    y_label="Achieved Efficiency (PCE %)", 
    title_prefix=f"Risk Topology: Efficiency vs. LCOE Penalty\nTarget Eg = {TARGET_BANDGAP} eV",
    y_min=10.0, y_max=y_max_limit_pce
)

# 2. MAE vs SQ Saturation vs LCOE Penalty (Zoomed to 35% - 102% Saturation)
plot_zoomed_hexbin(
    maes, y_sq_saturation, z_penalty_curr, z_penalty_fut, 
    y_label="% of Theoretical SQ Limit Achieved", 
    title_prefix=f"Risk Topology: SQ Saturation vs. LCOE Penalty\nTarget Eg = {TARGET_BANDGAP} eV",
    y_min=35.0, y_max=102.0
)