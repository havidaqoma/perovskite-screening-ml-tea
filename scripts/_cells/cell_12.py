# CELL 7E: DETERMINISTIC RISK PROFILER (FIXED DIMENSIONS)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# USER INPUTS
# ==========================================
TARGET_BANDGAP = 1.15  # The specific material target
# ==========================================

print(f"Calculating Deterministic Analytical Risk for Target Eg = {TARGET_BANDGAP} eV...")

# 1. Sweep Exact Prediction Error (Delta Eg) from -0.6 eV to +0.6 eV
# We use 500 points to ensure a perfectly smooth "Analytical" curve
delta_eg_sweep = np.linspace(-0.6, 0.6, 500)
actual_egs = TARGET_BANDGAP + delta_eg_sweep

# 2. Pure Physics Engine (Vectorized for the sweep)
def analytical_pce_calc(eg_array):
    # Theoretical SQ Curve
    eff = 33.0 - 15.0 * (eg_array - 1.34)**2
    eff = np.clip(eff, 0.0, 33.0)
    # The Physics Cliff: Outside [0.9, 2.5] eV, PV efficiency is zero
    eff[(eg_array < 0.5) | (eg_array > 2.5)] = 0.0 
    return (eff * 0.85) / 100.0 # Real-world 85% factor

# Calculate the sweep and the baseline
pce_curve_frac = analytical_pce_calc(actual_egs)
base_pce_frac = analytical_pce_calc(np.array([TARGET_BANDGAP]))[0]

# Calculate % of Target Efficiency Achieved (The Y-axis for Physics)
# This is the "Pure Calculation Relationship" you requested
sq_saturation_curve = (pce_curve_frac / base_pce_frac) * 100.0

# 3. Pure Economic Engine
def analytical_lcoe_calc(pce_array, future=False):
    # Clip efficiency slightly above zero to avoid division by zero errors in LCOE
    pce_safe = np.clip(pce_array, 0.005, None)
    kw = 1.0 * pce_safe
    if future:
        capex = 25.5 + 0.1 + (150.0 * kw)
        energy = 1471.0 * kw * 17.292 
        costs = capex + (10.0 * kw * 17.292) + (150.0 * kw * 0.555) 
    else:
        capex = 272.1 + 0.1 + (300.0 * kw)
        energy = 1471.0 * kw * 15.622 
        costs = capex + (20.0 * kw * 15.622) + (300.0 * kw * 0.555)
    return costs / energy

# Calculate LCOE Penalties
lcoe_curr_sweep = analytical_lcoe_calc(pce_curve_frac, future=False)
base_lcoe_curr = analytical_lcoe_calc(np.array([base_pce_frac]), future=False)[0]
penalty_curr_curve = np.clip(lcoe_curr_sweep - base_lcoe_curr, 0.0, None)

lcoe_fut_sweep = analytical_lcoe_calc(pce_curve_frac, future=True)
base_lcoe_fut = analytical_lcoe_calc(np.array([base_pce_frac]), future=True)[0]
penalty_fut_curve = np.clip(lcoe_fut_sweep - base_lcoe_fut, 0.0, None)


# --- PUBLICATION-QUALITY PLOTTING (DUAL-AXIS SENSITIVITY) ---
def plot_analytical_sensitivity(x_error, y_sq, y_penalty, title_tech, color_penalty, y_max_penalty):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axis 1: Thermodynamics (% SQ Saturation)
    color1 = 'teal'
    ax1.set_xlabel('Machine Learning Prediction Error ($\Delta E_g$ in eV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('% of Target Efficiency Achieved', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(x_error, y_sq, color=color1, lw=4, label='Efficiency Retained (%)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 125)
    ax1.axhline(100, color=color1, linestyle='--', alpha=0.5)

    # Axis 2: Economics (LCOE Penalty)
    ax2 = ax1.twinx()  
    color2 = color_penalty
    ax2.set_ylabel('LCOE Financial Penalty ($/kWh)', color=color2, fontsize=12, fontweight='bold')
    ax2.plot(x_error, y_penalty, color=color2, lw=4, label='Financial Penalty ($)')
    ax2.fill_between(x_error, 0, y_penalty, color=color2, alpha=0.15)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, y_max_penalty)

    # Highlight our specific XGBoost model error bounds (+/- 0.38 eV)
    ax1.axvspan(-model_mae, model_mae, color='gray', alpha=0.1, label=f'XGBoost Error Window ($\pm${model_mae:.2f} eV)')
    
    plt.title(f"Analytical Risk Profile: Physics vs. Economics\nTarget Eg = {TARGET_BANDGAP} eV | {title_tech}", fontsize=14, fontweight='bold')
    
    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', frameon=True, shadow=True, ncol=2)
    
    fig.tight_layout()
    plt.show()

# Run the plots
plot_analytical_sensitivity(delta_eg_sweep, sq_saturation_curve, penalty_curr_curve, "Current Glass-Based Tech", 'crimson', 0.025)
plot_analytical_sensitivity(delta_eg_sweep, sq_saturation_curve, penalty_fut_curve, "Future 2030 Flexible Tech", 'darkorange', 0.01)