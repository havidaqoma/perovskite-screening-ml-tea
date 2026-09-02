# FIGURE 3b: PUBLICATION-QUALITY SCATTER PLOT 
# Economic Risk vs. Optical Physics for 3,208 Novel Candidates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text # pip install adjustText for collision-free labels

# 1. Load the Full Statistics Data
# (Assuming the file was saved from Cell 5/6 of your pipeline)
df = pd.read_csv("xgboost_perovskite_discoveries_FullStats.csv")

# Sort by the Worst-Case Risk (Q90) to find the absolute safest economic bets
df_sorted = df.sort_values(by="LCOE_Q90_Worst", ascending=True).reset_index(drop=True)
top_10 = df_sorted.head(10)
the_rest = df_sorted.iloc[10:]

# 2. Setup Publication-Quality Aesthetics
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5
})
fig, ax = plt.subplots(figsize=(10, 7))

# 3. Plot the Bulk of the Candidates (Background)
# We use a scatter plot colored by Active Material Cost ($/m^2)
scatter = ax.scatter(
    the_rest['Predicted_Bandgap_eV'], 
    the_rest['LCOE_Q90_Worst'], 
    c=the_rest['Active_Material_Cost_m2'], 
    cmap='plasma', # Plasma provides great contrast for cost mapping
    s=40, 
    alpha=0.6, 
    edgecolors='none',
    label='Screened Candidates'
)

# 4. Highlight the Top 10 "Titanium-Clad" Discoveries
ax.scatter(
    top_10['Predicted_Bandgap_eV'], 
    top_10['LCOE_Q90_Worst'], 
    facecolors='cyan', 
    edgecolors='black', 
    linewidths=1.5,
    s=120, 
    marker='*', 
    zorder=5,
    label='Top 10 "Titanium-Clad" Candidates'
)

# 5. Add Target Reference Lines (Physics & Economics)
# Optimal SQ Peak (~1.34 eV)
ax.axvline(x=1.34, color='gray', linestyle='--', lw=1.5, zorder=1)
ax.text(1.35, ax.get_ylim()[1]*0.95, 'Ideal SQ Peak (1.34 eV)', color='gray', fontsize=10, rotation=270)

# 2030 SunShot Utility Target (~$0.03/kWh) or standard baseline
ax.axhline(y=0.03, color='crimson', linestyle=':', lw=2, zorder=1)
ax.text(ax.get_xlim()[0] + 0.05, 0.031, 'DOE 2030 Target ($0.03/kWh)', color='crimson', fontsize=10, fontweight='bold')

# 6. Add Smart Labels for the Top 10 (Using adjust_text to prevent overlap)
texts = []
for i, row in top_10.iterrows():
    texts.append(ax.text(
        row['Predicted_Bandgap_eV'], 
        row['LCOE_Q90_Worst'], 
        row['Formula'], 
        fontsize=9, 
        fontweight='bold',
        color='black'
    ))

# Repel overlapping labels for a clean, professional look
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))

# 7. Labels, Legends, and Colorbars
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Active Material Cost ($/m$^2$)', fontweight='bold', fontsize=12)

ax.set_title('Figure 3b: Economic Risk Profiling of 3,208 Unmapped Perovskites\n(Assuming Worst-Case Q90 ML Error Scenarios)', fontweight='bold', fontsize=14)
ax.set_xlabel('XGBoost Predicted Bandgap (eV)', fontweight='bold', fontsize=12)
ax.set_ylabel('Worst-Case LCOE Risk (Q90) ($/kWh)', fontweight='bold', fontsize=12)

# Set logical limits
ax.set_xlim(0.5, 2.5) # The physical PV boundary
ax.set_ylim(0.01, 0.15) # Zoom in on the highly viable economic zone

ax.legend(loc='upper right', frameon=True, edgecolor='black')
ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.show()

# (Optional) Save high-res for publication
# fig.savefig("Figure_3b_Risk_Scatter.png", dpi=300, bbox_inches='tight')
# %%