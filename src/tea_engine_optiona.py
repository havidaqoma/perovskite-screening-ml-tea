# CELL 3: UNCERTAINTY-AWARE TECHNO-ECONOMIC ENGINE (TEA) - LITERATURE ALIGNED
import numpy as np
from pymatgen.core import Composition

# -------------------------------------------------------------------------------------
# PART 1: COMMODITY & PROCESSING PRICE DICTIONARY ($/kg)
# -------------------------------------------------------------------------------------
ELEMENT_PRICES_KG = {
    "H": 1.39, "Li": 16.00, "O": 0.15, "F": 2.00, "Na": 3.00, "Mg": 2.32, "Al": 3.57, "S": 6.48, "Cl": 0.57, "K": 12.85, 
    "Ca": 2.28, "Sc": 9350.00, "Ti": 7.59, "Cr": 8.52, "Mn": 1.94, "Fe": 0.25, "Co": 56.28, "Ni": 17.47, "Cu": 13.06, 
    "Zn": 3.09, "Ga": 1331.58, "Ge": 5287.02, "Br": 3.78, "Rb": 10210.00, "Sr": 6.01, "Y": 33.00, "Zr": 23.14, 
    "Mo": 74.84, "Ru": 16155.50, "Rh": 76840.00, "Pd": 42326.00, "Ag": 1506.00, "Cd": 2.36, "In": 787.93, 
    "Sn": 34.13, "Sb": 51.80, "Te": 174.12, "I": 78.43, "Cs": 36994.83, "Ba": 0.26, "La": 4.00, "Ce": 4.36, 
    "Pr": 148.80, "Nd": 139.40, "Gd": 55.00, "Tb": 2289.25, "Dy": 640.35, "Hf": 6961.10, "Ta": 209.00, 
    "W": 32.07, "Re": 4012.15, "Pt": 47201.00, "Au": 96829.50, "Pb": 2.11, "Bi": 38.67
}

# -------------------------------------------------------------------------------------
# PART 2: ACTIVE MATERIAL COST CALCULATION (YIELD & SOLVENT CORRECTED)
# -------------------------------------------------------------------------------------
def calc_material_cost(formula, future=False):
    try:
        comp = Composition(formula)
        
        # Base physical requirements for a 500 nm film (~5.0 g/m^2 dense material)
        base_mass_g = 5.0
        
        # LITERATURE CORRECTION: Material Utilization Yield
        # Current (Spin-coating): ~30% utilization (70% wasted into spin bowl)
        # Future (Slot-die R2R): ~85% utilization (15% wasted overspray/edge trimming)
        utilization = 0.85 if future else 0.30
        required_mass_kg = (base_mass_g / utilization) / 1000.0
        
        # Calculate raw elemental cost based on weight fractions
        raw_cost = sum([ELEMENT_PRICES_KG.get(el.symbol, 50.0) * comp.get_wt_fraction(el) * required_mass_kg for el in comp.elements])
        
        # LITERATURE CORRECTION: Precursor Synthesis & Solvent Markup (Jean et al.)
        processing_markup = 1.50 if future else 2.50
        return raw_cost * processing_markup
    except: 
        return 1.5 # Default fallback penalty if formula parsing fails

# -------------------------------------------------------------------------------------
# PART 3: OPTICAL PHYSICS BOUNDARY (EXACT SHOCKLEY-QUEISSER INTERPOLATION)
# -------------------------------------------------------------------------------------
# Pre-calculated SQ maximum efficiencies under AM1.5G spectrum
# Data format: [Bandgap in eV], [Theoretical Max Efficiency in %]
SQ_EG_POINTS = np.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.34, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.4, 2.6, 3.0])
SQ_EFF_POINTS = np.array([0.0, 0.0, 14.5, 24.5, 30.0, 32.8, 33.6, 33.7, 33.4, 32.1, 30.2, 28.1, 25.8, 23.5, 21.3, 19.1, 17.0, 13.0, 9.5, 4.0])

def sq_limit(bandgap_array):
    # Use highly accurate linear interpolation of the real AM1.5G SQ limit.
    # Any bandgap outside 0.5 - 3.0 eV returns 0% efficiency.
    eff = np.interp(bandgap_array, SQ_EG_POINTS, SQ_EFF_POINTS, left=0.0, right=0.0)
    return eff / 100.0 # Return as a fraction

# -------------------------------------------------------------------------------------
# PART 4: THE MONTE CARLO TECHNO-ECONOMIC ENGINE (CHANG ET AL. CALIBRATED)
# -------------------------------------------------------------------------------------
def run_tea(
    formula,
    pred_Eg,
    pred_Ef,
    mae_error,
    iterations=50000,
    future=False,
    comp_factor=None,
    lifetime_years: float | None = None,
):
    """
    Option A: comp_factor replaces blanket pce_max when provided.
    Default None = legacy behavior (pce_max only).

    The prior Ef->lifetime mapping was degenerate (all candidates saturated at
    20-30 yr), so it is replaced by explicit scenario lifetimes per
    IMPROVEMENT_ACTION_PLAN.md P0-2.  Existing callers that omit
    ``lifetime_years`` retain deterministic current/future scenario defaults.
    """
    # Discount Rate / WACC (3-5% for mature tech, 4-7% for current emerging tech)
    rate = np.random.uniform(0.03 if future else 0.04, 0.05 if future else 0.07, iterations)
    mat_cost = calc_material_cost(formula, future=future)
    
    # ---------------------------------------------------------------------------------
    # MODULE & FACTORY COSTS ($/m^2) - Extracted from Chang et al. 2017
    # ---------------------------------------------------------------------------------
    if future:
        # FUTURE 2030 (Roll-to-Roll Flexible):
        # Substrate: PET ($3) | ETL/HTL: Slot-die SnO2/NiOx ($3) | Contacts: AgNW ($5) | Encapsulation: R2R film ($10)
        fixed_mod_cost = np.random.normal(31.0, 3.0, iterations) 
        
        # Area-dependent BOS (Racking, Wiring, Land) - $80/m^2 (SunShot target)
        area_bos = np.random.normal(80.0, 5.0, iterations)
        
        # Power-dependent Inverter/Electrical - $150/kW ($0.15/Wdc)
        inv_rate = 150.0 
        om_rate = 10.0 # O&M: $10/kW/yr
        pce_max = 0.85 # Real-world panel captures 85% of theoretical SQ limit
        
    else:
        # CURRENT TECH (Rigid FTO Glass, Sheet-to-Sheet):
        # Substrate: FTO ($11.70) | ETL/HTL: TiO2+P3HT ($22.40) | Contacts: Ag ($4) | Encapsulation ($10.40)
        fixed_mod_cost = np.random.normal(73.5, 5.0, iterations)
        
        # Area-dependent BOS - $136/m^2 (Current rigid standard)
        area_bos = np.random.normal(136.0, 10.0, iterations)
        
        # Power-dependent Inverter/Electrical - $300/kW ($0.30/Wdc)
        inv_rate = 300.0 
        om_rate = 20.0 # O&M: $20/kW/yr
        pce_max = 0.70 # Real-world panel captures 70% of theoretical SQ limit

    # Combine Base Module Cost + Active Material Cost
    mod_cost = fixed_mod_cost + mat_cost
    
    # Chemistry Yield Penalties
    if not future:
        if "Sn" in formula: mod_cost /= 0.85 # Sn2+ oxidation factory scrap
        if pred_Ef > 0.5: mod_cost /= 0.80   # Phase-instability scrap
        
    # ---------------------------------------------------------------------------------
    # PHYSICS & EFFICIENCY CALCULATION
    # ---------------------------------------------------------------------------------
    # Option A: comp_factor directly replaces pce_max when provided (NOT a multiplier on pce_max)
    effective_pce_max = comp_factor if comp_factor is not None else pce_max
    # Propagate AI bandgap error through exact AM1.5G SQ curve, adding environmental noise
    pce = np.clip(np.random.normal(sq_limit(np.random.normal(pred_Eg, mae_error, iterations)) * effective_pce_max, 0.02, iterations), 0.01, 0.33)
    
    # Output power per square meter (Assuming 1000 W/m^2 solar irradiance -> 1 kW/m^2 base)
    kw_per_m2 = 1.0 * pce
    
    # ---------------------------------------------------------------------------------
    # CAPEX & O&M 
    # ---------------------------------------------------------------------------------
    capex_m2 = mod_cost + area_bos + (kw_per_m2 * inv_rate)
    
    if lifetime_years is None:
        lifetime_years = 25.0 if future else 20.0
    life = float(lifetime_years)
    
    energy_m2 = np.zeros(iterations)
    costs_m2 = capex_m2.copy()
    
    # ---------------------------------------------------------------------------------
    # DISCOUNTED CASH FLOW (DCF) LOOP
    # ---------------------------------------------------------------------------------
    for yr in range(1, int(life) + 1):
        df = 1.0 / ((1.0 + rate)**yr)
        
        # Base annual degradation (0.74% to 4.4% range based on Chang et al.)
        deg_rate = np.clip(0.0074 + (pred_Ef * 0.02), 0.0074, 0.044)
        degradation = (1.0 - deg_rate)**yr
        
        # 1471 kWh/kW/yr represents average US insolation
        energy_m2 += (1471.0 * kw_per_m2 * degradation) * df
        costs_m2 += (om_rate * kw_per_m2) * df
        
        # Inverter replacement at Year 15
        if yr == 15: costs_m2 += (kw_per_m2 * inv_rate) * df
            
    # LCOE = Total Discounted Costs / Total Discounted Energy Generated
    lcoe = costs_m2 / np.clip(energy_m2, 1e-5, None)
    
    return lcoe, life, mat_cost, pce

print("Literature-Calibrated Techno-Economic Engine (Exact SQ) Initialized.")
# %%
