# CELL 1: ADVANCED PHYSICS + MAGPIE + MENDELEEV FEATURE EXTRACTION
import numpy as np
import torch
import warnings
from pymatgen.core import Composition, Element
from matminer.featurizers.composition import ElementProperty
warnings.filterwarnings("ignore")

print("Loading dataset...")
raw_dataset = torch.load("perovskite_dataset_9D_Charge.pt", weights_only=False)

# -------------------------------------------------------------------------------------
# PART 1: PHYSICS ENGINES INITIALIZATION
# We initialize the Magpie featurizer from the matminer package.
# Magpie uses a peer-reviewed database to extract 132 fundamental elemental properties 
# (e.g., covalent radius, melting point, valence s/p/d orbital electrons).
# -------------------------------------------------------------------------------------
print("Initializing Matminer Magpie Featurizer...")
ep_feat = ElementProperty.from_preset(preset_name="magpie")
magpie_feature_names = ep_feat.feature_labels()

# We preserve your excellent Mendeleev / Pettifor scale logic.
MENDELEEV = {'He': 1, 'Ne': 2, 'Ar': 3, 'Kr': 4, 'Xe': 5, 'Rn': 6, 'F': 7, 'Cl': 8, 'Br': 9, 'I': 10, 'O': 11, 'S': 12, 'Se': 13, 'Te': 14, 'N': 15, 'P': 16, 'As': 17, 'Sb': 18, 'Bi': 19, 'C': 20, 'Si': 21, 'Ge': 22, 'Sn': 23, 'Pb': 24, 'B': 25, 'Al': 26, 'Ga': 27, 'In': 28, 'Tl': 29, 'Zn': 30, 'Cd': 31, 'Hg': 32, 'Cu': 33, 'Ag': 34, 'Au': 35, 'Ni': 36, 'Pd': 37, 'Pt': 38, 'Co': 39, 'Rh': 40, 'Ir': 41, 'Fe': 42, 'Ru': 43, 'Os': 44, 'Mn': 45, 'Tc': 46, 'Re': 47, 'Cr': 48, 'Mo': 49, 'W': 50, 'V': 51, 'Nb': 52, 'Ta': 53, 'Ti': 54, 'Zr': 55, 'Hf': 56, 'Sc': 57, 'Y': 58, 'Lu': 73, 'Li': 92, 'Na': 93, 'K': 94, 'Rb': 95, 'Cs': 96}

def get_radius(el_sym):
    el = Element(el_sym)
    return el.average_ionic_radius if el.average_ionic_radius else el.atomic_radius

# -------------------------------------------------------------------------------------
# PART 2: THE "PSEUDO-3D" FEATURE EXTRACTOR
# This function combines Magpie (132 features) + Mendeleev (4 features) 
# + Delta X (1 feature) + Steric Geometry (2 features) = 139 Physical Dimensions.
# -------------------------------------------------------------------------------------
def get_ultimate_features(formula):
    comp = Composition(formula)
    elements = comp.elements
    fractions = [comp.get_atomic_fraction(el) for el in elements]
    
    # 1. Get Magpie Features (132 properties)
    magpie_feats = ep_feat.featurize(comp)
    
    # 2. Get Custom Mendeleev Features
    mn = [float(MENDELEEV.get(el.symbol, 50.0)) for el in elements]
    mn_w = np.average(mn, weights=fractions)
    mn_feats = [mn_w, np.max(mn), np.min(mn), np.average((mn - mn_w)**2, weights=fractions)]
    
    # 3. Get Electronegativity Difference (Delta X)
    X = [float(getattr(el, 'X', 0.0) or 0.0) for el in elements]
    delta_x = [np.max(X) - np.min(X)]
    
    # 4. Get Pseudo-3D Sterics (Goldschmidt and Octahedral constraints)
    try:
        sorted_els = sorted(elements, key=lambda e: getattr(e, 'X', 0.0) or 0.0)
        t_factor = (get_radius(sorted_els[0]) + get_radius(sorted_els[-1])) / (np.sqrt(2) * (get_radius(sorted_els[1]) + get_radius(sorted_els[-1])))
        mu_factor = get_radius(sorted_els[1]) / get_radius(sorted_els[-1])
        steric_feats = [t_factor, mu_factor]
    except: 
        steric_feats = [0.0, 0.0]

    # Combine everything into a single mathematical vector
    return np.array(magpie_feats + mn_feats + delta_x + steric_feats)

# The EXACT names of the 139 features, perfectly aligned for the SHAP Explainer later
final_feature_names = magpie_feature_names + ["Mendeleev_Mean", "Mendeleev_Max", "Mendeleev_Min", "Mendeleev_Var", "Delta_Electronegativity", "Goldschmidt_Tolerance", "Octahedral_Factor"]

X_features, y_bg, y_fe, is_semi_y, formulas = [], [], [], [], []

print("Extracting 139-Dimensional Physics Features... (This may take a minute)")
for data in raw_dataset:
    try:
        X_features.append(get_ultimate_features(data.formula))
        y_bg.append(data.y[0, 0].item())
        y_fe.append(data.y[0, 1].item())
        is_semi_y.append(int(data.y[0, 0].item() > 0.01))
        formulas.append(data.formula)
    except: continue

X_features, y_bg, y_fe, is_semi_y, formulas = np.array(X_features), np.array(y_bg), np.array(y_fe), np.array(is_semi_y), np.array(formulas)

np.random.seed(42)
indices = np.arange(len(X_features))
np.random.shuffle(indices)
split = int(0.8 * len(X_features))
train_idx, val_idx = indices[:split], indices[split:]
print(f"Extraction Complete. Train: {len(train_idx)} | Val: {len(val_idx)}")