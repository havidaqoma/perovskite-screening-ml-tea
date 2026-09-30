"""TEA v2 parameter registry. Every number carries kind + source + (for direct) a verbatim quote.

Boundary: utility-scale, 100-MWdc single-axis tracker plant, U.S. cost basis (NREL Q1-2023 benchmark),
2022 USD, simple real-dollar LCOE (no tax, no depreciation schedule; the convention IRENA uses).
Functional unit: $/kWh AC delivered. Plant sized per kWdc of module nameplate at STC (1 kW/m2).

Kinds: direct | figure | derived | scenario   (see data/sources/SOURCES.md)
"""
from __future__ import annotations

ETA_REF_CSI = 0.205       # NREL23 representative module efficiency (direct, see ETA_CSI below)
ILR = 1.34

P = {
    # ---------------- plant cost structure (NREL23 Fig. 13, PV only, MSP / MMP) ----------------
    "eta_csi": dict(value=0.205, kind="direct", src="NREL23", unit="-",
                    quote="20.5%-efficient, 2.57-m2 bifacial monocrystalline silicon modules"),
    "ilr": dict(value=1.34, kind="direct", src="ATB24", unit="Wdc/Wac", quote="ILR of 1.34"),
    "module_csi_kwdc": dict(msp=253, mmp=372, kind="figure", src="NREL23 Fig13", unit="$/kWdc"),
    "inverter_kwdc": dict(msp=35, mmp=48, kind="figure", src="NREL23 Fig13", unit="$/kWdc"),
    "ebos_kwdc": dict(msp=146, mmp=176, kind="figure", src="NREL23 Fig13", unit="$/kWdc"),
    "officework_kwdc": dict(msp=63, mmp=66, kind="figure", src="NREL23 Fig13", unit="$/kWdc"),
    "other_kwdc": dict(msp=111, mmp=134, kind="figure", src="NREL23 Fig13", unit="$/kWdc",
                       note="sales tax, contingency, management, profit"),
    "total_kwdc": dict(msp=973, mmp=1161, kind="figure", src="NREL23 Fig13", unit="$/kWdc"),
    # area-scaled items, converted at the benchmark efficiency (PV-only values)
    "sbos_m2": dict(msp=129 * 0.205, mmp=128 * 0.205, kind="derived", src="NREL23 Fig13",
                    unit="$/m2", formula="SBOS $/kWdc x 0.205 kWdc/m2 (PV only)"),
    "fieldwork_m2": dict(msp=236 * 0.205, mmp=236 * 0.205, kind="derived", src="NREL23 Fig13",
                         unit="$/m2", formula="fieldwork $/kWdc x 0.205 kWdc/m2 (PV only)"),
    "markup": dict(msp=1 + 111 / (973 - 111), mmp=1 + 134 / (1161 - 134), kind="derived", src="NREL23 Fig13",
                   unit="x", formula="1 + Other / (Total - Other): 'Other' applied as a proportional markup"),

    # ---------------- perovskite module manufacturing cost (EPFL25) ----------------
    "module_pvk_m2": dict(low=50.0, med=85.0, high=335.0, kind="direct", src="EPFL25", unit="$/m2",
                          quote="The module capex can be considered as $50/m2, $85/m2, and $335/m2"),

    # ---------------- energy yield ----------------
    "cf_ac": dict(low=0.15, high=0.30, kind="figure", src="LAZ24 p35", unit="-",
                  note="Solar PV Utility capacity factor, high-cost case 15%, low-cost case 30%; sampled uniform"),
    "plant_life": dict(value=35, kind="figure", src="LAZ24 p35", unit="yr", note="Facility Life, Utility"),
    "deg_csi": dict(value=0.007, kind="direct", src="ATB24", unit="1/yr", quote="degradation rate reduction from 0.7%/yr"),

    # ---------------- O&M and finance ----------------
    "fom_kwac": dict(low=11.0, high=22.0, kind="derived", src="LAZ24 p35 + ATB24", unit="$/kWac-yr",
                     formula="low = Lazard utility low-case Fixed O&M $11/kW-yr (figure); high = ATB 2023 FOM",
                     quote="The fixed O&M (FOM) cost of $22/kWAC-yr for 2023"),
    "discount_real": dict(low=0.04, high=0.08, kind="scenario", src="-", unit="1/yr",
                          note="real WACC, sampled uniform. Reference: Lazard 60% debt @8% + 40% equity @12% "
                               "= 9.6% nominal pre-tax (derived), i.e. above this real range"),

    # ---------------- perovskite-specific scenarios (no field data exist for these absorbers) ----------------
    "deg_pvk": dict(grid=[0.007, 0.02, 0.05], kind="scenario", src="-", unit="1/yr",
                    note="0.7% = c-Si parity (ATB24); 2% and 5% illustrate unproven stability"),
    "burnin_pvk": dict(grid=[0.0, 0.05], kind="scenario", src="-", unit="-"),
    "module_life_pvk": dict(grid=[5, 10, 15, 25, 35], kind="scenario", src="-", unit="yr",
                            note="modules replaced at end of life until plant life 35 yr"),
    "replace_labor_frac": dict(value=0.0, alt=0.5, kind="scenario", src="-", unit="-",
                               note="fraction of fieldwork $/m2 repeated at module replacement; 0 favours perovskite"),
    # realised fraction of the SQ limit at MODULE level (replaces Option A derating). Review 2026-09-29:
    # 0.82 is a lab-CELL figure with no cell-to-module loss and is reported only as an idealised ceiling;
    # the c-Si module parity level (eta_csi / SQ(1.12 eV) = 0.61, derived in s30) is the physical optimum used.
    "f_sq": dict(grid=[0.21, 0.50, 0.82], kind="derived", src="ZHANG22", unit="-",
                 formula="0.21 = 6.37% / SQ(1.64 eV) for hydrogenated Cs2AgBiBr6 (demonstrated CELL); 0.50 = "
                         "intermediate scenario; 0.82 = 25.7% / SQ(~1.55 eV) lead-halide perovskite CELL record "
                         "(band gap assumed) = idealised ceiling, not a module value",
                 quote="improved up to 6.37%"),
    "absorber_thickness_nm": dict(value=500, kind="scenario", src="legacy", unit="nm"),
    "absorber_density_gcm3": dict(value=5.0, kind="scenario", src="-", unit="g/cm3",
                                  note="typical halide double perovskite; the legacy 5 g/m2 implied 10 g/cm3"),
    "absorber_utilization": dict(value=0.85, kind="scenario", src="legacy", unit="-"),
}

# Element prices, $/kg. USGS24 2023 estimates where USGS publishes one (direct or derived from the printed unit);
# otherwise the legacy April value, flagged. Only the absorber's elemental cost uses these.
TROY_OZ_KG = 0.0311034768
LB_KG = 0.45359237
ELEMENT_PRICE = {
    "Ag": (23.40 / TROY_OZ_KG, "USGS24 silver bullion $23.40/troy oz (derived)"),
    "Bi": (4.10 / LB_KG, "USGS24 bismuth $4.10/lb (derived)"),
    "Sb": (5.60 / LB_KG, "USGS24 antimony metal $5.60/lb (derived)"),
    "Ga": (450.0, "USGS24 gallium high-purity $450/kg"),
    "Ge": (1400.0, "USGS24 germanium metal $1,400/kg"),
    "In": (240.0, "USGS24 indium U.S. warehouse $240/kg"),
    "I": (61.0, "USGS24 crude iodine $61/kg"),
    "Se": (23.0, "USGS24 selenium $23/kg"),
    "Sn": (13.00 / LB_KG, "USGS24 tin NY dealer 1,300 c/lb (derived)"),
    "Nb": (25.0, "USGS24 ferroniobium $25/kg (alloy basis, not pure Nb)"),
    "Ta": (190.0 / 0.8190, "USGS24 tantalite $190/kg Ta2O5 -> Ta basis (derived, x 1/0.819)"),
    "Li": (46000.0 / 1000 / 0.1878, "USGS24 Li2CO3 $46,000/t -> Li basis (derived, Li mass fraction 0.1878)"),
    "Cs": (142.00 / (50 * 0.6925) * 1000, "USGS24: no market price; reagent CsOAc 50 g $142 -> Cs basis (derived, upper-bound-like)"),
    "Rb": (63.10 / (10 * 0.7401) * 1000, "USGS24: no market price; reagent Rb2CO3 10 g $63.10 -> Rb basis (derived, upper-bound-like)"),
}
LEGACY_UNSOURCED = {"H": 1.39, "O": 0.15, "F": 2.00, "Na": 3.00, "S": 6.48, "Cl": 0.57, "K": 12.85, "Ti": 7.59,
                    "V": 50.0, "Cr": 8.52, "Mn": 1.94, "Fe": 0.25, "Cu": 13.06, "Zn": 3.09, "Br": 3.78, "Zr": 23.14,
                    "Mo": 74.84, "W": 32.07}


def price(el: str) -> tuple[float, str]:
    if el in ELEMENT_PRICE:
        return ELEMENT_PRICE[el]
    if el in LEGACY_UNSOURCED:
        return LEGACY_UNSOURCED[el], "legacy April value (unsourced)"
    raise KeyError(f"no price for element {el}")
