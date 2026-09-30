"""TEA v2 engine: simple real-dollar LCOE for a utility PV plant, sized per kWdc.

  A      = 1 / eta                       m2 of module per kWdc (STC 1 kW/m2)
  CAPEX  = markup * [ A * (module + absorber + sbos + fieldwork)  +  inverter + ebos + officework ]   $/kWdc
  Y      = cf_ac * 8760 / ilr            kWh_AC per kWdc per year (first year, before burn-in)
  E_t    = Y * (1 - burnin) * (1 - deg)^(age_t - 1),  age resets when modules are replaced
  REPL_t = markup * A * (module + absorber + labor_frac * fieldwork)   at t = L_mod, 2 L_mod, ... < L_plant
  LCOE   = (CAPEX + sum_t OM/(1+r)^t + sum_t REPL_t/(1+r)^t) / sum_t E_t/(1+r)^t        $/kWh

Area-scaled costs are divided by efficiency; power-scaled costs are not. This is exactly the
structure the April "thin-film decoupling" claim ignored. No formation energy enters anywhere.
All inputs broadcast (numpy arrays of shape (N,) or scalars).
"""
from __future__ import annotations

import numpy as np

HOURS = 8760.0


def capex_kwdc(eta, module_m2, sbos_m2, fieldwork_m2, inverter_kwdc, ebos_kwdc, officework_kwdc, markup,
               absorber_m2=0.0):
    eta = np.asarray(eta, float)
    area = np.where(eta > 0, 1.0 / np.where(eta > 0, eta, 1.0), np.inf)
    return markup * (area * (module_m2 + absorber_m2 + sbos_m2 + fieldwork_m2)
                     + inverter_kwdc + ebos_kwdc + officework_kwdc)


def lcoe(eta, module_m2, sbos_m2, fieldwork_m2, inverter_kwdc, ebos_kwdc, officework_kwdc, markup,
         cf_ac, ilr, fom_kwac, r, deg, plant_life: int, module_life: int | None = None, burnin=0.0,
         absorber_m2=0.0, replace_labor_frac=0.0, return_parts: bool = False):
    eta = np.asarray(eta, float)
    shape = np.broadcast(eta, module_m2, cf_ac, r, deg, fom_kwac).shape
    eta_b = np.broadcast_to(eta, shape)
    ok = eta_b > 0
    area = np.where(ok, 1.0 / np.where(ok, eta_b, 1.0), np.nan)
    capex = markup * (area * (module_m2 + absorber_m2 + sbos_m2 + fieldwork_m2)
                      + inverter_kwdc + ebos_kwdc + officework_kwdc)
    y1 = np.asarray(cf_ac, float) * HOURS / ilr
    om = np.asarray(fom_kwac, float) / ilr
    L = int(plant_life)
    Lm = L if module_life is None else int(module_life)
    repl_unit = markup * area * (module_m2 + absorber_m2 + replace_labor_frac * fieldwork_m2)
    pv_e = np.zeros(shape)
    pv_om = np.zeros(shape)
    pv_repl = np.zeros(shape)
    for t in range(1, L + 1):
        df = (1.0 + r) ** (-t)
        age = (t - 1) % Lm + 1
        pv_e = pv_e + y1 * (1.0 - burnin) * (1.0 - deg) ** (age - 1) * df
        pv_om = pv_om + om * df
        if t % Lm == 0 and t < L:
            pv_repl = pv_repl + repl_unit * df
    out = np.where(ok, (capex + pv_om + pv_repl) / pv_e, np.inf)
    if return_parts:
        return out, {"capex": capex, "pv_om": pv_om, "pv_repl": pv_repl, "pv_energy": pv_e}
    return out
