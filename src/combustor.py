"""Constant-pressure methane combustor with complete combustion, per kg of air.

CH4 + a(O2 + (79/21) N2) -> CO2 + 2 H2O + (a - 2) O2 + (79/21) a N2
Fuel enters at 298.15 K, so it carries zero sensible enthalpy.
"""

from scipy.optimize import brentq

from src.props.gas import AIR, MOLAR_MASS, GasMixture

LHV_CH4 = 50.0e6
ETA_F = 0.98

O2_PER_CH4 = 2.0
F_STOICH = MOLAR_MASS["CH4"] * AIR.mole_fractions["O2"] / (O2_PER_CH4 * AIR.M_mix)


def product_mixture(f):
    """Combustion products of 1 kg air + f kg CH4, as a mass-fraction GasMixture."""
    n_fuel = f / MOLAR_MASS["CH4"]
    n_O2_air = AIR.mass_fractions["O2"] / MOLAR_MASS["O2"]
    n_N2_air = AIR.mass_fractions["N2"] / MOLAR_MASS["N2"]
    n_O2_left = n_O2_air - O2_PER_CH4 * n_fuel
    if n_O2_left < 0.0:
        raise ValueError(f"f = {f} exceeds stoichiometric f = {F_STOICH}")
    masses = {
        "CO2": n_fuel * MOLAR_MASS["CO2"],
        "H2O": 2.0 * n_fuel * MOLAR_MASS["H2O"],
        "O2": n_O2_left * MOLAR_MASS["O2"],
        "N2": n_N2_air * MOLAR_MASS["N2"],
    }
    # Normalised by the summed product mass: the tabulated molar masses conserve mass
    # across the reaction only to ~1e-5 of the fuel mass.
    total = sum(masses.values())
    return GasMixture({k: m / total for k, m in masses.items()}, basis="mass")


def solve_fuel_air(T2, TIT, eta_f=ETA_F, LHV=LHV_CH4):
    """Fuel-air ratio from f·LHV·eta_f = (1 + f)·h_prod(TIT) - h_air(T2)."""
    h_air_2 = AIR.h(T2)

    def residual(f):
        return f * LHV * eta_f - ((1.0 + f) * product_mixture(f).h(TIT) - h_air_2)

    f = brentq(residual, 1e-5, F_STOICH, xtol=1e-15, rtol=1e-15)
    return {"f": f, "phi": f / F_STOICH, "products": product_mixture(f)}
