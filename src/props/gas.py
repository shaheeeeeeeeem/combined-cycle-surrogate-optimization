"""Ideal-gas properties with temperature-dependent cp from B&S Table A.6.

Units are SI throughout: T in K, P in Pa, h in J/kg, cp and s0 in J/kg·K.
Reference state: h = 0 and s0 = 0 at T_REF = 298.15 K for every species.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

TABLE_A6 = Path(__file__).resolve().parents[2] / "textbook_tables" / "BS7e_TableA6_cp_polynomials.csv"

R_U = 8314.46
T_REF = 298.15
T_VALID = (250.0, 1200.0)
T_BRACKET = (200.0, 3000.0)
MOLAR_MASS = {"N2": 28.013, "O2": 31.999, "CO2": 44.010, "H2O": 18.015, "CH4": 16.043}
SPECIES = ("N2", "O2", "CO2", "H2O")


class A6ExtrapolationWarning(UserWarning):
    pass


def _load_coefficients(path=TABLE_A6, formulas=SPECIES + ("Air",)):
    table = pd.read_csv(path).set_index("formula")
    coeffs = {}
    for formula in formulas:
        c0, c1, c2, c3 = table.loc[formula, ["C0", "C1", "C2", "C3"]].astype(float)
        coeffs[formula] = (1000.0 * c0, 1000.0 * c1, 1000.0 * c2, 1000.0 * c3)
    return coeffs


COEFFS = _load_coefficients()

_extrapolation_warned = False


def _check_range(T):
    global _extrapolation_warned
    if _extrapolation_warned:
        return
    T = np.asarray(T, dtype=float)
    if np.any((T < T_VALID[0]) | (T > T_VALID[1])):
        _extrapolation_warned = True
        warnings.warn(
            f"Table A.6 cp polynomial evaluated outside {T_VALID[0]:.0f}-{T_VALID[1]:.0f} K; "
            "values there are extrapolated.",
            A6ExtrapolationWarning,
            stacklevel=3,
        )


def _cp_poly(T, species):
    c0, c1, c2, c3 = COEFFS[species]
    theta = np.asarray(T, dtype=float) / 1000.0
    return c0 + c1 * theta + c2 * theta**2 + c3 * theta**3


def _h_poly(T, species):
    c0, c1, c2, c3 = COEFFS[species]
    theta = np.asarray(T, dtype=float) / 1000.0
    return 1000.0 * (c0 * theta + c1 * theta**2 / 2 + c2 * theta**3 / 3 + c3 * theta**4 / 4)


def _s0_poly(T, species):
    c0, c1, c2, c3 = COEFFS[species]
    T = np.asarray(T, dtype=float)
    theta = T / 1000.0
    return c0 * np.log(T) + c1 * theta + c2 * theta**2 / 2 + c3 * theta**3 / 3


def _h_species(T, species):
    return _h_poly(T, species) - _h_poly(T_REF, species)


def _s0_species(T, species):
    return _s0_poly(T, species) - _s0_poly(T_REF, species)


def cp(T, species):
    _check_range(T)
    return _cp_poly(T, species)


def h(T, species):
    _check_range(T)
    return _h_species(T, species)


def s0(T, species):
    _check_range(T)
    return _s0_species(T, species)


class GasMixture:
    """Ideal-gas mixture of A.6 species at fixed composition."""

    def __init__(self, fractions, basis="mole", molar_masses=None):
        if basis not in ("mole", "mass"):
            raise ValueError("basis must be 'mole' or 'mass'")
        masses = {**MOLAR_MASS, **(molar_masses or {})}
        self.species = tuple(fractions)
        x = np.array([fractions[k] for k in self.species], dtype=float)
        if not np.isclose(x.sum(), 1.0, atol=1e-9):
            raise ValueError(f"{basis} fractions sum to {x.sum()}, not 1")
        M = np.array([masses[k] for k in self.species], dtype=float)
        if basis == "mole":
            y = x
            self.M_mix = float(np.sum(y * M))
            c = y * M / self.M_mix
        else:
            c = x
            self.M_mix = float(1.0 / np.sum(c / M))
            y = c / M * self.M_mix
        self.mole_fractions = dict(zip(self.species, y))
        self.mass_fractions = dict(zip(self.species, c))
        self.R_mix = R_U / self.M_mix
        self.T_bracket = self._monotonic_bracket()

    def _monotonic_bracket(self):
        # Extrapolated A.6 cubics (N2, O2) give cp <= 0 above ~2200 K, where h(T) stops
        # increasing; cap the inversion bracket at the last T with cp > 0.
        lo, hi = T_BRACKET
        grid = np.arange(lo, hi + 1.0, 1.0)
        nonpositive = np.nonzero(self._cp(grid) <= 0.0)[0]
        if nonpositive.size == 0:
            return lo, hi
        return lo, float(grid[nonpositive[0] - 1])

    def _cp(self, T):
        return sum(c * _cp_poly(T, k) for k, c in self.mass_fractions.items())

    def _h(self, T):
        return sum(c * _h_species(T, k) for k, c in self.mass_fractions.items())

    def _s0(self, T):
        return sum(c * _s0_species(T, k) for k, c in self.mass_fractions.items())

    def cp(self, T):
        _check_range(T)
        return self._cp(T)

    def h(self, T):
        _check_range(T)
        return self._h(T)

    def s0(self, T):
        _check_range(T)
        return self._s0(T)

    def _invert(self, func, target):
        lo, hi = self.T_bracket
        solve = np.vectorize(lambda v: brentq(lambda T: func(T) - v, lo, hi, xtol=1e-10))
        T = solve(np.asarray(target, dtype=float))[()]
        _check_range(T)
        return T

    def T_from_h(self, h):
        return self._invert(self._h, h)

    def T_from_s0(self, s0):
        return self._invert(self._s0, s0)

    def isentropic_T(self, T1, P1, P2):
        # Frozen composition: the mixing term -R·Σ y_i ln y_i is the same at both
        # states and cancels, so s2 - s1 = s0(T2) - s0(T1) - R_mix·ln(P2/P1) = 0.
        return self.T_from_s0(self.s0(T1) + self.R_mix * np.log(np.asarray(P2) / np.asarray(P1)))

    def compress(self, T1, P1, P2, eta):
        """Adiabatic compression. w = h_out - h_in > 0 is the specific work input."""
        T_out_s = self.isentropic_T(T1, P1, P2)
        h_in, h_out_s = self.h(T1), self.h(T_out_s)
        h_out = h_in + (h_out_s - h_in) / eta
        return {"T_out": self.T_from_h(h_out), "T_out_s": T_out_s, "h_in": h_in,
                "h_out": h_out, "h_out_s": h_out_s, "w": h_out - h_in}

    def expand(self, T1, P1, P2, eta):
        """Adiabatic expansion. w = h_in - h_out > 0 is the specific work output."""
        T_out_s = self.isentropic_T(T1, P1, P2)
        h_in, h_out_s = self.h(T1), self.h(T_out_s)
        h_out = h_in - eta * (h_in - h_out_s)
        return {"T_out": self.T_from_h(h_out), "T_out_s": T_out_s, "h_in": h_in,
                "h_out": h_out, "h_out_s": h_out_s, "w": h_in - h_out}


AIR = GasMixture({"O2": 0.21, "N2": 0.79}, basis="mole")
