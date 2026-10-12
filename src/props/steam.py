"""Water/steam properties from CoolProp (IAPWS-95), SI units: P in Pa, T in K, h in J/kg, s in J/kg·K.

Quality functions return None for single-phase states (CoolProp reports Q = -1 there).
"""

from CoolProp.CoolProp import PropsSI

FLUID = "Water"


def Tsat(P):
    return PropsSI("T", "P", P, "Q", 0, FLUID)


def h_f(P):
    return PropsSI("H", "P", P, "Q", 0, FLUID)


def h_g(P):
    return PropsSI("H", "P", P, "Q", 1, FLUID)


def s_f(P):
    return PropsSI("S", "P", P, "Q", 0, FLUID)


def s_g(P):
    return PropsSI("S", "P", P, "Q", 1, FLUID)


def h_PT(P, T):
    return PropsSI("H", "P", P, "T", T, FLUID)


def s_PT(P, T):
    return PropsSI("S", "P", P, "T", T, FLUID)


def h_PS(P, s):
    return PropsSI("H", "P", P, "S", s, FLUID)


def T_PH(P, h):
    return PropsSI("T", "P", P, "H", h, FLUID)


def s_PH(P, h):
    return PropsSI("S", "P", P, "H", h, FLUID)


def _two_phase_quality(q):
    return q if 0.0 <= q <= 1.0 else None


def x_PH(P, h):
    return _two_phase_quality(PropsSI("Q", "P", P, "H", h, FLUID))


def x_PS(P, s):
    return _two_phase_quality(PropsSI("Q", "P", P, "S", s, FLUID))
