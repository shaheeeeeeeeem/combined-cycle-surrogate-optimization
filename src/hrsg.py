"""Single-pressure HRSG (economizer -> evaporator -> superheater) with pinch-point closure.

Gas states: 4 (inlet) -> 4a (after superheater) -> 4b (after evaporator, pinch) -> 4c (stack).
Steam states: 8 (feedwater) -> 8a (sat. liquid) -> 8b (sat. vapour) -> 5 (superheated).
Counterflow pairings: 4-5 (approach), 4a-8b, 4b-8a (pinch), 4c-8.
"""

import numpy as np

from src.props import steam
from src.rankine import rankine

DP_GAS = 3e3
N_PROFILE = 50


def hrsg(gt, Ps, Ts, Pcond, eta_st=0.85, eta_pump=0.80, dT_pinch=10.0):
    cycle = rankine(Ps, Ts, Pcond, eta_st=eta_st, eta_pump=eta_pump)
    st = cycle["states"]
    gas = gt["products"]
    m_gas = gt["m_gas"]
    T4 = gt["states"]["4"]["T"]
    h4 = gt["states"]["4"]["h"]
    T_sat = st["8a"]["T"]
    h8, h8a, h8b, h5 = st["8"]["h"], st["8a"]["h"], st["8b"]["h"], st["5"]["h"]

    T4b = T_sat + dT_pinch
    if T4b >= T4:
        raise ValueError(f"GT exhaust {T4:.1f} K is below pinch temperature {T4b:.1f} K: no steam can be raised")
    h4b = gas.h(T4b)
    m_s = m_gas * (h4 - h4b) / (h5 - h8a)

    # Ideal-gas enthalpy is independent of pressure, so the 3 kPa gas-side drop
    # (1.05 -> 1.02 bar) only sets the stack pressure, not the energy balances.
    h4a = h4 - m_s * (h5 - h8b) / m_gas
    h4c = h4b - m_s * (h8a - h8) / m_gas
    T4a = gas.T_from_h(h4a)
    T4c = gas.T_from_h(h4c)

    Q_sh = m_s * (h5 - h8b)
    Q_ev = m_s * (h8b - h8a)
    Q_ec = m_s * (h8a - h8)
    Q_total = Q_sh + Q_ev + Q_ec

    h_ec = np.linspace(h8, h8a, N_PROFILE)
    h_ev = np.linspace(h8a, h8b, N_PROFILE)
    h_sh = np.linspace(h8b, h5, N_PROFILE)
    T_ec = np.array([steam.T_PH(Ps, h) for h in h_ec[:-1]] + [T_sat])
    T_ev = np.full(N_PROFILE, T_sat)
    T_sh = np.array([T_sat] + [steam.T_PH(Ps, h) for h in h_sh[1:]])
    h_steam = np.concatenate([h_ec, h_ev, h_sh])
    T_steam = np.concatenate([T_ec, T_ev, T_sh])
    section = np.repeat(["economizer", "evaporator", "superheater"], N_PROFILE)
    Q = m_s * (h_steam - h8)
    T_gas = gas.T_from_h(h4c + Q / m_gas)

    dT = T_gas - T_steam
    i_min = int(np.argmin(dT))
    pinch_region = section != "superheater"
    i_pinch = int(np.flatnonzero(pinch_region)[np.argmin(dT[pinch_region])])

    def location(i):
        if np.isclose(Q[i], Q_ec, rtol=0.0, atol=1e-9 * Q_total):
            return "4b"
        if np.isclose(Q[i], Q_total, rtol=0.0, atol=1e-9 * Q_total):
            return "4"
        if np.isclose(Q[i], 0.0, rtol=0.0, atol=1e-9 * Q_total):
            return "4c"
        return str(section[i])

    h_gas_T8 = gas.h(st["8"]["T"])
    m_air = gt["m_air"]
    h_air_1 = gt["states"]["1"]["h"]
    return {
        "cycle": cycle, "m_s": m_s, "T4a": T4a, "T4b": T4b, "T4c": T4c,
        "h4a": h4a, "h4b": h4b, "h4c": h4c,
        "Q_sh": Q_sh, "Q_ev": Q_ev, "Q_ec": Q_ec, "Q_total": Q_total,
        "min_dT": dT[i_min], "min_dT_location": location(i_min), "min_dT_Q": Q[i_min],
        "min_dT_pinch": dT[i_pinch], "min_dT_pinch_location": location(i_pinch),
        "profile": {"Q": Q, "T_gas": T_gas, "T_steam": T_steam, "section": section},
        "Q_stack": m_gas * gas.h(T4c) - m_air * h_air_1,
        "Q_cond": m_s * (st["6"]["h"] - st["7"]["h"]),
        "eps_HRSG": (h4 - h4c) / (h4 - h_gas_T8),
        "P_gas_in": gt["states"]["4"]["P"], "P_gas_out": gt["states"]["4"]["P"] - DP_GAS,
    }
