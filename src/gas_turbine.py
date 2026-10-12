"""Methane-fired open Brayton cycle with variable-cp air and combustion products.

State numbering: 1 compressor inlet, 2 compressor exit, 3 turbine inlet (TIT), 4 turbine exhaust.
Specific entropy s = s0(T) - R·ln(P/P0) with P0 = 1 atm, excluding the mixing term.
"""

from pathlib import Path

import numpy as np

from src.combustor import ETA_F, LHV_CH4, solve_fuel_air
from src.props.gas import AIR

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
P0 = 101325.0
HRSG_INLET_LIMIT_C = 650.0


def _state(gas, T, P, h):
    return {"T": T, "P": P, "h": h, "s": gas.s0(T) - gas.R_mix * np.log(P / P0)}


def gas_turbine(rp, TIT, m_air=100.0, T1=298.0, P1=1.013e5, P_exh=1.05e5, eta_c=0.86, eta_t=0.89,
                eta_f=ETA_F, LHV=LHV_CH4):
    P2 = rp * P1
    comp = AIR.compress(T1, P1, P2, eta_c)
    T2 = comp["T_out"]

    comb = solve_fuel_air(T2, TIT, eta_f=eta_f, LHV=LHV)
    products = comb["products"]
    P3 = P2

    # Expansion ratio is P3/P_exh, not rp: the turbine exhausts to 1.05 bar to overcome
    # the HRSG gas-side pressure drop, while the compressor draws from 1.013 bar.
    turb = products.expand(TIT, P3, P_exh, eta_t)

    m_fuel = comb["f"] * m_air
    m_gas = m_air + m_fuel
    W_c = m_air * comp["w"]
    W_t = m_gas * turb["w"]
    W_GT_net = W_t - W_c
    return {
        "states": {
            "1": _state(AIR, T1, P1, comp["h_in"]),
            "2s": _state(AIR, comp["T_out_s"], P2, comp["h_out_s"]),
            "2": _state(AIR, T2, P2, comp["h_out"]),
            "3": _state(products, TIT, P3, turb["h_in"]),
            "4s": _state(products, turb["T_out_s"], P_exh, turb["h_out_s"]),
            "4": _state(products, turb["T_out"], P_exh, turb["h_out"]),
        },
        "f": comb["f"], "phi": comb["phi"], "m_air": m_air, "m_fuel": m_fuel, "m_gas": m_gas,
        "W_c": W_c, "W_t": W_t, "W_GT_net": W_GT_net, "w_net": W_GT_net / m_air,
        "eta_GT": W_GT_net / (m_fuel * LHV), "T4": turb["T_out"], "products": products,
    }


def eta_vs_rp_study(TITs=(1200.0, 1300.0, 1400.0, 1500.0), rp=np.arange(4.0, 60.01, 0.25),
                    fig_name="phase3_gt_eta_vs_rp.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    curves, rows = {}, []
    for TIT in TITs:
        runs = [gas_turbine(r, TIT) for r in rp]
        eta = np.array([g["eta_GT"] for g in runs])
        w_net = np.array([g["w_net"] for g in runs])
        T4_C = np.array([g["T4"] for g in runs]) - 273.15
        curves[TIT] = (eta, w_net, T4_C)
        i_eta, i_w = eta.argmax(), w_net.argmax()
        rows.append({"TIT [K]": TIT, "rp at max eta": rp[i_eta], "max eta_GT [%]": 100 * eta[i_eta],
                     "rp at max w_net": rp[i_w], "max w_net [kJ/kg air]": w_net[i_w] / 1e3})
    table = pd.DataFrame(rows)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_w = ax_a.twinx()
    for k, TIT in enumerate(TITs):
        eta, w_net, T4_C = curves[TIT]
        color = f"C{k}"
        ax_a.plot(rp, 100 * eta, "-", color=color, label=f"TIT = {TIT:.0f} K")
        ax_w.plot(rp, w_net / 1e3, "--", color=color, lw=1)
        i = eta.argmax()
        ax_a.plot(rp[i], 100 * eta[i], "o", color=color)
        ax_a.annotate(f"{rp[i]:.1f}", (rp[i], 100 * eta[i]), textcoords="offset points", xytext=(0, 6),
                      ha="center", fontsize=8, color=color)
        ax_b.plot(rp, T4_C, "-", color=color, label=f"TIT = {TIT:.0f} K")
    for ax in (ax_a, ax_b):
        ax.axvspan(8, 30, color="0.9", zorder=0, label="design range r$_p$ 8–30")
        ax.set_xlabel("Pressure ratio r$_p$")
        ax.set_xlim(rp[0], rp[-1])
        ax.grid(alpha=0.3)
    ax_a.set_ylabel("η$_{GT}$ [%]  (solid, ● = maximum)")
    ax_w.set_ylabel("w$_{net}$ [kJ/kg air]  (dashed)")
    ax_a.set_title("(a) Gas turbine efficiency and specific work")
    ax_a.legend(loc="lower right", fontsize=8)
    ax_b.axhline(HRSG_INLET_LIMIT_C, color="k", ls=":", label="HRSG inlet limit 650 °C")
    ax_b.set_ylabel("Turbine exhaust T$_4$ [°C]")
    ax_b.set_title("(b) Turbine exhaust temperature")
    ax_b.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)
    return table


if __name__ == "__main__":
    import warnings

    from src.props.gas import A6ExtrapolationWarning

    warnings.simplefilter("ignore", A6ExtrapolationWarning)
    print(eta_vs_rp_study().to_string(index=False, float_format=lambda v: f"{v:.2f}"))
