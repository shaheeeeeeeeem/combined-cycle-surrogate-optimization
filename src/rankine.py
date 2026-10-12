"""Single-pressure Rankine cycle on CoolProp water properties.

State numbering: 7 condenser exit (sat. liquid) -> 8 pump exit -> 8a economizer exit (sat. liquid at Ps)
-> 8b drum exit (sat. vapour at Ps) -> 5 superheater exit / turbine inlet -> 6 turbine exit -> 7.
All specific quantities are per kg of steam, SI units.
"""

from pathlib import Path

import numpy as np

from src.props import steam

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def _state(P, T, h, s, x):
    return {"P": P, "T": T, "h": h, "s": s, "x": x}


def _state_PH(P, h):
    return _state(P, steam.T_PH(P, h), h, steam.s_PH(P, h), steam.x_PH(P, h))


def rankine(Ps, Ts, Pcond, eta_st=0.85, eta_pump=0.80):
    if Pcond >= Ps:
        raise ValueError(f"Pcond ({Pcond} Pa) must be below Ps ({Ps} Pa)")
    T_sat_s = steam.Tsat(Ps)
    if Ts <= T_sat_s:
        raise ValueError(f"Ts ({Ts:.2f} K) must exceed Tsat(Ps) = {T_sat_s:.2f} K")

    s7 = steam.s_f(Pcond)
    h7 = steam.h_f(Pcond)
    st7 = _state(Pcond, steam.Tsat(Pcond), h7, s7, 0.0)

    h8s = steam.h_PS(Ps, s7)
    st8s = _state(Ps, steam.T_PH(Ps, h8s), h8s, s7, None)
    h8 = h7 + (h8s - h7) / eta_pump
    st8 = _state_PH(Ps, h8)

    st8a = _state(Ps, T_sat_s, steam.h_f(Ps), steam.s_f(Ps), 0.0)
    st8b = _state(Ps, T_sat_s, steam.h_g(Ps), steam.s_g(Ps), 1.0)

    h5, s5 = steam.h_PT(Ps, Ts), steam.s_PT(Ps, Ts)
    st5 = _state(Ps, Ts, h5, s5, None)

    h6s = steam.h_PS(Pcond, s5)
    st6s = _state(Pcond, steam.T_PH(Pcond, h6s), h6s, s5, steam.x_PS(Pcond, s5))
    h6 = h5 - eta_st * (h5 - h6s)
    st6 = _state_PH(Pcond, h6)

    w_st = h5 - h6
    w_pump = h8 - h7
    w_net = w_st - w_pump
    q_in = h5 - h8
    q_out = h6 - h7
    return {
        "states": {"7": st7, "8s": st8s, "8": st8, "8a": st8a, "8b": st8b, "5": st5, "6s": st6s, "6": st6},
        "w_st": w_st, "w_pump": w_pump, "w_net": w_net,
        "q_in": q_in, "q_out": q_out, "eta": w_net / q_in, "x6": st6["x"],
    }


def plot_Ts(cycle, fig_name="phase2_rankine_Ts.png", title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    draw_Ts(ax, cycle, title=title)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)


def draw_Ts(ax, cycle, title=None):
    from CoolProp.CoolProp import PropsSI

    st = cycle["states"]
    T_dome = np.linspace(273.16, PropsSI("Tcrit", steam.FLUID) - 0.01, 300)
    sf = [PropsSI("S", "T", T, "Q", 0, steam.FLUID) for T in T_dome]
    sg = [PropsSI("S", "T", T, "Q", 1, steam.FLUID) for T in T_dome]

    Ps = st["5"]["P"]
    T_econ = np.linspace(st["8"]["T"], st["8a"]["T"] - 1e-3, 40)
    T_sh = np.linspace(st["8b"]["T"] + 1e-3, st["5"]["T"], 40)
    s_econ = [steam.s_PT(Ps, T) for T in T_econ]
    s_sh = [steam.s_PT(Ps, T) for T in T_sh]

    kJ = 1e-3
    to_C = -273.15
    ax.plot(np.concatenate([sf, sg[::-1]]) * kJ, np.concatenate([T_dome, T_dome[::-1]]) + to_C,
            "k-", lw=0.8, label="saturation dome")

    path_s = ([st["7"]["s"], st["8"]["s"]] + s_econ + [st["8a"]["s"], st["8b"]["s"]] + s_sh
              + [st["5"]["s"], st["6"]["s"], st["7"]["s"]])
    path_T = ([st["7"]["T"], st["8"]["T"]] + list(T_econ) + [st["8a"]["T"], st["8b"]["T"]] + list(T_sh)
              + [st["5"]["T"], st["6"]["T"], st["7"]["T"]])
    ax.plot(np.array(path_s) * kJ, np.array(path_T) + to_C, "C0-", lw=1.8, label="cycle")
    ax.plot([st["5"]["s"] * kJ, st["6s"]["s"] * kJ], [st["5"]["T"] + to_C, st["6s"]["T"] + to_C],
            "C0--", lw=1.2, label="isentropic expansion")

    offsets = {"7": (-12, -12), "8": (-14, 4), "8a": (-18, 4), "8b": (6, 2), "5": (6, 0), "6": (6, -10),
               "6s": (-18, -12)}
    for name, (dx, dy) in offsets.items():
        ax.plot(st[name]["s"] * kJ, st[name]["T"] + to_C, "o", color="C3", ms=4)
        ax.annotate(name, (st[name]["s"] * kJ, st[name]["T"] + to_C), textcoords="offset points",
                    xytext=(dx, dy), fontsize=10)

    ax.set_xlabel("s [kJ/kg·K]")
    ax.set_ylabel("T [°C]")
    ax.set_title(title or "Rankine cycle T–s diagram")
    ax.set_xlim(0, 9.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
