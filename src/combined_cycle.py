"""Combined-cycle direct model: gas turbine + HRSG + Rankine bottoming cycle.

Inputs are SI: rp [-], TIT [K], Ps [Pa], Ts [K], Pcond [Pa].
Constraint margins are positive when satisfied, in natural units (K, quality, phi).
"""

from pathlib import Path

import numpy as np

from src.combustor import ETA_F, LHV_CH4
from src.gas_turbine import P0, gas_turbine
from src.hrsg import hrsg
from src.props import steam

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

T_HRSG_MAX = 923.15
TIT_MAX = 1550.0
T_STACK_MIN = 373.15
DT_PINCH_MIN = 10.0
DT_APPROACH_MIN = 5.0
X6_MIN = 0.88
PHI_MAX = 0.5
SUPERHEAT_MIN = 20.0
FEASIBILITY_TOL = 1e-6

MARGIN_KEYS = ("m_TIT", "m_hrsg_inlet", "m_stack", "m_pinch", "m_approach", "m_quality", "m_phi")

BASELINE = {"rp": 15.0, "TIT": 1400.0, "Ps": 30e5, "Ts": 450.0 + 273.15, "Pcond": 0.075e5}


def _gas_state(gas, T, P, h):
    return {"T": T, "P": P, "h": h, "s": gas.s0(T) - gas.R_mix * np.log(P / P0)}


def _violations(margins):
    return [k for k in MARGIN_KEYS if not margins[k] >= -FEASIBILITY_TOL]


def _evaluate(rp, TIT, Ps, Ts, Pcond, T_hrsg_max, gt=None):
    gt = gt or gas_turbine(rp, TIT)
    hr = hrsg(gt, Ps, Ts, Pcond)
    cycle = hr["cycle"]
    st_steam = cycle["states"]
    gas = gt["products"]

    m_s = hr["m_s"]
    W_GT_net = gt["W_GT_net"]
    W_ST_net = m_s * cycle["w_net"]
    W_net = W_GT_net + W_ST_net
    fuel_input = gt["m_fuel"] * LHV_CH4

    exhaust_superheated = cycle["x6"] is None
    x6 = 1.0 if exhaust_superheated else cycle["x6"]
    T4 = gt["T4"]
    margins = {
        "m_TIT": TIT_MAX - TIT,
        "m_hrsg_inlet": T_hrsg_max - T4,
        "m_stack": hr["T4c"] - T_STACK_MIN,
        # Pinch is checked over economizer + evaporator; the superheater hot end is
        # governed by the separate 5 K approach constraint.
        "m_pinch": hr["min_dT_pinch"] - DT_PINCH_MIN,
        "m_approach": (T4 - Ts) - DT_APPROACH_MIN,
        "m_quality": x6 - X6_MIN,
        "m_phi": PHI_MAX - gt["phi"],
    }
    violated = _violations(margins)

    P4, P4c = hr["P_gas_in"], hr["P_gas_out"]
    frac_4a = hr["Q_sh"] / hr["Q_total"]
    frac_4b = (hr["Q_sh"] + hr["Q_ev"]) / hr["Q_total"]
    states = dict(gt["states"])
    # Gas-side pressure assumed to fall linearly with heat removed through the HRSG.
    states["4a"] = _gas_state(gas, hr["T4a"], P4 - frac_4a * (P4 - P4c), hr["h4a"])
    states["4b"] = _gas_state(gas, hr["T4b"], P4 - frac_4b * (P4 - P4c), hr["h4b"])
    states["4c"] = _gas_state(gas, hr["T4c"], P4c, hr["h4c"])
    states.update(st_steam)

    combustor_heat = fuel_input * ETA_F
    outputs = W_GT_net + W_ST_net + hr["Q_stack"] + hr["Q_cond"]
    energy = {
        "fuel_input": fuel_input,
        "combustor_loss": fuel_input - combustor_heat,
        "combustor_heat": combustor_heat,
        "W_GT_net": W_GT_net,
        "W_ST_net": W_ST_net,
        "Q_stack": hr["Q_stack"],
        "Q_cond": hr["Q_cond"],
        "residual": combustor_heat - outputs,
        "residual_rel": (combustor_heat - outputs) / combustor_heat,
    }
    return {
        "eta_cc": W_net / fuel_input,
        "W_net": W_net, "W_GT_net": W_GT_net, "W_ST_net": W_ST_net, "w_net": W_net / gt["m_air"],
        "work_ratio": W_ST_net / W_GT_net, "eps_HRSG": hr["eps_HRSG"], "m_s": m_s,
        "f": gt["f"], "phi": gt["phi"], "m_fuel": gt["m_fuel"], "m_gas": gt["m_gas"], "m_air": gt["m_air"],
        "T4": T4, "T4c": hr["T4c"], "x6": x6, "exhaust_superheated": exhaust_superheated, "Ts": Ts,
        "eta_GT": gt["eta_GT"], "eta_rankine": cycle["eta"],
        "min_dT": hr["min_dT"], "min_dT_location": hr["min_dT_location"],
        "min_dT_pinch": hr["min_dT_pinch"], "min_dT_pinch_location": hr["min_dT_pinch_location"],
        "Q_sh": hr["Q_sh"], "Q_ev": hr["Q_ev"], "Q_ec": hr["Q_ec"],
        "margins": margins, "feasible": not violated, "reason": ", ".join(violated) if violated else "ok",
        "states": states, "profile": hr["profile"], "energy": energy, "products": gas,
    }


def _failed(reason, rp, TIT, Ts, T_hrsg_max):
    margins = {k: np.nan for k in MARGIN_KEYS}
    margins["m_TIT"] = TIT_MAX - TIT
    try:
        gt = gas_turbine(rp, TIT)
        margins["m_hrsg_inlet"] = T_hrsg_max - gt["T4"]
        margins["m_phi"] = PHI_MAX - gt["phi"]
    except Exception:
        pass
    return {"eta_cc": np.nan, "W_net": np.nan, "w_net": np.nan, "Ts": Ts,
            "margins": margins, "feasible": False, "reason": f"error: {reason}"}


def direct_model(rp, TIT, Ps, Ts, Pcond, T_hrsg_max=T_HRSG_MAX):
    try:
        return _evaluate(rp, TIT, Ps, Ts, Pcond, T_hrsg_max)
    except Exception as exc:
        return _failed(exc, rp, TIT, Ts, T_hrsg_max)


def Ts_from_tau(tau, Ps, T4):
    T_lo = steam.Tsat(Ps) + SUPERHEAT_MIN
    T_hi = T4 - DT_APPROACH_MIN
    return T_lo + tau * (T_hi - T_lo), T_hi >= T_lo


def direct_model_tau(rp, TIT, Ps, tau, Pcond, T_hrsg_max=T_HRSG_MAX):
    Ts = np.nan
    try:
        gt = gas_turbine(rp, TIT)
        Ts, window_ok = Ts_from_tau(tau, Ps, gt["T4"])
        if not window_ok:
            Ts = steam.Tsat(Ps) + SUPERHEAT_MIN
        result = _evaluate(rp, TIT, Ps, Ts, Pcond, T_hrsg_max, gt=gt)
        if not window_ok:
            result["feasible"] = False
            result["reason"] = "no superheat window" + (f"; {result['reason']}" if result["reason"] != "ok" else "")
        result["tau"] = tau
        return result
    except Exception as exc:
        result = _failed(exc, rp, TIT, Ts, T_hrsg_max)
        result["tau"] = tau
        return result


def baseline_summary(result):
    import pandas as pd

    rows = [
        ("eta_cc [%]", 100 * result["eta_cc"]),
        ("eta_GT [%]", 100 * result["eta_GT"]),
        ("eta_Rankine [%]", 100 * result["eta_rankine"]),
        ("W_net [MW]", result["W_net"] / 1e6),
        ("W_GT_net [MW]", result["W_GT_net"] / 1e6),
        ("W_ST_net [MW]", result["W_ST_net"] / 1e6),
        ("w_net [kJ/kg air]", result["w_net"] / 1e3),
        ("W_ST/W_GT [-]", result["work_ratio"]),
        ("eps_HRSG [-]", result["eps_HRSG"]),
        ("f [-]", result["f"]),
        ("phi [-]", result["phi"]),
        ("m_fuel [kg/s]", result["m_fuel"]),
        ("m_s [kg/s]", result["m_s"]),
        ("T4 [C]", result["T4"] - 273.15),
        ("T4c [C]", result["T4c"] - 273.15),
        ("Ts [C]", result["Ts"] - 273.15),
        ("x6 [-]", result["x6"]),
        ("min dT over HRSG profile [K]", result["min_dT"]),
        ("energy residual (relative) [-]", result["energy"]["residual_rel"]),
    ]
    units = {"m_TIT": "K", "m_hrsg_inlet": "K", "m_stack": "K", "m_pinch": "K", "m_approach": "K",
             "m_quality": "-", "m_phi": "-"}
    rows += [(f"{k} [{units[k]}]", v) for k, v in result["margins"].items()]
    rows += [("feasible", result["feasible"]), ("reason", result["reason"])]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def plot_TQ(result, fig_name="phase4_baseline_TQ.png", title="HRSG T–Q diagram, baseline"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = result["profile"]
    st = result["states"]
    Q = prof["Q"] / 1e6
    to_C = -273.15
    Q_ec, Q_ev, Q_sh = result["Q_ec"] / 1e6, result["Q_ev"] / 1e6, result["Q_sh"] / 1e6
    Q_tot = Q_ec + Q_ev + Q_sh

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(Q, prof["T_gas"] + to_C, "C3-", lw=2, label="gas (4 → 4a → 4b → 4c)")
    ax.plot(Q, prof["T_steam"] + to_C, "C0-", lw=2, label="water/steam (8 → 8a → 8b → 5)")
    gas_pts = {"4c": (0.0, st["4c"]["T"]), "4b": (Q_ec, st["4b"]["T"]),
               "4a": (Q_ec + Q_ev, st["4a"]["T"]), "4": (Q_tot, st["4"]["T"])}
    steam_pts = {"8": (0.0, st["8"]["T"]), "8a": (Q_ec, st["8a"]["T"]),
                 "8b": (Q_ec + Q_ev, st["8b"]["T"]), "5": (Q_tot, st["5"]["T"])}
    for name, (q, T) in gas_pts.items():
        ax.plot(q, T + to_C, "o", color="C3", ms=4)
        ax.annotate(name, (q, T + to_C), textcoords="offset points", xytext=(-4, 7), fontsize=10, color="C3")
    for name, (q, T) in steam_pts.items():
        ax.plot(q, T + to_C, "o", color="C0", ms=4)
        ax.annotate(name, (q, T + to_C), textcoords="offset points", xytext=(-4, -15), fontsize=10, color="C0")

    pinch = st["4b"]["T"] - st["8a"]["T"]
    approach = st["4"]["T"] - st["5"]["T"]
    ax.annotate("", (Q_ec, st["8a"]["T"] + to_C), (Q_ec, st["4b"]["T"] + to_C),
                arrowprops={"arrowstyle": "<->", "color": "k"})
    ax.annotate(f"pinch ΔT = {pinch:.1f} K", (Q_ec, (st["4b"]["T"] + st["8a"]["T"]) / 2 + to_C),
                textcoords="offset points", xytext=(8, -24), fontsize=9)
    ax.annotate("", (Q_tot, st["5"]["T"] + to_C), (Q_tot, st["4"]["T"] + to_C),
                arrowprops={"arrowstyle": "<->", "color": "k"})
    ax.annotate(f"approach ΔT = {approach:.1f} K", (Q_tot, (st["4"]["T"] + st["5"]["T"]) / 2 + to_C),
                textcoords="offset points", xytext=(6, -3), fontsize=9)
    for q in (Q_ec, Q_ec + Q_ev):
        ax.axvline(q, color="0.7", lw=0.8, ls=":")
    ax.set_xlim(-0.03 * Q_tot, 1.25 * Q_tot)
    ax.set_xlabel("Cumulative heat transferred from stack end, Q [MW]")
    ax.set_ylabel("T [°C]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)


def plot_Ts(result, fig_name="phase4_baseline_Ts.png", title_suffix="baseline"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.rankine import draw_Ts

    st = result["states"]
    rp_P = st["2"]["P"]
    products = result["products"]
    kJ, to_C = 1e-3, -273.15

    def s_line(gas, T, P):
        return (gas.s0(T) - gas.R_mix * np.log(P / P0)) * kJ

    fig, (ax_g, ax_s) = plt.subplots(1, 2, figsize=(13, 5))
    T_comb = np.linspace(st["2"]["T"], st["3"]["T"], 60)
    T_hrsg = np.linspace(st["4"]["T"], st["4c"]["T"], 60)
    P_hrsg = np.linspace(st["4"]["P"], st["4c"]["P"], 60)
    ax_g.plot([st["1"]["s"] * kJ, st["2"]["s"] * kJ], [st["1"]["T"] + to_C, st["2"]["T"] + to_C], "C0-", lw=2,
              label="compression 1 → 2 (air)")
    ax_g.plot(s_line(products, T_comb, rp_P), T_comb + to_C, "C3-", lw=2, label="combustion 2 → 3 (products)")
    ax_g.plot([st["3"]["s"] * kJ, st["4"]["s"] * kJ], [st["3"]["T"] + to_C, st["4"]["T"] + to_C], "C1-", lw=2,
              label="expansion 3 → 4")
    ax_g.plot(s_line(products, T_hrsg, P_hrsg), T_hrsg + to_C, "C2-", lw=2, label="HRSG 4 → 4c")
    ax_g.plot([st["1"]["s"] * kJ, st["2s"]["s"] * kJ], [st["1"]["T"] + to_C, st["2s"]["T"] + to_C], "C0--", lw=1)
    ax_g.plot([st["3"]["s"] * kJ, st["4s"]["s"] * kJ], [st["3"]["T"] + to_C, st["4s"]["T"] + to_C], "C1--", lw=1)
    for name in ("1", "2s", "2", "3", "4s", "4", "4a", "4b", "4c"):
        ax_g.plot(st[name]["s"] * kJ, st[name]["T"] + to_C, "ko", ms=3)
        ax_g.annotate(name, (st[name]["s"] * kJ, st[name]["T"] + to_C), textcoords="offset points",
                      xytext=(-16, -4) if name == "4s" else (5, -4), fontsize=9)
    ax_g.set_xlabel("s [kJ/kg·K]  (s° − R ln(P/1 atm), mixing term excluded)")
    ax_g.set_ylabel("T [°C]")
    ax_g.set_title(f"(a) Gas side, {title_suffix}")
    ax_g.grid(alpha=0.3)
    ax_g.legend(fontsize=8, loc="upper left")

    steam_states = {k: st[k] for k in ("7", "8s", "8", "8a", "8b", "5", "6s", "6")}
    draw_Ts(ax_s, {"states": steam_states}, title=f"(b) Steam side, {title_suffix}")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import warnings

    import pandas as pd

    from src.props.gas import A6ExtrapolationWarning

    warnings.simplefilter("ignore", A6ExtrapolationWarning)
    base = direct_model(**BASELINE)
    summary = baseline_summary(base)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "baseline_summary.csv", index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.5g}"):
        print(summary.to_string(index=False))
    plot_TQ(base)
    plot_Ts(base)
