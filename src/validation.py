"""Validation of the property routines against the project's hand-traceable checks."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.props.gas import AIR, T_REF, T_VALID, GasMixture
from src.rankine import plot_Ts, rankine

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
TABLE_A71 = ROOT / "textbook_tables" / "BS7e_TableA7.1_air_ideal_gas.csv"

TOL_GAS_PCT = 0.5
TOL_STEAM_PCT = 0.1

CHECK1_INPUTS = {"T1": 300.0, "P1": 100e3, "rp": 15.0, "T3": 1400.0, "eta_c": 0.86, "eta_t": 0.89}
CHECK1_HAND = {
    "T2 [K]": 694.7,
    "T4 [K]": 787.7,
    "w_c [kJ/kg]": 407.4,
    "w_t [kJ/kg]": 706.5,
    "w_net [kJ/kg]": 299.1,
    "q_h [kJ/kg]": 807.4,
    "eta_th [%]": 37.05,
}
CHECK1_HAND_H = {"h1 [kJ/kg]": 300.47, "h3 [kJ/kg]": 1515.27}

CHECK2_INPUTS = {"Ps": 30e5, "Ts": 450.0 + 273.15, "Pcond": 7.5e3, "eta_st": 0.85, "eta_pump": 0.80}
CHECK2_HAND = {
    "h5 [kJ/kg]": 3344.00,
    "s5 [kJ/kg·K]": 7.0833,
    "h6s [kJ/kg]": 2208.4,
    "x6s [-]": 0.848,
    "h6 [kJ/kg]": 2378.8,
    "x6 [-]": 0.919,
    "h7 [kJ/kg]": 168.77,
    "s7 [kJ/kg·K]": 0.5763,
    "h8 [kJ/kg]": 172.54,
    "w_st [kJ/kg]": 965.2,
    "w_pump [kJ/kg]": 3.77,
    "w_net [kJ/kg]": 961.4,
    "q_H [kJ/kg]": 3171.5,
    "eta_Rankine [%]": 30.3,
}

# Gas constant for the A.6 "Air" row, as used in textbook_tables/README.md.
R_AIR_ROW = 287.0
AIR_ROW = GasMixture({"Air": 1.0}, basis="mass", molar_masses={"Air": 8314.46 / R_AIR_ROW})


def load_table_a71():
    return pd.read_csv(TABLE_A71)


def a71_reference_values():
    row = load_table_a71().set_index("T_K").loc[T_REF]
    return float(row["h_kJkg"]), float(row["s0_kJkgK"])


def brayton_air_standard(gas, T1, P1, rp, T3, eta_c, eta_t):
    P2 = rp * P1
    comp = gas.compress(T1, P1, P2, eta_c)
    turb = gas.expand(T3, P2, P1, eta_t)
    h1, h3 = comp["h_in"], turb["h_in"]
    q_h = h3 - comp["h_out"]
    w_net = turb["w"] - comp["w"]
    return {
        "T2s": comp["T_out_s"], "T2": comp["T_out"], "T4s": turb["T_out_s"], "T4": turb["T_out"],
        "h1": h1, "h3": h3, "w_c": comp["w"], "w_t": turb["w"],
        "w_net": w_net, "q_h": q_h, "eta_th": w_net / q_h,
    }


def comparison_table(rows, tol_pct=TOL_GAS_PCT):
    table = pd.DataFrame(rows, columns=["Quantity", "Hand value", "Code output"])
    table["% difference"] = 100.0 * (table["Code output"] - table["Hand value"]) / table["Hand value"]
    table["Pass/Fail"] = np.where(table["% difference"].abs() <= tol_pct, "Pass", "Fail")
    return table


def check1(gas=AIR, csv_name="validation_check1.csv"):
    r = brayton_air_standard(gas, **CHECK1_INPUTS)
    code = {
        "T2 [K]": r["T2"],
        "T4 [K]": r["T4"],
        "w_c [kJ/kg]": r["w_c"] / 1e3,
        "w_t [kJ/kg]": r["w_t"] / 1e3,
        "w_net [kJ/kg]": r["w_net"] / 1e3,
        "q_h [kJ/kg]": r["q_h"] / 1e3,
        "eta_th [%]": 100.0 * r["eta_th"],
    }
    rows = [(q, CHECK1_HAND[q], code[q]) for q in CHECK1_HAND]

    # A.7.1 uses its own reference (h = 298.62 kJ/kg at 298.15 K); the code uses h = 0 there.
    h_shift, _ = a71_reference_values()
    shifted = {"h1 [kJ/kg]": r["h1"] / 1e3 + h_shift, "h3 [kJ/kg]": r["h3"] / 1e3 + h_shift}
    rows += [(f"{q} (reference-state shift)", CHECK1_HAND_H[q], shifted[q]) for q in CHECK1_HAND_H]

    table = comparison_table(rows)
    if csv_name:
        RESULTS_DIR.mkdir(exist_ok=True)
        table.to_csv(RESULTS_DIR / csv_name, index=False)
    return table


def check1_air_row():
    return check1(gas=AIR_ROW, csv_name="validation_check1_A6_air_row.csv")


def check2(csv_name="validation_check2.csv"):
    cycle = rankine(**CHECK2_INPUTS)
    st = cycle["states"]
    code = {
        "h5 [kJ/kg]": st["5"]["h"] / 1e3,
        "s5 [kJ/kg·K]": st["5"]["s"] / 1e3,
        "h6s [kJ/kg]": st["6s"]["h"] / 1e3,
        "x6s [-]": st["6s"]["x"],
        "h6 [kJ/kg]": st["6"]["h"] / 1e3,
        "x6 [-]": st["6"]["x"],
        "h7 [kJ/kg]": st["7"]["h"] / 1e3,
        "s7 [kJ/kg·K]": st["7"]["s"] / 1e3,
        "h8 [kJ/kg]": st["8"]["h"] / 1e3,
        "w_st [kJ/kg]": cycle["w_st"] / 1e3,
        "w_pump [kJ/kg]": cycle["w_pump"] / 1e3,
        "w_net [kJ/kg]": cycle["w_net"] / 1e3,
        "q_H [kJ/kg]": cycle["q_in"] / 1e3,
        "eta_Rankine [%]": 100.0 * cycle["eta"],
    }
    table = comparison_table([(q, CHECK2_HAND[q], code[q]) for q in CHECK2_HAND], tol_pct=TOL_STEAM_PCT)
    if csv_name:
        RESULTS_DIR.mkdir(exist_ok=True)
        table.to_csv(RESULTS_DIR / csv_name, index=False)
    return table


def a6_extrapolation_study(T_min=300.0, T_max=1600.0, fig_name="phase1_A6_vs_A71.png"):
    table = load_table_a71()
    table = table[(table["T_K"] >= T_min) & (table["T_K"] <= T_max)].reset_index(drop=True)
    h_shift, s_shift = a71_reference_values()
    T = table["T_K"].to_numpy()
    h_code = AIR.h(T) / 1e3 + h_shift
    s_code = AIR.s0(T) / 1e3 + s_shift
    study = pd.DataFrame({
        "T_K": T,
        "h_err_pct": 100.0 * (h_code - table["h_kJkg"]) / table["h_kJkg"],
        "s0_err_pct": 100.0 * (s_code - table["s0_kJkgK"]) / table["s0_kJkgK"],
    })

    in_range = study["T_K"] <= T_VALID[1]
    extrapolated = (study["T_K"] > T_VALID[1]) & (study["T_K"] <= 1500.0)
    summary = pd.DataFrame({
        "range": [f"{T_min:.0f}-{T_VALID[1]:.0f} K", f"{T_VALID[1]:.0f}-1500 K"],
        "max |h error| [%]": [study.loc[m, "h_err_pct"].abs().max() for m in (in_range, extrapolated)],
        "max |s0 error| [%]": [study.loc[m, "s0_err_pct"].abs().max() for m in (in_range, extrapolated)],
    })

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(T_VALID[1], T_max, color="0.85", label="A.6 extrapolated")
    ax.plot(study["T_K"], study["h_err_pct"], "o-", ms=3, label="h")
    ax.plot(study["T_K"], study["s0_err_pct"], "s-", ms=3, label="s°")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlim(T_min, T_max)
    ax.set_xlabel("T [K]")
    ax.set_ylabel("Error vs Table A.7.1 [%]")
    ax.set_title("Air (21% O$_2$ / 79% N$_2$) from A.6 polynomials vs B&S Table A.7.1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)
    return study, summary


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("Check-1: 21/79 O2/N2 mixture from A.6 N2 and O2 rows")
    print(check1().to_string(index=False))
    print("\nCheck-1 comparison: A.6 'Air' row (not used in the model)")
    print(check1_air_row().to_string(index=False))
    print("\nA.6 mixture vs Table A.7.1 (reference-shifted)")
    print(a6_extrapolation_study()[1].to_string(index=False))
    print("\nCheck-2: Rankine cycle, CoolProp vs B&S steam tables")
    print(check2().to_string(index=False))
    plot_Ts(rankine(**CHECK2_INPUTS), title="Check-2 Rankine cycle (30 bar, 450 °C, 7.5 kPa)")
