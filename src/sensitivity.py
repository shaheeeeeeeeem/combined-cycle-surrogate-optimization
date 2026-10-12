"""Sensitivity of the optimum to the HRSG-inlet limit and to individual bounds and limits.

(a) The surrogate GA is re-run at HRSG-inlet limits of 600, 650 and 700 C and verified with the direct model.
(b) One-at-a-time levers at the final optimum, direct model only: tau = 1 is held and no other variable is
    re-optimised, so each delta is a local derivative-like estimate, not a re-optimised gain.
"""

from pathlib import Path

import pandas as pd

from src.combined_cycle import direct_model_tau
from src.doe import MARGIN_SCALES, to_physical
from src.optimize import ACTIVE_FRACTION, GA_MARGINS, surrogate_optimisation
from src.surrogates import load_selection

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
HRSG_LIMITS_K = (873.15, 923.15, 973.15)


def active_constraints(margins):
    return [k for k in GA_MARGINS if abs(margins[k]) <= ACTIVE_FRACTION * MARGIN_SCALES[k]]


def hrsg_limit_study(data, selection, ann_params, limits=HRSG_LIMITS_K, seeds=(1, 2, 3)):
    rows = []
    for limit in limits:
        opt = surrogate_optimisation(data, selection, ann_params, T_hrsg_max=limit, seeds=seeds)
        rp, TIT, Ps_bar, tau, Pcond_bar = to_physical(opt["x"])[0]
        r = direct_model_tau(rp, TIT, Ps_bar * 1e5, tau, Pcond_bar * 1e5, T_hrsg_max=limit)
        rows.append({
            "T_hrsg_max [C]": limit - 273.15, "rp": rp, "TIT [K]": TIT, "Ps [bar]": Ps_bar, "tau": tau,
            "Ts [C]": r["Ts"] - 273.15, "Pcond [bar]": Pcond_bar, "eta_cc [%]": 100 * r["eta_cc"],
            "eta_sur [%]": opt["eta_sur"], "gap [pp]": opt["gap"], "W_net [MW]": r["W_net"] / 1e6,
            "T4 [C]": r["T4"] - 273.15, "m_hrsg_inlet [K]": r["margins"]["m_hrsg_inlet"],
            "refinement loops": int(opt["history"]["loop"].max()), "feasible": r["feasible"],
            "active constraints": ", ".join(active_constraints(r["margins"])) or "none",
        })
    return pd.DataFrame(rows)


def read_optimum(path=RESULTS_DIR / "optimum.csv"):
    table = pd.read_csv(path).set_index("Quantity")["Value"]
    return {"rp": float(table["rp [-]"]), "TIT": float(table["TIT [K]"]), "Ps": float(table["Ps [bar]"]) * 1e5,
            "tau": float(table["tau [-]"]), "Pcond": float(table["Pcond [bar]"]) * 1e5}


def lever_study(x_opt):
    base_kwargs = {**x_opt, "tau": 1.0}
    levers = [
        ("TIT +50 K (to 1550 K, A.6 extrapolated)", {"TIT": base_kwargs["TIT"] + 50.0}),
        ("Ps +10 bar", {"Ps": base_kwargs["Ps"] + 10e5}),
        ("Pcond -0.01 bar", {"Pcond": base_kwargs["Pcond"] - 0.01e5}),
        ("HRSG inlet limit -50 K", {"T_hrsg_max": 923.15 - 50.0}),
        ("HRSG inlet limit +50 K", {"T_hrsg_max": 923.15 + 50.0}),
        ("Superheater approach 5 -> 10 K", {"dT_approach": 10.0}),
        ("Evaporator pinch 10 -> 15 K", {"dT_pinch": 15.0}),
    ]
    base = direct_model_tau(**base_kwargs)
    rows = []
    for name, change in levers:
        r = direct_model_tau(**{**base_kwargs, **change})
        rows.append({
            "lever": name, "eta_cc [%]": 100 * r["eta_cc"], "W_net [MW]": r["W_net"] / 1e6,
            "d_eta_cc [pp]": 100 * (r["eta_cc"] - base["eta_cc"]), "d_W_net [MW]": (r["W_net"] - base["W_net"]) / 1e6,
            "feasible": r["feasible"], "reason": r["reason"],
        })
    table = pd.DataFrame(rows)
    return table, base


def plot_levers(table, fig_name="phase8_levers.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.sort_values("d_eta_cc [pp]")
    colors = ["C2" if v > 0 else "C3" if v < 0 else "0.6" for v in t["d_eta_cc [pp]"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(t["lever"], t["d_eta_cc [pp]"], color=colors)
    for y, (v, ok) in enumerate(zip(t["d_eta_cc [pp]"], t["feasible"])):
        label = f"{v:+.3f}" + ("" if ok else " (infeasible)")
        ax.annotate(label, (v, y), textcoords="offset points", xytext=(4 if v >= 0 else -4, 0),
                    ha="left" if v >= 0 else "right", va="center", fontsize=8)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Δη$_{cc}$ at the optimum [percentage points]")
    ax.set_title("One-at-a-time levers at the optimum (τ = 1 held, nothing re-optimised)")
    ax.grid(axis="x", alpha=0.3)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)


def main():
    import warnings

    from src.props.gas import A6ExtrapolationWarning

    warnings.simplefilter("ignore", A6ExtrapolationWarning)
    pd.set_option("display.width", 200)
    data = pd.read_csv(RESULTS_DIR / "doe.csv")
    selection, ann_params = load_selection()
    hrsg = hrsg_limit_study(data, selection, ann_params)
    hrsg.to_csv(RESULTS_DIR / "sensitivity_hrsg.csv", index=False)
    print("HRSG-inlet limit sensitivity:")
    print(hrsg.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    levers, base = lever_study(read_optimum())
    levers.to_csv(RESULTS_DIR / "sensitivity_bounds.csv", index=False)
    print(f"\nOne-at-a-time levers at the optimum (eta_cc {100 * base['eta_cc']:.3f} %, "
          f"W_net {base['W_net'] / 1e6:.2f} MW):")
    print(levers.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    plot_levers(levers)


if __name__ == "__main__":
    main()
