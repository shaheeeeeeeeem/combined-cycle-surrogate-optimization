"""Latin hypercube design of experiments over the 5-D design space.

Design vector in the unit cube: (rp, TIT, Ps, tau, Pcond). Ts follows from tau inside the
superheat window [Tsat(Ps) + 20 K, T4 - 5 K].
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.combined_cycle import MARGIN_KEYS, T_HRSG_MAX, direct_model_tau

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

VARIABLES = ("rp", "TIT", "Ps", "tau", "Pcond")
LOWER = np.array([8.0, 1200.0, 10.0, 0.0, 0.05])
UPPER = np.array([30.0, 1500.0, 60.0, 1.0, 0.15])
UNIT_COLUMNS = [f"u_{v}" for v in VARIABLES]
MARGIN_SCALES = {"m_quality": 0.05, "m_hrsg_inlet": 50.0, "m_stack": 50.0, "m_pinch": 10.0, "m_approach": 10.0}


def to_physical(U):
    """Unit cube -> (rp, TIT [K], Ps [bar], tau, Pcond [bar])."""
    return LOWER + np.atleast_2d(U) * (UPPER - LOWER)


def evaluate_unit(u, T_hrsg_max=T_HRSG_MAX):
    rp, TIT, Ps_bar, tau, Pcond_bar = to_physical(u)[0]
    start = time.perf_counter()
    r = direct_model_tau(rp, TIT, Ps_bar * 1e5, tau, Pcond_bar * 1e5, T_hrsg_max=T_hrsg_max)
    elapsed = time.perf_counter() - start
    row = dict(zip(UNIT_COLUMNS, u))
    row.update({"rp": rp, "TIT": TIT, "Ps_bar": Ps_bar, "tau": tau, "Pcond_bar": Pcond_bar})
    row.update({
        "Ts_C": r.get("Ts", np.nan) - 273.15,
        "eta_cc": 100.0 * r.get("eta_cc", np.nan),
        "W_net_MW": r.get("W_net", np.nan) / 1e6,
        "w_net_kJkg": r.get("w_net", np.nan) / 1e3,
        "T4": r.get("T4", np.nan),
        "work_ratio": r.get("work_ratio", np.nan),
        "eps_HRSG": r.get("eps_HRSG", np.nan),
    })
    row.update({k: r["margins"][k] for k in MARGIN_KEYS})
    row.update({"feasible": r["feasible"], "reason": r["reason"],
                "exhaust_superheated": r.get("exhaust_superheated", np.nan), "eval_time_s": elapsed})
    return row


def evaluate_points(U, T_hrsg_max=T_HRSG_MAX):
    return pd.DataFrame([evaluate_unit(u, T_hrsg_max) for u in np.atleast_2d(U)])


def lhs(n, seed=42):
    return qmc.LatinHypercube(d=len(VARIABLES), seed=seed).random(n)


def run_doe(n=500, seed=42, csv_name="doe.csv"):
    df = evaluate_points(lhs(n, seed))
    if csv_name:
        RESULTS_DIR.mkdir(exist_ok=True)
        df.to_csv(RESULTS_DIR / csv_name, index=False)
    return df


def summarize(df):
    failed = df["eta_cc"].isna()
    violations = {k: int((df[k] < -1e-6).sum()) for k in MARGIN_KEYS}
    margin_stats = df[list(MARGIN_KEYS)].agg(["min", "median"]).T
    return {
        "n": len(df),
        "feasible_fraction": float(df["feasible"].mean()),
        "n_feasible": int(df["feasible"].sum()),
        "n_physics_failures": int(failed.sum()),
        "n_no_superheat_window": int(df["reason"].str.startswith("no superheat window").sum()),
        "violations": violations,
        "margin_stats": margin_stats,
        "mean_eval_time_s": float(df["eval_time_s"].mean()),
    }


def plot_doe(df, fig_name="phase5_doe.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feasible = df["feasible"].astype(bool)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8))
    sc = ax_a.scatter(df.loc[feasible, "rp"], df.loc[feasible, "TIT"], c=df.loc[feasible, "eta_cc"],
                      cmap="viridis", s=22, label="feasible")
    ax_a.scatter(df.loc[~feasible, "rp"], df.loc[~feasible, "TIT"], c=df.loc[~feasible, "eta_cc"],
                 cmap="viridis", marker="x", s=22, label="infeasible", vmin=sc.get_clim()[0], vmax=sc.get_clim()[1])
    fig.colorbar(sc, ax=ax_a, label="η$_{cc}$ [%]")
    ax_a.set_xlabel("Pressure ratio r$_p$")
    ax_a.set_ylabel("TIT [K]")
    ax_a.set_title(f"(a) DOE, r$_p$–TIT projection ({feasible.sum()}/{len(df)} feasible)")
    ax_a.legend(loc="lower right", fontsize=8)

    violations = {k: int((df[k] < -1e-6).sum()) for k in MARGIN_KEYS}
    ax_b.bar(list(violations), list(violations.values()), color="C3")
    for i, v in enumerate(violations.values()):
        ax_b.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax_b.set_ylabel("Number of DOE points violating")
    ax_b.set_title("(b) Constraint violations (a point may violate several)")
    ax_b.tick_params(axis="x", rotation=30)
    ax_b.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import warnings

    from src.props.gas import A6ExtrapolationWarning

    warnings.simplefilter("ignore", A6ExtrapolationWarning)
    df = run_doe()
    s = summarize(df)
    print(f"DOE points: {s['n']}, feasible: {s['n_feasible']} ({100 * s['feasible_fraction']:.1f}%), "
          f"physics failures (NaN): {s['n_physics_failures']}, no superheat window: {s['n_no_superheat_window']}")
    print(f"mean direct-model time: {1e3 * s['mean_eval_time_s']:.1f} ms")
    print("violations per constraint:", s["violations"])
    print(s["margin_stats"].to_string(float_format=lambda v: f"{v:.4g}"))
    plot_doe(df)
