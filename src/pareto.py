"""Bi-objective NSGA-II on the surrogates: maximise eta_cc and W_net subject to all constraints."""

from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from src.combined_cycle import BASELINE, direct_model
from src.doe import MARGIN_SCALES, UNIT_COLUMNS, evaluate_unit, to_physical
from src.optimize import ACTIVE_FRACTION, GA_MARGINS, constraint_matrix, surrogate_margins
from src.surrogates import fit_surrogates, load_selection

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
HRSG_LIMITS_K = (873.15, 923.15, 973.15)
N_VERIFY = 5


class BiObjectiveProblem(Problem):
    def __init__(self, surrogates, T_hrsg_max):
        super().__init__(n_var=5, n_obj=2, n_ieq_constr=len(GA_MARGINS), xl=0.0, xu=1.0)
        self.surrogates = surrogates
        self.T_hrsg_max = T_hrsg_max

    def _evaluate(self, X, out, *args, **kwargs):
        pred = self.surrogates(X)
        out["F"] = np.column_stack([-pred["eta_cc"], -pred["W_net_MW"]])
        out["G"] = constraint_matrix(surrogate_margins(pred, self.T_hrsg_max))


def _active(margins, i=None):
    pick = (lambda v: v[i]) if i is not None else (lambda v: v)
    return ", ".join(k for k in GA_MARGINS if abs(pick(margins[k])) <= ACTIVE_FRACTION * MARGIN_SCALES[k]) or "none"


def knee_index(eta, W):
    e = (eta - eta.min()) / (eta.max() - eta.min())
    w = (W - W.min()) / (W.max() - W.min())
    i_a, i_b = int(np.argmax(w)), int(np.argmax(e))
    a, b = np.array([w[i_a], e[i_a]]), np.array([w[i_b], e[i_b]])
    d = b - a
    dist = np.abs(d[0] * (e - a[1]) - d[1] * (w - a[0])) / np.hypot(*d)
    return int(np.argmax(dist))


def pareto_front(surrogates, T_hrsg_max, pop_size=100, n_gen=200, seed=1):
    res = minimize(BiObjectiveProblem(surrogates, T_hrsg_max), NSGA2(pop_size=pop_size), ("n_gen", n_gen),
                   seed=seed, verbose=False)
    X = np.atleast_2d(res.X)
    order = np.argsort(-res.F[:, 1])
    X = X[order]
    pred = surrogates(X)
    margins = surrogate_margins(pred, T_hrsg_max)
    phys = to_physical(X)
    front = pd.DataFrame(X, columns=UNIT_COLUMNS)
    front["rp"], front["TIT"], front["Ps_bar"], front["tau"], front["Pcond_bar"] = phys.T
    front["eta_sur"] = pred["eta_cc"]
    front["W_net_sur"] = pred["W_net_MW"]
    front["T4_sur"] = pred["T4"]
    front["m_hrsg_inlet_sur"] = T_hrsg_max - pred["T4"]
    front["active_sur"] = [_active(margins, i) for i in range(len(front))]
    front["verified"] = False
    for col in ("eta_dir", "W_net_dir", "Ts_C", "gap_eta_pp", "gap_W_MW", "m_hrsg_inlet_dir", "feasible_dir",
                "active_dir", "reason_dir"):
        front[col] = "" if col in ("active_dir", "reason_dir") else np.nan
    front["feasible_dir"] = front["feasible_dir"].astype(object)
    for i in np.unique(np.round(np.linspace(0, len(front) - 1, N_VERIFY)).astype(int)):
        row = evaluate_unit(X[i], T_hrsg_max)
        front.loc[i, ["verified", "eta_dir", "W_net_dir", "Ts_C"]] = [True, row["eta_cc"], row["W_net_MW"], row["Ts_C"]]
        front.loc[i, "gap_eta_pp"] = abs(row["eta_cc"] - front.loc[i, "eta_sur"])
        front.loc[i, "gap_W_MW"] = abs(row["W_net_MW"] - front.loc[i, "W_net_sur"])
        front.loc[i, "feasible_dir"] = bool(row["feasible"])
        front.loc[i, "reason_dir"] = row["reason"]
        front.loc[i, "m_hrsg_inlet_dir"] = row["m_hrsg_inlet"]
        front.loc[i, "active_dir"] = _active({k: row[k] for k in GA_MARGINS})
    front["knee"] = False
    front.loc[knee_index(front["eta_sur"].to_numpy(), front["W_net_sur"].to_numpy()), "knee"] = True
    return front


def plot_fronts(fronts, optimum, baseline, fig_name="phase8_pareto.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.6))
    inset = ax.inset_axes([0.42, 0.08, 0.55, 0.55])
    knee = fronts[923.15][fronts[923.15]["knee"]].iloc[0]
    widths = {873.15: 1.2, 923.15: 2.6, 973.15: 4.2}
    for target in (ax, inset):
        for k, (limit, front) in enumerate(fronts.items()):
            color = f"C{k}"
            target.plot(front["W_net_sur"], front["eta_sur"], "-", color=color, lw=widths[limit], alpha=0.85,
                        zorder=5 - k, label=f"Pareto front, HRSG inlet ≤ {limit - 273.15:.0f} °C")
            v = front[front["verified"]]
            target.plot(v["W_net_dir"], v["eta_dir"], "o", mfc="none", mec=color, ms=8, mew=1.4, zorder=6)
        target.plot(knee["W_net_sur"], knee["eta_sur"], "D", color="k", ms=6, zorder=7)
        target.plot(optimum["W_net"], optimum["eta"], "*", color="gold", mec="k", ms=15, zorder=8,
                    label="single-objective optimum")
        target.grid(alpha=0.3)
    ax.plot(baseline["W_net"], baseline["eta"], "x", color="k", ms=10, mew=2, label="baseline")
    ax.annotate(f"baseline\n{baseline['W_net']:.1f} MW, {baseline['eta']:.2f} %", (baseline["W_net"], baseline["eta"]),
                textcoords="offset points", xytext=(10, -4), fontsize=8)
    inset.annotate(f"knee (650 °C)\n{knee['W_net_sur']:.1f} MW, {knee['eta_sur']:.2f} %",
                   (knee["W_net_sur"], knee["eta_sur"]), textcoords="offset points", xytext=(-80, -38), fontsize=8,
                   arrowprops={"arrowstyle": "-", "color": "0.3"})
    inset.set_title("zoom on the fronts", fontsize=8)
    inset.tick_params(labelsize=7)
    ax.indicate_inset_zoom(inset, edgecolor="0.5")
    ax.plot([], [], "o", mfc="none", mec="0.3", label="direct-model verification")
    ax.plot([], [], "D", color="k", label="knee of the 650 °C front")
    ax.set_xlabel("W$_{net}$ [MW]  (100 kg/s air)")
    ax.set_ylabel("η$_{cc}$ [%]")
    ax.set_title("η$_{cc}$ vs W$_{net}$ Pareto fronts (NSGA-II on surrogates)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)


def summary_sentences(fronts):
    f600, f650, f700 = (fronts[k] for k in HRSG_LIMITS_K)
    hi_w = f650.loc[f650["W_net_sur"].idxmax()]
    hi_eta = f650.loc[f650["eta_sur"].idxmax()]
    knee = f650[f650["knee"]].iloc[0]
    return [
        f"Along the 650 °C front the high-W_net end ({hi_w['W_net_sur']:.1f} MW, {hi_w['eta_sur']:.2f} %, rp "
        f"{hi_w['rp']:.1f}) is bounded by the HRSG-inlet limit (T4 margin {hi_w['m_hrsg_inlet_sur']:.1f} K on the "
        f"surrogate, {hi_w['m_hrsg_inlet_dir']:.1f} K on the direct model) on top of the pinch and approach, while "
        f"the high-efficiency end ({hi_eta['W_net_sur']:.1f} MW, {hi_eta['eta_sur']:.2f} %, rp {hi_eta['rp']:.1f}) "
        f"is set by the TIT, Ps and Pcond bounds with only [{hi_eta['active_dir']}] active (direct model).",
        f"Relaxing the HRSG-inlet limit from 600 to 650 to 700 °C raises the maximum W_net from "
        f"{f600['W_net_sur'].max():.1f} to {f650['W_net_sur'].max():.1f} to {f700['W_net_sur'].max():.1f} MW, while the "
        f"maximum eta_cc stays at {f600['eta_sur'].max():.2f} / {f650['eta_sur'].max():.2f} / "
        f"{f700['eta_sur'].max():.2f} %, so the tube limit only shapes the high-power end of the front.",
        f"We recommend operating near the 650 °C knee at rp {knee['rp']:.1f}, TIT {knee['TIT']:.0f} K, "
        f"Ps {knee['Ps_bar']:.1f} bar, Pcond {knee['Pcond_bar']:.3f} bar ({knee['W_net_sur']:.1f} MW, "
        f"{knee['eta_sur']:.2f} %), which gives up {hi_eta['eta_sur'] - knee['eta_sur']:.2f} pp of efficiency "
        f"against the efficiency end for {knee['W_net_sur'] - hi_eta['W_net_sur']:+.1f} MW of net power.",
    ]


def main():
    import sys
    import warnings

    from src.props.gas import A6ExtrapolationWarning

    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore", A6ExtrapolationWarning)
    pd.set_option("display.width", 200)
    selection, ann_params = load_selection()
    surrogates = fit_surrogates(pd.read_csv(RESULTS_DIR / "doe.csv"), selection, ann_params)
    fronts = {}
    for limit in HRSG_LIMITS_K:
        front = pareto_front(surrogates, limit)
        front.to_csv(RESULTS_DIR / f"pareto_{limit - 273.15:.0f}.csv", index=False)
        fronts[limit] = front
        v = front[front["verified"]]
        print(f"HRSG limit {limit - 273.15:.0f} C: {len(front)} front points, W_net "
              f"{front['W_net_sur'].min():.2f}-{front['W_net_sur'].max():.2f} MW, eta_cc "
              f"{front['eta_sur'].min():.3f}-{front['eta_sur'].max():.3f} %; verified gaps: eta max "
              f"{v['gap_eta_pp'].max():.3f} pp, W_net max {v['gap_W_MW'].max():.3f} MW, "
              f"all feasible: {bool(v['feasible_dir'].all())}")
        print(v[["rp", "TIT", "Ps_bar", "tau", "Pcond_bar", "eta_sur", "eta_dir", "W_net_sur", "W_net_dir",
                 "m_hrsg_inlet_dir", "feasible_dir", "reason_dir"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    knee = fronts[923.15][fronts[923.15]["knee"]].iloc[0]
    print(f"\nKnee (650 C): rp {knee['rp']:.2f}, TIT {knee['TIT']:.1f} K, Ps {knee['Ps_bar']:.2f} bar, tau "
          f"{knee['tau']:.3f}, Pcond {knee['Pcond_bar']:.4f} bar -> W_net {knee['W_net_sur']:.2f} MW, "
          f"eta_cc {knee['eta_sur']:.3f} % (surrogate)")
    opt = pd.read_csv(RESULTS_DIR / "optimum.csv").set_index("Quantity")["Value"]
    optimum = {"W_net": float(opt["W_net [MW]"]), "eta": float(opt["eta_cc [%]"])}
    base = direct_model(**BASELINE)
    plot_fronts(fronts, optimum, {"W_net": base["W_net"] / 1e6, "eta": 100 * base["eta_cc"]})
    print()
    for sentence in summary_sentences(fronts):
        print(f"- {sentence}")


if __name__ == "__main__":
    main()
