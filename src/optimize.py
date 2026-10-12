"""Surrogate-assisted GA optimisation of eta_cc with direct-model verification and adaptive refinement.

Design vector: unit cube (rp, TIT, Ps, tau, Pcond). Constraints G <= 0 are normalised margins:
-m_quality/0.05, -(T_hrsg_max - T4)/50, -m_stack/50, -m_pinch/10, -m_approach/10.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.optimize import minimize

from src.combined_cycle import T_HRSG_MAX, direct_model_tau
from src.doe import MARGIN_SCALES, UNIT_COLUMNS, evaluate_points, evaluate_unit, lhs, to_physical
from src.surrogates import fit_surrogates, load_selection

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

GA_MARGINS = ("m_quality", "m_hrsg_inlet", "m_stack", "m_pinch", "m_approach")
GAP_TARGET_PP = 0.2
MAX_REFINEMENTS = 3
REFINE_POINTS = 50
REFINE_HALF_WIDTH = 0.10
ACTIVE_FRACTION = 0.02
# The true pinch margin is identically 0 by construction unless the economizer cold end
# crosses; the surrogate reproduces that zero only to ~1e-4 K, which would randomly mark
# feasible designs infeasible. A predicted m_pinch >= -0.05 K counts as satisfied: ~20x the
# ANN test RMSE, and far smaller than real cold-end crosses (-0.7 to -47 K in the DOE).
PINCH_NOISE_BAND = 0.05


def surrogate_margins(pred, T_hrsg_max):
    return {"m_quality": pred["m_quality"], "m_hrsg_inlet": T_hrsg_max - pred["T4"], "m_stack": pred["m_stack"],
            "m_pinch": pred["m_pinch"] + PINCH_NOISE_BAND, "m_approach": pred["m_approach"]}


def constraint_matrix(margins):
    return np.column_stack([-np.asarray(margins[k], dtype=float) / MARGIN_SCALES[k] for k in GA_MARGINS])


class SurrogateProblem(Problem):
    def __init__(self, surrogates, T_hrsg_max=T_HRSG_MAX):
        super().__init__(n_var=5, n_obj=1, n_ieq_constr=len(GA_MARGINS), xl=0.0, xu=1.0)
        self.surrogates = surrogates
        self.T_hrsg_max = T_hrsg_max

    def _evaluate(self, X, out, *args, **kwargs):
        pred = self.surrogates(X)
        out["F"] = -pred["eta_cc"]
        out["G"] = constraint_matrix(surrogate_margins(pred, self.T_hrsg_max))


class DirectProblem(ElementwiseProblem):
    def __init__(self, T_hrsg_max=T_HRSG_MAX):
        super().__init__(n_var=5, n_obj=1, n_ieq_constr=len(GA_MARGINS), xl=0.0, xu=1.0)
        self.T_hrsg_max = T_hrsg_max

    def _evaluate(self, x, out, *args, **kwargs):
        rp, TIT, Ps_bar, tau, Pcond_bar = to_physical(x)[0]
        r = direct_model_tau(rp, TIT, Ps_bar * 1e5, tau, Pcond_bar * 1e5, T_hrsg_max=self.T_hrsg_max)
        if not np.isfinite(r["eta_cc"]):
            out["F"], out["G"] = [0.0], np.full(len(GA_MARGINS), 1e3)
            return
        g = constraint_matrix({k: [r["margins"][k]] for k in GA_MARGINS})[0]
        out["F"] = [-100.0 * r["eta_cc"]]
        out["G"] = np.where(np.isfinite(g), g, 1e3)


class BestHistory(Callback):
    def __init__(self):
        super().__init__()
        self.best = []

    def notify(self, algorithm):
        opt = algorithm.opt[0]
        self.best.append(-opt.F[0] if opt.CV[0] <= 0.0 else np.nan)


def run_ga(problem, seeds, pop_size=100, n_gen=200):
    runs = []
    for seed in seeds:
        history = BestHistory()
        res = minimize(problem, GA(pop_size=pop_size, eliminate_duplicates=True), ("n_gen", n_gen),
                       seed=seed, callback=history, verbose=False)
        feasible = res.X is not None and res.CV is not None and res.CV[0] <= 0.0
        runs.append({"seed": seed, "x": res.X if feasible else None, "eta": -res.F[0] if feasible else np.nan,
                     "history": history.best})
    best = max((r for r in runs if r["x"] is not None), key=lambda r: r["eta"], default=None)
    return best, runs


def verify(x, surrogates, T_hrsg_max=T_HRSG_MAX):
    row = evaluate_unit(x, T_hrsg_max)
    eta_sur = float(surrogates(np.atleast_2d(x))["eta_cc"][0])
    return row, eta_sur, abs(eta_sur - row["eta_cc"])


def _history_row(loop, x, eta_sur, row, gap, n_samples):
    phys = to_physical(x)[0]
    return {"loop": loop, **dict(zip(UNIT_COLUMNS, x)), "rp": phys[0], "TIT": phys[1], "Ps_bar": phys[2],
            "tau": phys[3], "Pcond_bar": phys[4], "Ts_C": row["Ts_C"], "eta_sur": eta_sur,
            "eta_dir": row["eta_cc"], "gap_pp": gap, "feasible": bool(row["feasible"]),
            "reason": row["reason"], "n_samples": n_samples}


def surrogate_optimisation(data, selection, ann_params, T_hrsg_max=T_HRSG_MAX, seeds=(1, 2, 3, 4, 5),
                           refine_seeds=(1, 2, 3), pop_size=100, n_gen=200):
    surrogates = fit_surrogates(data, selection, ann_params)
    best, runs = run_ga(SurrogateProblem(surrogates, T_hrsg_max), seeds, pop_size, n_gen)
    row, eta_sur, gap = verify(best["x"], surrogates, T_hrsg_max)
    history = [_history_row(0, best["x"], eta_sur, row, gap, len(data))]
    x_best = best["x"]
    for loop in range(1, MAX_REFINEMENTS + 1):
        if gap <= GAP_TARGET_PP and row["feasible"]:
            break
        lo = np.clip(x_best - REFINE_HALF_WIDTH, 0.0, 1.0)
        hi = np.clip(x_best + REFINE_HALF_WIDTH, 0.0, 1.0)
        local = lo + lhs(REFINE_POINTS, seed=100 + loop) * (hi - lo)
        data = pd.concat([data, evaluate_points(local, T_hrsg_max)], ignore_index=True)
        surrogates = fit_surrogates(data, selection, ann_params)
        best, _ = run_ga(SurrogateProblem(surrogates, T_hrsg_max), refine_seeds, pop_size, n_gen)
        x_best = best["x"]
        row, eta_sur, gap = verify(x_best, surrogates, T_hrsg_max)
        history.append(_history_row(loop, x_best, eta_sur, row, gap, len(data)))
    return {"x": x_best, "row": row, "eta_sur": eta_sur, "gap": gap, "history": pd.DataFrame(history),
            "runs": runs, "data": data, "surrogates": surrogates}


def direct_optimisation(T_hrsg_max=T_HRSG_MAX, pop_size=60, n_gen=80, seed=1):
    best, runs = run_ga(DirectProblem(T_hrsg_max), [seed], pop_size, n_gen)
    return best, runs[0]


def optimum_table(x, T_hrsg_max=T_HRSG_MAX):
    rp, TIT, Ps_bar, tau, Pcond_bar = to_physical(x)[0]
    r = direct_model_tau(rp, TIT, Ps_bar * 1e5, tau, Pcond_bar * 1e5, T_hrsg_max=T_hrsg_max)
    active = [k for k in GA_MARGINS if abs(r["margins"][k]) <= ACTIVE_FRACTION * MARGIN_SCALES[k]]
    rows = [("rp [-]", rp), ("TIT [K]", TIT), ("Ps [bar]", Ps_bar), ("Ts [C]", r["Ts"] - 273.15),
            ("Pcond [bar]", Pcond_bar), ("tau [-]", tau), ("eta_cc [%]", 100 * r["eta_cc"]),
            ("W_net [MW]", r["W_net"] / 1e6), ("w_net [kJ/kg air]", r["w_net"] / 1e3),
            ("W_ST/W_GT [-]", r["work_ratio"]), ("eps_HRSG [-]", r["eps_HRSG"]),
            ("eta_GT [%]", 100 * r["eta_GT"]), ("m_s [kg/s]", r["m_s"]), ("T4 [C]", r["T4"] - 273.15),
            ("T4c [C]", r["T4c"] - 273.15), ("x6 [-]", r["x6"]), ("phi [-]", r["phi"])]
    rows += [(k, v) for k, v in r["margins"].items()]
    rows += [("feasible", r["feasible"]), ("active constraints", ", ".join(active) if active else "none")]
    return pd.DataFrame(rows, columns=["Quantity", "Value"]), r, active


def plot_convergence(runs, direct_run=None, fig_name="phase7_ga_convergence.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        ax.plot(np.arange(1, len(run["history"]) + 1), run["history"], lw=1.2, label=f"surrogate GA, seed {run['seed']}")
    if direct_run is not None:
        ax.plot(np.arange(1, len(direct_run["history"]) + 1), direct_run["history"], "k--", lw=1.5,
                label="direct-model GA (reference)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best feasible η$_{cc}$ [%]")
    ax.set_title("GA convergence")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)


def main(direct_reference=True):
    import warnings

    from src.combined_cycle import plot_TQ, plot_Ts
    from src.props.gas import A6ExtrapolationWarning

    warnings.simplefilter("ignore", A6ExtrapolationWarning)
    pd.set_option("display.width", 160)
    data = pd.read_csv(RESULTS_DIR / "doe.csv")
    selection, ann_params = load_selection()

    t0 = time.perf_counter()
    opt = surrogate_optimisation(data, selection, ann_params)
    t_sur = time.perf_counter() - t0
    opt["history"].to_csv(RESULTS_DIR / "optimization_history.csv", index=False)
    print(f"Surrogate optimisation: {t_sur:.0f} s, direct-model calls: {len(opt['data'])}")
    print(opt["history"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    table, result, active = optimum_table(opt["x"])
    table.to_csv(RESULTS_DIR / "optimum.csv", index=False)
    print("\nFinal optimum:")
    print(table.to_string(index=False))

    if direct_reference:
        t0 = time.perf_counter()
        direct_best, direct_run = direct_optimisation()
        t_dir = time.perf_counter() - t0
        direct_table, _, _ = optimum_table(direct_best["x"])
        comparison = pd.DataFrame({"Quantity": table["Quantity"], "surrogate GA": table["Value"],
                                   "direct GA": direct_table["Value"]})
        comparison.to_csv(RESULTS_DIR / "optimum_vs_direct_ga.csv", index=False)
        print(f"\nDirect-model GA reference: {t_dir:.0f} s, {60 * 80} direct-model calls")
        print(comparison.to_string(index=False))
        plot_convergence(opt["runs"], direct_run)
    else:
        print("\nDirect-model reference GA skipped: optimum_vs_direct_ga.csv and the convergence figure are kept.")
    plot_TQ(result, fig_name="phase7_optimum_TQ.png", title="HRSG T–Q diagram, optimum")
    plot_Ts(result, fig_name="phase7_optimum_Ts.png", title_suffix="optimum")


if __name__ == "__main__":
    main()
