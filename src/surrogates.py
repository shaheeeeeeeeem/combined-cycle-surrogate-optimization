"""RSM and ANN surrogates of the combined-cycle direct model.

Inputs are the 5 unit-cube design coordinates (rp, TIT, Ps, tau, Pcond), coded to [-1, 1].
Targets: eta_cc [%], W_net [MW], T4 [K] and the margins m_quality, m_stack, m_pinch, m_approach.
- T4 is modelled instead of m_hrsg_inlet, so any HRSG-inlet limit follows as T_hrsg_max - T4 without refitting.
- m_TIT needs no surrogate: it is exact from the TIT input.
- m_phi needs no surrogate: phi <= 0.407 over the whole design box, so phi < 0.5 never binds.
Infeasible DOE points are kept: the margin surrogates must see both sides of each constraint boundary.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

from src.doe import MARGIN_SCALES, UNIT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

TARGETS = ("eta_cc", "W_net_MW", "T4", "m_quality", "m_stack", "m_pinch", "m_approach")
CODED = ("rp", "TIT", "Ps", "tau", "Pcond")
RSM_FORMULA = "y ~ ({}) ** 2 + {}".format(" + ".join(CODED), " + ".join(f"I({c} ** 2)" for c in CODED))
ANN_GRID = {
    "regressor__hidden_layer_sizes": [(16, 16), (32, 32), (64, 64)],
    "regressor__activation": ["tanh", "relu"],
    "regressor__alpha": [1e-4, 1e-3, 1e-2],
}
SEED = 42


def coded_frame(U):
    return pd.DataFrame(2.0 * np.atleast_2d(U) - 1.0, columns=CODED)


def clean(df):
    return df.dropna(subset=list(UNIT_COLUMNS) + list(TARGETS)).reset_index(drop=True)


class RSM:
    def fit(self, U, y):
        data = coded_frame(U)
        data["y"] = np.asarray(y)
        self.model = ols(RSM_FORMULA, data=data).fit()
        return self

    def predict(self, U):
        return np.asarray(self.model.predict(coded_frame(U)))


def _ann(params=None):
    mlp = MLPRegressor(solver="lbfgs", max_iter=5000, random_state=SEED)
    model = TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
    return model.set_params(**params) if params else model


class ANN:
    def __init__(self, params=None):
        self.params = params

    def fit(self, U, y):
        X = coded_frame(U).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if self.params is None:
                search = GridSearchCV(_ann(), ANN_GRID, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
                search.fit(X, np.asarray(y))
                self.params = search.best_params_
                self.model = search.best_estimator_
            else:
                self.model = _ann(self.params).fit(X, np.asarray(y))
        return self

    def predict(self, U):
        return self.model.predict(coded_frame(U).to_numpy())


class Surrogates:
    """Selected surrogate per target; call with a (n, 5) unit-cube array to get a dict of predictions."""

    def __init__(self, models, selection):
        self.models = models
        self.selection = selection

    def __call__(self, U):
        return {t: m.predict(U) for t, m in self.models.items()}


def split(df):
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    return idx_train, idx_test


def evaluate_surrogates(df):
    """Fit RSM and ANN per target on the shared 80/20 split; return metrics, selection and test predictions."""
    df = clean(df)
    U = df[UNIT_COLUMNS].to_numpy()
    i_tr, i_te = split(df)
    rows, selection, ann_params, predictions, rsm_models = [], {}, {}, {}, {}
    for target in TARGETS:
        y = df[target].to_numpy()
        fitted = {"RSM": RSM().fit(U[i_tr], y[i_tr]), "ANN": ANN().fit(U[i_tr], y[i_tr])}
        rsm_models[target] = fitted["RSM"]
        ann_params[target] = fitted["ANN"].params
        rmse = {}
        for name, model in fitted.items():
            p_tr, p_te = model.predict(U[i_tr]), model.predict(U[i_te])
            predictions[(target, name)] = p_te
            rmse[name] = root_mean_squared_error(y[i_te], p_te)
            rows.append({"target": target, "model": name,
                         "train_R2": r2_score(y[i_tr], p_tr), "test_R2": r2_score(y[i_te], p_te),
                         "test_RMSE": rmse[name], "test_MAE": mean_absolute_error(y[i_te], p_te)})
        selection[target] = min(rmse, key=rmse.get)
    metrics = pd.DataFrame(rows)
    metrics["selected"] = [selection[t] == m for t, m in zip(metrics["target"], metrics["model"])]
    return {"df": df, "train_idx": i_tr, "test_idx": i_te, "metrics": metrics, "selection": selection,
            "ann_params": ann_params, "test_predictions": predictions, "rsm_models": rsm_models}


def fit_surrogates(df, selection, ann_params):
    """Refit the selected model of every target on all valid rows (ANN with its tuned hyperparameters)."""
    df = clean(df)
    U = df[UNIT_COLUMNS].to_numpy()
    models = {}
    for target in TARGETS:
        y = df[target].to_numpy()
        model = RSM() if selection[target] == "RSM" else ANN(ann_params[target])
        models[target] = model.fit(U, y)
    return Surrogates(models, selection)


def save_selection(ev, path=RESULTS_DIR / "surrogate_selection.json"):
    payload = {}
    for target in TARGETS:
        params = {k: list(v) if isinstance(v, tuple) else v for k, v in ev["ann_params"][target].items()}
        payload[target] = {"model": ev["selection"][target], "ann_params": params}
    path.write_text(json.dumps(payload, indent=2))


def load_selection(path=RESULTS_DIR / "surrogate_selection.json"):
    payload = json.loads(Path(path).read_text())
    selection = {t: v["model"] for t, v in payload.items()}
    ann_params = {t: {k: tuple(p) if isinstance(p, list) else p for k, p in v["ann_params"].items()}
                  for t, v in payload.items()}
    return selection, ann_params


def min_normalized_margin(df):
    # m_pinch is zero by construction wherever the cold end does not cross, so it would pin
    # every such point to distance 0; pinch crosses are marked separately instead.
    scales = {k: s for k, s in MARGIN_SCALES.items() if k != "m_pinch"}
    return pd.concat([df[k] / s for k, s in scales.items()], axis=1).min(axis=1)


def plot_parity(ev, fig_name="phase6_parity.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, i_te = ev["df"], ev["test_idx"]
    feasible = df["feasible"].to_numpy(dtype=bool)[i_te]
    metrics = ev["metrics"].set_index(["target", "model"])
    fig, axes = plt.subplots(4, 4, figsize=(15, 14))
    panels = [(t, m) for t in TARGETS for m in ("RSM", "ANN")]
    for ax, (target, name) in zip(axes.flat, panels):
        y = df[target].to_numpy()[i_te]
        p = ev["test_predictions"][(target, name)]
        ax.scatter(y[feasible], p[feasible], s=12, marker="o", label="feasible")
        ax.scatter(y[~feasible], p[~feasible], s=14, marker="x", color="C3", label="infeasible")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        r = metrics.loc[(target, name)]
        star = " (selected)" if ev["selection"][target] == name else ""
        ax.set_title(f"{target}: {name}{star}\nR²={r['test_R2']:.3f}, RMSE={r['test_RMSE']:.3g}", fontsize=9)
        ax.set_xlabel("direct model", fontsize=8)
        ax.set_ylabel("surrogate", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("Surrogate parity on the 20% test set", fontsize=13)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=150)
    plt.close(fig)


def failure_analysis(ev, fig_name="phase6_failure.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, i_te = ev["df"], ev["test_idx"]
    test = df.iloc[i_te].reset_index(drop=True)
    distance = min_normalized_margin(test).to_numpy()
    superheated = test["exhaust_superheated"].astype(bool).to_numpy()
    pinch_cross = (test["m_pinch"] < -1e-6).to_numpy()
    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, target in zip(axes, ("eta_cc", "m_quality")):
        name = ev["selection"][target]
        err = np.abs(ev["test_predictions"][(target, name)] - test[target].to_numpy())
        normal = ~superheated & ~pinch_cross
        ax.scatter(distance[normal], err[normal], s=14, label="other test points")
        ax.scatter(distance[superheated], err[superheated], s=30, marker="^", color="C2",
                   label="superheated ST exhaust (x6 capped at 1)")
        ax.scatter(distance[pinch_cross], err[pinch_cross], s=40, marker="s", color="C3",
                   label="cold-end pinch cross")
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("min normalised constraint margin (< 0: infeasible)")
        ax.set_ylabel(f"|test error| of {target} ({name})")
        ax.set_title(f"{target}: error vs distance to feasibility boundary")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        near = np.abs(distance) < 0.5
        stats[target] = {
            "model": name,
            "median_near": float(np.median(err[near])) if near.any() else np.nan,
            "median_far": float(np.median(err[~near])) if (~near).any() else np.nan,
            "median_superheated": float(np.median(err[superheated])) if superheated.any() else np.nan,
            "median_pinch_cross": float(np.median(err[pinch_cross])) if pinch_cross.any() else np.nan,
            "n_superheated": int(superheated.sum()), "n_pinch_cross": int(pinch_cross.sum()),
        }
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=200)
    plt.close(fig)
    return stats


def failure_sentences(ev, stats):
    m = ev["metrics"].set_index(["target", "model"])
    eta, q = stats["eta_cc"], stats["m_quality"]
    ratio = eta["median_near"] / eta["median_far"]
    comparison = "no larger than" if ratio <= 1.0 else f"only {ratio:.1f}x" if ratio < 2.0 else f"{ratio:.1f}x"
    return [
        f"eta_cc: the ANN reaches a test RMSE of {m.loc[('eta_cc', 'ANN'), 'test_RMSE']:.3f} pp against "
        f"{m.loc[('eta_cc', 'RSM'), 'test_RMSE']:.3f} pp for the full quadratic RSM; the RSM residual is the "
        "higher-order curvature of eta in rp and TIT (the interior eta maximum moves with TIT), which a "
        "second-order polynomial cannot follow.",
        f"m_pinch is the clearest RSM failure (test R2 {m.loc[('m_pinch', 'RSM'), 'test_R2']:.2f}): the pinch is "
        "imposed, so the margin is exactly zero everywhere except where the economizer cold end crosses "
        "(low rp, high TIT, low tau); a step-like response like this cannot be fitted by a smooth quadratic.",
        f"m_quality has a kink where the steam-turbine exhaust becomes superheated and x6 is capped at 1; the "
        f"{q['n_superheated']} such test point(s) have a median ANN error of {q['median_superheated']:.2g} "
        f"against {q['median_far']:.2g} elsewhere, and the cold-end pinch-cross test point(s) give an eta_cc "
        f"error of {eta['median_pinch_cross']:.2g} pp against {eta['median_far']:.2g} pp elsewhere.",
        f"Close to the feasibility boundary (|min normalised margin| < 0.5) the eta_cc error "
        f"({eta['median_near']:.2g} pp median) is {comparison} the error far from it "
        f"({eta['median_far']:.2g} pp), whereas the pinch-cross point is about "
        f"{eta['median_pinch_cross'] / eta['median_far']:.0f}x worse, so the largest failures sit at the physics "
        "kinks; both kink regions are sparsely sampled, which is why the GA optimum must be re-checked with the "
        "direct model.",
    ]


def save_rsm_tables(ev):
    model = ev["rsm_models"]["eta_cc"].model
    anova = sm.stats.anova_lm(model, typ=2)
    anova.to_csv(RESULTS_DIR / "rsm_anova_eta.csv", index_label="term")
    coeffs = pd.DataFrame({"coef": model.params, "std_err": model.bse, "p_value": model.pvalues})
    coeffs.to_csv(RESULTS_DIR / "rsm_coeffs_eta.csv", index_label="term")
    return anova


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    doe = pd.read_csv(RESULTS_DIR / "doe.csv")
    ev = evaluate_surrogates(doe)
    RESULTS_DIR.mkdir(exist_ok=True)
    ev["metrics"].to_csv(RESULTS_DIR / "surrogate_metrics.csv", index=False)
    print(ev["metrics"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nselected:", ev["selection"])
    print("ANN hyperparameters:", ev["ann_params"])
    save_selection(ev)
    anova = save_rsm_tables(ev)
    top = anova.drop(index="Residual").sort_values("F", ascending=False).head(5)
    print("\nTop 5 ANOVA terms for eta_cc (RSM):")
    print(top.to_string(float_format=lambda v: f"{v:.4g}"))
    plot_parity(ev)
    stats = failure_analysis(ev)
    print("\nWhere and why the surrogates fail:")
    for sentence in failure_sentences(ev, stats):
        print(f"- {sentence}")
