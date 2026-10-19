"""
plots.py
Visualization utilities for APMR Regression experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


def _save(name: str):
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# APMR OPTIMIZATION DIAGNOSTICS
# ============================================================

def plot_convergence(history):
    """
    APMR objective convergence.
    """

    objective = np.asarray(history["objective"])

    plt.figure(figsize=(6.5, 4.5))

    plt.plot(
        np.arange(1, len(objective) + 1),
        objective,
        linewidth=1.8,
    )

    plt.xlabel("Outer Iteration")
    plt.ylabel("Median Absolute Residual")
    plt.title("APMR Objective Convergence")

    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    _save("convergence")


def plot_step_norm(history):
    """
    APMR step-size convergence.
    """

    step_norm = np.asarray(history["step_norm"])

    plt.figure(figsize=(6.5, 4.5))

    plt.semilogy(
        np.arange(1, len(step_norm) + 1),
        step_norm,
        linewidth=1.8,
    )

    plt.xlabel("Outer Iteration")
    plt.ylabel(r"$\|x^{k+1}-x^k\|_2$")
    plt.title("APMR Step Norm")

    plt.grid(True, alpha=0.3)

    _save("step_norm")


def plot_gradient_norm(history):
    """
    APMR gradient/subgradient norm.
    """

    gradient_norm = np.asarray(history["gradient_norm"])

    plt.figure(figsize=(6.5, 4.5))

    plt.semilogy(
        np.arange(1, len(gradient_norm) + 1),
        gradient_norm,
        linewidth=1.8,
    )

    plt.xlabel("Outer Iteration")
    plt.ylabel(r"$\|g_k\|_2$")
    plt.title("APMR Gradient Norm")

    plt.grid(True, alpha=0.3)

    _save("gradient_norm")


def plot_active_set(history):
    """
    Evolution of the active median set.
    """

    active_set = np.asarray(history["active_set_size"])

    plt.figure(figsize=(6.5, 4.5))

    plt.plot(
        np.arange(1, len(active_set) + 1),
        active_set,
        linewidth=1.8,
    )

    plt.xlabel("Outer Iteration")
    plt.ylabel(r"$|\mathcal{I}_k|$")
    plt.title("Active Median Set Size")

    plt.grid(True, alpha=0.3)

    _save("active_set")


# ============================================================
# RESIDUAL / PREDICTION DIAGNOSTICS
# ============================================================

def plot_residual_histogram(
    model,
    A,
    b,
):
    """
    Residual distribution for APMR.
    """

    residuals = model.residuals(
        A,
        b,
        model.coef_,
    )

    plt.figure(figsize=(6.5, 4.5))

    plt.hist(
        residuals,
        bins=30,
        alpha=0.75,
    )

    plt.axvline(
        np.median(np.abs(residuals)),
        linewidth=2,
        linestyle="--",
        label="Median absolute residual",
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("APMR Residual Distribution")

    plt.legend()
    plt.grid(True, alpha=0.3)

    _save("residual_histogram")


def plot_prediction(
    y_true,
    y_pred,
):
    """
    Prediction versus true response.
    """

    plt.figure(figsize=(5.5, 5.5))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6,
    )

    mn = min(
        np.min(y_true),
        np.min(y_pred),
    )

    mx = max(
        np.max(y_true),
        np.max(y_pred),
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
        linewidth=2,
        linestyle="--",
        label="Ideal prediction",
    )

    plt.xlabel("True Response")
    plt.ylabel("Predicted Response")
    plt.title("APMR Prediction Accuracy")

    plt.legend()
    plt.grid(True, alpha=0.3)

    _save("prediction")


# ============================================================
# METHOD COMPARISON
# ============================================================

METHOD_STYLES = {
    "APMR": {
        "marker": "o",
        "linestyle": "-",
    },
    "LAD": {
        "marker": "s",
        "linestyle": "--",
    },
    "LeastSquares": {
        "marker": "^",
        "linestyle": ":",
    },
    "Huber": {
        "marker": "D",
        "linestyle": "-.",
    },
    "TukeyLasso": {
        "marker": "P",
        "linestyle": "-",
    },
    "LMS": {
        "marker": "X",
        "linestyle": "--",
    },
    "LQS": {
        "marker": "X",
        "linestyle": "--",
    },
}


def _plot_comparison(
    results,
    metric,
    ylabel,
    title,
    filename,
    log_scale=False,
    y_min=None,
):
    """
    Plot mean performance versus contamination level,
    with standard-deviation error bars across random seeds.
    """

    plt.figure(figsize=(8, 5.5))

    methods = list(results["Method"].unique())

    for method in methods:

        subset = results[
            results["Method"] == method
        ]

        grouped = (
            subset
            .groupby("OutlierFraction")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        style = METHOD_STYLES.get(
            method,
            {
                "marker": "o",
                "linestyle": "-",
            },
        )

        plt.errorbar(
            grouped["OutlierFraction"],
            grouped["mean"],
            yerr=grouped["std"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6,
            capsize=3,
            alpha=0.9,
            label=method,
        )

    plt.xlabel("Outlier Fraction")
    plt.ylabel(ylabel)
    plt.title(title)

    if log_scale:
        plt.yscale("log")

    if y_min is not None:
        plt.ylim(bottom=y_min)

    plt.grid(True, alpha=0.3)
    plt.legend(
        ncol=2,
        fontsize=9,
    )

    _save(filename)


# ============================================================
# PERFORMANCE COMPARISONS
# ============================================================

def plot_mar(results):
    """
    Median absolute residual versus contamination.
    This is the primary objective of the proposed method.
    """

    _plot_comparison(
        results,
        metric="MedianResidual",
        ylabel="Median Absolute Residual",
        title="Robustness to Increasing Outlier Contamination",
        filename="mar_comparison",
        log_scale=False,
    )


def plot_mse(results):
    """
    MSE versus contamination.
    """

    _plot_comparison(
        results,
        metric="MSE",
        ylabel="Mean Squared Error",
        title="Prediction Error under Increasing Contamination",
        filename="mse_comparison",
        log_scale=False,
    )


def plot_mae(results):
    """
    MAE versus contamination.
    """

    _plot_comparison(
        results,
        metric="MAE",
        ylabel="Mean Absolute Error",
        title="Mean Absolute Prediction Error",
        filename="mae_comparison",
        log_scale=True,
    )


# ============================================================
# RELATIVE PERFORMANCE
# ============================================================

def plot_relative_mse(
    results,
    baseline="LeastSquares",
):
    """
    Relative MSE compared with a baseline method.

    Values below 1 indicate improvement over the baseline.
    """

    plt.figure(figsize=(8, 5.5))

    methods = [
        method
        for method in results["Method"].unique()
        if method != baseline
    ]

    baseline_data = (
        results[results["Method"] == baseline]
        .groupby("OutlierFraction")["MSE"]
        .mean()
    )

    for method in methods:

        method_data = (
            results[results["Method"] == method]
            .groupby("OutlierFraction")["MSE"]
            .mean()
        )

        common = (
            method_data.index
            .intersection(baseline_data.index)
        )

        ratio = (
            method_data.loc[common]
            / baseline_data.loc[common]
        )

        style = METHOD_STYLES.get(
            method,
            {
                "marker": "o",
                "linestyle": "-",
            },
        )

        plt.plot(
            common,
            ratio,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6,
            label=method,
        )

    plt.axhline(
        1.0,
        linewidth=1.5,
        linestyle="--",
        label=f"{baseline} baseline",
    )

    plt.xlabel("Outlier Fraction")
    plt.ylabel("MSE / Baseline MSE")

    plt.title(
        f"Relative Prediction Error Compared with {baseline}"
    )

    plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend(
        ncol=2,
        fontsize=9,
    )

    _save("relative_mse")


def plot_relative_mar(
    results,
    baseline="LAD",
):
    """
    Relative MAR compared with a baseline.

    Values below 1 indicate improvement over the baseline.
    """

    plt.figure(figsize=(8, 5.5))

    methods = [
        method
        for method in results["Method"].unique()
        if method != baseline
    ]

    baseline_data = (
        results[results["Method"] == baseline]
        .groupby("OutlierFraction")["MedianResidual"]
        .mean()
    )

    for method in methods:

        method_data = (
            results[results["Method"] == method]
            .groupby("OutlierFraction")["MedianResidual"]
            .mean()
        )

        common = (
            method_data.index
            .intersection(baseline_data.index)
        )

        ratio = (
            method_data.loc[common]
            / baseline_data.loc[common]
        )

        style = METHOD_STYLES.get(
            method,
            {
                "marker": "o",
                "linestyle": "-",
            },
        )

        plt.plot(
            common,
            ratio,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6,
            label=method,
        )

    plt.axhline(
        1.0,
        linewidth=1.5,
        linestyle="--",
        label=f"{baseline} baseline",
    )

    plt.xlabel("Outlier Fraction")
    plt.ylabel("Median Residual / Baseline")

    plt.title(
        f"Relative MAR Compared with {baseline}"
    )

    plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend(
        ncol=2,
        fontsize=9,
    )

    _save("relative_mar")


# ============================================================
# RANKING PLOT
# ============================================================

def plot_method_ranking(
    results,
    metric="MSE",
):
    """
    Average rank of each method over all contamination levels.

    Lower rank is better.
    """

    ranking = []

    for fraction in sorted(
        results["OutlierFraction"].unique()
    ):

        subset = results[
            results["OutlierFraction"] == fraction
        ]

        mean_values = (
            subset
            .groupby("Method")[metric]
            .mean()
            .sort_values()
        )

        ranks = mean_values.rank(
            method="average"
        )

        for method, rank in ranks.items():

            ranking.append(
                {
                    "Method": method,
                    "Rank": rank,
                }
            )

    ranking = pd.DataFrame(ranking)

    average_rank = (
        ranking
        .groupby("Method")["Rank"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        average_rank.index,
        average_rank.values,
    )

    plt.xlabel("Average Rank")
    plt.ylabel("Method")

    plt.title(
        f"Average Method Ranking Based on {metric}"
    )

    plt.gca().invert_yaxis()

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    _save(
        f"method_ranking_{metric.lower()}"
    )


# ============================================================
# COMPUTATIONAL PERFORMANCE
# ============================================================

def plot_iterations(results):
    """
    APMR and other iterative methods.

    Only methods with iteration information are displayed.
    """

    iterative = results[
        results["OuterIterations"].notna()
    ]

    if iterative.empty:
        return

    plt.figure(figsize=(8, 5))

    methods = (
        iterative["Method"]
        .unique()
    )

    for method in methods:

        subset = iterative[
            iterative["Method"] == method
        ]

        grouped = (
            subset
            .groupby("OutlierFraction")["OuterIterations"]
            .agg(["mean", "std"])
            .reset_index()
        )

        style = METHOD_STYLES.get(
            method,
            {
                "marker": "o",
                "linestyle": "-",
            },
        )

        plt.errorbar(
            grouped["OutlierFraction"],
            grouped["mean"],
            yerr=grouped["std"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6,
            capsize=3,
            label=method,
        )

    plt.xlabel("Outlier Fraction")
    plt.ylabel("Number of Iterations")

    plt.title(
        "Computational Iterations"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    _save("iterations")


# ============================================================
# COEFFICIENTS
# ============================================================

def plot_coefficients(model):
    """
    Estimated regression coefficients for APMR.
    """

    plt.figure(figsize=(9, 4.5))

    plt.stem(
        np.arange(len(model.coef_)),
        model.coef_,
    )

    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")

    plt.title(
        "APMR Estimated Regression Coefficients"
    )

    plt.grid(True, alpha=0.3)

    _save("coefficients")


# ============================================================
# OBJECTIVE
# ============================================================

def plot_objective(
    model,
    save_path="figures/objective.png",
):
    """
    Objective history for APMR.
    """

    plt.figure(figsize=(6.5, 4.5))

    plt.plot(
        model.history["objective"],
        linewidth=1.8,
    )

    plt.xlabel("Outer Iteration")
    plt.ylabel("Median Absolute Residual")

    plt.title(
        "APMR Objective History"
    )

    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_parameter_recovery(results, x_true):
    """
    Parameter Recovery Error (L2 norm) versus contamination.
    Shows how well each method recovers the clean-data ground truth (x_true).
    """
    if "_coef" not in results.columns:
        print("'_coef' column not found in results. Skipping recovery plot.")
        return

    # Create a copy to avoid SettingWithCopyWarning
    df_plot = results.copy()

    # Calculate the L2 parameter recovery error for every single experiment run
    df_plot["ParameterError"] = df_plot["_coef"].apply(
        lambda coef: np.linalg.norm(np.asarray(coef) - np.asarray(x_true))
    )

    _plot_comparison(
        df_plot,
        metric="ParameterError",
        ylabel=r"$\|x_{est} - x_{true}\|_2$",
        title="Parameter Recovery Under Severe Contamination",
        filename="parameter_recovery",
        log_scale=True, # Log scale is essential here since LS will explode
    )


def plot_coefficient_comparison(apmr_model, ls_model, x_true):
    """
    Side-by-side comparison of the estimated coefficients at high contamination.
    Shows how APMR matches the clean ground truth while LS diverges.
    """
    if apmr_model.coef_ is None or ls_model.coef_ is None or x_true is None:
        return
   
    # Slightly wider figure to accommodate the gaps nicely
    plt.figure(figsize=(12, 5))
    
    indices = np.arange(len(x_true))
    width = 0.2   # Made the columns thinner
    offset = 0.2  # Increased the offset to create a clean gap between bars
    
    # Plot side-by-side grouped bars shifted mathematically to prevent overlap
    plt.bar(indices- offset, apmr_model.coef_, width, label='APMR (Contaminated)', color='blue', alpha=0.8)
    plt.bar(indices , x_true, width, label='Clean Ground Truth ($x_{true}$)', color='green', alpha=0.7)
    plt.bar(indices + offset, ls_model.coef_, width, label='Least Squares (Contaminated)', color='red', alpha=0.6)

    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient Value")
    plt.yscale('symlog', linthresh=0.1)
    plt.title("Coefficient Recovery (APMR vs Least Squares)")
    
    # Anchor the x-ticks perfectly to the center of each group
    plt.xticks(indices)

    # Optional: Log scale if coefficients vary wildly
    # plt.yscale('symlog', linthresh=1.0) 

    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()

    _save("coefficient_comparison")

def plot_dual_candidate_objective(history):
    """
    APMR objective convergence highlighting the dual-candidate selection.

    A: Base APMR candidate chosen.
    E: Geometric Memory candidate chosen.
    """

    if "candidate_chosen" not in history:
        print("Dual-candidate tracking not found in history.")
        return

    iterations = np.asarray(history["iteration"])
    objectives = np.asarray(history["objective"])
    choices = np.asarray(history["candidate_chosen"])

    print(
        f"iterations={len(iterations)}, "
        f"objectives={len(objectives)}, "
        f"choices={len(choices)}"
    )

    # ---------------------------------------------------------
    # The choice at iteration k produces x^{k+1}.
    # Therefore choice[k] corresponds to objective[k+1].
    #
    # The final choice has no corresponding objective if the
    # algorithm terminated at that iteration.
    # ---------------------------------------------------------

    n_transitions = min(
        len(choices),
        len(iterations) - 1
    )

    plot_iters = iterations[1:n_transitions + 1]
    plot_objs = objectives[1:n_transitions + 1]
    plot_choices = choices[:n_transitions]

    # Separate indices according to selected candidate
    idx_A = np.where(plot_choices == "A")[0]
    idx_E = np.where(plot_choices == "E")[0]

    plt.figure(figsize=(8, 5.5))

    # Full objective trajectory
    plt.plot(
        iterations,
        objectives,
        color="black",
        linewidth=1.2,
        alpha=0.4,
        zorder=1
    )

    # Base APMR candidate
    if len(idx_A) > 0:
        plt.scatter(
            plot_iters[idx_A],
            plot_objs[idx_A],
            color="blue",
            marker="o",
            s=35,
            alpha=0.8,
            label="Base APMR Candidate ($x_A$)",
            zorder=2
        )

    # Geometric memory candidate
    if len(idx_E) > 0:
        plt.scatter(
            plot_iters[idx_E],
            plot_objs[idx_E],
            color="red",
            marker="o",
            s=35,
            alpha=0.8,
            label="Memory Candidate ($x_E$)",
            zorder=3
        )

    # Initialization
    plt.scatter(
        iterations[0],
        objectives[0],
        color="black",
        marker="s",
        s=45,
        label="Initialization",
        zorder=4
    )

    plt.xlabel("Outer Iteration")
    plt.ylabel("Exact MAR Objective")
    plt.title("Dual-Candidate Selection Trajectory")

    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()

    _save("dual_candidate_objective")


# ============================================================
# MASTER FUNCTION
# ============================================================

def generate_all_plots(
    apmr_model,
    A_test,
    b_test,
    results,
    x_true,
    ls_model,
):
    """
    Generate all figures used for the experiments.
    """

    prediction = apmr_model.predict(A_test)

    # ------------------------------------------
    # APMR optimization behavior
    # ------------------------------------------

    plot_convergence(
        apmr_model.history
    )

    plot_dual_candidate_objective(
        apmr_model.history
    )

    plot_step_norm(
        apmr_model.history
    )

    plot_gradient_norm(
        apmr_model.history
    )

    plot_active_set(
        apmr_model.history
    )

    # ------------------------------------------
    # APMR diagnostic behavior
    # ------------------------------------------

    plot_residual_histogram(
        apmr_model,
        A_test,
        b_test,
    )

    plot_prediction(
        b_test,
        prediction,
    )

    plot_coefficients(
        apmr_model
    )

    # ------------------------------------------
    # Method comparisons
    # ------------------------------------------

    plot_mar(results)

    plot_mse(results)

    plot_mae(results)

    # ------------------------------------------
    # Relative comparisons
    # ------------------------------------------

    plot_relative_mse(
        results,
        baseline="LeastSquares",
    )

    plot_relative_mar(
        results,
        baseline="LAD",
    )

    # ------------------------------------------
    # Overall rankings
    # ------------------------------------------

    plot_method_ranking(
        results,
        metric="MSE",
    )

    plot_method_ranking(
        results,
        metric="MedianResidual",
    )

    # ------------------------------------------
    # Computational comparison
    # ------------------------------------------

    plot_iterations(results)

    plot_coefficient_comparison(apmr_model, ls_model, x_true)

    plot_parameter_recovery(results,x_true)
