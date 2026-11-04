"""
plots.py
Publication-grade visualization suite for the Memory-Augmented Surrogate-Based
Proximal Framework for Median Absolute Residual (APMR) Regression.

Generates the 6 core methodological figures for the manuscript as well as
individual standalone figures for backward compatibility with existing scripts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

# Consistent, colorblind-safe Okabe-Ito publication palette and marker styles
METHOD_STYLES = {
    "APMR": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.2,
        "zorder": 5,
    },
    "LAD": {
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder": 4,
    },
    "Huber": {
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-.",
        "linewidth": 1.8,
        "zorder": 4,
    },
    "LeastSquares": {
        "color": "#D55E00",
        "marker": "^",
        "linestyle": ":",
        "linewidth": 1.8,
        "zorder": 3,
    },
    "LS": {
        "color": "#D55E00",
        "marker": "^",
        "linestyle": ":",
        "linewidth": 1.8,
        "zorder": 3,
    },
    "TukeyLasso": {
        "color": "#CC79A7",
        "marker": "P",
        "linestyle": "-",
        "linewidth": 1.7,
        "zorder": 3,
    },
    "LMS": {
        "color": "#8C564B",
        "marker": "X",
        "linestyle": "--",
        "linewidth": 1.7,
        "zorder": 3,
    },
    "LQS": {
        "color": "#8C564B",
        "marker": "X",
        "linestyle": "--",
        "linewidth": 1.7,
        "zorder": 3,
    },
    "LMS/LQS": {
        "color": "#8C564B",
        "marker": "X",
        "linestyle": "--",
        "linewidth": 1.7,
        "zorder": 3,
    },
    "RANSAC": {
        "color": "#56B4E9",
        "marker": "v",
        "linestyle": "-.",
        "linewidth": 1.7,
        "zorder": 3,
    },
}


def _get_style(method_name: str) -> Dict:
    return METHOD_STYLES.get(
        method_name,
        {
            "color": "#333333",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 1.8,
            "zorder": 2,
        },
    )


def _save(name: str):
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# 1. STATISTICAL ROBUSTNESS & PARAMETER RECOVERY (FIGURES 1 & 2)
# ============================================================

def _draw_metric_panel(
    ax: plt.Axes,
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    log_scale: bool = True,
    show_bands: bool = True,
):
    """
    Draw a single contamination comparison curve panel with shaded dispersion
    bands that remain strictly positive on logarithmic axes.
    """
    methods = list(results["Method"].unique())

    for method in methods:
        subset = results[results["Method"] == method]
        grouped = (
            subset.groupby("OutlierFraction")[metric]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )

        style = _get_style(method)
        x_vals = grouped["OutlierFraction"].to_numpy(dtype=float)
        mean_vals = grouped["mean"].to_numpy(dtype=float)
        std_vals = np.nan_to_num(grouped["std"].to_numpy(dtype=float), nan=0.0)

        ax.plot(
            x_vals,
            mean_vals,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=6,
            alpha=0.95,
            label=method,
            zorder=style["zorder"],
        )

        if show_bands and np.any(std_vals > 0):
            if log_scale:
                # Clip lower band to positive values so log scale never fails
                lower = np.maximum(
                    grouped["min"].to_numpy(dtype=float) * 0.8,
                    mean_vals - 0.5 * std_vals,
                )
                lower = np.maximum(lower, 1e-6)
            else:
                lower = mean_vals - 0.5 * std_vals
            upper = mean_vals + 0.5 * std_vals
            ax.fill_between(
                x_vals,
                lower,
                upper,
                color=style["color"],
                alpha=0.12,
                zorder=style["zorder"] - 1,
            )

    ax.set_xlabel("Outlier Fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")


def _compute_parameter_error_df(
    results: pd.DataFrame,
    x_true: np.ndarray,
) -> Optional[pd.DataFrame]:
    if "_coef" not in results.columns or x_true is None:
        return None
    x_true_arr = np.asarray(x_true, dtype=float).ravel()
    df_plot = results.copy()

    def _err(coef):
        c_arr = np.asarray(coef, dtype=float).ravel()
        d_min = min(len(c_arr), len(x_true_arr))
        return float(np.linalg.norm(c_arr[:d_min] - x_true_arr[:d_min]))

    df_plot["ParameterError"] = df_plot["_coef"].apply(_err)
    return df_plot


def plot_robustness_benchmark(
    results: pd.DataFrame,
    x_true: Optional[np.ndarray] = None,
):
    """
    FIGURE 1 (1x3 multi-panel):
    (a) Test MAR vs. Outlier Fraction
    (b) Test MSE vs. Outlier Fraction
    (c) Parameter Recovery Error ||x_est - x_true||_2 vs. Outlier Fraction
    """
    df_param = _compute_parameter_error_df(results, x_true)
    n_cols = 3 if df_param is not None else 2

    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.6))
    if n_cols == 1:
        axes = [axes]

    _draw_metric_panel(
        axes[0],
        results,
        metric="MedianResidual",
        ylabel="Test MAR",
        title="(a) Median Absolute Residual (MAR)",
        log_scale=True,
    )

    _draw_metric_panel(
        axes[1],
        results,
        metric="MSE",
        ylabel="Test MSE",
        title="(b) Mean Squared Error (MSE)",
        log_scale=True,
    )

    if df_param is not None:
        _draw_metric_panel(
            axes[2],
            df_param,
            metric="ParameterError",
            ylabel=r"$\|\hat{x} - x_{\mathrm{true}}\|_2$",
            title="(c) Parameter Recovery Error",
            log_scale=True,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=min(len(labels), 7),
        fontsize=9.5,
        frameon=True,
    )

    _save("robustness_benchmark")


def plot_mar(results: pd.DataFrame):
    """Standalone Test MAR vs. Contamination plot (log-scaled)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _draw_metric_panel(
        ax,
        results,
        metric="MedianResidual",
        ylabel="Median Absolute Residual (MAR)",
        title="Robustness to Increasing Outlier Contamination (MAR)",
        log_scale=True,
    )
    ax.legend(ncol=2, fontsize=9)
    _save("mar_comparison")


def plot_mse(results: pd.DataFrame):
    """Standalone Test MSE vs. Contamination plot (log-scaled)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _draw_metric_panel(
        ax,
        results,
        metric="MSE",
        ylabel="Mean Squared Error (MSE)",
        title="Prediction Error under Increasing Contamination (MSE)",
        log_scale=True,
    )
    ax.legend(ncol=2, fontsize=9)
    _save("mse_comparison")


def plot_parameter_recovery(results: pd.DataFrame, x_true: np.ndarray):
    """Standalone Parameter Recovery Error ||x_est - x_true||_2 plot."""
    df_param = _compute_parameter_error_df(results, x_true)
    if df_param is None:
        print("'_coef' column or x_true not found. Skipping parameter_recovery plot.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _draw_metric_panel(
        ax,
        df_param,
        metric="ParameterError",
        ylabel=r"$\|\hat{x} - x_{\mathrm{true}}\|_2$",
        title="Parameter Recovery Error under Severe Contamination",
        log_scale=True,
    )
    ax.legend(ncol=2, fontsize=9)
    _save("parameter_recovery")


def plot_coefficient_comparison(
    apmr_model,
    ls_model,
    x_true: np.ndarray,
    lad_model=None,
):
    """
    FIGURE 2:
    Side-by-side grouped bar chart comparing Clean Ground Truth (x_true),
    APMR, LAD (optional), and Least Squares at severe contamination.
    """
    if apmr_model is None or ls_model is None or x_true is None:
        return
    if getattr(apmr_model, "coef_", None) is None or getattr(ls_model, "coef_", None) is None:
        return

    x_true_arr = np.asarray(x_true, dtype=float).ravel()
    d = len(x_true_arr)
    apmr_c = np.asarray(apmr_model.coef_, dtype=float).ravel()[:d]
    ls_c = np.asarray(ls_model.coef_, dtype=float).ravel()[:d]

    plt.figure(figsize=(11.5, 4.8))
    indices = np.arange(d)

    if lad_model is not None and getattr(lad_model, "coef_", None) is not None:
        lad_c = np.asarray(lad_model.coef_, dtype=float).ravel()[:d]
        width = 0.19
        plt.bar(
            indices - 1.5 * width,
            x_true_arr,
            width,
            label=r"Ground Truth ($x_{\mathrm{true}}$)",
            color="#009E73",
            alpha=0.85,
        )
        plt.bar(
            indices - 0.5 * width,
            apmr_c,
            width,
            label="APMR (Contaminated)",
            color="#0072B2",
            alpha=0.9,
        )
        plt.bar(
            indices + 0.5 * width,
            lad_c,
            width,
            label="LAD (Contaminated)",
            color="#E69F00",
            alpha=0.8,
        )
        plt.bar(
            indices + 1.5 * width,
            ls_c,
            width,
            label="Least Squares (Contaminated)",
            color="#D55E00",
            alpha=0.75,
        )
    else:
        width = 0.24
        plt.bar(
            indices - width,
            x_true_arr,
            width,
            label=r"Ground Truth ($x_{\mathrm{true}}$)",
            color="#009E73",
            alpha=0.85,
        )
        plt.bar(
            indices,
            apmr_c,
            width,
            label="APMR (Contaminated)",
            color="#0072B2",
            alpha=0.9,
        )
        plt.bar(
            indices + width,
            ls_c,
            width,
            label="Least Squares (Contaminated)",
            color="#D55E00",
            alpha=0.75,
        )

    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient Value (symlog scale)")
    plt.yscale("symlog", linthresh=0.1)
    plt.title("Component-Wise Parameter Recovery under Severe Contamination")
    plt.xticks(indices)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend(frameon=True)

    _save("coefficient_comparison")


# ============================================================
# 2. DUAL-CANDIDATE & GEOMETRIC MEMORY DYNAMICS (FIGURES 3 & 4)
# ============================================================

def _draw_dual_candidate_panel(ax: plt.Axes, history: Dict):
    iterations = np.asarray(history["iteration"])
    objectives = np.asarray(history["objective"])
    choices = np.asarray(history.get("candidate_chosen", ["A"] * len(iterations)))

    n_transitions = min(len(choices), len(iterations) - 1)
    plot_iters = iterations[1 : n_transitions + 1]
    plot_objs = objectives[1 : n_transitions + 1]
    plot_choices = choices[:n_transitions]

    idx_A = np.where(plot_choices == "A")[0]
    idx_E = np.where(plot_choices == "E")[0]
    pct_E = (100.0 * len(idx_E) / max(1, n_transitions))

    ax.plot(
        iterations,
        objectives,
        color="#222222",
        linewidth=1.4,
        alpha=0.45,
        zorder=1,
    )

    if len(idx_A) > 0:
        ax.scatter(
            plot_iters[idx_A],
            plot_objs[idx_A],
            color="#0072B2",
            marker="o",
            s=30,
            alpha=0.8,
            label=f"Base Proximal Candidate $x_A$ ({len(idx_A)})",
            zorder=2,
        )

    if len(idx_E) > 0:
        ax.scatter(
            plot_iters[idx_E],
            plot_objs[idx_E],
            color="#D55E00",
            marker="^",
            s=42,
            alpha=0.95,
            label=f"Geometric Memory Candidate $x_E$ ({len(idx_E)}, {pct_E:.1f}%)",
            zorder=3,
        )

    ax.scatter(
        iterations[0],
        objectives[0],
        color="black",
        marker="s",
        s=45,
        label="Initialization $x^0$",
        zorder=4,
    )

    ax.set_xlabel("Outer Iteration $k$")
    ax.set_ylabel("Exact MAR Objective $f(x^k)$")
    ax.set_title("Dual-Candidate Selection Trajectory")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8.5, frameon=True)


def plot_dual_candidate_objective(history: Dict):
    """
    Standalone plot of exact MAR objective convergence with Candidate A vs. E markers.
    Also saves to 'convergence.png' for backward compatibility with existing TeX files.
    """
    if "candidate_chosen" not in history:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _draw_dual_candidate_panel(ax, history)
    _save("dual_candidate_objective")

    # Also save as convergence.png so existing \includegraphics{figures/convergence.png} works
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _draw_dual_candidate_panel(ax, history)
    _save("convergence")


def plot_memory_ablation(
    history_full: Dict,
    history_base_only: Optional[Dict] = None,
    history_unconstrained: Optional[Dict] = None,
):
    """
    FIGURE 3 (1x2 multi-panel when ablation histories are provided, or standalone):
    Compares Full Dual-Candidate APMR vs. Base Trust-Region APMR Only (no memory)
    vs. Unconstrained Subgradient/Momentum.
    """
    if history_base_only is None and history_unconstrained is None:
        # Still generate the dual-candidate objective panel cleanly
        plot_dual_candidate_objective(history_full)
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    # Left panel: Dual-candidate selection on full APMR
    _draw_dual_candidate_panel(axes[0], history_full)
    axes[0].set_title("(a) Dual-Candidate Selection ($x_A$ vs. $x_E$)")

    # Right panel: Ablation study curves
    axes[1].plot(
        np.asarray(history_full["iteration"]) + 1,
        np.asarray(history_full["objective"]),
        color="#0072B2",
        linewidth=2.2,
        label="Full Dual-Candidate APMR ($x_A$ + Safeguarded $x_E$)",
    )

    if history_base_only is not None:
        axes[1].plot(
            np.asarray(history_base_only["iteration"]) + 1,
            np.asarray(history_base_only["objective"]),
            color="#009E73",
            linestyle="--",
            linewidth=1.9,
            label=r"Base Trust-Region APMR Only ($x_A$ only, $E^k \equiv 0$)",
        )

    if history_unconstrained is not None:
        axes[1].plot(
            np.asarray(history_unconstrained["iteration"]) + 1,
            np.asarray(history_unconstrained["objective"]),
            color="#D55E00",
            linestyle=":",
            linewidth=1.6,
            alpha=0.85,
            label=r"Unconstrained Subgradient (No Trust Ball $\mathcal{B}_k$)",
        )

    axes[1].set_xlabel("Outer Iteration $k$")
    axes[1].set_ylabel("Exact MAR Objective $f(x^k)$")
    axes[1].set_title("(b) Ablation of Memory & Trust-Region Components")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=8.5, frameon=True)

    _save("dual_candidate_dynamics")


def plot_geometric_memory_dynamics(history: Dict, a_bound: Optional[float] = None):
    """
    FIGURE 4 (1x2 multi-panel):
    (a) Available ray distance rho_k(u^k) vs. Trust radius R_k vs. Memory norm ||E^k||_2.
    (b) Squared-displacement efficiency scores Q(x_A^{k+1}) and Q(x_E^{k+1}) vs. KL H1 bound 'a'.
    """
    has_geom = "trust_radius" in history and "available_room" in history
    has_q = "q_A" in history and "q_E" in history

    if not has_geom and not has_q:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    iters = np.asarray(history["iteration"]) + 1

    # Panel (a): Geometric room vs. Trust radius
    if has_geom:
        r_k = np.asarray(history["trust_radius"], dtype=float)
        rho_k = np.asarray(history["available_room"], dtype=float)
        axes[0].semilogy(
            iters,
            np.maximum(rho_k, 1e-12),
            color="#009E73",
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label=r"Available Boundary Distance $\rho_k(u^k)$",
        )
        axes[0].semilogy(
            iters,
            np.maximum(r_k, 1e-12),
            color="#0072B2",
            linewidth=1.9,
            label=r"Trust-Region Radius $R_k = \eta \Delta_k$",
        )
        if "memory_norm" in history:
            e_norm = np.asarray(history["memory_norm"], dtype=float)
            axes[0].semilogy(
                iters,
                np.maximum(e_norm, 1e-12),
                color="#D55E00",
                linewidth=1.8,
                label=r"Accumulated Memory Norm $\|E^k\|_2$",
            )
        axes[0].set_xlabel("Outer Iteration $k$")
        axes[0].set_ylabel("Euclidean Distance (log scale)")
        axes[0].set_title(r"(a) Unexploited Geometric Room ($\rho_k(u^k)$ vs. $R_k$)")
        axes[0].grid(True, alpha=0.3, which="both")
        axes[0].legend(fontsize=8.5, frameon=True)

    # Panel (b): Q-score competition (H1 sufficient decrease ratio)
    if has_q:
        q_A = np.asarray(history["q_A"], dtype=float)
        q_E = np.asarray(history["q_E"], dtype=float)

        valid_A = np.isfinite(q_A) & (q_A > 0)
        valid_E = np.isfinite(q_E) & (q_E > 0)

        if np.any(valid_A):
            axes[1].scatter(
                iters[valid_A],
                q_A[valid_A],
                color="#0072B2",
                marker="o",
                s=22,
                alpha=0.7,
                label=r"Base Efficiency $Q(x_A^{k+1})$",
            )
        if np.any(valid_E):
            axes[1].scatter(
                iters[valid_E],
                q_E[valid_E],
                color="#D55E00",
                marker="^",
                s=30,
                alpha=0.85,
                label=r"Memory Efficiency $Q(x_E^{k+1})$",
            )
        if a_bound is not None and a_bound > 0:
            axes[1].axhline(
                a_bound,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=rf"KL Sufficient Decrease Bound $a = {a_bound:.2e}$",
            )

        axes[1].set_xlabel("Outer Iteration $k$")
        axes[1].set_ylabel(r"Efficiency Score $Q(z) = (f(x^k) - f(z)) / \|z - x^k\|_2^2$")
        axes[1].set_title("(b) Squared-Displacement Efficiency Competition (H1)")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3, which="both")
        axes[1].legend(fontsize=8.5, frameon=True)

    _save("geometric_room_and_qscore")


# ============================================================
# 3. KL CONVERGENCE & COMPLEXITY VERIFICATION (FIGURE 5)
# ============================================================

def plot_kl_convergence_verification(history: Dict):
    """
    FIGURE 5 (1x2 multi-panel):
    (a) Step Norm ||x^{k+1} - x^k||_2 (log scale) and Cumulative Trajectory Length
        sum_{j=0}^k ||x^{j+1} - x^j||_2 (proving finite trajectory length under KL).
    (b) Representative Clarke Subgradient Norm ||g_k||_2 and Active Median Set Size |M_k|.
    """
    step_norm = np.asarray(history["step_norm"], dtype=float)
    grad_norm = np.asarray(history["gradient_norm"], dtype=float)
    active_set = np.asarray(history["active_set_size"], dtype=float)
    iters = np.arange(1, len(step_norm) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # Panel (a): Step norm + Cumulative arc length
    ax1 = axes[0]
    line1 = ax1.semilogy(
        iters,
        np.maximum(step_norm, 1e-15),
        color="#0072B2",
        linewidth=1.8,
        label=r"Step Norm $\|x^{k+1} - x^k\|_2$",
    )
    ax1.set_xlabel("Outer Iteration $k$")
    ax1.set_ylabel(r"Step Displacement $\|x^{k+1} - x^k\|_2$", color="#0072B2")
    ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax1.grid(True, alpha=0.3, which="both")

    ax1_twin = ax1.twinx()
    cum_length = np.cumsum(step_norm)
    line2 = ax1_twin.plot(
        iters,
        cum_length,
        color="#D55E00",
        linestyle="--",
        linewidth=2.0,
        label=r"Cumulative Length $\sum_{j=0}^k \|x^{j+1} - x^j\|_2$",
    )
    ax1_twin.set_ylabel("Cumulative Trajectory Length", color="#D55E00")
    ax1_twin.tick_params(axis="y", labelcolor="#D55E00")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=8.5, frameon=True)
    ax1.set_title("(a) Finite-Length Trajectory Verification (KL Property)")

    # Panel (b): Subgradient norm ||g_k||_2 and Active Set Size |M_k|
    ax2 = axes[1]
    line3 = ax2.semilogy(
        iters,
        np.maximum(grad_norm, 1e-15),
        color="#009E73",
        linewidth=1.8,
        label=r"Representative Subgradient Norm $\|g_k\|_2$",
    )
    ax2.set_xlabel("Outer Iteration $k$")
    ax2.set_ylabel(r"Clarke Subgradient Norm $\|g_k\|_2$", color="#009E73")
    ax2.tick_params(axis="y", labelcolor="#009E73")
    ax2.grid(True, alpha=0.3, which="both")

    ax2_twin = ax2.twinx()
    line4 = ax2_twin.plot(
        iters,
        active_set,
        color="#CC79A7",
        linestyle=":",
        linewidth=1.8,
        label=r"Active Median Set Size $|\mathcal{M}_k|$",
    )
    ax2_twin.set_ylabel(r"Active Set Cardinality $|\mathcal{M}_k|$", color="#CC79A7")
    ax2_twin.tick_params(axis="y", labelcolor="#CC79A7")

    lines_b = line3 + line4
    labels_b = [l.get_label() for l in lines_b]
    ax2.legend(lines_b, labels_b, loc="upper right", fontsize=8.5, frameon=True)
    ax2.set_title(r"(b) Subgradient Norm $\|g_k\|_2$ & Active Set $|\mathcal{M}_k|$")

    _save("kl_convergence_verification")


def plot_step_norm(history: Dict):
    """Standalone Step Norm plot (retained for backward compatibility)."""
    step_norm = np.asarray(history["step_norm"], dtype=float)
    plt.figure(figsize=(6.5, 4.5))
    plt.semilogy(
        np.arange(1, len(step_norm) + 1),
        np.maximum(step_norm, 1e-15),
        color="#0072B2",
        linewidth=1.8,
    )
    plt.xlabel("Outer Iteration $k$")
    plt.ylabel(r"$\|x^{k+1}-x^k\|_2$")
    plt.title("APMR Step Displacement Norm")
    plt.grid(True, alpha=0.3, which="both")
    _save("step_norm")


def plot_gradient_norm(history: Dict):
    """Standalone Subgradient Norm plot (retained for backward compatibility)."""
    gradient_norm = np.asarray(history["gradient_norm"], dtype=float)
    plt.figure(figsize=(6.5, 4.5))
    plt.semilogy(
        np.arange(1, len(gradient_norm) + 1),
        np.maximum(gradient_norm, 1e-15),
        color="#009E73",
        linewidth=1.8,
    )
    plt.xlabel("Outer Iteration $k$")
    plt.ylabel(r"$\|g_k\|_2$")
    plt.title("APMR Representative Subgradient Norm")
    plt.grid(True, alpha=0.3, which="both")
    _save("gradient_norm")


def plot_active_set(history: Dict):
    """Standalone Active Median Set Size |M_k| plot."""
    active_set = np.asarray(history["active_set_size"], dtype=float)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(
        np.arange(1, len(active_set) + 1),
        active_set,
        color="#CC79A7",
        linewidth=1.8,
    )
    plt.xlabel("Outer Iteration $k$")
    plt.ylabel(r"$|\mathcal{M}_k|$")
    plt.title("Active Median Set Cardinality")
    plt.grid(True, alpha=0.3)
    _save("active_set")


def plot_iterations(results: pd.DataFrame):
    """Standalone Outer Iterations vs. Outlier Fraction plot."""
    if "OuterIterations" not in results.columns:
        return
    iterative = results[results["OuterIterations"].notna()]
    if iterative.empty:
        return

    plt.figure(figsize=(7.5, 4.8))
    for method in iterative["Method"].unique():
        subset = iterative[iterative["Method"] == method]
        grouped = (
            subset.groupby("OutlierFraction")["OuterIterations"]
            .agg(["mean", "std"])
            .reset_index()
        )
        style = _get_style(method)
        plt.errorbar(
            grouped["OutlierFraction"],
            grouped["mean"],
            yerr=grouped["std"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=6,
            capsize=3,
            label=method,
        )

    plt.xlabel("Outlier Fraction")
    plt.ylabel("Number of Outer Iterations")
    plt.title("Computational Effort Across Contamination Levels")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    _save("iterations")


# ============================================================
# 4. 2D PIECEWISE-AFFINE LANDSCAPE & TRAJECTORY (FIGURE 6)
# ============================================================

def plot_2d_trajectory(
    A_2d: np.ndarray,
    b_2d: np.ndarray,
    apmr_traj: np.ndarray,
    ls_traj: Optional[np.ndarray] = None,
    lad_traj: Optional[np.ndarray] = None,
    x_true_2d: Optional[np.ndarray] = None,
    intercept: float = 0.0,
    grid_res: int = 180,
):
    """
    FIGURE 6:
    Plots the exact nonconvex piecewise-affine lower-median contour landscape
    for a 2D regression problem and overlays the optimization trajectories of
    APMR, LAD, and LS from a shared initialization x^0.
    """
    A_arr = np.asarray(A_2d, dtype=float)
    b_arr = np.asarray(b_2d, dtype=float).ravel()
    apmr_path = np.asarray(apmr_traj, dtype=float)

    all_pts = [apmr_path]
    if ls_traj is not None:
        all_pts.append(np.asarray(ls_traj, dtype=float))
    if lad_traj is not None:
        all_pts.append(np.asarray(lad_traj, dtype=float))
    if x_true_2d is not None:
        all_pts.append(np.asarray(x_true_2d, dtype=float).reshape(1, 2))

    stacked = np.vstack(all_pts)
    x_min, y_min = np.min(stacked, axis=0) - 1.0
    x_max, y_max = np.max(stacked, axis=0) + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid_pts = np.c_[xx.ravel(), yy.ravel()]

    # Evaluate exact lower-median MAR objective across the 2D grid
    res_grid = np.abs(A_arr[:, :2] @ grid_pts.T + intercept - b_arr[:, None])
    h = int(np.ceil(len(b_arr) / 2.0)) - 1
    zz = np.partition(res_grid, h, axis=0)[h].reshape(xx.shape)

    plt.figure(figsize=(7.5, 6.0))
    cf = plt.contourf(xx, yy, zz, levels=35, cmap="viridis", alpha=0.82)
    plt.contour(xx, yy, zz, levels=35, colors="black", linewidths=0.35, alpha=0.4)
    cbar = plt.colorbar(cf)
    cbar.set_label("Exact MAR Objective $f(x)$")

    if ls_traj is not None:
        ls_path = np.asarray(ls_traj, dtype=float)
        plt.plot(
            ls_path[:, 0],
            ls_path[:, 1],
            color="#D55E00",
            linestyle=":",
            marker="^",
            markersize=4,
            linewidth=1.8,
            label="Least Squares (LS)",
        )

    if lad_traj is not None:
        lad_path = np.asarray(lad_traj, dtype=float)
        plt.plot(
            lad_path[:, 0],
            lad_path[:, 1],
            color="#E69F00",
            linestyle="--",
            marker="s",
            markersize=4,
            linewidth=1.8,
            label="Least Absolute Deviations (LAD)",
        )

    plt.plot(
        apmr_path[:, 0],
        apmr_path[:, 1],
        color="#0072B2",
        linestyle="-",
        marker="o",
        markersize=4,
        linewidth=2.2,
        label="Proposed APMR Trajectory",
    )

    plt.scatter(
        apmr_path[0, 0],
        apmr_path[0, 1],
        color="white",
        edgecolor="black",
        marker="s",
        s=75,
        zorder=6,
        label="Initial Point $x^0$",
    )

    if x_true_2d is not None:
        plt.scatter(
            x_true_2d[0],
            x_true_2d[1],
            color="#009E73",
            edgecolor="black",
            marker="*",
            s=160,
            zorder=7,
            label=r"Clean Ground Truth $x_{\mathrm{true}}$",
        )

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Optimization Trajectories on the Nonconvex MAR Landscape")
    plt.legend(loc="best", fontsize=9, frameon=True)
    _save("trajectory")


# ============================================================
# MASTER ENTRY POINT
# ============================================================

def generate_all_plots(
    apmr_model,
    A_test,
    b_test,
    results: pd.DataFrame,
    x_true: Optional[np.ndarray] = None,
    ls_model=None,
    lad_model=None,
    history_base_only: Optional[Dict] = None,
    history_unconstrained: Optional[Dict] = None,
    a_bound: Optional[float] = None,
):
    """
    Master function generating the complete 6-figure methodological suite
    plus standalone figures for full backward compatibility.
    """
    # 1. Multi-panel & standalone statistical robustness benchmarks (Figure 1)
    plot_robustness_benchmark(results, x_true=x_true)
    plot_mar(results)
    plot_mse(results)
    if x_true is not None:
        plot_parameter_recovery(results, x_true)

    # 2. Component-wise coefficient recovery bar chart (Figure 2)
    if ls_model is not None and x_true is not None:
        plot_coefficient_comparison(
            apmr_model,
            ls_model,
            x_true,
            lad_model=lad_model,
        )

    # 3. Dual-candidate selection & ablation dynamics (Figure 3)
    if getattr(apmr_model, "history", None) is not None:
        plot_dual_candidate_objective(apmr_model.history)
        plot_memory_ablation(
            apmr_model.history,
            history_base_only=history_base_only,
            history_unconstrained=history_unconstrained,
        )

        # 4. Geometric room (rho_k vs R_k) and Q-score competition (Figure 4)
        plot_geometric_memory_dynamics(apmr_model.history, a_bound=a_bound)

        # 5. KL finite-length & subgradient bound verification (Figure 5)
        plot_kl_convergence_verification(apmr_model.history)
        plot_step_norm(apmr_model.history)
        plot_gradient_norm(apmr_model.history)
        plot_active_set(apmr_model.history)

    # 6. Computational iterations across outlier fractions
    plot_iterations(results)