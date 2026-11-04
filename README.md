# A Surrogate-Based Proximal Framework for Median Absolute Residual Regression

This repository implements the **Memory-Augmented Approximate Proximal Median of Residuals (APMR)** framework, a robust linear regression estimator that directly minimizes the exact lower-median of absolute residuals (MAR). Because the MAR objective depends strictly on the $50\text{th}$ order statistic ($h = \lceil n/2 \rceil$), APMR fits the clean majority of observations and achieves the maximum $50\%$ breakdown point against extreme, heavy-tailed, and asymmetric outliers.

To overcome the nonconvex, nonsmooth, and piecewise-affine nature of the MAR objective, APMR employs a **dual-candidate iterative optimization architecture** guaranteed to converge under the Kurdyka–Łojasiewicz (KL) framework:
1. **Base Proximal Candidate ($x_A^{k+1}$):** Constructed by freezing the active median set $\mathcal{M}_k$ inside a dynamically scaled iteration-local trust-region ball $\mathcal{B}_k = \overline{B}(x^k, R_k)$ determined by the distance to the nearest switching hyperplane in $\mathcal{H}_k$, and solving the constrained surrogate proximal subproblem in $\mathcal{O}(T_k \vert{}\mathcal{M}_k\vert{} d)$ operations.
2. **Geometric Memory Candidate ($x_E^{k+1}$):** Constructed by accumulating unexploited directional movement $c_k = \max\{0, \rho_k(u^k) - \Vert{}s^k\Vert{}_2\}$ up to the true switching boundaries in $\mathcal{H}_k$ into a recursive memory vector $E^k$.
3. **Stationarity-Safeguarded Selection Rule:** Both candidates compete on the exact global MAR objective via the squared-displacement efficiency score $Q(z) = (f(x^k) - f(z)) / \Vert{}z - x^k\Vert{}_2^2 > 0$ (enforcing KL Sufficient Decrease **H1**) coupled with an explicit subgradient safeguard $\Vert{}g_E\Vert{}_2 \le b \Vert{}E^k\Vert{}_2$ (enforcing KL Stationarity Bound **H2**).

---

## Key Features

- **Exact Lower-Median Order-Statistic Optimization**: Minimizes $f(x) = r_{(h)}(x)$ with $h = \lceil n/2 \rceil$, assigning zero subgradient weight to extreme upper-tail residuals.
- **$\mathcal{O}(nd + T_k\vert{}\mathcal{M}_k\vert{}d)$ Per-Iteration Complexity**: Vectorized $\mathcal{O}(nd)$ trust-radius and ray-intersection geometry paired with an $\mathcal{O}(T_k\vert{}\mathcal{M}_k\vert{}d)$ active-set inner projected subgradient solver.
- **Provable KL Global Convergence**: Strictly enforces Conditions **H1–H3**, guaranteeing finite trajectory length ($\sum_{k=0}^\infty \Vert{}x^{k+1}-x^k\Vert{}_2 < \infty$) and finite-step convergence to a Clarke stationary point ($\mathbf{0} \in \partial f(x^\star)$).
- **Comprehensive Benchmarking Pipeline**: Evaluates APMR on standardized California Housing ($n = 20,640$, $d = 12$) and synthetic regression benchmarks under asymmetric heavy-tailed Student-$t$ contamination against 7+ competing estimators.
- **Publication-Grade Visualization Suite**: Colorblind-safe (Okabe–Ito) multi-panel plots for statistical robustness, parameter recovery, dual-candidate selection trajectories, and KL convergence verification.

---

## Project Structure

```text
├── README.md                    # Project documentation
├── requirements.txt             # Python package dependencies
├── apmr.py                      # Core Memory-Augmented APMRRegression implementation
├── datasets.py                  # Synthetic data generation, splitting, and asymmetric outlier injection
├── experiments.py               # California Housing & synthetic benchmark pipeline
├── plots.py                     # Publication-grade visualization suite
├── test.py                      # 2D piecewise-affine landscape & trajectory comparison script
├── comparison/                  # Competing robust regression implementations
│   ├── __init__.py
│   ├── lms_lqs.py               # Approximate Least Median / Quantile of Squares (LMS/LQS)
│   ├── mio.py                   # Mixed-Integer Optimization (SOS1) LQS via Gurobi
│   └── tukey_lasso.py           # Nonconvex Tukey-Lasso via proximal gradient descent
├── data sets/
│   └── housing.csv              # California Housing benchmark dataset
├── figures/                     # Generated 300 DPI visualization outputs
│   ├── robustness_benchmark.png
│   ├── mar_comparison.png
│   ├── mse_comparison.png
│   ├── parameter_recovery.png
│   ├── coefficient_comparison.png
│   ├── dual_candidate_objective.png
│   ├── convergence.png
│   ├── kl_convergence_verification.png
│   ├── step_norm.png
│   ├── gradient_norm.png
│   ├── active_set.png
│   ├── iterations.png
│   └── trajectory.png
└── Data & Results Files (CSV)
    ├── A.csv                    # Preprocessed feature matrix
    ├── b.csv                    # Preprocessed target response vector
    ├── x.csv                    # Clean reference coefficient vector (x_true)
    ├── apmr_trajectory.csv      # Outer-iteration coefficient trajectory of APMR
    ├── results.csv              # Full run-by-run metrics and estimated coefficients (TSV)
    └── summary.csv              # Aggregated mean metrics grouped by (OutlierFraction, Method)