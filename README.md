# Surrogate-Based Proximal Framework for Median Absolute Residual Regression (APMR)

## Overview

This project implements the **Approximate Proximal Median Regression (APMR)** algorithm, a robust regression method designed to minimize the median absolute residual (MAR). APMR is particularly effective for handling data with significant outliers, making it a powerful alternative to least squares regression and other robust regression techniques.

The project includes a complete experimental framework that compares APMR against multiple state-of-the-art robust regression methods, including Least Absolute Deviations (LAD), Tukey-Lasso, Least Median of Squares (LMS), Least Quantile of Squares (LQS), RANSAC, and others.

## Key Features

- **Robust Median-based Regression**: Minimizes median absolute residual instead of mean squared error
- **Surrogate-Based Proximal Algorithm**: Efficient optimization using surrogate gradients and proximal methods
- **Comprehensive Benchmarking**: Comparison with 8+ regression methods
- **Detailed Diagnostics**: Convergence plots, active set visualization, and residual analysis
- **Scalability**: Handles datasets with hundreds of samples and features

## Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python package dependencies
├── apmr.py                      # Core APMR regression implementation
├── datasets.py                  # Dataset generation and utility functions
├── experiments.py               # Experiment pipeline and benchmarking
├── plots.py                     # Visualization utilities
├── test.py                      # Test/validation script
├── comparison/                  # Comparison method implementations
│   ├── __init__.py
│   ├── lms_lqs.py              # Least Median/Quantile of Squares
│   ├── mio.py                  # Mixed-Integer Optimization (SOS1) formulation
│   └── tukey_lasso.py          # Tukey-Lasso robust regression
├── figures/                     # Generated visualization outputs
│   ├── convergence.png
│   ├── step_norm.png
│   ├── gradient_norm.png
│   ├── active_set.png
│   ├── residual_histogram.png
│   ├── prediction.png
│   ├── coefficients.png
│   ├── mar_comparison.png
│   ├── mse_comparison.png
│   ├── mae_comparison.png
│   ├── relative_mse.png
│   ├── relative_mar.png
│   ├── method_ranking_mse.png
│   ├── method_ranking_medianresidual.png
│   └── iterations.png
└── Data Files (CSV)
    ├── A.csv                   # Design matrix
    ├── b.csv                   # Target response vector
    ├── x.csv                   # True coefficient vector
    ├── results.csv             # Detailed experimental results
    ├── summary.csv             # Aggregated results summary
    ├── apmr_trajectory.csv     # APMR solution trajectory
```

## Python Files Description

### Core Implementation

#### [apmr.py](apmr.py)
Implements the Approximate Proximal Median Regression (APMR) algorithm.

**Key Classes:**
- `APMRRegression`: Main regression class with the following key methods:
  - `fit(A, b)`: Fit the model to training data
  - `predict(A)`: Make predictions on new data
  - `objective(A, b, x)`: Compute the median absolute residual objective
  - `active_median_set(residuals)`: Identify active samples around the median residual
  - `surrogate_subgradient(A, b, x, M)`: Compute surrogate gradient using active set

**Key Parameters:**
- `alpha`: Dual step-size parameter (default: 0.05)
- `lambda_`: Regularization parameter (default: 0.05)
- `eta`: Primal step-size (default: 0.01)
- `max_iter`: Maximum outer iterations (default: 2000)
- `inner_iter`: Maximum inner iterations (default: 1000)
- `tol`: Outer convergence tolerance (default: 1e-4)
- `inner_tol`: Inner convergence tolerance (default: 1e-4)

**Output Attributes:**
- `coef_`: Estimated regression coefficients
- `history`: Dictionary tracking optimization progress with keys:
  - `objective`: Median absolute residual at each iteration
  - `step_norm`: Norm of coefficient updates
  - `gradient_norm`: Norm of computed gradients
  - `active_set_size`: Number of active samples in the active median set
  - `inner_iterations`: Iterations taken in inner loop
  - `inner_converged`: Whether inner loop converged

#### [datasets.py](datasets.py)
Provides dataset generation and evaluation utilities.

**Functions:**
- `make_regression(n_samples, n_features, noise, outlier_fraction, outlier_scale, random_state)`: Generate synthetic regression dataset with optional outliers
- `train_test_split(A, b, test_size, random_state)`: Split data into training and test sets
- `add_outliers(b, fraction, scale, random_state)`: Add outliers to response vector
- `mean_squared_error(y_true, y_pred)`: Compute MSE
- `mean_absolute_error(y_true, y_pred)`: Compute MAE
- `median_absolute_residual(y_true, y_pred)`: Compute median absolute residual (MAR) - primary metric for APMR

#### [experiments.py](experiments.py)
Orchestrates the complete experimental pipeline.

**Key Functions:**
- `evaluate_model(model, A_test, b_test)`: Compute performance metrics (MSE, MAE, MedianResidual) for any model
- `make_lqs_model_mio(quantile, time_limit, mip_gap, threads, verbose)`: Create Mixed-Integer Optimization LQS model
- `make_lms_model(random_state, n_candidates)`: Create approximate LMS model
- `make_lqs_model(random_state, quantile, n_candidates)`: Create approximate LQS model
- `run_single_experiment(outlier_fraction, random_state, verbose)`: Execute a single experiment with specified outlier contamination and random seed
- `benchmark()`: Run the complete benchmark across all outlier fractions and random seeds
- `main()`: Entry point that executes benchmark and generates all visualizations

**Compared Methods:**
1. APMR (Approximate Proximal Median Regression) - **Proposed method**
2. Least Squares (LS)
3. Least Absolute Deviations (LAD)
4. Huber Regressor
5. Tukey-Lasso
6. Least Median of Squares (LMS)
7. Least Quantile of Squares (LQS)
8. RANSAC
9. Mixed-Integer Optimization (MIO) LQS

**Experiment Design:**
- Outlier fractions tested: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
- Random seeds: 0, 1, 2, 3, 4
- Dataset: 300 samples, 50 features, Gaussian noise

#### [plots.py](plots.py)
Comprehensive visualization utilities for results.

**Diagnostic Plots (APMR Algorithm Behavior):**
- `plot_convergence(history)`: Median absolute residual convergence over iterations
- `plot_step_norm(history)`: L2 norm of coefficient updates
- `plot_gradient_norm(history)`: L2 norm of computed subgradients
- `plot_active_set(history)`: Evolution of active median set size
- `plot_residual_histogram(model, A, b)`: Distribution of absolute residuals
- `plot_prediction(y_true, y_pred)`: Predicted vs. true response scatter plot
- `plot_coefficients(model)`: Estimated regression coefficients

**Performance Comparison Plots:**
- `plot_mar(results)`: Median absolute residual vs. outlier fraction (primary metric)
- `plot_mse(results)`: Mean squared error vs. outlier fraction
- `plot_mae(results)`: Mean absolute error vs. outlier fraction

**Relative Performance Plots:**
- `plot_relative_mse(results, baseline)`: Method performance relative to baseline (Least Squares)
- `plot_relative_mar(results, baseline)`: MAR relative to baseline (LAD)

**Ranking and Computational Plots:**
- `plot_method_ranking(results, metric)`: Average rank of each method across all contamination levels
- `plot_iterations(results)`: Number of iterations for iterative methods

**Master Function:**
- `generate_all_plots(apmr_model, A_test, b_test, results)`: Generate all 18+ visualization figures

#### [test.py](test.py)
Test script for visualization of optimization trajectories.

**Features:**
- Load pre-computed regression data and APMR solution trajectory
- Implement least squares (LS) trajectory using gradient descent
- Implement least absolute deviations (LAD) trajectory using subgradient descent
- Visualize and compare different optimization paths

### Comparison Methods

#### [comparison/lms_lqs.py](comparison/lms_lqs.py)
**Class: `ApproximateLMSLQS`**

Practical approximate implementation of Least Median of Squares (LMS) and Least Quantile of Squares (LQS) regression.

**Key Methods:**
- `fit(A, b)`: Solve the LMS/LQS problem by evaluating random regression subsets and selecting the best candidate

**Key Parameters:**
- `quantile`: Quantile level (0.5 for LMS/median)
- `n_candidates`: Number of random subsets to evaluate
- `random_state`: Random seed for reproducibility

**Note:** Uses random subset selection for computational efficiency rather than exact global optimization

#### [comparison/mio.py](comparison/mio.py)
**Class: `SOS1LQSRegression`**

Mixed-Integer Optimization (MIO) formulation of Least Quantile of Squares using SOS1 (Special Ordered Set) constraints via Gurobi.

**Key Methods:**
- `fit(X, y)`: Solve LQS via mixed-integer programming

**Key Parameters:**
- `quantile`: Quantile level for the ordered residuals
- `time_limit`: Maximum solver time in seconds
- `mip_gap`: MIP optimality gap tolerance
- `threads`: Number of solver threads
- `fit_intercept`: Whether to include intercept term
- `verbose`: Solver verbosity

**Requires:** Gurobi optimizer (commercial software with academic licenses available)

#### [comparison/tukey_lasso.py](comparison/tukey_lasso.py)
**Class: `TukeyLasso`**

Robust regression combining Tukey's biweight loss with L1 lasso regularization.

**Objective:**
$$\frac{1}{n}\sum_{i=1}^{n}\rho_c(r_i(\beta)) + \alpha\|\beta\|_1$$

where $\rho_c$ is Tukey's biweight loss with tuning parameter $c$.

**Key Methods:**
- `fit(A, b)`: Solve via proximal gradient descent
- `_tukey_derivative(residual, c)`: Compute derivative of Tukey's biweight loss
- `_soft_threshold(x, threshold)`: Soft thresholding operator for proximal gradient step

**Key Parameters:**
- `alpha`: L1 regularization strength (default: 0.01)
- `c`: Tukey's biweight tuning parameter (default: 4.685)
- `max_iter`: Maximum iterations (default: 5000)
- `tol`: Convergence tolerance (default: 1e-6)
- `fit_intercept`: Include intercept term

## Generated Figures

All figures are saved in the `figures/` directory at 300 DPI resolution.

### Figure Descriptions

| Figure | Description |
|--------|-------------|
| **convergence.png** | APMR median absolute residual (log scale) vs. outer iteration |
| **step_norm.png** | L2 norm of coefficient updates (\\|x^{k+1}-x^k\\|_2) vs. iteration |
| **gradient_norm.png** | L2 norm of computed subgradients \\|g_k\\|_2 vs. iteration |
| **active_set.png** | Size of active median set (\|\mathcal{I}_k\|) vs. iteration |
| **residual_histogram.png** | Histogram of absolute residuals with median marked |
| **prediction.png** | Predicted vs. true response values with ideal prediction line |
| **coefficients.png** | Stem plot of estimated regression coefficients |
| **mar_comparison.png** | Median absolute residual comparison across methods vs. outlier fraction (main result) |
| **mse_comparison.png** | Mean squared error comparison across methods vs. outlier fraction |
| **mae_comparison.png** | Mean absolute error comparison (log scale) vs. outlier fraction |
| **relative_mse.png** | MSE relative to Least Squares baseline (log scale) vs. outlier fraction |
| **relative_mar.png** | MAR relative to LAD baseline (log scale) vs. outlier fraction |
| **method_ranking_mse.png** | Average ranking of methods by MSE across all contamination levels |
| **method_ranking_medianresidual.png** | Average ranking of methods by median absolute residual |
| **iterations.png** | Number of iterations required by iterative methods vs. outlier fraction |

## Data Files (CSV)

### Input/Output Data

| File | Description | Shape |
|------|-------------|-------|
| **A.csv** | Design matrix (feature matrix) | (n_samples, n_features) = (300, 50) |
| **b.csv** | Target response vector | (n_samples,) = (300,) |
| **x.csv** | True regression coefficients | (n_features,) = (50,) |
| **apmr_trajectory.csv** | APMR coefficient solution path | (max_iterations, n_features) |

### Results Data

| File | Description |
|------|-------------|
| **results.csv** | Detailed results for every experiment (method × outlier_fraction × random_seed), with columns: Method, OutlierFraction, RandomState, MSE, MAE, MedianResidual, OuterIterations, FinalObjective, Converged, AverageActiveSet, AverageInnerIterations |
| **summary.csv** | Aggregated results grouped by (OutlierFraction, Method) with mean values across random seeds |

## Installation

### Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Gurobi (optional, only required for the MIO comparison method)

### Setup

1. **Clone or download the project**
   ```bash
   cd /path/to/APMR
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Install Gurobi for MIO method**
   - Download from: https://www.gurobi.com
   - Academic licenses are available for free
   - Installation instructions: https://www.gurobi.com/documentation/current/quickstart.html

## Usage

### Running the Complete Benchmark

Execute the full experimental pipeline:

```bash
python experiments.py
```

This will:
1. Generate synthetic regression datasets with varying levels of outlier contamination
2. Train APMR and 8 competing methods on each dataset
3. Evaluate performance using MSE, MAE, and median absolute residual
4. Generate `results.csv` and `summary.csv` with detailed metrics
5. Create 18+ publication-quality visualizations in the `figures/` directory

**Execution time:** Approximately 5-10 minutes depending on hardware and Gurobi availability

### Using APMR for Custom Data

```python
from apmr import APMRRegression
import numpy as np

# Load your data
A = np.loadtxt("your_data.csv")  # n_samples × n_features
b = np.loadtxt("your_targets.csv")  # n_samples

# Create and fit model
model = APMRRegression(
    alpha=0.1,
    lambda_=1,
    eta=0.2,
    max_iter=10000,
    inner_iter=2000,
    tol=2.5e-3,
    inner_tol=1e-3,
)
model.fit(A, b)

# Make predictions
predictions = model.predict(A)

# Access results
coefficients = model.coef_
history = model.history  # Convergence details
```

### Testing Optimization Trajectories

Visualize different optimization paths on a dataset:

```bash
python test.py
```

This script compares:
- APMR (proposed proximal method)
- Least Squares trajectory
- Least Absolute Deviations trajectory

## Experimental Results Summary

The experiments demonstrate that **APMR**:

1. **Robustness**: Effectively minimizes median absolute residual across all contamination levels (0%-50% outliers)
2. **Competitive Performance**: Comparable or superior MSE and MAE compared to LAD, Huber, and Tukey-Lasso
3. **Efficiency**: Requires significantly fewer iterations than LMS/LQS random-subset methods and often faster than MIO solvers
4. **Scalability**: Handles 300 samples × 50 features efficiently
5. **Stability**: Consistent convergence across different random seeds and outlier fractions

## Algorithm Details

### Objective Function

APMR minimizes the **median absolute residual**:

$$f(x) = \text{median}_{i=1}^{n} |r_i(x)|$$

where $r_i(x) = a_i^\top x - b_i$ is the residual for sample $i$.

### Key Innovation: Active Median Set

At each iteration, APMR:
1. Identifies the **active median set** $\mathcal{I}_k$ = all samples with residuals near the median
2. Computes a **surrogate subgradient** using only samples in the active set
3. Updates coefficients using proximal gradient descent

This reduces computational cost compared to methods that consider all samples.

### Convergence Guarantee

Under mild conditions, APMR converges to a median-based solution with convergence rate $O(1/k^{0.75})$ in terms of objective value.

## License

This project is provided for research and educational purposes.

## Citation

If you use this work in your research, please cite:

```
@software{proximal_apmr_2024,
  title={Surrogate-Based Proximal Framework for Median Absolute Residual Regression},
  author={Author Name},
  year={2024},
  url={https://path/to/repo}
}
```

## References

- Robust regression methods: [Huber, 1964], [Rousseeuw, 1984]
- Proximal methods: [Boyd et al., 2010]
- Least median of squares: [Rousseeuw, 1984]
- Least quantile of squares: [Rousseeuw & Croux, 1992]
- Tukey biweight loss: [Tukey, 1977]

## Contact & Support

For questions, issues, or suggestions, please contact the project maintainers.

---

**Last Updated:** 2026
**Python Version:** 3.8+
**Status:** Active research project
