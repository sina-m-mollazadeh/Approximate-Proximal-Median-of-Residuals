"""
experiments.py
Numerical experiments for Surrogate-Based Proximal Framework for Median Absolute Residual Regression.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    QuantileRegressor,
    HuberRegressor,
    RANSACRegressor
)

import gurobipy as gp
from gurobipy import GRB

from apmr import APMRRegression


from datasets import (
    make_regression,
    train_test_split,
    add_outliers,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_residual,
)

from plots import generate_all_plots

def evaluate_model(
    model,
    A_test,
    b_test,
):
    prediction = model.predict(
        A_test
    )

    metrics = {
        "MSE":
            mean_squared_error(
                b_test,
                prediction,
            ),


        "MAE":
            mean_absolute_error(
                b_test,
                prediction,
            ),


        "MedianResidual":
            median_absolute_residual(
                b_test,
                prediction,
            ),
    }
    return metrics

class CoefficientModel:
    """
    Small wrapper around a coefficient vector so that
    all methods expose the same predict() interface.
    """
    def __init__(self, coef):
        self.coef_ = np.asarray(coef)
    def predict(self, A):
        return A @ self.coef_

class TukeyLasso:
    """
    Tukey-Lasso solved by proximal gradient descent.
    Objective:
        (1/n) sum_i rho_c(r_i(beta))
        + alpha * ||beta||_1
    where rho_c is Tukey's biweight loss.
    The intercept is not penalized if fit_intercept=True.
    """
    def __init__(
        self,
        alpha=0.01,
        c=4.685,
        max_iter=5000,
        tol=1e-6,
        fit_intercept=True,
    ):
        self.alpha = alpha
        self.c = c
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
    @staticmethod
    def _tukey_derivative(residual, c):
        """
        Derivative of Tukey's biweight loss.
        """
        u = residual / c
        derivative = np.zeros_like(residual)
        mask = np.abs(u) <= 1.0
        derivative[mask] = (
            residual[mask]
            * (1.0 - u[mask] ** 2) ** 2
        )
        return derivative
    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(
            np.abs(x) - threshold,
            0.0,
        )
    def fit(self, A, b):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        n, p = A.shape
        # ----------------------------------------------------
        # Add intercept column if requested
        # ----------------------------------------------------
        if self.fit_intercept:
            X = np.column_stack(
                [np.ones(n), A]
            )
        else:
            X = A
        q = X.shape[1]
        # ----------------------------------------------------
        # Robust initial scale
        # ----------------------------------------------------
        ls_coef = np.linalg.lstsq(
            X,
            b,
            rcond=None,
        )[0]
        initial_residuals = b - X @ ls_coef
        mad = np.median(
            np.abs(
                initial_residuals
                - np.median(initial_residuals)
            )
        )
        scale = 1.4826 * mad
        if scale < 1e-8:
            scale = np.std(initial_residuals)
        if scale < 1e-8:
            scale = 1.0
        c = self.c * scale
        # ----------------------------------------------------
        # Initial point
        # ----------------------------------------------------
        beta = ls_coef.copy()
        # ----------------------------------------------------
        # Lipschitz constant approximation
        #
        # rho'' for Tukey's biweight is bounded in magnitude
        # by a small absolute constant. We use 1.0 as a
        # conservative/simple upper bound here.
        # ----------------------------------------------------
        spectral_norm = np.linalg.norm(
            X,
            ord=2,
        )
        L = (
            spectral_norm ** 2
            / n
        )
        if L < 1e-12:
            L = 1.0
        step = 1.0 / L
        # ----------------------------------------------------
        # Proximal gradient
        # ----------------------------------------------------
        self.history_ = []
        for k in range(self.max_iter):
            residual = X @ beta - b
            grad = (
                X.T
                @ self._tukey_derivative(
                    residual,
                    c,
                )
            ) / n
            beta_old = beta.copy()
            # Gradient step
            beta_temp = beta - step * grad
            # Do not penalize intercept
            if self.fit_intercept:
                beta_new = beta_temp.copy()
                beta_new[0] = beta_temp[0]
                beta_new[1:] = self._soft_threshold(
                    beta_temp[1:],
                    step * self.alpha,
                )
            else:
                beta_new = self._soft_threshold(
                    beta_temp,
                    step * self.alpha,
                )
            beta = beta_new
            relative_change = (
                np.linalg.norm(beta - beta_old)
                / max(
                    1.0,
                    np.linalg.norm(beta_old),
                )
            )
            self.history_.append(
                relative_change
            )
            if relative_change < self.tol:
                break
        self.n_iter_ = k + 1
        self.scale_ = scale
        self.tukey_c_ = c
        # Store coefficients in the same format as sklearn
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self
    def predict(self, A):
        return (
            self.intercept_
            + np.asarray(A) @ self.coef_
        )


class ApproximateLMSLQS:
    """
    Practical approximate LMS/LQS estimator.
    LMS:
        minimize the median squared residual.
    LQS:
        minimize the h-th ordered squared residual.
    Because the exact LMS/LQS problem is highly nonconvex,
    this implementation uses random regression subsets and
    selects the best candidate.
    This should therefore be described in the paper as an
    approximate/random-subset LMS/LQS implementation rather
    than an exact global optimizer.
    """
    def __init__(
        self,
        quantile=0.5,
        n_candidates=100,
        random_state=0,
    ):
        self.quantile = quantile
        self.n_candidates = n_candidates
        self.random_state = random_state
    def fit(self, A, b):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        n, p = A.shape
        rng = np.random.default_rng(
            self.random_state
        )
        # Number of residuals used by LQS.
        h = int(
            np.ceil(
                self.quantile * n
            )
        )
        # Regression needs at least p observations.
        subset_size = min(
            n,
            max(p, h),
        )
        best_beta = None
        best_objective = np.inf
        # ----------------------------------------------------
        # Include ordinary LS as one candidate.
        # ----------------------------------------------------
        candidate_betas = []
        try:
            beta_ls = np.linalg.lstsq(
                A,
                b,
                rcond=None,
            )[0]
            candidate_betas.append(
                beta_ls
            )
        except np.linalg.LinAlgError:
            pass
        # ----------------------------------------------------
        # Random subset candidates
        # ----------------------------------------------------
        for _ in range(
            self.n_candidates
        ):
            indices = rng.choice(
                n,
                size=subset_size,
                replace=False,
            )
            X_sub = A[indices]
            y_sub = b[indices]
            try:
                beta = np.linalg.lstsq(
                    X_sub,
                    y_sub,
                    rcond=None,
                )[0]
                candidate_betas.append(
                    beta
                )
            except np.linalg.LinAlgError:
                continue
        # ----------------------------------------------------
        # Evaluate all candidates
        # ----------------------------------------------------
        for beta in candidate_betas:
            residuals = np.abs(
                b - A @ beta
            )
            ordered = np.sort(
                residuals ** 2
            )
            objective = ordered[h - 1]
            if objective < best_objective:
                best_objective = objective
                best_beta = beta.copy()
        if best_beta is None:
            raise RuntimeError(
                "LMS/LQS failed to produce "
                "a valid regression estimate."
            )
        self.coef_ = best_beta
        self.objective_ = best_objective
        self.h_ = h
        return self
    def predict(self, A):
        return np.asarray(A) @ self.coef_

class SOS1LQSRegression:
    def __init__(
        self,
        quantile=0.5,
        time_limit=None,
        mip_gap=0.0,
        threads=0,
        verbose=False,
        fit_intercept=True,
    ):

        if not (0 < quantile < 1):
            raise ValueError("quantile must be between 0 and 1.")

        self.quantile = quantile
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.threads = threads
        self.verbose = verbose
        self.fit_intercept = fit_intercept

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] != len(y):
            raise ValueError("X and y have incompatible dimensions.")

        n, p = X.shape

        if self.fit_intercept:
            X_model = np.column_stack([X, np.ones(n)])
            p_model = p + 1
        else:
            X_model = X
            p_model = p

        q = int(np.ceil(n * self.quantile))

        # Avoid q = n for quantiles extremely close to 1.
        q = min(max(q, 1), n)

        self.q_ = q
        self.n_samples_in_ = n
        self.n_features_in_ = p


        model = gp.Model("SOS1_LQS")

        model.Params.OutputFlag = 1 if self.verbose else 0
        model.Params.Threads = self.threads
        model.Params.MIPGap = self.mip_gap

        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit

        beta = model.addMVar(
            shape=p_model,
            lb=-GRB.INFINITY,
            name="beta",
        )

        gamma = model.addVar(
            lb=0.0,
            name="gamma",
        )

        r_plus = model.addVars(
            n,
            lb=0.0,
            name="r_plus",
        )

        r_minus = model.addVars(
            n,
            lb=0.0,
            name="r_minus",
        )

        mu = model.addVars(
            n,
            lb=0.0,
            name="mu",
        )

        mu_bar = model.addVars(
            n,
            lb=0.0,
            name="mu_bar",
        )

        z = model.addVars(
            n,
            vtype=GRB.BINARY,
            name="z",
        )


        for i in range(n):

            model.addConstr(
                r_plus[i] - r_minus[i]
                ==
                float(y[i]) - X_model[i, :] @ beta,
                name=f"residual_{i}",
            )
        for i in range(n):

            model.addConstr(
                r_plus[i]
                + r_minus[i]
                - gamma
                ==
                mu_bar[i] - mu[i],
                name=f"quantile_{i}",
            )

        model.addConstr(
            gp.quicksum(z[i] for i in range(n)) == q,
            name="selection",
        )

        for i in range(n):

            model.addConstr(
                mu[i] <= gamma,
                name=f"mu_bound_{i}",
            )

        for i in range(n):

            model.addSOS(
                GRB.SOS_TYPE1,
                [mu_bar[i], mu[i]],
                [1.0, 2.0],
            )


        for i in range(n):

            model.addSOS(
                GRB.SOS_TYPE1,
                [r_plus[i], r_minus[i]],
                [1.0, 2.0],
            )

    
        for i in range(n):

            model.addSOS(
                GRB.SOS_TYPE1,
                [z[i], mu_bar[i]],
                [1.0, 2.0],
            )


        model.setObjective(
            gamma,
            GRB.MINIMIZE,
        )

        self.model_ = model

        model.optimize()

        self.status_ = model.Status
        self.runtime_ = model.Runtime
        self.node_count_ = model.NodeCount
        self.mip_gap_ = model.MIPGap

        if model.SolCount == 0:
            raise RuntimeError(
                f"Gurobi failed to find a feasible solution. "
                f"Status code: {model.Status}"
            )


        beta_solution = np.array(
            [beta[j].X for j in range(p_model)]
        )

        if self.fit_intercept:
            self.coef_ = beta_solution[:-1]
            self.intercept_ = beta_solution[-1]
        else:
            self.coef_ = beta_solution
            self.intercept_ = 0.0

        self.objective_ = gamma.X

    
        self.gamma_ = gamma.X

        self.r_plus_ = np.array(
            [r_plus[i].X for i in range(n)]
        )

        self.r_minus_ = np.array(
            [r_minus[i].X for i in range(n)]
        )

        self.mu_ = np.array(
            [mu[i].X for i in range(n)]
        )

        self.mu_bar_ = np.array(
            [mu_bar[i].X for i in range(n)]
        )

        self.z_ = np.array(
            [round(z[i].X) for i in range(n)],
            dtype=int,
        )
        
        residuals = y - self.predict(X)

        self.residuals_ = residuals
        self.abs_residuals_ = np.abs(residuals)

        # Actual order statistic
        self.order_statistic_ = np.sort(
            self.abs_residuals_
        )[n - q]

        # For median case, this should agree with gamma.
        self.order_statistic_error_ = (
            self.objective_ - self.order_statistic_
        )

        self.converged_ = (
            model.Status == GRB.OPTIMAL
        )


        print("MIO status:", model.Status)
        print("MIO status name:", {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
        }.get(model.Status, model.Status))
    
        print("MIO objective:", model.ObjVal)
        print("MIO bound:", model.ObjBound)
        print("MIO gap:", model.MIPGap)
        print("MIO runtime:", model.Runtime)
        print("MIO nodes:", model.NodeCount)

        return self

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        prediction = X @ self.coef_

        if self.fit_intercept:
            prediction = prediction + self.intercept_

        return prediction

def make_lqs_model_mio(
    random_state=None,
    quantile=0.5,
    time_limit=None,
    mip_gap=0.0,
    threads=0,
    verbose=False,
):
    return SOS1LQSRegression(
        quantile=quantile,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        verbose=verbose,
        fit_intercept=True,
    )
def make_lms_model(
    random_state,
    n_candidates=100,
):
    return ApproximateLMSLQS(
        quantile=0.5,
        n_candidates=n_candidates,
        random_state=random_state,
    )
def make_lqs_model(
    random_state,
    quantile=0.5,
    n_candidates=100,
):
    return ApproximateLMSLQS(
        quantile=quantile,
        n_candidates=n_candidates,
        random_state=random_state,
    )
    
def run_single_experiment(outlier_fraction,random_state=0,verbose=False):

    A, b, x_true = make_regression(
        n_samples=300,
        n_features=50,
        noise=0.1,
        outlier_fraction=0,
        random_state=
            random_state,
    )

    # A, b, x_true = make_regression(
    #         n_samples=10,
    #         n_features=5,
    #         noise=0.1,
    #         outlier_fraction=0,
    #         random_state=
    #             random_state,
    #     )
    
    np.savetxt("A.csv", A)
    np.savetxt("x.csv",x_true)
    np.savetxt("b.csv",b)

    (A_train, A_test, b_train, b_test,) = train_test_split(A, b, test_size=0.25, random_state=random_state,)

    if outlier_fraction > 0:
        b_train = add_outliers(b_train, fraction=outlier_fraction, scale=250, random_state=random_state)    

    signal = A_train @ x_true
    noise = b_train - signal
    
    snr_db = 10 * np.log10(
        np.var(signal) / np.var(noise))
    print(f"Signal to Noise Ratio(dB): {snr_db}")

    rng = np.random.default_rng(random_state)
    apmr = APMRRegression(
        alpha=0.1,
        lambda_=1,
        eta=0.2,
        max_iter=10000,
        inner_iter=2000,
        tol=2.5*1e-3,
        inner_tol=1e-3,
        random_state=random_state,
        initial_point = rng.standard_normal(A.shape[1]),
        verbose=verbose
    )
    # apmr = APMRRegression(
    #         alpha=0.1,
    #         lambda_=1,
    #         eta=0.2,
    #         max_iter=10000,
    #         inner_iter=2000,
    #         tol=1e-6,
    #         inner_tol=1e-6,
    #         random_state=random_state,
    #         initial_point = rng.standard_normal(A.shape[1]),
    #         verbose=verbose
    #     )

    apmr.fit( A_train, b_train)

    ls = LinearRegression()

    ls.fit(A_train,b_train)

    lad = QuantileRegressor(
        quantile=0.5,
        alpha=0.0,
        solver="highs",
    )

    lad.fit(A_train,b_train)

    huber = HuberRegressor(
        epsilon=1.35,
        alpha=0.0,
        max_iter=1000,
        tol=1e-6,
    )
    huber.fit(
        A_train,
        b_train,
    )

    tukey_lasso = TukeyLasso(
        alpha=0.01,
        c=4.685,
        max_iter=5000,
        tol=1e-6,
    )
    tukey_lasso.fit(
        A_train,
        b_train,
    )

    lms = make_lms_model(
        random_state=random_state,
        n_candidates=100,
    )
    lms.fit(
        A_train,
        b_train,
    )

    lqs = make_lqs_model(
        random_state=random_state,
        quantile=0.5,
        n_candidates=100,
    )
    lqs.fit(
        A_train,
        b_train,
    )

    lqs_mio=make_lqs_model_mio(
        random_state=random_state,
        quantile=0.5,
        time_limit=7200,
        mip_gap=0.0,
        threads=0,
        verbose=verbose
    )
    lqs_mio.fit(A_train,b_train)
    lqs.fit(
    A_train,
    b_train,
    )
    print("gamma:", lqs_mio.gamma_)
    print("objective:", lqs_mio.objective_)
    print("actual median:", np.median(lqs_mio.abs_residuals_))
    print("actual min:", np.min(lqs_mio.abs_residuals_))
    print("sorted residuals:", np.sort(lqs_mio.abs_residuals_))
    print("q:", lqs_mio.q_)
    print("selected:", np.sum(lqs_mio.z_))
    print("status:", lqs_mio.status_)

    

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=0.5,
        residual_threshold=None,
        max_trials=5000,
        max_skips=10000,
        stop_probability=0.999,
        random_state=random_state,
    )

    ransac.fit(A_train, b_train)



    results = []
    models = [
        ("APMR", apmr),
        ("LeastSquares", ls),
        ("LAD", lad),
        ("Huber", huber),
        ("TukeyLasso", tukey_lasso),
        ("LMS", lms),
        ("LQS", lqs),
        ("RANSAC",ransac),
        ("MIO",lqs_mio)
    ]

    for name, model in models:

        metrics = evaluate_model(model, A_test, b_test,)
        metrics["Method"] = name
        metrics["OutlierFraction"] = (outlier_fraction)
        metrics["RandomState"] = (random_state)

        if name == "APMR":
            metrics["OuterIterations"] = (apmr.n_iter_)
            metrics["FinalObjective"] = (apmr.objective_)
            metrics["Converged"] = (apmr.converged_)
            metrics["AverageActiveSet"] = (np.mean(apmr.history["active_set_size"]))
            metrics["AverageInnerIterations"] = (np.mean(apmr.history["inner_iterations"]))

        elif name == "TukeyLasso":
            metrics["OuterIterations"] = (
                tukey_lasso.n_iter_
            )
            metrics["FinalObjective"] = np.nan
            metrics["Converged"] = (
                tukey_lasso.n_iter_
                < tukey_lasso.max_iter
            )
            metrics["AverageActiveSet"] = np.nan
            metrics["AverageInnerIterations"] = np.nan

        elif name in ["LMS", "LQS"]:
            metrics["OuterIterations"] = (
                lms.n_candidates
                if name == "LMS"
                else lqs.n_candidates
            )
            metrics["FinalObjective"] = (
                model.objective_
            )
            metrics["Converged"] = np.nan
            metrics["AverageActiveSet"] = np.nan
            metrics["AverageInnerIterations"] = np.nan

        elif name == "RANSAC":
            metrics["OuterIterations"] = ransac.n_trials_
            metrics["FinalObjective"] = np.nan
            metrics["Converged"] = ransac.n_trials_ < ransac.max_trials
            metrics["AverageActiveSet"] = np.nan
            metrics["AverageInnerIterations"] = np.nan

        elif name == "MIO":
            metrics["OuterIterations"] = lqs_mio.node_count_
            metrics["FinalObjective"] = lqs_mio.objective_
            metrics["Converged"] = lqs_mio.converged_
            metrics["AverageActiveSet"] = np.sum(lqs_mio.z_)
            metrics["AverageInnerIterations"] = np.nan

        else:
            metrics["OuterIterations"] = np.nan
            metrics["FinalObjective"] = np.nan
            metrics["Converged"] = np.nan
            metrics["AverageActiveSet"] = np.nan
            metrics["AverageInnerIterations"] = np.nan

        results.append(
            metrics
        )

    return (apmr,pd.DataFrame(results),A_test,b_test)

def benchmark():
    """
    Run all experiments.
    """
    fractions = [
        # 0.0, 
        # 0.1, 
        # 0.2, 
        # 0.3, 
        # 0.4, 
        0.5
    ]

    seeds = [
        # 0, 
        # 1, 
        # 2, 
        # 3, 
        4
   ]

    frames = []

    apmr_model = None
    A_test = None
    b_test = None

    for seed in seeds:
       for fraction in fractions:

            print("Running experiment | "f"outliers={fraction:.1f} "f"seed={seed}")

            (model,table,A_test_run,b_test_run,) = run_single_experiment(fraction,random_state=seed,verbose=False)

            frames.append(table)

            if seed == 0 and fraction == 0.5:                
                apmr_model = model
                A_test = A_test_run
                b_test = b_test_run

    results = pd.concat(frames, ignore_index=True)

    results.to_csv("results.csv", index=False)
    summary = (
        results
        .groupby(["OutlierFraction", "Method"], as_index=False)
        .agg({
            "MSE": "mean",
            "MAE": "mean",
            "MedianResidual": "mean",
            "OuterIterations": "mean",
            "FinalObjective": "mean",
            "AverageActiveSet": "mean",
            "AverageInnerIterations": "mean",
            "Converged": "mean",
        })
    )
    print(summary)

    return (apmr_model, A_test, b_test, results)

def main():
    (apmr_model,A_test,b_test,results,) = benchmark()

    print(results)

    generate_all_plots(apmr_model, A_test, b_test, results)

if __name__ == "__main__":
    main()