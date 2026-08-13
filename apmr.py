"""
apmr.py
Algorithn of Surrogate-Based Proximal Framework for Median Absolute Residual Regression.
"""

import logging
from typing import Dict, Optional
import numpy as np

class APMRRegression:
    def __init__(
        self,
        alpha=0.05,
        lambda_=0.05,
        eta=0.01,
        inner_iter=1000,
        max_iter=2000,
        tol=1e-4,
        inner_tol=1e-4,
        random_state=None,
        initial_point=None,
        verbose=False
    ):

        self.alpha = alpha
        self.lambda_ = lambda_
        self.eta = eta

        self.max_iter = max_iter
        self.inner_iter = inner_iter

        self.tol = tol
        self.inner_tol = inner_tol

        self.random_state = random_state
        self.initial_point = initial_point

        self.coef_ = None
        self.history: Optional[Dict] = None

        self.logger = logging.getLogger(
            self.__class__.__name__
        )
        self.verbose=verbose


    def __repr__(self):
        return (
            "MARRegression("
            f"alpha={self.alpha}, "
            f"lambda={self.lambda_}, "
            f"eta={self.eta}, "
            f"max_iter={self.max_iter}, "
            f"inner_iter={self.inner_iter})"
        )

    def median(self,residuals)-> int:
        med=np.median(residuals)
        mini=np.argmin(np.abs(residuals-med))
        return residuals[mini]
    
    def residuals(self, A, b, x):
        return np.abs(A @ x - b)

    def objective(self, A, b, x):
        return self.median(self.residuals(A,b,x))
    
    def active_median_set(self, residuals: np.ndarray) -> np.ndarray:
        median = self.median(residuals)
        tol = 1e-4 * max(1,abs(median))
        idx = np.where(np.abs(residuals - median)<= tol)[0]

        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(residuals - median)))])

        return idx

    def surrogate_subgradient(self,A,b,x,M):
        residuals = A[M] @ x - b[M]
        return (np.sign(residuals) @ A[M]) / len(M)

    def fit(self,A,b):

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        if A.ndim != 2:
            raise ValueError("A must be 2-dimensional.")

        if b.ndim != 1:
            raise ValueError("b must be one-dimensional.")

        if A.shape[0] != b.shape[0]:
            raise ValueError("Number of samples in A and b differ.")

        rng = np.random.default_rng(self.random_state)

        _, d = A.shape

        if self.initial_point is None:
            x = rng.normal(0, 1e-3, size=d)
        else:
            x = np.asarray(self.initial_point, dtype=float)

        history = {
            "iteration": [],
            "objective": [],
            "step_norm": [],
            "gradient_norm": [],
            "inner_iterations": [],
            "proximal_step_norm": [],
            "active_set_size": [],
            "median": [],
            "coefficients": [],
            "inner_converged": [],
        }

        outer_converged = False
        best_objective = np.inf
        best_x = x.copy()
        k = -1
        trajectory = [x.copy()]

        ############
        # Outer Iter
        ############
        for k in range(self.max_iter):
            residuals = self.residuals(A,b,x,)

            median = self.median(residuals)

            history["iteration"].append(k)
            history["objective"].append(median)
            history["median"].append(median)
            history["coefficients"].append(x.copy())

            if median < best_objective:
                best_objective = median
                best_x = x.copy()

            M = self.active_median_set(residuals)

            history["active_set_size"].append(len(M))

            g = self.surrogate_subgradient(A, b, x, M)

            history["gradient_norm"].append(np.linalg.norm(g))

            alpha_k = (self.alpha/(k + 1) ** 0.75)

            z = x - alpha_k * g
            u = z.copy()
            inner_converged = False
            inner_count = 0

            ############
            # Inner Iter
            ############
            for t in range(self.inner_iter):

                s=self.surrogate_subgradient(
                    A,
                    b,
                    u,
                    M
                )
                lambda_k = self.lambda_ * (1 + 0.015 * k)
                proximal_gradient = (
                    s
                    +
                    (u - z)
                    /
                    lambda_k
                )

                eta_t = self.eta/(t+1)

                u_new = (
                    u
                    -
                    eta_t * proximal_gradient
                )

                relative_step = (
                    np.linalg.norm(u_new - u)
                    /
                    max(1.0, np.linalg.norm(u))
                )

                inner_count = t + 1

                if relative_step < self.inner_tol:
                    u = u_new
                    inner_converged = True
                    break

                u = u_new
                trajectory.append(u)
            
            x_new = u

            history["inner_iterations"].append(inner_count)

            history["inner_converged"].append(inner_converged)

            history["proximal_step_norm"].append(np.linalg.norm(x_new-z))

            step = np.linalg.norm(x_new-x)

            history["step_norm"].append(step)

            relative_step = (step / max(1,np.linalg.norm(x)))
            print(f"at iteration {k} the relative step is {relative_step}") if self.verbose else ""
            print(f"at iteration {k} the parameters are lambda:{lambda_k}, eta:{eta_t}, alpha:{alpha_k}, ") if self.verbose else ""


            if relative_step < self.tol:
                x = x_new
                outer_converged = True
                break

            x = x_new

            self.logger.info("Iter %4d | ""obj %.6f | ""||g|| %.4e | ""step %.4e | ""|M| %d",k,median,np.linalg.norm(g),step,len(M),)
            trajectory.append(x)
        
        final_median = self.objective(A, b, x)

        history["iteration"].append(k + 1)
        history["objective"].append(final_median)
        history["median"].append(final_median)
        history["coefficients"].append(x.copy())

        if final_median < best_objective:
            best_objective = final_median
            best_x = x.copy()

        np.savetxt("apmr_trajectory.csv",trajectory,delimiter=",")

        self.coef_ = best_x
        self.history = history
        self.n_iter_ = k + 1
        self.objective_ = best_objective
        self.converged_ = outer_converged
    
        return self

    def predict(self, A):

        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted.")
        return A @ self.coef_

    def score(self, A, b):

        if self.coef_ is None:
            raise RuntimeError(
                "Model has not been fitted."
            )

        return -self.objective(A, b, self.coef_)