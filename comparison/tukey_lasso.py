import numpy as np

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