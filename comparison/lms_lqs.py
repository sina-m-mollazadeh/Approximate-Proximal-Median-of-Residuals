import numpy as np

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