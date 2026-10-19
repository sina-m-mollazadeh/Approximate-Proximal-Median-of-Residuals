"""
apmr.py
Algorithm of Memory-Augmented Surrogate-Based Proximal Framework 
for Median Absolute Residual Regression.
"""
import logging
from typing import Dict, Optional
import numpy as np

class APMRRegression:
    def __init__(
        self,
        alpha=2.0,
        lambda_=1.0,
        eta=0.5,
        trust_scaling=0.9,
        gamma=0.85,
        b_safeguard=1e10,
        inner_iter=50,
        max_iter=500,
        tol=1e-6,
        inner_tol=1e-5,
        fit_intercept=True,
        random_state=None,
        initial_point=None,
        verbose=False,
    ):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.eta = eta
        self.trust_scaling = trust_scaling
        self.gamma = gamma
        self.b_safeguard = b_safeguard
        self.max_iter = max_iter
        self.inner_iter = inner_iter
        self.tol = tol
        self.inner_tol = inner_tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.initial_point = initial_point
        self.coef_ = None
        self.intercept_ = 0.0
        self.history: Optional[Dict] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.verbose = verbose

    def __repr__(self):
        return (
            "APMRRegression("
            f"alpha={self.alpha}, lambda={self.lambda_}, eta={self.eta}, "
            f"trust_scaling={self.trust_scaling}, gamma={self.gamma}, "
            f"max_iter={self.max_iter}, inner_iter={self.inner_iter})"
        )

    def _augment(self, A):
        if not self.fit_intercept:
            return A
        return np.column_stack([A, np.ones(A.shape[0], dtype=A.dtype)])

    def residuals(self, A, b, x):
        """Calculate absolute residuals."""
        return np.abs(A @ x - b)

    def objective(self, A, b, x):
        """Strict Lower-Median Convention for the MAR objective."""
        res = self.residuals(A, b, x)
        h = int(np.ceil(len(res) / 2.0)) - 1
        return float(np.partition(res, h)[h])
    
    def active_median_set(self, A, b, x) -> np.ndarray:
        """Find the active set of indices exactly at the median."""
        res = self.residuals(A, b, x)
        val = self.objective(A, b, x)
        tol = 1e-2 * max(1.0, val) # Numerical tolerance for equality
        idx = np.where(np.abs(res - val) <= tol)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(res - val)))])
        return idx

    def surrogate_subgradient(self, A, b, x, M):
        """Construct the representative subgradient (arithmetic mean of active set)."""
        active_residuals = A[M] @ x - b[M]
        signs=np.sign(active_residuals)
        # This is Performed so it wont assign weight zero to residuals that are zero. it is choosen 1 and it is within the clarke bounds
        signs[signs==0]=1
        return (signs @ A[M]) / len(M)

    def compute_trust_radius(self, A, b, x, M):
        """
        Vectorized O(N*d) trust radius calculation.
        Uses the minimum distance to zero-crossing and signed switching
        hyperplanes, without percentile relaxation.
        """
        signed_res = A @ x - b
        abs_res = np.abs(signed_res)
        signs = np.where(signed_res >= 0.0, 1.0, -1.0)

        # 1. Minimum distance to zero-crossing hyperplanes for active set M
        norms_M = np.linalg.norm(A[M], axis=1)
        valid_M = (norms_M > 1e-12) & (abs_res[M] > 1e-8)

        d0 = (
            float(np.min(abs_res[M][valid_M] / norms_M[valid_M]))
            if np.any(valid_M)
            else np.inf
        )

        # 2. Minimum distance to signed switching hyperplanes outside M
        mask_inactive = np.ones(A.shape[0], dtype=bool)
        mask_inactive[M] = False
        J = np.where(mask_inactive)[0]

        if J.size == 0:
            return self.trust_scaling * (
                d0 if not np.isinf(d0) else 1.0
            )

        h = int(np.ceil(len(abs_res) / 2.0)) - 1
        val = np.partition(abs_res, h)[h]
        i_med = int(np.argmin(np.abs(abs_res - val)))

        # Signed switching hyperplanes:
        # (s_i a_i - s_j a_j)^T x = s_i b_i - s_j b_j
        diff_A = signs[i_med] * A[i_med] - signs[J, None] * A[J]
        norms_diff = np.linalg.norm(diff_A, axis=1)
        num_diff = np.abs(abs_res[i_med] - abs_res[J])

        valid_switch = (norms_diff > 1e-12) & (num_diff > 1e-4)

        d_switch = (
            float(np.min(num_diff[valid_switch] / norms_diff[valid_switch]))
            if np.any(valid_switch)
            else np.inf
        )

        # Select the nearest valid hyperplane
        min_dist = min(d0, d_switch)

        if np.isinf(min_dist):
            min_dist = 1.0

        return self.trust_scaling * max(
            min_dist,
            1e-3 * max(1.0, val)
        )

    def compute_available_room(self, A, b, x, u_dir, M):
        """Vectorized unexploited geometric movement rho_k(u^k) along u_dir."""
        e = A @ x - b
        v = A @ u_dir
        abs_e = np.abs(e)

        # 1. Zero-crossing ray intersections for active set
        v_M, e_M = v[M], e[M]
        valid_0 = np.abs(v_M) > 1e-12

        alphas_0 = -e_M[valid_0] / v_M[valid_0]
        alphas_0 = alphas_0[alphas_0 > 1e-6]

        a0 = float(np.min(alphas_0)) if alphas_0.size > 0 else np.inf

        # 2. Inactive indices
        mask_inactive = np.ones(A.shape[0], dtype=bool)
        mask_inactive[M] = False
        J = np.where(mask_inactive)[0]

        if J.size == 0:
            return a0 if not np.isinf(a0) else 0.0

        # 3. Identify the median-residual index
        h = int(np.ceil(len(e) / 2.0)) - 1
        val = np.partition(abs_e, h)[h]
        i_med = int(np.argmin(np.abs(abs_e - val)))

        # 4. Signed switching hyperplane ray intersections
        signs = np.where(e >= 0.0, 1.0, -1.0)

        den = signs[i_med] * v[i_med] - signs[J] * v[J]
        num = abs_e[J] - abs_e[i_med]

        valid_s = (np.abs(den) > 1e-12) & (np.abs(num) > 1e-4)

        alphas_s = num[valid_s] / den[valid_s]
        alphas_s = alphas_s[alphas_s > 1e-6]

        a_switch = (
            float(np.min(alphas_s))
            if alphas_s.size > 0
            else np.inf
        )

        # 5. Nearest positive intersection, without quantile relaxation
        res_alpha = min(a0, a_switch)

        return res_alpha if not np.isinf(res_alpha) else 0.0

    def q_score(self, A, b, x_base, x_cand):
        """Calculate Squared-Displacement Efficiency Score (H1 Metric)."""
        diff = x_cand - x_base
        norm_sq = np.sum(diff ** 2)
        if norm_sq < 1e-14:
            return -np.inf
        return (self.objective(A, b, x_base) - self.objective(A, b, x_cand)) / norm_sq

    def fit(self, A, b):
        A_raw = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        if A_raw.ndim != 2:
            raise ValueError("A must be 2-dimensional.")
        if b.ndim != 1:
            raise ValueError("b must be one-dimensional.")
        if A_raw.shape[0] != b.shape[0]:
            raise ValueError("Samples in A and b differ.")

        A_aug = self._augment(A_raw)
        rng = np.random.default_rng(self.random_state)
        n, d = A_aug.shape

        # Warm-start using trimmed least-squares on the bottom 50% residuals at 0
        # + small random perturbation to break exact censored-target ties at k=0
        res_init = np.abs(b)
        h = int(np.ceil(n / 2.0)) - 1
        inlier_idx = np.argpartition(res_init, h)[: h + 1]
        x_warm, _, _, _ = np.linalg.lstsq(A_aug[inlier_idx], b[inlier_idx], rcond=None)
        x = x_warm + rng.normal(0, 1e-4, size=d)

        if self.initial_point is not None:
            init_pt = np.asarray(self.initial_point, dtype=float)
            if self.fit_intercept and init_pt.size == A_raw.shape[1]:
                init_pt = np.append(init_pt, np.median(b))
            # If user passed exact zeros, blend with warm start or perturb to avoid censored tie trap
            if np.linalg.norm(init_pt) < 1e-12:
                x = x_warm + rng.normal(0, 1e-4, size=d)
            else:
                x = init_pt.copy()

        # Initialize Memory Vector Safely (Prevents division by zero errors early on)
        E_k = np.zeros(d)

        history = {
            "iteration": [], "objective": [], "step_norm": [],
            "gradient_norm": [], "inner_iterations": [], "active_set_size": [],
            "candidate_chosen": []
        }

        outer_converged = False
        best_objective = self.objective(A_aug, b, x)
        best_x = x.copy()
        trajectory = [x[: A_raw.shape[1]].copy()]

        for k in range(self.max_iter):
            f_xk = self.objective(A_aug, b, x)
            M_k = self.active_median_set(A_aug, b, x)

            if f_xk < best_objective:
                best_objective = f_xk
                best_x = x.copy()

            # --- 1. Trust Region Calculation ---
            R_k = self.compute_trust_radius(A_aug, b, x, M_k)

            # --- 2. Base Forward Step ---
            g_k = self.surrogate_subgradient(A_aug, b, x, M_k)
            alpha_k = self.alpha / ((k + 1) ** 0.75)
            z_k = x - alpha_k * g_k

            # --- 3. Approximate Proximal Solver (with Euclidean Projection) ---
            u = z_k.copy()
            best_u = u.copy()
            best_u_obj = np.inf
            lambda_k = self.lambda_ * (1 + 0.015 * k)
            inner_count = 0

            for t in range(self.inner_iter):
                s = self.surrogate_subgradient(A_aug, b, u, M_k)
                prox_grad = s + (u - z_k) / lambda_k
                eta_t = self.eta / (t + 1)
                v = u - eta_t * prox_grad

                # Euclidean Projection onto Trust Ball B(x^k, R_k)
                diff = v - x
                dist_v = np.linalg.norm(diff)
                if dist_v > R_k:
                    u_new = x + (R_k / dist_v) * diff
                else:
                    u_new = v

                obj_u = self.objective(A_aug, b, u_new)
                if obj_u < best_u_obj:
                    best_u_obj = obj_u
                    best_u = u_new.copy()

                if np.linalg.norm(u_new - u) / max(1.0, np.linalg.norm(u)) < self.inner_tol:
                    u = u_new
                    inner_count = t + 1
                    break
                u = u_new
                inner_count = t + 1

            x_A = best_u if best_u_obj < self.objective(A_aug, b, u) else u.copy()
            Q_A = self.q_score(A_aug, b, x, x_A)

            # --- 4. Geometric Memory Candidate ---
            x_E = x + E_k
            Q_E = self.q_score(A_aug, b, x, x_E)

            # --- 5. Stationarity-Safeguarded Selection Rule ---
            norm_E = np.linalg.norm(E_k)
            S_E = False

            # Safe checking prevents breaking when ||E^k|| = 0
            if norm_E > 1e-14:
                M_E = self.active_median_set(A_aug, b, x_E)
                g_E = self.surrogate_subgradient(A_aug, b, x_E, M_E)
                # Ensure the H2 Stationarity Bound holds
                S_E = np.linalg.norm(g_E) <= self.b_safeguard * norm_E

                if self.verbose and k % 10 == 0:
                    print(f"  [Debug] Q_E: {Q_E:.2e} vs Q_A: {Q_A:.2e} | "
                          f"||g_E||: {np.linalg.norm(g_E):.2f} <= {self.b_safeguard * norm_E:.2f} (S_E={S_E})")

            if Q_E > Q_A and S_E:
                x_next = x_E.copy()
                chosen = "E"
            else:
                x_next = x_A.copy()
                chosen = "A"

            # --- 6. Memory Update ---
            s_k = x_A - x
            norm_sk = np.linalg.norm(s_k)
            if norm_sk > 1e-14:
                u_dir = s_k / norm_sk
                rho_k = self.compute_available_room(A_aug, b, x, u_dir, M_k)
                c_k = max(0.0, rho_k - norm_sk)
            else:
                u_dir = np.zeros_like(x)
                c_k = 0.0

            E_k = self.gamma * E_k + (1 - self.gamma) * (c_k * u_dir)

            # --- 7. Metrics & Convergence Check ---
            step = np.linalg.norm(x_next - x)
            history["iteration"].append(k)
            history["objective"].append(f_xk)
            history["active_set_size"].append(len(M_k))
            history["gradient_norm"].append( np.linalg.norm(g_k))
            history["inner_iterations"].append(inner_count)
            history["step_norm"].append(step)
            history["candidate_chosen"].append(chosen)
            trajectory.append(x_next[: A_raw.shape[1]].copy())

            relative_step = step / max(1.0, np.linalg.norm(x))

            if self.verbose:
                print(f"Iter {k:4d} | Obj: {f_xk:.6f} | Best: {best_objective:.6f} | Choice: {chosen} | RelStep: {relative_step:.4e}")

            if relative_step < self.tol and k >= 10:
                x = x_next
                outer_converged = True
                break

            x = x_next

        # Finalize
        final_median = self.objective(A_aug, b, x)
        if final_median < best_objective:
            best_objective = final_median
            best_x = x.copy()

        np.savetxt("apmr_trajectory.csv", trajectory, delimiter=",")
        if self.fit_intercept:
            self.coef_ = best_x[:-1]
            self.intercept_ = float(best_x[-1])
        else:
            self.coef_ = best_x
            self.intercept_ = 0.0

        self.history = history
        self.n_iter_ = k + 1
        self.objective_ = best_objective
        self.converged_ = outer_converged

        return self

    def predict(self, A):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted.")
        return A @ self.coef_ + self.intercept_

    def score(self, A, b):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted.")
        res = np.abs(self.predict(A) - b)
        h = int(np.ceil(len(res) / 2.0)) - 1
        return -float(np.partition(res, h)[h])