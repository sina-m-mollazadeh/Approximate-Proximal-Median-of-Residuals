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
        lambda_max=100.0,
        rho=0.015,
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
        self.lambda_max = lambda_max
        self.rho = rho
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

    def surrogate_objective(self, A, b, x, M) -> float:
        """Evaluate the frozen local surrogate \tilde{f}_k(x) in O(|M_k| * d) time."""
        return float(np.mean(np.abs(A[M] @ x - b[M])))

    def surrogate_subgradient(self, A, b, x, M):
        """Construct the representative subgradient (arithmetic mean of active set)."""
        active_residuals = A[M] @ x - b[M]
        signs=np.sign(active_residuals)
        # This is Performed so it wont assign weight zero to residuals that are zero. it is choosen 1 and it is within the clarke bounds
        signs[signs==0]=1
        return (signs @ A[M]) / len(M)

    def compute_trust_radius(self, A, b, x, M):
        """
        Vectorized O(n*d) trust radius calculation.
        Checks zero-crossing hyperplanes and both (a_i - a_j) and (a_i + a_j)
        order-switching hyperplanes.
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

        # 2. Minimum distance to switching hyperplanes outside M
        mask_inactive = np.ones(A.shape[0], dtype=bool)
        mask_inactive[M] = False
        J = np.where(mask_inactive)[0]

        if J.size == 0:
            return self.trust_scaling * (d0 if not np.isinf(d0) else 1.0)

        h = int(np.ceil(len(abs_res) / 2.0)) - 1
        val = np.partition(abs_res, h)[h]
        i_med = int(np.argmin(np.abs(abs_res - val)))

        # 2a. Same-sign switching hyperplanes: (s_i a_i - s_j a_j)^T x = s_i b_i - s_j b_j
        diff_A_minus = signs[i_med] * A[i_med] - signs[J, None] * A[J]
        norms_minus = np.linalg.norm(diff_A_minus, axis=1)
        num_minus = np.abs(abs_res[i_med] - abs_res[J])
        valid_minus = (norms_minus > 1e-12) & (num_minus > 1e-4)
        d_minus = (
            float(np.min(num_minus[valid_minus] / norms_minus[valid_minus]))
            if np.any(valid_minus)
            else np.inf
        )

        # 2b. Opposite-sign switching hyperplanes: (s_i a_i + s_j a_j)^T x = s_i b_i + s_j b_j
        diff_A_plus = signs[i_med] * A[i_med] + signs[J, None] * A[J]
        norms_plus = np.linalg.norm(diff_A_plus, axis=1)
        num_plus = abs_res[i_med] + abs_res[J]
        valid_plus = (norms_plus > 1e-12) & (num_plus > 1e-4)
        d_plus = (
            float(np.min(num_plus[valid_plus] / norms_plus[valid_plus]))
            if np.any(valid_plus)
            else np.inf
        )

        # Select the nearest valid hyperplane across all families
        min_dist = min(d0, d_minus, d_plus)

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

        signs = np.where(e >= 0.0, 1.0, -1.0)

        # 4a. Same-sign switching ray intersections: (s_i v_i - s_j v_j) alpha = |e_j| - |e_i|
        den_minus = signs[i_med] * v[i_med] - signs[J] * v[J]
        num_minus = abs_e[J] - abs_e[i_med]
        valid_minus = (np.abs(den_minus) > 1e-12) & (np.abs(num_minus) > 1e-4)
        alphas_minus = num_minus[valid_minus] / den_minus[valid_minus]
        alphas_minus = alphas_minus[alphas_minus > 1e-6]
        a_minus = float(np.min(alphas_minus)) if alphas_minus.size > 0 else np.inf

        # 4b. Opposite-sign switching ray intersections: (s_i v_i + s_j v_j) alpha = -(|e_i| + |e_j|)
        den_plus = signs[i_med] * v[i_med] + signs[J] * v[J]
        num_plus = -(abs_e[J] + abs_e[i_med])
        valid_plus = (np.abs(den_plus) > 1e-12) & (np.abs(num_plus) > 1e-4)
        alphas_plus = num_plus[valid_plus] / den_plus[valid_plus]
        alphas_plus = alphas_plus[alphas_plus > 1e-6]
        a_plus = float(np.min(alphas_plus)) if alphas_plus.size > 0 else np.inf

        # 5. Nearest positive intersection across all hyperplanes in H_k
        res_alpha = min(a0, a_minus, a_plus)

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

            # --- 3. Approximate Proximal Solver (O(|M_k| * d) per inner step) ---
            lambda_k = min(self.lambda_max, self.lambda_ * (1.0 + self.rho * k))

            # Anchor best_u to x^k so Phi_k(x_A^{k+1}) <= Phi_k(x^k) is guaranteed (H1)
            phi_xk = self.surrogate_objective(A_aug, b, x, M_k) + (
                0.5 / lambda_k
            ) * float(np.sum((x - z_k) ** 2))
            best_u = x.copy()
            best_u_phi = phi_xk

            # Project initial forward point z_k onto Trust Ball B(x^k, R_k)
            diff_z = z_k - x
            dist_z = np.linalg.norm(diff_z)
            u = x + (R_k / dist_z) * diff_z if dist_z > R_k else z_k.copy()

            phi_u0 = self.surrogate_objective(A_aug, b, u, M_k) + (
                0.5 / lambda_k
            ) * float(np.sum((u - z_k) ** 2))
            if phi_u0 < best_u_phi:
                best_u_phi = phi_u0
                best_u = u.copy()

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

                # O(|M_k| * d) surrogate proximal objective evaluation
                phi_u = self.surrogate_objective(A_aug, b, u_new, M_k) + (
                    0.5 / lambda_k
                ) * float(np.sum((u_new - z_k) ** 2))
                if phi_u < best_u_phi:
                    best_u_phi = phi_u
                    best_u = u_new.copy()

                if np.linalg.norm(u_new - u) / max(1.0, np.linalg.norm(u)) < self.inner_tol:
                    u = u_new
                    inner_count = t + 1
                    break
                u = u_new
                inner_count = t + 1

            # Select inner candidate with lowest surrogate proximal objective Phi_k
            x_A = best_u.copy()
            f_xA = self.objective(A_aug, b, x_A)

            # If the trust-radius floor allowed x_A to cross a switching kink and increase f,
            # backtrack along the segment (x_A - x) inside the true cell until f(x_A) < f(x^k)
            if f_xA >= f_xk:
                step_vec = u - x
                for bt in range(1, 12):
                    cand_bt = x + (0.5 ** bt) * step_vec
                    f_bt = self.objective(A_aug, b, cand_bt)
                    if f_bt < f_xk:
                        x_A = cand_bt
                        f_xA = f_bt
                        break
                else:
                    # At exact local minimum kink: remain at x^k (zero displacement)
                    x_A = x.copy()
                    f_xA = f_xk

            disp_A_sq = float(np.sum((x_A - x) ** 2))
            Q_A = (f_xk - f_xA) / disp_A_sq if disp_A_sq > 1e-18 else 0.0

            # --- 4. Geometric Memory Candidate ---
            x_E = x + E_k
            Q_E = self.q_score(A_aug, b, x, x_E)

            # --- 5. Stationarity-Safeguarded Selection Rule (Strictly Enforcing H1: Q > 0) ---
            norm_E = np.linalg.norm(E_k)
            S_E = False

            if norm_E > 1e-14 and Q_E > 0.0:
                M_E = self.active_median_set(A_aug, b, x_E)
                g_E = self.surrogate_subgradient(A_aug, b, x_E, M_E)
                S_E = np.linalg.norm(g_E) <= self.b_safeguard * norm_E

            if Q_E > max(0.0, Q_A) and S_E:
                x_next = x_E.copy()
                chosen = "E"
            elif Q_A > 0.0:
                x_next = x_A.copy()
                chosen = "A"
            else:
                x_next = x.copy()
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