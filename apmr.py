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
        alpha=0.05,
        lambda_=0.05,
        eta=0.01,
        trust_scaling=0.9,
        gamma=0.9,
        b_safeguard=1.0,
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
        self.trust_scaling = trust_scaling
        self.gamma = gamma
        self.b_safeguard = b_safeguard
        self.max_iter = max_iter
        self.inner_iter = inner_iter
        self.tol = tol
        self.inner_tol = inner_tol
        self.random_state = random_state
        self.initial_point = initial_point
        self.coef_ = None
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

    def residuals(self, A, b, x):
        """Calculate absolute residuals."""
        return np.abs(A @ x - b)

    def objective(self, A, b, x):
        """Strict Lower-Median Convention for the MAR objective."""
        res = self.residuals(A, b, x)
        h = int(np.ceil(len(res) / 2.0)) - 1
        return np.sort(res)[h]
    
    def active_median_set(self, A, b, x) -> np.ndarray:
        """Find the active set of indices exactly at the median."""
        res = self.residuals(A, b, x)
        val = self.objective(A, b, x)
        tol = 1e-6 * max(1.0, val) # Numerical tolerance for equality
        idx = np.where(np.abs(res - val) <= tol)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(res - val)))])
        return idx

    def surrogate_subgradient(self, A, b, x, M):
        """Construct the representative subgradient (arithmetic mean of active set)."""
        residuals = A[M] @ x - b[M]
        return (np.sign(residuals) @ A[M]) / len(M)

    def compute_trust_radius(self, A, b, x, M):
        """Calculate distance to nearest switching hyperplane to define R_k."""
        min_dist = np.inf
        
        for i in M:
            # 1. Distance to Zero-Crossing Hyperplanes
            norm_a = np.linalg.norm(A[i])
            if norm_a > 1e-12:
                d0 = np.abs(A[i] @ x - b[i]) / norm_a
                if d0 > 1e-8: # Ignore if currently on this boundary
                    min_dist = min(min_dist, d0)
            
            # 2. Distance to Relative Switching Hyperplanes
            for j in range(A.shape[0]):
                if j in M: 
                    continue
                
                # Positive alignment (a_i + a_j)^T x = b_i + b_j
                diff_a_plus = A[i] + A[j]
                norm_plus = np.linalg.norm(diff_a_plus)
                if norm_plus > 1e-12:
                    d_plus = np.abs((A[i] + A[j]) @ x - (b[i] + b[j])) / norm_plus
                    if d_plus > 1e-8:
                        min_dist = min(min_dist, d_plus)
                        
                # Negative alignment (a_i - a_j)^T x = b_i - b_j
                diff_a_minus = A[i] - A[j]
                norm_minus = np.linalg.norm(diff_a_minus)
                if norm_minus > 1e-12:
                    d_minus = np.abs((A[i] - A[j]) @ x - (b[i] - b[j])) / norm_minus
                    if d_minus > 1e-8:
                        min_dist = min(min_dist, d_minus)
        
        if np.isinf(min_dist):
            min_dist = 1.0 # Safe fallback if no boundaries exist
            
        return self.trust_scaling * min_dist

    def compute_available_room(self, A, b, x, u_dir, M):
        """Compute unexploited geometric movement rho_k(u^k) along direction."""
        e = A @ x - b
        v = A @ u_dir
        min_alpha = np.inf
        
        for i in M:
            # Zero crossing intersection
            if np.abs(v[i]) > 1e-12:
                alpha = -e[i] / v[i]
                if alpha > 1e-8: 
                    min_alpha = min(min_alpha, alpha)
            
            # Switch crossing intersection
            for j in range(A.shape[0]):
                if j in M: 
                    continue
                    
                den_minus = v[i] - v[j]
                if np.abs(den_minus) > 1e-12:
                    alpha = (e[j] - e[i]) / den_minus
                    if alpha > 1e-8: 
                        min_alpha = min(min_alpha, alpha)
                
                den_plus = v[i] + v[j]
                if np.abs(den_plus) > 1e-12:
                    alpha = (-e[i] - e[j]) / den_plus
                    if alpha > 1e-8: 
                        min_alpha = min(min_alpha, alpha)
        
        return min_alpha if not np.isinf(min_alpha) else 0.0

    def q_score(self, A, b, x_base, x_cand):
        """Calculate Squared-Displacement Efficiency Score (H1 Metric)."""
        diff = x_cand - x_base
        norm_sq = np.sum(diff ** 2)
        if norm_sq < 1e-14:
            return -np.inf
        return (self.objective(A, b, x_base) - self.objective(A, b, x_cand)) / norm_sq

    def fit(self, A, b):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        
        if A.ndim != 2: raise ValueError("A must be 2-dimensional.")
        if b.ndim != 1: raise ValueError("b must be one-dimensional.")
        if A.shape[0] != b.shape[0]: raise ValueError("Samples in A and b differ.")
            
        rng = np.random.default_rng(self.random_state)
        n, d = A.shape
        
        if self.initial_point is None:
            x = rng.normal(0, 1e-3, size=d)
        else:
            x = np.asarray(self.initial_point, dtype=float)
            
        # Initialize Memory Vector Safely (Prevents division by zero errors early on)
        E_k = np.zeros(d)
        
        history = {
            "iteration": [], "objective": [], "step_norm": [],
            "gradient_norm": [], "inner_iterations": [], "active_set_size": [],
            "candidate_chosen": []
        }
        
        outer_converged = False
        best_objective = np.inf
        best_x = x.copy()
        trajectory = [x.copy()]

        for k in range(self.max_iter):
            f_xk = self.objective(A, b, x)
            M_k = self.active_median_set(A, b, x)
            
            if f_xk < best_objective:
                best_objective = f_xk
                best_x = x.copy()

            # --- 1. Trust Region Calculation ---
            R_k = self.compute_trust_radius(A, b, x, M_k)

            # --- 2. Base Forward Step ---
            g_k = self.surrogate_subgradient(A, b, x, M_k)
            alpha_k = self.alpha / ((k + 1) ** 0.75)
            z_k = x - alpha_k * g_k
            
            # --- 3. Approximate Proximal Solver (with Euclidean Projection) ---
            u = z_k.copy()
            lambda_k = self.lambda_ * (1 + 0.015 * k)
            inner_count = 0
            
            for t in range(self.inner_iter):
                s = self.surrogate_subgradient(A, b, u, M_k)
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
                    
                if np.linalg.norm(u_new - u) / max(1.0, np.linalg.norm(u)) < self.inner_tol:
                    u = u_new
                    inner_count = t + 1
                    break
                u = u_new
                inner_count = t + 1
                
            x_A = u.copy()
            Q_A = self.q_score(A, b, x, x_A)

            # --- 4. Geometric Memory Candidate ---
            x_E = x + E_k
            Q_E = self.q_score(A, b, x, x_E)
            
            # --- 5. Stationarity-Safeguarded Selection Rule ---
            norm_E = np.linalg.norm(E_k)
            S_E = False
            
            # Safe checking prevents breaking when ||E^k|| = 0
            if norm_E > 1e-14:
                M_E = self.active_median_set(A, b, x_E)
                g_E = self.surrogate_subgradient(A, b, x_E, M_E)
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
                rho_k = self.compute_available_room(A, b, x, u_dir, M_k)
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
            history["gradient_norm"].append(np.linalg.norm(g_k))
            history["inner_iterations"].append(inner_count)
            history["step_norm"].append(step)
            history["candidate_chosen"].append(chosen)
            trajectory.append(x_next)
            
            relative_step = step / max(1.0, np.linalg.norm(x))
            
            if self.verbose:
                print(f"Iter {k:4d} | Obj: {f_xk:.6f} | Choice: {chosen} | RelStep: {relative_step:.4e}")
                
            if relative_step < self.tol:
                x = x_next
                outer_converged = True
                break
                
            x = x_next

        # Finalize
        final_median = self.objective(A, b, x)
        if final_median < best_objective:
            best_objective = final_median
            best_x = x.copy()
            
        np.savetxt("apmr_trajectory.csv", trajectory, delimiter=",")
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
            raise RuntimeError("Model has not been fitted.")
        return -self.objective(A, b, self.coef_)