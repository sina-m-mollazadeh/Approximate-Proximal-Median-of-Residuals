
import gurobipy as gp
from gurobipy import GRB
import numpy as np

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
