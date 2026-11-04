"""
experiments.py
Numerical experiments for Surrogate-Based Proximal Framework for Median Absolute Residual Regression.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    QuantileRegressor,
    HuberRegressor,
    RANSACRegressor
)

from comparison.lms_lqs import ApproximateLMSLQS
from comparison.tukey_lasso import TukeyLasso
from comparison.mio import SOS1LQSRegression
from apmr import APMRRegression

from datasets import (
    make_regression,
    train_test_split,
    add_outliers,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_residual,
)
def load_california_housing_with_outliers(
    csv_path,
    data_percentage=100,
    random_state=42,
    scale=True
):
    """
    Load the California Housing dataset, preprocess categorical features,
    optionally sample the dataset, and optionally standardize features
    and target values.

    Parameters
    ----------
    csv_path : str
        Path to the California Housing CSV file.
    data_percentage : float, default=100
        Percentage of rows to retain, between 0 and 100.
    random_state : int, default=42
        Random seed for reproducible sampling.
    scale : bool, default=True
        Whether to standardize A and b.

    Returns
    -------
    A : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    b : np.ndarray, shape (n_samples,)
        Target vector.
    x_true : None
        Placeholder retained for compatibility with existing code.
    scaler : StandardScaler or None
        Fitted feature scaler.
    scaler_y : StandardScaler or None
        Fitted target scaler.
    """

    # 1. Validate parameters
    if not 0 < data_percentage <= 100:
        raise ValueError("data_percentage must be in (0, 100].")

    # 2. Load data and remove missing values
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset is empty after removing missing values.")

    # 3. Separate target and features
    if "median_house_value" not in df.columns:
        raise ValueError(
            "The dataset must contain 'median_house_value'."
        )

    b = df["median_house_value"].to_numpy(dtype=float)
    X_df = df.drop(columns=["median_house_value"]).copy()

    # 4. Encode categorical features using a fixed category set
    if "ocean_proximity" in X_df.columns:
        ocean_categories = [
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN",
        ]

        unknown_categories = (
            set(X_df["ocean_proximity"].unique())
            - set(ocean_categories)
        )

        if unknown_categories:
            raise ValueError(
                f"Unexpected ocean_proximity categories: "
                f"{unknown_categories}"
            )

        X_df["ocean_proximity"] = pd.Categorical(
            X_df["ocean_proximity"],
            categories=ocean_categories
        )

        X_df = pd.get_dummies(
            X_df,
            columns=["ocean_proximity"],
            drop_first=True,
            dtype=float
        )

    # Keep a fixed feature ordering
    feature_columns = X_df.columns.tolist()
    A = X_df.to_numpy(dtype=float)

    # 5. Sample rows AFTER feature construction
    if data_percentage < 100:
        n_samples = max(
            1,
            int(len(b) * data_percentage / 100)
        )

        rng = np.random.RandomState(random_state)
        indices = rng.choice(
            len(b),
            size=n_samples,
            replace=False
        )

        A = A[indices]
        b = b[indices]

    # 6. Standardize features and target
    scaler = None
    scaler_y = None

    if scale:
        scaler = StandardScaler()
        A = scaler.fit_transform(A)

        scaler_y = StandardScaler()
        b = scaler_y.fit_transform(
            b.reshape(-1, 1)
        ).ravel()

    # 7. Ground-truth coefficients are defined externally
    x_true = None

    # 8. Return values in the original order
    return A, b, x_true, scaler, scaler_y


from plots import generate_all_plots

def evaluate_model(
    model,
    A_test,
    b_test,
):
    prediction = model.predict(A_test)

    metrics = {
        "MSE":mean_squared_error(b_test,prediction),

        "MAE":mean_absolute_error(b_test,prediction),

        "MedianResidual":median_absolute_residual(b_test,prediction),
    }
    return metrics



def make_lqs_model_mio(
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
    synthetic = False
    if synthetic:
        A, b, x_true = make_regression(
                n_samples=1000,
                n_features=20,
                noise=0.1,
                outlier_fraction=0,
                random_state=random_state,
            )
    else:
        scale = True
        A, b, x_true,scaler,scaler_y = load_california_housing_with_outliers(csv_path="data sets/housing.csv",data_percentage=100,scale=scale)
        x_true = np.array([
            -38842.167557, -35829.136219, 9856.017359, -21012.224327, 
            45615.736073, -39188.338792, 18967.609144, 77469.652853, 
            -20179.158814, 3106.787118, -2126.142932, 3015.681664
        ])
        if scale:
            x_true = x_true * scaler.scale_ / scaler_y.scale_[0]

    np.savetxt("A.csv", A)
    np.savetxt("x.csv",x_true)
    np.savetxt("b.csv",b)

    (A_train, A_test, b_train, b_test,) = train_test_split(A, b, test_size=0.2, random_state=random_state,)

    if outlier_fraction > 0:
        noise_scale = 500 * np.std(b_train)

        b_train = add_outliers(
            b_train,
            fraction=outlier_fraction,
            scale=noise_scale,
            random_state=random_state
        )
    signal = A_train @ x_true
    noise = b_train - signal

    snr_db = 10 * np.log10(np.var(signal) / np.var(noise))
    print(f"Signal to Noise Ratio(dB): {snr_db}")

    rng = np.random.default_rng(random_state)

    apmr = APMRRegression(
        alpha=0.5,
        lambda_=1.0,
        eta=0.25,
        trust_scaling=0.9,
        gamma=0.85,
        b_safeguard=1e8,
        inner_iter=100,
        max_iter=10000,
        tol=1e-10,
        lambda_max=100.0,
        inner_tol=1e-10,
        fit_intercept=True,
        initial_point=np.ones(A[0].shape),
        random_state=random_state,
        verbose=verbose,
    )

    run_methods = {
        "APMR": True,
        "LeastSquares": True,
        "LAD": True,
        "Huber": True,
        "TukeyLasso": True,
        "LMS": True,
        "LQS": True,
        "MIO": False,
        "RANSAC": True,
    }

    models = {}
    times_exec = {}

    # Approximate Proximal Median Regression (APMR) model
    if run_methods["APMR"]:
        start_time = time.time()
        apmr.fit(A_train, b_train)
        times_exec["APMR"] = time.time() - start_time
        models["APMR"] = apmr

    # Least Squares (LS) model
    if run_methods["LeastSquares"]:
        ls = LinearRegression()
        start_time = time.time()
        ls.fit(A_train, b_train)
        times_exec["LeastSquares"] = time.time() - start_time
        models["LeastSquares"] = ls

    # Least Absolute Deviations (LAD) model
    if run_methods["LAD"]:
        lad = QuantileRegressor(
            quantile=0.5,
            alpha=0.0,
            solver="highs",
        )
        start_time = time.time()
        lad.fit(A_train, b_train)
        times_exec["LAD"] = time.time() - start_time
        models["LAD"] = lad

    # Huber
    if run_methods["Huber"]:
        huber = HuberRegressor(
            epsilon=1.35,
            alpha=0.0,
            max_iter=500,
            tol=1e-3,
        )
        start_time = time.time()
        huber.fit(A_train, b_train)
        times_exec["Huber"] = time.time() - start_time
        models["Huber"] = huber

    # Tukey Lasso
    if run_methods["TukeyLasso"]:
        tukey_lasso = TukeyLasso(
            alpha=0.01,
            c=4.685,
            max_iter=5000,
            tol=1e-6,
        )
        start_time = time.time()
        tukey_lasso.fit(A_train, b_train)
        times_exec["TukeyLasso"] = time.time() - start_time
        models["TukeyLasso"] = tukey_lasso

    # LMS
    if run_methods["LMS"]:
        lms = make_lms_model(
            random_state=random_state,
            n_candidates=100,
        )
        start_time = time.time()
        lms.fit(A_train, b_train)
        times_exec["LMS"] = time.time() - start_time
        models["LMS"] = lms

    # LQS
    if run_methods["LQS"]:
        lqs = make_lqs_model(
            random_state=random_state,
            quantile=0.5,
            n_candidates=100,
        )
        start_time = time.time()
        lqs.fit(A_train, b_train)
        times_exec["LQS"] = time.time() - start_time
        models["LQS"] = lqs

    # MIO
    if run_methods["MIO"]:
        lqs_mio=make_lqs_model_mio(
            quantile=0.5,
            time_limit=10000,
            mip_gap=0.0,
            threads=0,
            verbose=verbose
        )
        start_time = time.time()
        lqs_mio.fit(A_train,b_train)
        times_exec["MIO"] = time.time() - start_time
        models["MIO"] = lqs_mio

    # RANSAC
    if run_methods["RANSAC"]:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=0.5,
            residual_threshold=None,
            max_trials=5000,
            max_skips=10000,
            stop_probability=0.999,
            random_state=random_state,
        )
        start_time = time.time()
        ransac.fit(A_train, b_train)
        times_exec["RANSAC"] = time.time() - start_time
        models["RANSAC"] = ransac

    results = []

    for name, model in models.items():

        metrics = evaluate_model(model, A_test, b_test,)
        metrics["Method"] = name
        metrics["OutlierFraction"] = (outlier_fraction)
        metrics["RandomState"] = (random_state)
        metrics["Time"] = times_exec[name]


        if name == "APMR":
            metrics["OuterIterations"] = (apmr.n_iter_)
            metrics["FinalObjective"] = (apmr.objective_)
            metrics["Converged"] = (apmr.converged_)
            metrics["AverageActiveSet"] = (np.mean(apmr.history["active_set_size"]))
            metrics["AverageInnerIterations"] = (np.mean(apmr.history["inner_iterations"]))
            metrics['_coef']=apmr.coef_.copy()


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
            metrics['_coef']=tukey_lasso.coef_.copy()


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
            metrics['_coef']=model.coef_.copy()
      

        elif name == "RANSAC":
            metrics["OuterIterations"] = ransac.n_trials_
            metrics["FinalObjective"] = np.nan
            metrics["Converged"] = ransac.n_trials_ < ransac.max_trials
            metrics["AverageActiveSet"] = np.nan
            metrics["AverageInnerIterations"] = np.nan
            metrics['_coef']=ransac.estimator_.coef_.copy()


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
            metrics['_coef']=model.coef_.copy()

        results.append(
            metrics
        )

    return (apmr,pd.DataFrame(results),A_test,b_test,x_true,apmr)

def benchmark():
    """
    Run all experiments.
    """
    fractions = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5
    ]

    seeds = [
        0, 
        1,
        2,
        3,
        4
   ]

    frames = []

    apmr_model = None
    A_test = None
    b_test = None

    for seed in seeds:
       for fraction in fractions:

            print("Running experiment | "f"outliers={fraction} "f"seed={seed}")

            (model,table,A_test_run,b_test_run,x_true,ls) = run_single_experiment(fraction,random_state=seed,verbose=False)

            frames.append(table)

            if seed == seeds[0] and fraction == fractions[-1]:                
                apmr_model = model
                A_test = A_test_run
                b_test = b_test_run

    results = pd.concat(frames, ignore_index=True)

    results.to_csv("results.csv", index=False,sep='\t')
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
    summary.to_csv("summary.csv", index=False)


    return (apmr_model, A_test, b_test, results,x_true,ls)

def main():
    (apmr_model,A_test,b_test,results,x_true,ls) = benchmark()

    print(results)

    generate_all_plots(apmr_model, A_test, b_test, results,x_true,ls)

if __name__ == "__main__":
    main()