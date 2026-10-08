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
def load_california_housing_with_outliers(csv_path):
    """
    Loads California Housing, preprocesses features, and injects outliers.
    """
    # 1. Load data
    df = pd.read_csv(csv_path)
    
    # 2. Handle missing data (total_bedrooms has NaNs)
    df = df.dropna().reset_index(drop=True)
    
    # 3. Separate target (b)
    b = df['median_house_value'].values
    
    # 4. Separate features (A) and handle categorical ocean_proximity
    X_df = df.drop(columns=['median_house_value'])
    X_df = pd.get_dummies(X_df, columns=['ocean_proximity'], drop_first=True, dtype=float)
    A = X_df.values
    
    # 5. Standardize A (Crucial for APMR step norms and gradient calculations)
    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    scaler_y = StandardScaler()
    b = scaler_y.fit_transform(b.reshape(-1, 1)).flatten()
    
    x_true = None 
    
    return A, b, x_true,scaler,scaler_y

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
                n_features=500,
                noise=0.1,
                outlier_fraction=0,
                random_state=
                    random_state,
            )
    else:
        A, b, x_true,scaler,scaler_y = load_california_housing_with_outliers(csv_path="data sets/housing.csv")
        x_true = np.array([
            -38842.167557, -35829.136219, 9856.017359, -21012.224327, 
            45615.736073, -39188.338792, 18967.609144, 77469.652853, 
            -20179.158814, 3106.787118, -2126.142932, 3015.681664
        ])
        x_true = x_true * scaler.scale_ / scaler_y.scale_[0]

    
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
        alpha=50.0,             # Stable forward thrust
        lambda_=10.0,           
        eta=1.0,               
        
        # 1. FEED THE MEMORY: Stop the base step halfway to the wall. 
        # This leaves a massive 50% of unexploited room (c_k) to inject into E.
        trust_scaling=0.9,    
        
        # 2. RETAIN THE MEMORY: Keep 60% of the historical momentum. 
        # Combined with the massive c_k from above, E will grow incredibly powerful.
        gamma=0.85,             
        
        # 3. OPEN THE GATE: Since we are in a dense piecewise-linear space, 
        # completely remove the artificial H2 mathematical choke.
        b_safeguard=1e10,       
        
        inner_iter=100,        
        max_iter=1000,
        tol=1e-8,              
        inner_tol=1e-3,
        random_state=random_state,
        initial_point=np.zeros(A_train.shape[1]), 
        verbose=True
    )

    times_exec=[]

    # Approximate Proximal Median Regression (APMR) model
    start_time = time.time()
    apmr.fit( A_train, b_train)
    end_time = time.time()
    times_exec.append(end_time - start_time)

    # Least Squares (LS) model
    ls = LinearRegression()
    start_time = time.time()
    ls.fit(A_train,b_train)
    end_time = time.time()
    times_exec.append(end_time - start_time)

    # Least Absolute Deviations (LAD) model
    lad = QuantileRegressor(
        quantile=0.5,
        alpha=0.0,
        solver="highs",
    )
    start_time = time.time()
    lad.fit(A_train,b_train)
    end_time = time.time()
    times_exec.append(end_time - start_time)

    # Huber model
    huber = HuberRegressor(
        epsilon=1.35,
        alpha=0.0,
        max_iter=1000,
        tol=1e-6,
    )
    start_time = time.time()
    huber.fit(
        A_train,
        b_train,
    )
    end_time = time.time()
    times_exec.append(end_time - start_time)

    # Tukey Lasso model
    tukey_lasso = TukeyLasso(
        alpha=0.01,
        c=4.685,
        max_iter=5000,
        tol=1e-6,
    )
    start_time = time.time()
    tukey_lasso.fit(
        A_train,
        b_train,
    )
    end_time = time.time()
    times_exec.append(end_time - start_time)

    # Least Median of Squares (LMS) model
    lms = make_lms_model(
        random_state=random_state,
        n_candidates=100,
    )
    start_time = time.time()
    lms.fit(
        A_train,
        b_train,
    )
    end_time = time.time()
    times_exec.append(end_time - start_time)

    # Least Quantile of Squares (LQS) model
    lqs = make_lqs_model(
        random_state=random_state,
        quantile=0.5,
        n_candidates=100,
    )
    start_time = time.time()
    lqs.fit(
        A_train,
        b_train,
    )
    end_time = time.time()
    times_exec.append(end_time - start_time)

    # MIO model
    # lqs_mio=make_lqs_model_mio(
    #     quantile=0.5,
    #     time_limit=times_exec[0],
    #     mip_gap=0.0,
    #     threads=0,
    #     verbose=verbose
    # )
    # start_time = time.time()
    # lqs_mio.fit(A_train,b_train)
    # end_time = time.time()
    # times_exec.append(end_time - start_time)


    # RANSAC model
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
    end_time = time.time()
    times_exec.append(end_time - start_time)

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
        # ("MIO",lqs_mio)
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
            metrics["Time"] = (times_exec[0])
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
            metrics["Time"] = (times_exec[4])
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
            metrics["Time"] = (times_exec[5]
                if name == "LMS"
                else times_exec[6]) 

        elif name == "RANSAC":
            metrics["OuterIterations"] = ransac.n_trials_
            metrics["FinalObjective"] = np.nan
            metrics["Converged"] = ransac.n_trials_ < ransac.max_trials
            metrics["AverageActiveSet"] = np.nan
            metrics["AverageInnerIterations"] = np.nan
            metrics['_coef']=ransac.estimator_.coef_.copy()
            # metrics["Time"] = (times_exec[8])
            metrics["Time"] = (times_exec[7])



        # elif name == "MIO":
        #     metrics["OuterIterations"] = lqs_mio.node_count_
        #     metrics["FinalObjective"] = lqs_mio.objective_
        #     metrics["Converged"] = lqs_mio.converged_
        #     metrics["AverageActiveSet"] = np.sum(lqs_mio.z_)
        #     metrics["AverageInnerIterations"] = np.nan
        #     metrics["Time"] = (times_exec[7])

        else:
            metrics["OuterIterations"] = np.nan
            metrics["FinalObjective"] = np.nan
            metrics["Converged"] = np.nan
            metrics["AverageActiveSet"] = np.nan
            metrics["AverageInnerIterations"] = np.nan
            metrics['_coef']=model.coef_.copy()
            if name == "LeastSquares":
                metrics["Time"] = (times_exec[1])
            elif name == "LAD":
                metrics["Time"] = (times_exec[2])
            elif name == "Huber":
                metrics["Time"] = (times_exec[3])

        results.append(
            metrics
        )

    return (apmr,pd.DataFrame(results),A_test,b_test,x_true,ls)

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
        0, 
        # # 1, 
        # # 2, 
        # # 3, 
        # 4
   ]

    frames = []

    apmr_model = None
    A_test = None
    b_test = None

    for seed in seeds:
       for fraction in fractions:

            print("Running experiment | "f"outliers={fraction:.1f} "f"seed={seed}")

            (model,table,A_test_run,b_test_run,x_true,ls) = run_single_experiment(fraction,random_state=seed,verbose=False)

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
    summary.to_csv("summary.csv", index=False)


    return (apmr_model, A_test, b_test, results,x_true,ls)

def main():
    (apmr_model,A_test,b_test,results,x_true,ls) = benchmark()

    print(results)

    generate_all_plots(apmr_model, A_test, b_test, results,x_true,ls)

if __name__ == "__main__":
    main()