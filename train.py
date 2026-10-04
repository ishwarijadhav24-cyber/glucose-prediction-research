import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import warnings

# Suppress specific warnings about empty features
warnings.filterwarnings('ignore', message='Skipping features without any observed values')

# Local imports
from data_processing import load_subject_data, create_features

def evaluate_model(y_true, y_pred):
    """Return RMSE, MAE and R² for predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def clarke_error_grid(y_true, y_pred):
    """Simple Clarke Error Grid classification.
    Returns percentages for zones A and B.
    """
    a_count = b_count = 0
    total = len(y_true)
    for gt, pr in zip(y_true, y_pred):
        if gt <= 100:
            if abs(pr - gt) <= 18:
                a_count += 1
            elif abs(pr - gt) <= 0.2 * gt:
                b_count += 1
        else:
            if abs(pr - gt) <= 0.2 * gt:
                a_count += 1
            elif abs(pr - gt) <= 0.2 * gt + 20:
                b_count += 1
    a_perc = 100 * a_count / total
    b_perc = 100 * b_count / total
    return a_perc, b_perc

def main():
    # Choose a subject for a quick sanity‑check; the full benchmark uses all subjects inside load_subject_data.
    subject_id = "559"
    data_dir = os.path.join(os.getcwd(), "data", "OhioT1DM_2018")
    df = load_subject_data(subject_id, data_dir=data_dir)
    features_df = create_features(df)

    X = features_df.drop(columns=["target", "ts"]).values
    y = features_df["target"].values

    # Report NaNs before imputation
    print(f"NaN values in X before imputation: {np.isnan(X).sum()}")

    # Train‑test split (80/20)
    split_idx = int(0.8 * len(features_df))
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Impute missing values – fit only on training data
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)
    # Ensure completely empty columns (e.g., heart_rate) become zeros
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"NaN after imputation (train, test): {np.isnan(X_train).sum()}, {np.isnan(X_test).sum()}")

    results = []

    # Persistence baseline (glucose 6 steps earlier)
    pers_pred = features_df["glucose"].values[split_idx - 6 : -6]
    rmse, mae, r2 = evaluate_model(y_test, pers_pred)
    a_perc, b_perc = clarke_error_grid(y_test, pers_pred)
    results.append(["Persistence", rmse, mae, r2, a_perc, b_perc])

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    rmse, mae, r2 = evaluate_model(y_test, pred)
    a_perc, b_perc = clarke_error_grid(y_test, pred)
    results.append(["Ridge", rmse, mae, r2, a_perc, b_perc])

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=4)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    rmse, mae, r2 = evaluate_model(y_test, pred)
    a_perc, b_perc = clarke_error_grid(y_test, pred)
    results.append(["RandomForest", rmse, mae, r2, a_perc, b_perc])

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    pred = mlp.predict(X_test)
    rmse, mae, r2 = evaluate_model(y_test, pred)
    a_perc, b_perc = clarke_error_grid(y_test, pred)
    results.append(["MLP", rmse, mae, r2, a_perc, b_perc])

    # LightGBM with callback API
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.05,
              "num_leaves": 31, "seed": 42, "verbose": -1}
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_eval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    rmse, mae, r2 = evaluate_model(y_test, pred)
    a_perc, b_perc = clarke_error_grid(y_test, pred)
    results.append(["LightGBM", rmse, mae, r2, a_perc, b_perc])

    # Stacking ensemble (Ridge + RF + LightGBM)
    estimators = [
        ("ridge", Ridge(alpha=1.0)),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=4)),
        ("lgb", lgb.LGBMRegressor(**params, n_estimators=500)),
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    stack.fit(X_train, y_train)
    pred = stack.predict(X_test)
    rmse, mae, r2 = evaluate_model(y_test, pred)
    a_perc, b_perc = clarke_error_grid(y_test, pred)
    results.append(["Stacking", rmse, mae, r2, a_perc, b_perc])

    # Leaderboard
    leaderboard = pd.DataFrame(
        results,
        columns=["Model", "RMSE (mg/dL)", "MAE (mg/dL)", "R2", "EGA Zone A %", "EGA Zone B %"],
    ).sort_values("RMSE (mg/dL)")
    print("=== Model Benchmark Leaderboard ===")
    print(leaderboard.to_string(index=False))

    # Save the best model
    best_model_name = leaderboard.iloc[0]["Model"]
    print(f"\nBest model: {best_model_name}")
    model_map = {
        "Ridge": ridge,
        "RandomForest": rf,
        "MLP": mlp,
        "LightGBM": gbm,
        "Stacking": stack,
    }
    best_model = model_map.get(best_model_name)
    if best_model is not None:
        artifact_path = os.path.join(os.getcwd(), "model_artifact.pkl")
        joblib.dump(
            {
                "model": best_model,
                "features": list(features_df.drop(columns=["target", "ts"]).columns),
                "imputer": imputer,
                "iob_params": {"dia": 240, "t_peak": 60},
            },
            artifact_path,
        )
        print(f"Saved best model to {artifact_path}")
    else:
        print("Best model is Persistence baseline; nothing to save.")

if __name__ == "__main__":
    main()
