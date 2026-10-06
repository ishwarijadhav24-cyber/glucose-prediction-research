"""
Objective 3 shared training code (OhioT1DM only).

evaluate_loso.py builds its models inline inside run_loso_evaluation(), so they
cannot be imported. The settings below are copied EXACTLY from evaluate_loso.py
(lines 278-370). Data loading, feature engineering, metrics and the Clarke grid
are imported unchanged from evaluate_loso.py / data_processing.py.
"""

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from evaluate_loso import discover_patient_ids, load_all_patients

OHIO_DIR = "data/OhioT1DM"
RANDOM_STATE = 42
MODEL_NAMES = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"]
TRAINED_MODELS = MODEL_NAMES[1:]

MODEL_SETTINGS = {
    "imputer": "SimpleImputer(strategy='mean', keep_empty_features=True) + np.nan_to_num(nan=0.0)",
    "scaler": "StandardScaler()",
    "Ridge": {"alpha": 1.0},
    "RandomForest": {"n_estimators": 50, "max_depth": 15, "min_samples_leaf": 5, "max_samples": 0.5, "n_jobs": -1},
    "MLP": {"hidden_layer_sizes": [128, 64], "max_iter": 150, "early_stopping": True, "n_iter_no_change": 10},
    "LightGBM": {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
                 "n_jobs": -1, "verbose": -1, "early_stopping": "none (plain fit, as Objective 1)"},
    "Stacking": {"estimators": "Ridge(1.0) + RF(30, depth 12, leaf 5, max_samples 0.5) + LGBM(100, lr 0.05, 31 leaves)",
                 "final_estimator": "Ridge(alpha=1.0)", "cv": 3, "n_jobs": 1},
    "Persistence": "prediction = glucose at time t",
    "random_state": RANDOM_STATE,
}


def load_ohio_features(data_dir=OHIO_DIR):
    """{patient: create_features(load_subject_data(patient))}, exactly as Objective 1 (no cache)."""
    return load_all_patients(discover_patient_ids(data_dir), data_dir, cache_file=None)


def feature_columns(patient_data):
    sample_df = next(iter(patient_data.values()))
    return [c for c in sample_df.columns if c not in ["target", "ts"]]


def fit_preprocessing(X_train_raw):
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X_imp = np.nan_to_num(imputer.fit_transform(X_train_raw), nan=0.0)
    scaler = StandardScaler()
    return imputer, scaler, scaler.fit_transform(X_imp)


def apply_preprocessing(imputer, scaler, X_raw):
    return scaler.transform(np.nan_to_num(imputer.transform(X_raw), nan=0.0))


def build_model(name, random_state=RANDOM_STATE):
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=random_state)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=50, max_depth=15, min_samples_leaf=5, max_samples=0.5,
                                     n_jobs=-1, random_state=random_state)
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=150, early_stopping=True,
                            n_iter_no_change=10, random_state=random_state)
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                 n_jobs=-1, random_state=random_state, verbose=-1)
    if name == "Stacking":
        estimators = [
            ("ridge", Ridge(alpha=1.0, random_state=random_state)),
            ("rf", RandomForestRegressor(n_estimators=30, max_depth=12, min_samples_leaf=5, max_samples=0.5,
                                         n_jobs=-1, random_state=random_state)),
            ("lgb", lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, n_jobs=-1,
                                      random_state=random_state, verbose=-1)),
        ]
        return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0, random_state=random_state),
                                 cv=3, n_jobs=1)
    raise ValueError(f"Unknown model {name!r}")
