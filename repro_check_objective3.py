"""
Reproducibility check for Objective 3: retrain Ridge and LightGBM on 11 OhioT1DM
patients (all except 559), evaluate on 559 exactly as evaluate_loso.py does
(common mask: target and glucose at t finite), and compare with
results_causal/loso_patient_results.csv (read only).
"""

import json
import os

import numpy as np
import pandas as pd

from evaluate_loso import calculate_mard, calculate_rmse
from objective3_common import (apply_preprocessing, build_model, feature_columns,
                               fit_preprocessing, load_ohio_features)

HELD_OUT = "559"
TOLERANCE_MARD = 0.5

data = load_ohio_features()
cols = feature_columns(data)
train = pd.concat([d for p, d in data.items() if p != HELD_OUT], ignore_index=True)
test = data[HELD_OUT]
imputer, scaler, X_train = fit_preprocessing(train[cols].values)
X_test = apply_preprocessing(imputer, scaler, test[cols].values)
y_test = test["target"].values
mask = np.isfinite(y_test) & np.isfinite(test["glucose"].values)

ref = pd.read_csv("results_causal/loso_patient_results.csv")
ref = ref[ref["patient"].astype(str) == HELD_OUT].set_index("model")

rows = []
for name in ("Ridge", "LightGBM"):
    model = build_model(name)
    model.fit(X_train, train["target"].values)
    pred = model.predict(X_test)[mask]
    mard, rmse = calculate_mard(y_test[mask], pred), calculate_rmse(y_test[mask], pred)
    rows.append({"model": name, "n": int(mask.sum()), "n_obj1": int(ref.loc[name, "n_test_samples"]),
                 "MARD_now": mard, "MARD_obj1": ref.loc[name, "MARD"], "MARD_diff": mard - ref.loc[name, "MARD"],
                 "RMSE_now": rmse, "RMSE_obj1": ref.loc[name, "RMSE"], "RMSE_diff": rmse - ref.loc[name, "RMSE"]})

res = pd.DataFrame(rows)
res["PASS"] = res["MARD_diff"].abs() <= TOLERANCE_MARD
os.makedirs("results_objective3", exist_ok=True)
res.to_csv("results_objective3/reproducibility_check_559.csv", index=False)
print(res.round(4).to_string(index=False))
print("REPRODUCIBILITY:", "PASS" if res["PASS"].all() else "FAIL")
