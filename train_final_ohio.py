"""
Objective 3, Phase 3: train the final models on ALL 12 OhioT1DM patients (all data).

Same 6 models, hyperparameters, preprocessing and random_state as Objective 1
(evaluate_loso.py; settings copied in objective3_common.py because the models are
built inline there). The imputer and scaler are fitted on the OhioT1DM training
data only. Only OhioT1DM data is used: no external data is read here.
"""

import hashlib
import json
import os
import platform
import sys

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

from objective3_common import (MODEL_NAMES, MODEL_SETTINGS, OHIO_DIR, RANDOM_STATE, TRAINED_MODELS,
                               build_model, feature_columns, fit_preprocessing, load_ohio_features)

MODELS_DIR = "models_objective3"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    data = load_ohio_features(OHIO_DIR)
    cols = feature_columns(data)
    train = pd.concat(data.values(), ignore_index=True)
    X_raw, y = train[cols].values, train["target"].values
    print(f"Training on {len(data)} OhioT1DM patients, {len(y):,} samples, {len(cols)} features")

    imputer, scaler, X = fit_preprocessing(X_raw)
    joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    for name in TRAINED_MODELS:
        model = build_model(name)
        model.fit(X, y)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
        print(f"  {name} trained and saved")

    with open(os.path.join(MODELS_DIR, "feature_list.json"), "w", encoding="utf-8") as fh:
        json.dump(cols, fh, indent=1)
    settings = {
        "training_data": f"OhioT1DM, all 12 patients, all files in {OHIO_DIR} (training + testing XML)",
        "patients": sorted(data.keys()),
        "n_training_samples": int(len(y)),
        "n_features": len(cols),
        "models": MODEL_NAMES,
        "model_settings": MODEL_SETTINGS,
        "random_state": RANDOM_STATE,
        "versions": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                     "scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "python_executable": sys.executable,
        "note": "Objective 1 was run in a different environment; its LightGBM version was not recorded.",
    }
    with open(os.path.join(MODELS_DIR, "settings.json"), "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=1)

    hashes = {f: sha256(os.path.join(MODELS_DIR, f)) for f in sorted(os.listdir(MODELS_DIR))
              if f != "model_hashes.json"}
    with open(os.path.join(MODELS_DIR, "model_hashes.json"), "w", encoding="utf-8") as fh:
        json.dump(hashes, fh, indent=1)
    print(f"Saved {len(hashes)} files with SHA256 hashes")


if __name__ == "__main__":
    main()
