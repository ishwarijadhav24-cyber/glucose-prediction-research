"""
evaluate_loso.py - True 12-Patient Leave-One-Subject-Out (LOSO) Cross-Validation
Rigorous evaluation for Research Objective 1: Subject-Independent Glucose Prediction.

Key Enhancements & Specifications:
1. Resumable & Checkpointed:
   - Checkpoint saved to `results/loso_checkpoint.json` after every completed fold.
   - Per-patient fold predictions saved immediately to `results/fold_{pid}_predictions.csv`.
   - Per-patient evaluation metrics updated immediately to `results/loso_patient_results.csv`.
   - On startup, automatically detects completed folds and resumes seamlessly from the next patient.
2. Memory Efficient:
   - Evaluates one patient fold at a time, writes predictions to disk, releases memory via gc.collect().
3. Computationally Optimized:
   - RF: 50 trees, max_depth=15, min_samples_leaf=5, max_samples=0.5, n_jobs=-1, random_state=42.
   - MLP: (128, 64), max_iter=150, early_stopping=True, n_iter_no_change=10, random_state=42.
   - LightGBM: 150 trees, lr=0.05, num_leaves=31, min_child_samples=20, n_jobs=-1, random_state=42.
   - Stacking: Ridge + RF(30 trees) + LightGBM(100 trees) with Ridge meta-estimator, cv=3, n_jobs=1.
   - Persistence: Causal naive baseline y(t) -> y(t+30).
4. Scientifically Rigorous:
   - All 12 OhioT1DM patients (540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596).
   - Zero Data Leakage: Preprocessing (SimpleImputer, StandardScaler) fitted ONLY on the 11 training patients.
   - Standard Clarke Error Grid Analysis (Clarke et al., 1987) with published zone boundary equations.
   - Patient-level aggregation: Mean +/- SD calculated across the 12 patient folds.
"""

import os
import gc
import glob
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*empty features.*')

from data_processing import load_subject_data, create_features

# Attempt import of verified error_grids package; fallback to exact canonical implementation
try:
    from error_grids import clarke_error_zone_detailed
except ImportError:
    def _clarke_zone(act, pred):
        # Zone A
        if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
            return 0
        # Zone E - left upper (act hypo, pred hyper)
        if act <= 70 and pred >= 180:
            return 8
        # Zone E - right lower (act hyper, pred hypo)
        if act >= 180 and pred <= 70:
            return 7
        # Zone D - right (failure to detect hyper)
        if act >= 240 and 70 <= pred <= 180:
            return 6
        # Zone D - left (failure to detect hypo)
        if act <= 70 <= pred <= 180:
            return 5
        # Zone C - upper (overcorrecting normal/mild to hyper)
        if 70 <= act <= 290 and pred >= act + 110:
            return 4
        # Zone C - lower (overcorrecting normal/mild to hypo)
        if 130 <= act <= 180 and pred <= (7/5) * act - 182:
            return 3
        # Zone B - upper
        if act < pred:
            return 2
        # Zone B - lower
        return 1

    clarke_error_zone_detailed = np.vectorize(_clarke_zone)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def calculate_r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def calculate_mard(y_true, y_pred, eps=1e-6):
    """
    Mean Absolute Relative Difference (MARD) in percentage:
    MARD = 100% * (1/N) * sum(|y_true - y_pred| / max(|y_true|, eps))
    """
    rel_errors = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))
    return float(100.0 * np.mean(rel_errors))


def evaluate_clarke_ega(y_true, y_pred):
    """
    Standard Clarke Error Grid Analysis (Clarke et al., Diabetes Care 1987).
    Categorizes each point into Zones A, B, C, D, E.
    Returns: (zone_A_pct, zone_B_pct, zone_AB_pct, zone_labels)
    """
    raw_zones = clarke_error_zone_detailed(y_true, y_pred)
    n = len(raw_zones)
    # Zone 0 -> A; Zone 1, 2 -> B; Zone 3, 4 -> C; Zone 5, 6 -> D; Zone 7, 8 -> E
    a_count = int(np.sum(raw_zones == 0))
    b_count = int(np.sum((raw_zones == 1) | (raw_zones == 2)))
    a_pct = float(100.0 * a_count / n)
    b_pct = float(100.0 * b_count / n)
    ab_pct = float(a_pct + b_pct)

    # Convert to letter labels for plotting/export
    label_map = {0: 'A', 1: 'B', 2: 'B', 3: 'C', 4: 'C', 5: 'D', 6: 'D', 7: 'E', 8: 'E'}
    labels = [label_map.get(z, 'B') for z in raw_zones]
    return a_pct, b_pct, ab_pct, labels


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                return data.get("completed_patients", [])
        except Exception as e:
            print(f"[Warning] Failed to read checkpoint {checkpoint_path}: {e}")
    return []


def save_checkpoint(checkpoint_path, completed_patients, random_state=42):
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({
            "completed_patients": sorted(list(set(completed_patients))),
            "random_state": random_state,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    os.replace(tmp_path, checkpoint_path)


# ---------------------------------------------------------------------------
# Patient Discovery & Pre-caching
# ---------------------------------------------------------------------------
def discover_patient_ids(data_dir):
    files = glob.glob(os.path.join(data_dir, "*-ws-*.xml"))
    patient_ids = sorted(list(set(os.path.basename(f).split('-')[0] for f in files)))
    return patient_ids


def load_all_patients(patient_ids, data_dir, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        try:
            import pickle
            print(f"Loading pre-computed causal patient features from {cache_file}...")
            with open(cache_file, "rb") as f:
                patient_data = pickle.load(f)
            total_samples = sum(len(df) for df in patient_data.values())
            print(f"Loaded {len(patient_data)} patients from cache. Total samples: {total_samples:,}\n")
            return patient_data
        except Exception as e:
            print(f"[Warning] Failed to load cache {cache_file}: {e}. Recomputing from raw files...")

    patient_data = {}
    print("=" * 78)
    print(f"Loading and generating features for {len(patient_ids)} OhioT1DM patients...")
    print("=" * 78)

    t0 = time.time()
    for pid in patient_ids:
        raw_df = load_subject_data(pid, data_dir=data_dir)
        feat_df = create_features(raw_df)
        patient_data[pid] = feat_df
        print(f"  Patient {pid}: {len(raw_df):6d} raw records -> {len(feat_df):6d} feature rows (target: 30m ahead)")

    total_samples = sum(len(df) for df in patient_data.values())
    print(f"\nAll {len(patient_ids)} patients cached in {time.time() - t0:.1f}s. Total samples: {total_samples:,}\n")

    if cache_file:
        try:
            import pickle
            with open(cache_file, "wb") as f:
                pickle.dump(patient_data, f)
            print(f"Saved patient features cache to {cache_file}")
        except Exception as e:
            print(f"[Warning] Failed to save cache {cache_file}: {e}")

    return patient_data


# ---------------------------------------------------------------------------
# LOSO Evaluation Engine
# ---------------------------------------------------------------------------
def run_loso_evaluation(data_dir="data/OhioT1DM", results_dir="results", outputs_dir="outputs", random_state=42):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    patient_ids = discover_patient_ids(data_dir)
    print(f"Discovered {len(patient_ids)} patients: {patient_ids}")
    if len(patient_ids) != 12:
        warnings.warn(f"Expected 12 patients, found {len(patient_ids)}.")

    checkpoint_path = os.path.join(results_dir, "loso_checkpoint.json")
    completed_patients = load_checkpoint(checkpoint_path)
    # Ensure checkpoint file exists immediately for status inspection
    save_checkpoint(checkpoint_path, completed_patients, random_state=random_state)

    patient_results_path = os.path.join(results_dir, "loso_patient_results.csv")
    if os.path.exists(patient_results_path):
        df_existing_results = pd.read_csv(patient_results_path)
    else:
        df_existing_results = pd.DataFrame(columns=[
            "patient", "model", "RMSE", "MAE", "R2", "MARD", "EGA_A", "EGA_B", "EGA_A+B", "n_test_samples"
        ])

    print("=" * 78)
    print(f"CHECKPOINT STATUS: {len(completed_patients)}/12 completed folds: {completed_patients}")
    remaining_patients = [p for p in patient_ids if p not in completed_patients]
    print(f"REMAINING PATIENTS ({len(remaining_patients)}): {remaining_patients}")
    print("=" * 78)

    # Pre-cache patient data (strict per-patient feature engineering)
    cache_path = os.path.join(results_dir, "patient_features_causal.pkl")
    patient_data = load_all_patients(patient_ids, data_dir, cache_file=cache_path)
    sample_df = next(iter(patient_data.values()))
    feature_cols = [c for c in sample_df.columns if c not in ["target", "ts"]]
    print(f"Total features per record ({len(feature_cols)}): {feature_cols[:6]}... (+ {len(feature_cols)-6} more)")

    model_names = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"]

    # -----------------------------------------------------------------------
    # 12-Fold LOSO Loop (Skips Completed)
    # -----------------------------------------------------------------------
    total_start = time.time()

    for fold_idx, held_out in enumerate(patient_ids, 1):
        if held_out in completed_patients:
            print(f"\n[Fold {fold_idx:02d}/12] Patient {held_out} already completed in checkpoint. Skipping.")
            continue

        fold_start = time.time()
        print("\n" + "=" * 78)
        print(f"[Fold {fold_idx:02d}/12] Starting held-out test patient: {held_out}")
        print("=" * 78)

        # Build training set from other 11 patients
        train_dfs = [patient_data[p] for p in patient_ids if p != held_out]
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = patient_data[held_out]

        X_train_raw = train_df[feature_cols].values
        y_train = train_df["target"].values
        X_test_raw = test_df[feature_cols].values
        y_test = test_df["target"].values
        n_test = len(y_test)

        print(f"  Training samples: {len(y_train):,} | Test samples: {n_test:,}")

        # -------------------------------------------------------------------
        # Preprocessing: Fit ONLY on training fold (zero leakage)
        # -------------------------------------------------------------------
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        X_train_imp = np.nan_to_num(imputer.fit_transform(X_train_raw), nan=0.0)
        X_test_imp = np.nan_to_num(imputer.transform(X_test_raw), nan=0.0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_imp)
        X_test = scaler.transform(X_test_imp)

        # -------------------------------------------------------------------
        # Model 1: Persistence Baseline
        # Strictly causal: predicts y(t+30) using current glucose at t
        # -------------------------------------------------------------------
        t_m = time.time()
        pred_persistence = test_df["glucose"].values
        print(f"  [1/6] Persistence baseline ready in {time.time() - t_m:.2f}s")

        # -------------------------------------------------------------------
        # Model 2: Ridge Regression
        # -------------------------------------------------------------------
        t_m = time.time()
        ridge = Ridge(alpha=1.0, random_state=random_state)
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)
        print(f"  [2/6] Ridge fitted in {time.time() - t_m:.2f}s")

        # -------------------------------------------------------------------
        # Model 3: Random Forest Regressor
        # Optimized: 50 trees, max_depth=15, min_samples_leaf=5, max_samples=0.5
        # -------------------------------------------------------------------
        t_m = time.time()
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=15,
            min_samples_leaf=5,
            max_samples=0.5,
            n_jobs=-1,
            random_state=random_state
        )
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        print(f"  [3/6] RandomForest fitted in {time.time() - t_m:.2f}s")

        # -------------------------------------------------------------------
        # Model 4: Multi-Layer Perceptron (MLP)
        # Optimized: (128, 64), max_iter=150, early_stopping=True
        # -------------------------------------------------------------------
        t_m = time.time()
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=150,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_state
        )
        mlp.fit(X_train, y_train)
        pred_mlp = mlp.predict(X_test)
        print(f"  [4/6] MLP fitted in {time.time() - t_m:.2f}s")

        # -------------------------------------------------------------------
        # Model 5: LightGBM Regressor
        # Optimized: 150 trees, lr=0.05, num_leaves=31, min_child_samples=20
        # -------------------------------------------------------------------
        t_m = time.time()
        lgbm = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
        lgbm.fit(X_train, y_train)
        pred_lgb = lgbm.predict(X_test)
        print(f"  [5/6] LightGBM fitted in {time.time() - t_m:.2f}s")

        # -------------------------------------------------------------------
        # Model 6: Stacking Regressor
        # Base estimators: Ridge, RF(30 trees), LightGBM(100 trees)
        # Final estimator: Ridge. cv=3. Avoids nested parallelism deadlock.
        # -------------------------------------------------------------------
        t_m = time.time()
        stack_estimators = [
            ("ridge", Ridge(alpha=1.0, random_state=random_state)),
            ("rf", RandomForestRegressor(n_estimators=30, max_depth=12, min_samples_leaf=5, max_samples=0.5, n_jobs=-1, random_state=random_state)),
            ("lgb", lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=random_state, verbose=-1)),
        ]
        stack = StackingRegressor(
            estimators=stack_estimators,
            final_estimator=Ridge(alpha=1.0, random_state=random_state),
            cv=3,
            n_jobs=1
        )
        stack.fit(X_train, y_train)
        pred_stack = stack.predict(X_test)
        print(f"  [6/6] Stacking fitted in {time.time() - t_m:.2f}s")

        fold_preds = {
            "Persistence": pred_persistence,
            "Ridge": pred_ridge,
            "RandomForest": pred_rf,
            "MLP": pred_mlp,
            "LightGBM": pred_lgb,
            "Stacking": pred_stack,
        }

        # -------------------------------------------------------------------
        # Fold Diagnostics & Common Evaluation Mask
        # Causal preprocessing preserves real-world CGM dropouts (>30m), leaving
        # exactly those current glucose timestamps as NaN. For fair, rigorous
        # benchmarking, all models are evaluated on the exact same common mask
        # where both ground truth and the persistence reference are available.
        # -------------------------------------------------------------------
        X_test_processed = X_test
        print("\n" + "=" * 80)
        print(f"=== FOLD {fold_idx} ({held_out}) DIAGNOSTICS & COMMON EVALUATION MASK ===")
        print("=" * 80)
        print("y_test NaN:", np.isnan(y_test).sum())
        print("X_test BEFORE imputation NaN:", np.isnan(X_test_raw).sum())
        print("X_test_processed NaN:", np.isnan(X_test_processed).sum())
        print("X_test_processed finite:", np.isfinite(X_test_processed).all())
        print("raw test_df['glucose'] NaN:", np.isnan(test_df['glucose'].values).sum())
        print("-" * 80)

        # 1. Create common valid evaluation mask for the fold
        common_mask = np.isfinite(y_test) & np.isfinite(pred_persistence)
        n_total_test = len(y_test)
        n_eval = int(np.sum(common_mask))
        n_excluded = n_total_test - n_eval

        print(f"Total test rows: {n_total_test}")
        print(f"Common evaluation rows: {n_eval}")
        print(f"Excluded rows: {n_excluded}")
        print("Reason: current glucose unavailable for Persistence")
        print("-" * 80)

        # 2. Check each model's full prediction array and finite status on common_mask
        print("Model prediction diagnostics:")
        for model_name in model_names:
            preds_all = np.asarray(fold_preds[model_name])
            finite_all = preds_all[np.isfinite(preds_all)]
            preds_on_mask = preds_all[common_mask]
            is_finite_on_mask = bool(np.all(np.isfinite(preds_on_mask)))

            print(
                f"  {model_name:<15}: n={len(preds_all)}, "
                f"NaN={np.isnan(preds_all).sum()}, "
                f"+/-Inf={np.isinf(preds_all).sum()}, "
                f"min={finite_all.min() if len(finite_all) else 'NONE':.4f}, "
                f"max={finite_all.max() if len(finite_all) else 'NONE':.4f} | "
                f"finite on common_mask: {is_finite_on_mask}"
            )
        print("-" * 80)

        # -------------------------------------------------------------------
        # Evaluate All Models & Print Table using common_mask
        # -------------------------------------------------------------------
        print(f"\n  Fold {fold_idx} ({held_out}) Performance on Common Mask (n={n_eval}):")
        print(f"  {'Model':<15} {'MARD (%)':>10} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'EGA A%':>8} {'EGA B%':>8} {'EGA A+B%':>10}")
        print("  " + "-" * 76)

        fold_metrics = []
        fold_prediction_rows = []

        y_eval = y_test[common_mask]

        for m_name in model_names:
            preds_all = np.asarray(fold_preds[m_name])
            preds_eval = preds_all[common_mask]

            rmse = calculate_rmse(y_eval, preds_eval)
            mae = calculate_mae(y_eval, preds_eval)
            r2 = calculate_r2(y_eval, preds_eval)
            mard_val = calculate_mard(y_eval, preds_eval)
            ega_a, ega_b, ega_ab, zone_labels = evaluate_clarke_ega(y_eval, preds_eval)

            fold_metrics.append({
                "patient": held_out,
                "model": m_name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "MARD": mard_val,
                "EGA_A": ega_a,
                "EGA_B": ega_b,
                "EGA_A+B": ega_ab,
                "n_test_samples": n_eval
            })

            # Record per-sample predictions for this fold (preserve all original test rows)
            for ts, yt, yp in zip(test_df["ts"], y_test, preds_all):
                fold_prediction_rows.append({
                    "patient": held_out,
                    "timestamp": ts,
                    "model": m_name,
                    "y_true": float(yt),
                    "y_pred": float(yp) if np.isfinite(yp) else np.nan
                })

            print(f"  {m_name:<15} {mard_val:>10.2f} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f} {ega_a:>8.2f} {ega_b:>8.2f} {ega_ab:>10.2f}%")

        # -------------------------------------------------------------------
        # Save Fold Predictions Immediately to Separate File
        # -------------------------------------------------------------------
        fold_pred_df = pd.DataFrame(fold_prediction_rows)
        fold_pred_path = os.path.join(results_dir, f"fold_{held_out}_predictions.csv")
        fold_pred_df.to_csv(fold_pred_path, index=False)
        print(f"  [Saved] Predictions -> {fold_pred_path}")

        # Update cumulative predictions CSV
        cum_pred_path = os.path.join(results_dir, "loso_predictions.csv")
        if not os.path.exists(cum_pred_path):
            fold_pred_df.to_csv(cum_pred_path, index=False)
        else:
            fold_pred_df.to_csv(cum_pred_path, mode="a", header=False, index=False)

        # -------------------------------------------------------------------
        # Update Cumulative Patient Results CSV Immediately
        # -------------------------------------------------------------------
        df_new_metrics = pd.DataFrame(fold_metrics)
        # Remove any existing rows for this patient to prevent duplicates
        df_existing_results = df_existing_results[df_existing_results["patient"] != held_out]
        df_existing_results = pd.concat([df_existing_results, df_new_metrics], ignore_index=True)
        df_existing_results.to_csv(patient_results_path, index=False)
        print(f"  [Saved] Patient results -> {patient_results_path}")

        # -------------------------------------------------------------------
        # Update Checkpoint
        # -------------------------------------------------------------------
        completed_patients.append(held_out)
        save_checkpoint(checkpoint_path, completed_patients, random_state=random_state)
        print(f"  [Checkpoint] Saved {held_out} to {checkpoint_path}")

        fold_elapsed = time.time() - fold_start
        print(f"  Fold {fold_idx} completed in {fold_elapsed:.1f}s ({fold_elapsed/60:.2f} minutes)")

        # -------------------------------------------------------------------
        # Memory Cleanup
        # -------------------------------------------------------------------
        del train_df, X_train_raw, y_train, X_test_raw, y_test, X_train, X_test
        del ridge, rf, mlp, lgbm, stack, fold_preds, fold_prediction_rows, fold_pred_df
        gc.collect()



    print("\n" + "=" * 78)
    print(f"ALL 12 LOSO FOLDS COMPLETED in {(time.time() - total_start)/60:.2f} minutes.")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # Final Summary Table (Patient-level Mean +/- SD across 12 patients)
    # -----------------------------------------------------------------------
    df_patient_results = pd.read_csv(patient_results_path)
    summary_list = []
    for m_name in model_names:
        sub = df_patient_results[df_patient_results["model"] == m_name]
        summary_list.append({
            "model": m_name,
            "MARD_mean": float(sub["MARD"].mean()),
            "MARD_std": float(sub["MARD"].std(ddof=1)),
            "RMSE_mean": float(sub["RMSE"].mean()),
            "RMSE_std": float(sub["RMSE"].std(ddof=1)),
            "MAE_mean": float(sub["MAE"].mean()),
            "MAE_std": float(sub["MAE"].std(ddof=1)),
            "R2_mean": float(sub["R2"].mean()),
            "R2_std": float(sub["R2"].std(ddof=1)),
            "EGA_A_mean": float(sub["EGA_A"].mean()),
            "EGA_A_std": float(sub["EGA_A"].std(ddof=1)),
            "EGA_B_mean": float(sub["EGA_B"].mean()),
            "EGA_B_std": float(sub["EGA_B"].std(ddof=1)),
            "EGA_A+B_mean": float(sub["EGA_A+B"].mean()),
            "EGA_A+B_std": float(sub["EGA_A+B"].std(ddof=1)),
        })

    df_summary = pd.DataFrame(summary_list).sort_values("MARD_mean")
    summary_path = os.path.join(results_dir, "loso_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[Saved] LOSO Summary table -> {summary_path}")

    # Reconstruct/verify cumulative predictions from individual fold files
    all_fold_files = sorted(glob.glob(os.path.join(results_dir, "fold_*_predictions.csv")))
    if all_fold_files:
        df_all_preds = pd.concat([pd.read_csv(fp) for fp in all_fold_files], ignore_index=True)
        predictions_path = os.path.join(results_dir, "loso_predictions.csv")
        df_all_preds.to_csv(predictions_path, index=False)
        print(f"[Reconstructed] Full predictions ({len(df_all_preds):,} rows) -> {predictions_path}")
    else:
        df_all_preds = pd.read_csv(os.path.join(results_dir, "loso_predictions.csv"))

    # -----------------------------------------------------------------------
    # Visualizations (Generated Only When All 12 Folds Are Complete)
    # -----------------------------------------------------------------------
    print("\nGenerating final evaluation plots in outputs/...")

    # Plot 1: Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("mako", len(df_summary))

    sns.barplot(data=df_summary, x="model", y="MARD_mean", ax=axes[0], palette=palette)
    axes[0].errorbar(
        x=range(len(df_summary)),
        y=df_summary["MARD_mean"],
        yerr=df_summary["MARD_std"],
        fmt='none', c='black', capsize=5, elinewidth=1.5
    )
    axes[0].set_title("LOSO: Mean MARD across 12 Patients (Lower is Better)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("MARD (%)", fontsize=11)
    axes[0].set_xlabel("Model", fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)

    sns.barplot(data=df_summary, x="model", y="RMSE_mean", ax=axes[1], palette=palette)
    axes[1].errorbar(
        x=range(len(df_summary)),
        y=df_summary["RMSE_mean"],
        yerr=df_summary["RMSE_std"],
        fmt='none', c='black', capsize=5, elinewidth=1.5
    )
    axes[1].set_title("LOSO: Mean RMSE across 12 Patients (Lower is Better)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("RMSE (mg/dL)", fontsize=11)
    axes[1].set_xlabel("Model", fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    comp_plot_path = os.path.join(outputs_dir, "loso_model_comparison.png")
    fig.savefig(comp_plot_path, dpi=300)
    plt.close(fig)
    print(f"  -> {comp_plot_path}")

    # Plot 2: Per-Patient MARD Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=df_patient_results,
        x="patient",
        y="MARD",
        hue="model",
        ax=ax,
        palette="tab10"
    )
    ax.set_title("Per-Patient MARD across all 12 OhioT1DM Patients (LOSO)", fontsize=13, fontweight='bold')
    ax.set_ylabel("MARD (%)", fontsize=11)
    ax.set_xlabel("Held-Out Patient ID", fontsize=11)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    per_patient_path = os.path.join(outputs_dir, "loso_per_patient_mard.png")
    fig.savefig(per_patient_path, dpi=300)
    plt.close(fig)
    print(f"  -> {per_patient_path}")

    # Identify Best Model by lowest mean MARD
    best_model_name = df_summary.iloc[0]["model"]
    best_preds_df = df_all_preds[df_all_preds["model"] == best_model_name].copy()

    # Plot 3: Actual vs Predicted Glucose
    sample_sub = best_preds_df.sample(n=min(10000, len(best_preds_df)), random_state=random_state)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(sample_sub["y_true"], sample_sub["y_pred"], alpha=0.25, color="#1f77b4", s=12, label="Predictions")
    min_val = min(sample_sub["y_true"].min(), sample_sub["y_pred"].min())
    max_val = max(sample_sub["y_true"].max(), sample_sub["y_pred"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Identity (y = x)")
    ax.set_title(f"Actual vs. Predicted Glucose - {best_model_name} (LOSO 12 Patients)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Actual Blood Glucose (mg/dL)", fontsize=11)
    ax.set_ylabel("Predicted Blood Glucose (mg/dL)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    act_vs_pred_path = os.path.join(outputs_dir, "loso_actual_vs_predicted.png")
    fig.savefig(act_vs_pred_path, dpi=300)
    plt.close(fig)
    print(f"  -> {act_vs_pred_path}")

    # Plot 4: Standard Clarke Error Grid
    clarke_sample = best_preds_df.sample(n=min(15000, len(best_preds_df)), random_state=random_state).copy()
    _, _, _, sample_zones = evaluate_clarke_ega(clarke_sample["y_true"].values, clarke_sample["y_pred"].values)
    clarke_sample["zone"] = sample_zones

    fig, ax = plt.subplots(figsize=(9, 9))
    # Standard Clarke boundaries
    ax.plot([0, 450], [0, 450], 'k:', linewidth=1)
    ax.plot([0, 450], [0, 450 * 1.2], 'k--', alpha=0.6, linewidth=1)
    ax.plot([0, 450], [0, 450 * 0.8], 'k--', alpha=0.6, linewidth=1)
    ax.axvline(70, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(70, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(180, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(180, color='gray', linestyle=':', alpha=0.5)

    zone_colors = {'A': '#2ca02c', 'B': '#1f77b4', 'C': '#ff7f0e', 'D': '#d62728', 'E': '#9467bd'}
    for z in ['A', 'B', 'C', 'D', 'E']:
        z_pts = clarke_sample[clarke_sample["zone"] == z]
        if len(z_pts) > 0:
            ax.scatter(z_pts["y_true"], z_pts["y_pred"], c=zone_colors.get(z, 'gray'), alpha=0.35, s=14, label=f"Zone {z} ({100*len(z_pts)/len(clarke_sample):.1f}%)")

    best_summary = df_summary[df_summary["model"] == best_model_name].iloc[0]
    ax.set_title(
        f"Standard Clarke Error Grid - {best_model_name}\n"
        f"Zone A: {best_summary['EGA_A_mean']:.1f}% | Zone B: {best_summary['EGA_B_mean']:.1f}% | Zone A+B: {best_summary['EGA_A+B_mean']:.1f}%",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=11)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=11)
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 420)
    ax.legend(title="Clarke Zones", loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    clarke_path = os.path.join(outputs_dir, "loso_clarke_error_grid.png")
    fig.savefig(clarke_path, dpi=300)
    plt.close(fig)
    print(f"  -> {clarke_path}")

    # -----------------------------------------------------------------------
    # Final Leaderboard Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL 12-PATIENT LOSO BENCHMARK LEADERBOARD (RESEARCH OBJECTIVE 1)")
    print("=" * 90)
    header = (
        f"{'Model':<16} "
        f"{'MARD (%)':>15} "
        f"{'RMSE (mg/dL)':>16} "
        f"{'MAE (mg/dL)':>16} "
        f"{'R2':>14} "
        f"{'Clarke A+B (%)':>16}"
    )
    print(header)
    print("-" * 90)
    for _, row in df_summary.iterrows():
        line = (
            f"{row['model']:<16} "
            f"{row['MARD_mean']:>6.2f} +/- {row['MARD_std']:<5.2f} "
            f"{row['RMSE_mean']:>6.2f} +/- {row['RMSE_std']:<5.2f} "
            f"{row['MAE_mean']:>6.2f} +/- {row['MAE_std']:<5.2f} "
            f"{row['R2_mean']:>5.3f} +/- {row['R2_std']:<5.3f} "
            f"{row['EGA_A+B_mean']:>6.2f} +/- {row['EGA_A+B_std']:<5.2f}%"
        )
        print(line)
    print("=" * 90)
    print(f"\nBest Model across 12-Patient LOSO: {best_model_name} (MARD: {best_summary['MARD_mean']:.2f}% +/- {best_summary['MARD_std']:.2f}%)")


if __name__ == "__main__":
    import sys
    res_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", "results_causal")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUTPUTS_DIR", "outputs_causal")
    np.random.seed(42)
    run_loso_evaluation(results_dir=res_dir, outputs_dir=out_dir)
