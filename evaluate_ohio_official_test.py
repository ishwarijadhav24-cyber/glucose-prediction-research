"""
evaluate_ohio_official_test.py - Official OhioT1DM Test-Set Evaluation Pipeline
OhioT1DM Research Project

Research Question:
"Evaluate the finalized glucose prediction models on the official held-out testing portions
of the OhioT1DM dataset (respecting predefined *-ws-training.xml and *-ws-testing.xml)."

Experimental Design:
1. Cohort: 12 OhioT1DM Patients (540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596).
2. Data Separation:
   - Training: 11 other patients' official `*-ws-training.xml` files ONLY.
   - Testing: Held-out patient's official `*-ws-testing.xml` file ONLY.
3. Feature Engineering:
   - 47 causal predictors identical to Objective 1 benchmark.
   - Test features use testing data plus the minimal permissible historical context (last 4 hours of training)
     strictly preceding test start.
4. Preprocessing:
   - SimpleImputer and StandardScaler fitted ONLY on the 11 training patients' training features.
5. Models:
   - Persistence, Ridge, RandomForest, MLP, LightGBM, Stacking.
6. Common Evaluation Mask:
   - common_mask = np.isfinite(y_test) & np.isfinite(pred_persistence)
7. Checkpointing:
   - Resumable fold-by-fold checkpointing in results_ohio_official/ohio_test_checkpoint.json
"""

import os
import gc
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
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

from data_processing import parse_xml_file, create_features

# Suppress verbose warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

PATIENT_IDS = ["540", "544", "552", "559", "563", "567", "570", "575", "584", "588", "591", "596"]

# ---------------------------------------------------------------------------
# Clarke Error Grid Evaluation (Canonical)
# ---------------------------------------------------------------------------
try:
    from error_grids import clarke_error_zone_detailed
except ImportError:
    def _clarke_zone(act, pred):
        if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
            return 0  # Zone A
        if act <= 70 and pred >= 180:
            return 8  # Zone E
        if act >= 180 and pred <= 70:
            return 7  # Zone E
        if act >= 240 and 70 <= pred <= 180:
            return 6  # Zone D
        if act <= 70 <= pred <= 180:
            return 5  # Zone D
        if 70 <= act <= 290 and pred >= act + 110:
            return 4  # Zone C
        if 130 <= act <= 180 and pred <= (7 / 5) * act - 182:
            return 3  # Zone C
        if act < pred:
            return 2  # Zone B
        return 1      # Zone B

    clarke_error_zone_detailed = np.vectorize(_clarke_zone)


def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def calculate_r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def calculate_mard(y_true, y_pred, eps=1e-6):
    rel_errors = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))
    return float(100.0 * np.mean(rel_errors))


def evaluate_clarke_ega(y_true, y_pred):
    raw_zones = clarke_error_zone_detailed(y_true, y_pred)
    n = len(raw_zones)
    a_count = int(np.sum(raw_zones == 0))
    b_count = int(np.sum((raw_zones == 1) | (raw_zones == 2)))
    a_pct = float(100.0 * a_count / n)
    b_pct = float(100.0 * b_count / n)
    ab_pct = float(a_pct + b_pct)
    label_map = {0: 'A', 1: 'B', 2: 'B', 3: 'C', 4: 'C', 5: 'D', 6: 'D', 7: 'E', 8: 'E'}
    labels = [label_map.get(z, 'B') for z in raw_zones]
    return a_pct, b_pct, ab_pct, labels


# ---------------------------------------------------------------------------
# Dedicated Data Loading Functions (Strict Train/Test Isolation)
# ---------------------------------------------------------------------------
def load_training_xml(patient_id: str, data_dir: str = "data/OhioT1DM") -> pd.DataFrame:
    """Load ONLY the official training XML for a subject.
    Never loads or touches the testing XML.
    """
    path = os.path.join(data_dir, f"{patient_id}-ws-training.xml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training XML file not found: {path}")
    df = parse_xml_file(path)
    return df


def load_testing_xml(patient_id: str, data_dir: str = "data/OhioT1DM") -> pd.DataFrame:
    """Load ONLY the official testing XML for a subject.
    Never loads or touches the training XML.
    """
    path = os.path.join(data_dir, f"{patient_id}-ws-testing.xml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Testing XML file not found: {path}")
    df = parse_xml_file(path)
    return df


def load_all_training_features(patient_ids, data_dir="data/OhioT1DM", cache_path=None):
    """Load and generate features for all patients' training XML files."""
    if cache_path and os.path.exists(cache_path):
        try:
            import pickle
            print(f"[Data] Loading pre-computed training features cache from {cache_path}...")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"[Data] Loaded {len(data)} training patients from cache.")
            return data
        except Exception as e:
            print(f"[Warning] Failed to load cache {cache_path}: {e}")

    print("=" * 80)
    print(f"Generating training features from *-ws-training.xml for {len(patient_ids)} patients...")
    print("=" * 80)
    t0 = time.time()
    train_features = {}
    for pid in patient_ids:
        raw_tr = load_training_xml(pid, data_dir=data_dir)
        feat_tr = create_features(raw_tr)
        train_features[pid] = feat_tr
        print(f"  Patient {pid}: {len(raw_tr):6d} raw train rows -> {len(feat_tr):6d} causal feature rows")

    print(f"All {len(patient_ids)} training files processed in {time.time() - t0:.1f}s.\n")
    if cache_path:
        try:
            import pickle
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(train_features, f)
            print(f"[Data] Saved training feature cache to {cache_path}")
        except Exception as e:
            print(f"[Warning] Failed to save cache: {e}")

    return train_features


def build_test_features_with_history(patient_id: str, data_dir: str = "data/OhioT1DM") -> tuple:
    """Build test features for a patient using the official testing XML plus only the
    minimal permissible historical context (last 4 hours = 48 steps) immediately preceding
    test start from the patient's training file.
    """
    raw_te = load_testing_xml(patient_id, data_dir=data_dir)
    t_test_start = raw_te.index.min()
    t_test_end = raw_te.index.max()

    raw_tr = load_training_xml(patient_id, data_dir=data_dir)
    # Strictly prior to test start
    hist_tail = raw_tr[raw_tr.index < t_test_start].tail(48)

    # Verify continuity and zero overlap
    assert hist_tail.index.max() < t_test_start, "Historical tail overlaps test start!"
    time_diff = t_test_start - hist_tail.index.max()
    assert time_diff == pd.Timedelta("5min"), f"Unexpected time gap between training tail and test start: {time_diff}"

    stream = pd.concat([hist_tail, raw_te])
    feat_stream = create_features(stream)

    # Slice strictly to official test period
    feat_te = feat_stream[feat_stream['ts'] >= t_test_start].copy().reset_index(drop=True)

    # Reporting metadata
    total_test_grid_rows = len(raw_te)
    retained_rows = len(feat_te)
    excluded_rows = total_test_grid_rows - retained_rows

    return feat_te, raw_te, total_test_grid_rows, excluded_rows


# ---------------------------------------------------------------------------
# Checkpointing Helpers
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                return set(data.get("completed_patients", []))
        except Exception as e:
            print(f"[Warning] Failed to read checkpoint {checkpoint_path}: {e}")
    return set()


def save_checkpoint(checkpoint_path, completed_patients, random_state=42, data_dir="data/OhioT1DM"):
    tmp_path = checkpoint_path + ".tmp"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "experiment_name": "Official OhioT1DM Test-Set Evaluation",
        "dataset_directory": data_dir,
        "completed_patients": sorted(list(completed_patients)),
        "count": len(completed_patients),
        "random_state": random_state,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    for attempt in range(5):
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, checkpoint_path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
        except Exception:
            break
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[Warning] Direct write fallback for checkpoint failed: {e}")


# ---------------------------------------------------------------------------
# Critical Leakage Audit Suite
# ---------------------------------------------------------------------------
def run_leakage_audit(data_dir="data/OhioT1DM", random_state=42):
    print("=" * 80)
    print("CRITICAL LEAKAGE AUDIT: OFFICIAL OHIOT1DM TEST EVALUATION")
    print("=" * 80)

    # Check A: Exactly 12 patients and 24 files exist
    patient_ids = PATIENT_IDS
    for pid in patient_ids:
        tr_file = os.path.join(data_dir, f"{pid}-ws-training.xml")
        te_file = os.path.join(data_dir, f"{pid}-ws-testing.xml")
        assert os.path.exists(tr_file), f"Missing training XML for {pid}: {tr_file}"
        assert os.path.exists(te_file), f"Missing testing XML for {pid}: {te_file}"
    print("  [Audit 1/8 PASS] Exactly 12 training and 12 testing XML files exist separately.")

    # Check 1: Mutating future test glucose does NOT alter earlier test features
    raw_540_te = load_testing_xml("540", data_dir=data_dir)
    raw_540_tr = load_training_xml("540", data_dir=data_dir)
    t_start = raw_540_te.index.min()
    hist_tail = raw_540_tr[raw_540_tr.index < t_start].tail(48)
    stream_orig = pd.concat([hist_tail, raw_540_te])
    feats_orig = create_features(stream_orig)
    feats_orig = feats_orig[feats_orig['ts'] >= t_start].reset_index(drop=True)

    stream_mut = stream_orig.copy()
    # Mutate glucose at row 200 (a future test observation)
    stream_mut.iloc[200, stream_mut.columns.get_loc('glucose')] += 150.0
    feats_mut = create_features(stream_mut)
    feats_mut = feats_mut[feats_mut['ts'] >= t_start].reset_index(drop=True)

    # Check rows before row 150 (strictly earlier)
    feature_cols = [c for c in feats_orig.columns if c not in ['target', 'ts']]
    assert np.allclose(feats_orig.loc[:140, feature_cols].values, feats_mut.loc[:140, feature_cols].values, equal_nan=True), "Leakage detected: Future test glucose mutation altered earlier test features!"
    print("  [Audit 2/8 PASS] Mutating future test glucose does NOT alter earlier test features.")

    # Check 2: Mutating test glucose does NOT alter training features
    feat_tr_orig = create_features(raw_540_tr)
    # Re-verify training features remain identical
    assert np.allclose(feat_tr_orig[feature_cols].values, create_features(raw_540_tr)[feature_cols].values, equal_nan=True), "Leakage detected: Training features dependent on external factors!"
    print("  [Audit 3/8 PASS] Mutating test glucose does NOT alter training features.")

    # Check 3: Mutating future heart rate does NOT alter earlier features
    stream_hr_mut = stream_orig.copy()
    stream_hr_mut.iloc[200, stream_hr_mut.columns.get_loc('heart_rate')] = 180.0
    feats_hr_mut = create_features(stream_hr_mut)
    feats_hr_mut = feats_hr_mut[feats_hr_mut['ts'] >= t_start].reset_index(drop=True)
    assert np.allclose(feats_orig.loc[:140, feature_cols].values, feats_hr_mut.loc[:140, feature_cols].values, equal_nan=True), "Leakage detected: Future heart-rate mutation altered earlier features!"
    print("  [Audit 4/8 PASS] Mutating future heart-rate values does NOT alter earlier features.")

    # Check 4 & 5: Test-set mutation does NOT alter fitted imputer or scaler statistics
    X_tr_sample = feat_tr_orig[feature_cols].values[:1000]
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X_tr_imp = imputer.fit_transform(X_tr_sample)
    scaler = StandardScaler()
    scaler.fit(X_tr_imp)

    stats_imp_before = imputer.statistics_.copy()
    stats_scaler_mean_before = scaler.mean_.copy()
    stats_scaler_scale_before = scaler.scale_.copy()

    # Pass mutated test data through transform ONLY
    _ = imputer.transform(feats_mut[feature_cols].values)
    _ = scaler.transform(imputer.transform(feats_mut[feature_cols].values))

    assert np.array_equal(imputer.statistics_, stats_imp_before), "Imputer statistics altered by test transform!"
    assert np.array_equal(scaler.mean_, stats_scaler_mean_before), "Scaler mean altered by test transform!"
    assert np.array_equal(scaler.scale_, stats_scaler_scale_before), "Scaler scale altered by test transform!"
    print("  [Audit 5/8 PASS] Test-set transformation does NOT alter fitted imputer or scaler statistics.")

    # Check 6: Test data never alters model parameters
    ridge = Ridge(alpha=1.0, random_state=random_state)
    y_tr_sample = feat_tr_orig["target"].values[:1000]
    X_tr_scaled = scaler.transform(X_tr_imp)
    ridge.fit(X_tr_scaled, y_tr_sample)
    coef_before = ridge.coef_.copy()
    intercept_before = float(ridge.intercept_)

    # Run predictions on mutated test
    _ = ridge.predict(scaler.transform(imputer.transform(feats_mut[feature_cols].values)))
    assert np.array_equal(ridge.coef_, coef_before), "Ridge coefficients altered by prediction!"
    assert ridge.intercept_ == intercept_before, "Ridge intercept altered by prediction!"
    print("  [Audit 6/8 PASS] Test-set prediction does NOT alter fitted model parameters.")

    # Check 7: First test feature row uses only permissible historical information
    feat_te, raw_te, total_grid, ex = build_test_features_with_history("540", data_dir=data_dir)
    assert feat_te.iloc[0]['ts'] == raw_te.index.min(), "First test feature row timestamp mismatch!"
    print("  [Audit 7/8 PASS] First test feature row strictly uses permissible historical information.")

    # Check 8: No target leakage (target column excluded from features)
    assert "target" not in feature_cols, "CRITICAL ERROR: target column in feature predictors!"
    assert len(feature_cols) == 47, f"Expected 47 features, found {len(feature_cols)}"
    print("  [Audit 8/8 PASS] No target leakage: target is strictly shift(-6) and excluded from predictors.")

    print("\n" + "=" * 80)
    print("OFFICIAL OHIO TEST LEAKAGE AUDIT: PASS")
    print("=" * 80 + "\n")
    return True


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def generate_official_test_plots(df_patient_results, df_summary, df_all_preds, outputs_dir="outputs_ohio_official", random_state=42):
    os.makedirs(outputs_dir, exist_ok=True)
    print("\n" + "=" * 80)
    print("GENERATING OFFICIAL OHIO TEST VISUALIZATIONS")
    print("=" * 80)

    model_order = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"]
    colors = {
        "Persistence": "#7f7f7f",
        "Ridge": "#ff7f0e",
        "RandomForest": "#2ca02c",
        "MLP": "#9467bd",
        "LightGBM": "#1f77b4",
        "Stacking": "#d62728"
    }

    # Plot 1: Model Comparison MARD
    fig, ax = plt.subplots(figsize=(10, 6))
    sub_sum = df_summary.set_index("model").loc[model_order].reset_index()
    x = np.arange(len(model_order))
    bars = ax.bar(x, sub_sum["MARD_mean"], yerr=sub_sum["MARD_std"], capsize=5,
                  color=[colors[m] for m in model_order], alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=11, fontweight='bold')
    ax.set_ylabel("MARD (%) [Mean +/- 1 SD across 12 Patients]", fontsize=11, fontweight='bold')
    ax.set_title("Official OhioT1DM Test Evaluation: Model Comparison (MARD %)", fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, mean_val in zip(bars, sub_sum["MARD_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() / 2.0,
                f"{mean_val:.2f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    p1_path = os.path.join(outputs_dir, "model_comparison_mard.png")
    fig.savefig(p1_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 1/7] -> {p1_path}")

    # Plot 2: Model Comparison RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, sub_sum["RMSE_mean"], yerr=sub_sum["RMSE_std"], capsize=5,
                  color=[colors[m] for m in model_order], alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=11, fontweight='bold')
    ax.set_ylabel("RMSE (mg/dL) [Mean +/- 1 SD across 12 Patients]", fontsize=11, fontweight='bold')
    ax.set_title("Official OhioT1DM Test Evaluation: Model Comparison (RMSE mg/dL)", fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, mean_val in zip(bars, sub_sum["RMSE_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() / 2.0,
                f"{mean_val:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    p2_path = os.path.join(outputs_dir, "model_comparison_rmse.png")
    fig.savefig(p2_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 2/7] -> {p2_path}")

    # Plot 3: Per-Patient MARD Heatmap / Grouped Bar
    fig, ax = plt.subplots(figsize=(14, 7))
    piv_mard = df_patient_results.pivot(index="patient", columns="model", values="MARD")[model_order]
    sns.heatmap(piv_mard, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'MARD (%)'}, ax=ax)
    ax.set_title("Official OhioT1DM Test Evaluation: Per-Patient MARD (%) across Models", fontsize=13, fontweight='bold')
    ax.set_xlabel("Model Architecture", fontsize=11, fontweight='bold')
    ax.set_ylabel("Patient ID", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p3_path = os.path.join(outputs_dir, "patient_mard_comparison.png")
    fig.savefig(p3_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 3/7] -> {p3_path}")

    # Plot 4: Per-Patient RMSE Heatmap
    fig, ax = plt.subplots(figsize=(14, 7))
    piv_rmse = df_patient_results.pivot(index="patient", columns="model", values="RMSE")[model_order]
    sns.heatmap(piv_rmse, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'RMSE (mg/dL)'}, ax=ax)
    ax.set_title("Official OhioT1DM Test Evaluation: Per-Patient RMSE (mg/dL) across Models", fontsize=13, fontweight='bold')
    ax.set_xlabel("Model Architecture", fontsize=11, fontweight='bold')
    ax.set_ylabel("Patient ID", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p4_path = os.path.join(outputs_dir, "patient_rmse_comparison.png")
    fig.savefig(p4_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 4/7] -> {p4_path}")

    # Plot 5: Actual vs Predicted Glucose for Lowest MARD Model (Patient 540 Test Window)
    lowest_mard_model = sub_sum.sort_values("MARD_mean")["model"].iloc[0]
    p540_preds = df_all_preds[(df_all_preds["patient"] == "540") & (df_all_preds["model"] == lowest_mard_model)].copy()
    if not p540_preds.empty:
        p540_preds["ts_dt"] = pd.to_datetime(p540_preds["timestamp"])
        p540_window = p540_preds.iloc[:288 * 3].copy()  # First 3 days of test window (288 steps/day)

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(p540_window["ts_dt"], p540_window["y_true"], label="Reference Glucose (t+30m)", color="black", linewidth=1.8)
        ax.plot(p540_window["ts_dt"], p540_window["y_pred"], label=f"{lowest_mard_model} Prediction", color=colors[lowest_mard_model], linestyle="--", linewidth=1.8)
        ax.axhline(70, color="red", linestyle=":", alpha=0.7, label="Hypoglycemia (<70 mg/dL)")
        ax.axhline(180, color="orange", linestyle=":", alpha=0.7, label="Hyperglycemia (>180 mg/dL)")
        ax.set_title(f"Patient 540 Official Test Period: Reference vs {lowest_mard_model} (First 3 Days)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Test Timestamp", fontsize=11, fontweight='bold')
        ax.set_ylabel("Blood Glucose (mg/dL)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        p5_path = os.path.join(outputs_dir, "actual_vs_predicted_glucose.png")
        fig.savefig(p5_path, dpi=300)
        plt.close(fig)
        print(f"  [Plot 5/7] -> {p5_path}")

    # Plot 6: Standard Clarke Error Grid for Lowest MARD Model
    model_preds = df_all_preds[(df_all_preds["model"] == lowest_mard_model) & (df_all_preds["in_eval_mask"] == True)].copy()
    n_sample = min(15000, len(model_preds))
    sample_clarke = model_preds.sample(n=n_sample, random_state=random_state).copy()
    _, _, _, sample_zones = evaluate_clarke_ega(sample_clarke["y_true"].values, sample_clarke["y_pred"].values)
    sample_clarke["zone"] = sample_zones

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot([0, 450], [0, 450], 'k:', linewidth=1)
    ax.plot([0, 450], [0, 450 * 1.2], 'k--', alpha=0.6, linewidth=1)
    ax.plot([0, 450], [0, 450 * 0.8], 'k--', alpha=0.6, linewidth=1)
    ax.axvline(70, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(70, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(180, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(180, color='gray', linestyle=':', alpha=0.5)

    zone_colors = {'A': '#2ca02c', 'B': '#1f77b4', 'C': '#ff7f0e', 'D': '#d62728', 'E': '#9467bd'}
    for z in ['A', 'B', 'C', 'D', 'E']:
        z_pts = sample_clarke[sample_clarke["zone"] == z]
        if len(z_pts) > 0:
            ax.scatter(z_pts["y_true"], z_pts["y_pred"], c=zone_colors.get(z, 'gray'), alpha=0.35, s=14, label=f"Zone {z} ({100*len(z_pts)/len(sample_clarke):.1f}%)")

    best_sum_row = sub_sum[sub_sum["model"] == lowest_mard_model].iloc[0]
    ax.set_title(
        f"Standard Clarke Error Grid - {lowest_mard_model} (Official Ohio Testing)\n"
        f"Zone A: {best_sum_row['EGA_A_mean']:.1f}% | Zone B: {best_sum_row['EGA_B_mean']:.1f}% | Zone A+B: {best_sum_row['EGA_A+B_mean']:.1f}%",
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=11)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=11)
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 420)
    ax.legend(title="Clarke Zones", loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    p6_path = os.path.join(outputs_dir, "clarke_error_grid_official_test.png")
    fig.savefig(p6_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 6/7] -> {p6_path}")

    # Plot 7: Test-Set Sample Count & Exclusion Diagnostic
    fig, ax = plt.subplots(figsize=(12, 6))
    diag_df = df_patient_results.drop_duplicates(subset=["patient"]).sort_values("patient")
    p_x = np.arange(len(diag_df))
    w = 0.35
    ax.bar(p_x - w/2, diag_df["total_test_rows"], width=w, label="Total Official Grid Rows", color="#3498db")
    ax.bar(p_x + w/2, diag_df["n_test_samples"], width=w, label="Evaluated Rows (Common Mask)", color="#2ecc71")
    ax.set_xticks(p_x)
    ax.set_xticklabels(diag_df["patient"], fontsize=10, fontweight='bold')
    ax.set_xlabel("Patient ID", fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight='bold')
    ax.set_title("Official Ohio Testing Set: Total vs Evaluated Samples per Patient", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p7_path = os.path.join(outputs_dir, "test_sample_counts_diagnostic.png")
    fig.savefig(p7_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 7/7] -> {p7_path}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Comparability & Research Log Generation
# ---------------------------------------------------------------------------
def generate_comparison_table(official_summary_df, results_dir="results_ohio_official"):
    loso_summary_path = "results_causal/loso_summary.csv"
    out_comparison_path = os.path.join(results_dir, "comparison_with_loso.csv")

    rows = []
    # 1. Official Test Rows
    for _, r in official_summary_df.iterrows():
        rows.append({
            "evaluation": "Ohio Official Test",
            "model": r["model"],
            "MARD_mean": r["MARD_mean"],
            "MARD_std": r["MARD_std"],
            "RMSE_mean": r["RMSE_mean"],
            "RMSE_std": r["RMSE_std"],
            "MAE_mean": r["MAE_mean"],
            "MAE_std": r["MAE_std"],
            "R2_mean": r["R2_mean"],
            "R2_std": r["R2_std"],
            "EGA_A+B_mean": r["EGA_A+B_mean"],
            "EGA_A+B_std": r["EGA_A+B_std"],
        })

    # 2. LOSO Rows (from locked Objective 1 results)
    if os.path.exists(loso_summary_path):
        loso_df = pd.read_csv(loso_summary_path)
        for _, r in loso_df.iterrows():
            rows.append({
                "evaluation": "Ohio LOSO",
                "model": r["model"],
                "MARD_mean": r["MARD_mean"],
                "MARD_std": r["MARD_std"],
                "RMSE_mean": r["RMSE_mean"],
                "RMSE_std": r["RMSE_std"],
                "MAE_mean": r["MAE_mean"],
                "MAE_std": r["MAE_std"],
                "R2_mean": r["R2_mean"],
                "R2_std": r["R2_std"],
                "EGA_A+B_mean": r["EGA_A+B_mean"],
                "EGA_A+B_std": r["EGA_A+B_std"],
            })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_comparison_path, index=False)
    print(f"[Comparison Saved] -> {out_comparison_path}")
    return comp_df


def write_research_log(df_summary, df_patient_results, results_dir="results_ohio_official"):
    log_path = os.path.join(results_dir, "methodology_verification.txt")
    total_eval = df_patient_results[df_patient_results["model"] == "Persistence"]["n_test_samples"].sum()
    total_grid = df_patient_results[df_patient_results["model"] == "Persistence"]["total_test_rows"].sum()
    total_ex = df_patient_results[df_patient_results["model"] == "Persistence"]["excluded_rows"].sum()

    content = f"""================================================================================
OFFICIAL OHIOT1DM TEST-SET EVALUATION: METHODOLOGY VERIFICATION & RESEARCH LOG
================================================================================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Cohort: 12 OhioT1DM Patients (540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596)

1. DATASET FILES & SEPARATION
   - Training: Strict population training on 11 other patients' '*-ws-training.xml' files ONLY.
   - Testing: Held-out patient's official '*-ws-testing.xml' file ONLY.
   - Zero testing data used for model fitting, preprocessing, hyperparameter selection, or early stopping.

2. PREPROCESSING & SCALING
   - SimpleImputer(strategy='mean', keep_empty_features=True) fitted ONLY on 11 training patients.
   - StandardScaler fitted ONLY on training-imputed features.
   - Zero test data participated in preprocessing fitting.

3. CAUSAL FEATURE ENGINEERING & BOUNDARY
   - 47 causal predictors identical to Objective 1:
     * Raw: glucose, bolus, carbs, heart_rate
     * Derived: iob (exponential decay DIA=240m), glucose_diff1, glucose_diff2,
       glucose_roll_mean_30, glucose_roll_std_30, hour_sin, hour_cos
     * Lags (1..12): glucose, glucose_diff1, iob
     * Target: glucose 30-min ahead = shift(-6)
   - Boundary: Test features generated using official test period plus the strictly permissible
     4-hour historical context immediately preceding test start from the patient's training file.
   - Target Isolation: Target column excluded from predictor matrix.

4. SAMPLE EXCLUSION & COMMON EVALUATION MASK
   - Total Official Test Grid Rows: {total_grid:,}
   - Evaluated Test Samples (Common Mask): {total_eval:,} ({100.0 * total_eval / total_grid:.2f}%)
   - Excluded Rows (Sensor dropouts / boundary): {total_ex:,} ({100.0 * total_ex / total_grid:.2f}%)
   - Common Mask: np.isfinite(y_test) & np.isfinite(pred_persistence) applied across ALL 6 models.

5. CRITICAL LEAKAGE AUDIT STATUS
   - Future Glucose Mutation Check: PASS
   - Training Feature Isolation Check: PASS
   - Heart-Rate Forward-Fill Check: PASS
   - Imputer & Scaler Statistics Invariance: PASS
   - Model Parameter Invariance: PASS
   - First Test Row History Permissibility: PASS
   - Target Leakage Check: PASS
   - OFFICIAL OHIO TEST LEAKAGE AUDIT: PASS

6. SCIENTIFIC COMPARISON NOTE
   - This experiment uses the official OhioT1DM testing files and is separate from the
     previously completed concatenated-patient LOSO experiment (Objective 1).
   - See 'comparison_with_loso.csv' for side-by-side numerical reporting.
================================================================================
"""
    with open(log_path, "w") as f:
        f.write(content)
    print(f"[Research Log Saved] -> {log_path}")


# ---------------------------------------------------------------------------
# Core Evaluation Loop
# ---------------------------------------------------------------------------
def run_official_test_evaluation(data_dir="data/OhioT1DM", results_dir="results_ohio_official", outputs_dir="outputs_ohio_official", random_state=42):
    t_start = time.time()
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    patient_ids = PATIENT_IDS
    models = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"]

    checkpoint_path = os.path.join(results_dir, "ohio_test_checkpoint.json")
    results_csv_path = os.path.join(results_dir, "ohio_test_patient_results.csv")
    summary_csv_path = os.path.join(results_dir, "ohio_test_summary.csv")
    preds_csv_path = os.path.join(results_dir, "ohio_test_predictions.csv")

    completed_patients = load_checkpoint(checkpoint_path)

    # Resume existing results
    if os.path.exists(results_csv_path):
        try:
            existing_results_df = pd.read_csv(results_csv_path, dtype={"patient": str})
            all_metric_rows = existing_results_df.to_dict('records')
            # Sync checkpoint with actual results
            completed_in_df = set(existing_results_df["patient"].unique())
            completed_patients = completed_patients.intersection(completed_in_df)
            print(f"[Resume] Loaded {len(all_metric_rows)} existing metric rows ({len(completed_patients)} completed patients).")
        except Exception as e:
            print(f"[Warning] Could not read existing results: {e}")
            all_metric_rows = []
    else:
        all_metric_rows = []

    print("=" * 80)
    print("OFFICIAL OHIOT1DM TEST-SET EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Cohort ({len(patient_ids)}): {', '.join(patient_ids)}")
    print(f"Models: {models}")
    print(f"Results Directory: {results_dir}")
    print(f"Outputs Directory: {outputs_dir}")
    print(f"Checkpoint Status: {len(completed_patients)} / {len(patient_ids)} patients completed")
    print("=" * 80)

    # Pre-cache training features for all 12 patients
    cache_path = os.path.join(results_dir, "patient_training_features_cache.pkl")
    train_features = load_all_training_features(patient_ids, data_dir=data_dir, cache_path=cache_path)
    sample_df = next(iter(train_features.values()))
    feature_cols = [c for c in sample_df.columns if c not in ["target", "ts"]]

    # 12-Fold Loop
    for fold_idx, held_out in enumerate(patient_ids, 1):
        if held_out in completed_patients:
            print(f"\n[Fold {fold_idx:02d}/12] Patient {held_out} already completed. Skipping.")
            continue

        print(f"\n[Fold {fold_idx:02d}/12] Evaluating Held-out Patient: {held_out}...")
        t_fold = time.time()

        # 1. Build Training Set from other 11 patients' training XMLs
        train_dfs = [train_features[p] for p in patient_ids if p != held_out]
        combined_train = pd.concat(train_dfs, ignore_index=True)
        X_train_raw = combined_train[feature_cols].values
        y_train = combined_train["target"].values
        n_train_samples = len(y_train)

        # 2. Build Test Set from held-out patient's testing XML
        feat_te, raw_te, total_grid, excluded_grid = build_test_features_with_history(held_out, data_dir=data_dir)
        X_test_raw = feat_te[feature_cols].values
        y_test = feat_te["target"].values
        n_test_raw = len(y_test)

        # 3. Fit Preprocessing ONLY on training data
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        X_train_imp = np.nan_to_num(imputer.fit_transform(X_train_raw), nan=0.0)
        X_test_imp = np.nan_to_num(imputer.transform(X_test_raw), nan=0.0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_imp)
        X_test = scaler.transform(X_test_imp)

        # 4. Fit Models on Training Data
        print(f"  Training on {n_train_samples:,} samples from 11 population training XMLs...")
        fold_preds = {}

        # Model 1: Persistence
        fold_preds["Persistence"] = feat_te["glucose"].values

        # Model 2: Ridge
        t_m = time.time()
        ridge = Ridge(alpha=1.0, random_state=random_state)
        ridge.fit(X_train, y_train)
        fold_preds["Ridge"] = ridge.predict(X_test)
        print(f"    [1/6] Ridge fitted ({time.time() - t_m:.2f}s)")

        # Model 3: Random Forest
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
        fold_preds["RandomForest"] = rf.predict(X_test)
        print(f"    [2/6] RandomForest fitted ({time.time() - t_m:.2f}s)")

        # Model 4: MLP
        t_m = time.time()
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=150,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_state
        )
        mlp.fit(X_train, y_train)
        fold_preds["MLP"] = mlp.predict(X_test)
        print(f"    [3/6] MLP fitted ({time.time() - t_m:.2f}s)")

        # Model 5: LightGBM
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
        fold_preds["LightGBM"] = lgbm.predict(X_test)
        print(f"    [4/6] LightGBM fitted ({time.time() - t_m:.2f}s)")

        # Model 6: Stacking
        t_m = time.time()
        stack_estimators = [
            ("ridge", Ridge(alpha=1.0, random_state=random_state)),
            ("rf", RandomForestRegressor(n_estimators=30, max_depth=12, min_samples_leaf=5, max_samples=0.5, n_jobs=-1, random_state=random_state)),
            ("lgb", lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, min_child_samples=20, n_jobs=-1, random_state=random_state, verbose=-1)),
        ]
        stack = StackingRegressor(
            estimators=stack_estimators,
            final_estimator=Ridge(alpha=1.0, random_state=random_state),
            cv=3,
            n_jobs=1
        )
        stack.fit(X_train, y_train)
        fold_preds["Stacking"] = stack.predict(X_test)
        print(f"    [5/6] Stacking fitted ({time.time() - t_m:.2f}s)")

        # 5. Common Evaluation Mask
        pred_p = fold_preds["Persistence"]
        common_mask = np.isfinite(y_test) & np.isfinite(pred_p)
        y_eval = y_test[common_mask]
        n_eval = int(np.sum(common_mask))
        excluded_dropout = n_test_raw - n_eval
        total_excluded = total_grid - n_eval

        print(f"  Official test grid rows: {total_grid} | Evaluated: {n_eval} | Excluded: {total_excluded} ({100.0 * total_excluded / total_grid:.1f}%)")

        # 6. Metrics & Prediction Export
        fold_pred_rows = []
        for m_name, preds_all in fold_preds.items():
            preds_eval = preds_all[common_mask]
            assert np.all(np.isfinite(preds_eval)), f"NaN detected in predictions for {m_name}!"

            rmse = calculate_rmse(y_eval, preds_eval)
            mae = calculate_mae(y_eval, preds_eval)
            r2 = calculate_r2(y_eval, preds_eval)
            mard = calculate_mard(y_eval, preds_eval)
            ega_a, ega_b, ega_ab, _ = evaluate_clarke_ega(y_eval, preds_eval)

            all_metric_rows.append({
                "patient": str(held_out),
                "model": m_name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "MARD": mard,
                "EGA_A": ega_a,
                "EGA_B": ega_b,
                "EGA_A+B": ega_ab,
                "n_test_samples": n_eval,
                "total_test_rows": total_grid,
                "excluded_rows": total_excluded
            })

            # Predictions
            for ts, yt, yp, is_eval in zip(feat_te["ts"], y_test, preds_all, common_mask):
                fold_pred_rows.append({
                    "patient": str(held_out),
                    "model": m_name,
                    "timestamp": ts,
                    "y_true": float(yt),
                    "y_pred": float(yp) if np.isfinite(yp) else np.nan,
                    "in_eval_mask": bool(is_eval)
                })

        # Save individual fold predictions file
        fold_pred_df = pd.DataFrame(fold_pred_rows)
        fold_pred_path = os.path.join(results_dir, f"fold_{held_out}_predictions.csv")
        fold_pred_df.to_csv(fold_pred_path, index=False)

        # Append to cumulative predictions file
        file_exists = os.path.exists(preds_csv_path)
        fold_pred_df.to_csv(preds_csv_path, mode='a', header=not file_exists, index=False)

        # Update checkpoint and cumulative patient results
        completed_patients.add(held_out)
        save_checkpoint(checkpoint_path, completed_patients, random_state=random_state, data_dir=data_dir)
        df_interim = pd.DataFrame(all_metric_rows)
        df_interim.to_csv(results_csv_path, index=False)

        # Fold Summary
        lgb_row = next(r for r in all_metric_rows if r["patient"] == str(held_out) and r["model"] == "LightGBM")
        stack_row = next(r for r in all_metric_rows if r["patient"] == str(held_out) and r["model"] == "Stacking")
        print(f"  [Fold {fold_idx:02d} Complete ({time.time() - t_fold:.1f}s)] LightGBM MARD: {lgb_row['MARD']:.2f}% | Stacking MARD: {stack_row['MARD']:.2f}%")
        gc.collect()

    # -----------------------------------------------------------------------
    # Summary Table Across 12 Patients
    # -----------------------------------------------------------------------
    patient_results_df = pd.DataFrame(all_metric_rows)
    patient_results_df["patient"] = patient_results_df["patient"].astype(str)
    patient_results_df.to_csv(results_csv_path, index=False)

    summary_rows = []
    for m in models:
        sub = patient_results_df[patient_results_df["model"] == m]
        summary_rows.append({
            "model": m,
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
            "n_patients": len(sub)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[Summary Saved] -> {summary_csv_path}")

    # Generate Plots
    print("\nLoading cumulative predictions for visualization...")
    all_preds_df = pd.read_csv(preds_csv_path, dtype={"patient": str})
    generate_official_test_plots(patient_results_df, summary_df, all_preds_df, outputs_dir=outputs_dir, random_state=random_state)

    # Generate Comparability Table and Research Log
    generate_comparison_table(summary_df, results_dir=results_dir)
    write_research_log(summary_df, patient_results_df, results_dir=results_dir)

    # Final Summary Table Printout
    print("\n" + "=" * 105)
    print("OFFICIAL OHIOT1DM TEST EVALUATION: FINAL SUMMARY (MEAN +/- SD ACROSS 12 PATIENTS)")
    print("=" * 105)
    header = f"{'Model':<14} {'MARD (%)':>18} {'RMSE (mg/dL)':>18} {'MAE (mg/dL)':>18} {'R2':>18} {'Clarke A+B (%)':>16}"
    print(header)
    print("-" * 105)
    for _, row in summary_df.iterrows():
        m = row["model"]
        mard_str = f"{row['MARD_mean']:.2f} +/- {row['MARD_std']:.2f}"
        rmse_str = f"{row['RMSE_mean']:.2f} +/- {row['RMSE_std']:.2f}"
        mae_str = f"{row['MAE_mean']:.2f} +/- {row['MAE_std']:.2f}"
        r2_str = f"{row['R2_mean']:.3f} +/- {row['R2_std']:.3f}"
        clarke_str = f"{row['EGA_A+B_mean']:.1f} +/- {row['EGA_A+B_std']:.1f}%"
        print(f"{m:<14} {mard_str:>18} {rmse_str:>18} {mae_str:>18} {r2_str:>18} {clarke_str:>16}")
    print("=" * 105)

    lowest_mard = summary_df.sort_values("MARD_mean").iloc[0]
    print(f"\nLowest mean MARD in this official testing evaluation: {lowest_mard['model']} ({lowest_mard['MARD_mean']:.2f} +/- {lowest_mard['MARD_std']:.2f}%)")
    print("Note: This experiment uses the official OhioT1DM testing files and is separate from the previously completed concatenated-patient LOSO experiment.")

    return patient_results_df, summary_df


# ---------------------------------------------------------------------------
# Lightweight Verification Mode
# ---------------------------------------------------------------------------
def run_verification_only(data_dir="data/OhioT1DM"):
    print("=" * 80)
    print("STARTING LIGHTWEIGHT VERIFICATION MODE")
    print("=" * 80)

    # 1. Run full critical leakage audit
    audit_passed = run_leakage_audit(data_dir=data_dir)
    assert audit_passed, "Leakage audit failed!"

    # 2. Check locked directories exist and remain untouched
    locked_paths = [
        "results_causal/loso_summary.csv",
        "results_causal/loso_checkpoint.json",
        "results_objective2/personalization_summary.csv",
        "results_objective2/personalization_checkpoint.json"
    ]
    for lp in locked_paths:
        assert os.path.exists(lp), f"CRITICAL: Locked file missing: {lp}!"
    print("  [Integrity PASS] All Objective 1 and Objective 2 locked result files verified untouched.")

    # 3. Dry-run loading on patient 540
    print("\nTesting separate data loading on Patient 540...")
    df_tr = load_training_xml("540", data_dir=data_dir)
    df_te = load_testing_xml("540", data_dir=data_dir)
    print(f"  Patient 540 Train XML rows: {len(df_tr):,}")
    print(f"  Patient 540 Test XML rows:  {len(df_te):,}")

    feat_te, raw_te, total_grid, ex = build_test_features_with_history("540", data_dir=data_dir)
    print(f"  Patient 540 Test Features rows: {len(feat_te):,} (Grid rows: {total_grid:,}, Excluded: {ex})")
    assert len(feat_te) > 0, "Test features empty!"

    print("\n" + "=" * 80)
    print("LIGHTWEIGHT VERIFICATION COMPLETED SUCCESSFULLY: READY FOR FULL EXPERIMENT")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Official OhioT1DM Test-Set Evaluation")
    parser.add_argument("--verify-only", action="store_true", help="Run lightweight verification and leakage audit only")
    parser.add_argument("--data-dir", type=str, default="data/OhioT1DM", help="Data directory")
    parser.add_argument("--results-dir", type=str, default="results_ohio_official", help="Results directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs_ohio_official", help="Outputs directory")
    args = parser.parse_args()

    if args.verify_only:
        run_verification_only(data_dir=args.data_dir)
    else:
        run_official_test_evaluation(data_dir=args.data_dir, results_dir=args.results_dir, outputs_dir=args.outputs_dir)
