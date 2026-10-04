"""
evaluate_personalization.py - Objective 2: Patient-Specific Glucose Prediction Personalization
OhioT1DM Research Project

Research Question:
"Does increasing the amount of patient-specific historical data improve 30-minute-ahead glucose prediction?"

Experimental Design:
1. Cohort: 12 OhioT1DM Patients (540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596).
2. Fixed Chronological Evaluation Set: Final 50% of each patient's usable sequence [50%, 100%].
3. Personalization Levels: P in {0%, 5%, 10%, 20%, 30%, 50%} drawn from the first 50% sequence.
4. Preprocessing: Population-fixed SimpleImputer and StandardScaler fitted ONLY on 11 population subjects.
5. Primary Model: LightGBM Continued Boosting (init_model=base_lgb).
6. Secondary Model: Ridge Prior-Regularized Adaptation (regularized toward base weights w_base).
7. Baseline Reference: Persistence on the fixed evaluation set.
8. Common Evaluation Mask: common_mask = np.isfinite(y_test) & np.isfinite(pred_persistence).
9. Checkpointing: Per-combination checkpointing (216 combinations total).
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from data_processing import load_subject_data, create_features

# Suppress verbose warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

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
# Data Caching & Discovery
# ---------------------------------------------------------------------------
def discover_patient_ids(data_dir="data/OhioT1DM"):
    files = glob.glob(os.path.join(data_dir, "*-ws-*.xml"))
    patient_ids = sorted(list(set(os.path.basename(f).split('-')[0] for f in files)))
    return patient_ids


def load_all_patients(patient_ids, data_dir="data/OhioT1DM", cache_file=None):
    alt_caches = ["results_causal/patient_features_causal.pkl", "results_objective2/patient_features_causal.pkl"]
    if cache_file:
        alt_caches.insert(0, cache_file)

    for cf in alt_caches:
        if os.path.exists(cf):
            try:
                import pickle
                print(f"[Data] Loading pre-computed causal patient features from {cf}...")
                with open(cf, "rb") as f:
                    patient_data = pickle.load(f)
                total_samples = sum(len(df) for df in patient_data.values())
                print(f"[Data] Loaded {len(patient_data)} patients. Total samples: {total_samples:,}\n")
                return patient_data
            except Exception as e:
                print(f"[Warning] Failed to load cache {cf}: {e}")

    patient_data = {}
    print("=" * 78)
    print(f"Generating causal features for {len(patient_ids)} OhioT1DM patients...")
    print("=" * 78)
    t0 = time.time()
    for pid in patient_ids:
        raw_df = load_subject_data(pid, data_dir=data_dir)
        feat_df = create_features(raw_df)
        patient_data[pid] = feat_df
        print(f"  Patient {pid}: {len(raw_df):6d} raw -> {len(feat_df):6d} causal feature rows")

    total_samples = sum(len(df) for df in patient_data.values())
    print(f"\nAll {len(patient_ids)} patients processed in {time.time() - t0:.1f}s. Total samples: {total_samples:,}\n")

    if cache_file:
        try:
            import pickle
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(patient_data, f)
            print(f"[Data] Saved cache to {cache_file}")
        except Exception as e:
            print(f"[Warning] Failed to save cache {cache_file}: {e}")

    return patient_data


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                return set(data.get("completed_keys", []))
        except Exception as e:
            print(f"[Warning] Failed to read checkpoint {checkpoint_path}: {e}")
    return set()


def save_checkpoint(checkpoint_path, completed_keys, random_state=42):
    tmp_path = checkpoint_path + ".tmp"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "completed_keys": sorted(list(completed_keys)),
        "count": len(completed_keys),
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
# Prior-Regularized Ridge Adaptation
# ---------------------------------------------------------------------------
def fit_prior_regularized_ridge(X_personal, y_personal, w_base, b_base, lam=50.0):
    """Solve the prior-regularized adaptation problem:
        min_{w, b} ||y_personal - (X_personal w + b)||^2 + lambda ||w - w_base||^2 + lambda (b - b_base)^2

    Mathematically equivalent to solving for residual correction:
        r = y_personal - (X_personal w_base + b_base)
        Delta_theta = (Z^T Z + lambda I)^{-1} Z^T r
        theta = theta_base + Delta_theta
    where Z = [X_personal, 1].
    """
    n, d = X_personal.shape
    Z = np.hstack([X_personal, np.ones((n, 1))])
    theta_base = np.append(w_base, b_base)

    Lambda = np.eye(d + 1) * lam
    r = y_personal - (Z @ theta_base)
    delta_theta = np.linalg.solve(Z.T @ Z + Lambda, Z.T @ r)
    theta_adapted = theta_base + delta_theta

    w_adapted = theta_adapted[:d]
    b_adapted = float(theta_adapted[d])
    return w_adapted, b_adapted


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------
def generate_plots(patient_results_df, summary_df, predictions_df, outputs_dir="outputs_objective2", random_state=42):
    os.makedirs(outputs_dir, exist_ok=True)
    print("\n" + "=" * 80)
    print("GENERATING OBJECTIVE 2 VISUALIZATIONS")
    print("=" * 80)

    # 1. Personalization Curve: MARD vs P
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"LightGBM": "#1f77b4", "Ridge": "#ff7f0e", "Persistence": "#7f7f7f"}
    markers = {"LightGBM": "o", "Ridge": "s", "Persistence": "^"}
    linestyles = {"LightGBM": "-", "Ridge": "--", "Persistence": ":"}

    for m in ["LightGBM", "Ridge", "Persistence"]:
        sub = summary_df[summary_df["model"] == m].sort_values("p_level")
        p = sub["p_level"].values
        mean = sub["MARD_mean"].values
        std = sub["MARD_std"].values
        ax.plot(p, mean, label=f"{m} (Mean)", color=colors[m], marker=markers[m], linestyle=linestyles[m], linewidth=2.2, markersize=7)
        ax.fill_between(p, mean - std, mean + std, color=colors[m], alpha=0.15)

    ax.set_title("Objective 2: 30-Minute Blood Glucose Prediction vs Personalization Data (MARD)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Personalization Data Available (% of Patient Timeline)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean Absolute Relative Difference - MARD (%) [± 1 SD]", fontsize=11, fontweight='bold')
    ax.set_xticks([0, 5, 10, 20, 30, 50])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    mard_plot_path = os.path.join(outputs_dir, "personalization_curve_mard.png")
    fig.savefig(mard_plot_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 1/4] -> {mard_plot_path}")

    # 2. Personalization Curve: RMSE vs P
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in ["LightGBM", "Ridge", "Persistence"]:
        sub = summary_df[summary_df["model"] == m].sort_values("p_level")
        p = sub["p_level"].values
        mean = sub["RMSE_mean"].values
        std = sub["RMSE_std"].values
        ax.plot(p, mean, label=f"{m} (Mean)", color=colors[m], marker=markers[m], linestyle=linestyles[m], linewidth=2.2, markersize=7)
        ax.fill_between(p, mean - std, mean + std, color=colors[m], alpha=0.15)

    ax.set_title("Objective 2: 30-Minute Blood Glucose Prediction vs Personalization Data (RMSE)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Personalization Data Available (% of Patient Timeline)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Root Mean Squared Error - RMSE (mg/dL) [± 1 SD]", fontsize=11, fontweight='bold')
    ax.set_xticks([0, 5, 10, 20, 30, 50])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    rmse_plot_path = os.path.join(outputs_dir, "personalization_curve_rmse.png")
    fig.savefig(rmse_plot_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 2/4] -> {rmse_plot_path}")

    # 3. Patient Adaptation Heatmap: Delta MARD
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    piv_lgb = patient_results_df[patient_results_df["model"] == "LightGBM"].pivot(index="patient", columns="p_level", values="Delta_MARD")[[5, 10, 20, 30, 50]]
    piv_ridge = patient_results_df[patient_results_df["model"] == "Ridge"].pivot(index="patient", columns="p_level", values="Delta_MARD")[[5, 10, 20, 30, 50]]

    vmax = max(abs(piv_lgb.values.max()), abs(piv_lgb.values.min()), abs(piv_ridge.values.max()), abs(piv_ridge.values.min()), 1.0)
    sns.heatmap(piv_lgb, annot=True, fmt="+.2f", cmap="vlag", center=0, vmin=-vmax, vmax=vmax, ax=ax1, cbar_kws={'label': 'Δ MARD (%) [Positive = Improvement]'})
    ax1.set_title("LightGBM Personalization Gain (Δ MARD %)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Personalization Level P (%)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Patient ID", fontsize=10, fontweight='bold')

    sns.heatmap(piv_ridge, annot=True, fmt="+.2f", cmap="vlag", center=0, vmin=-vmax, vmax=vmax, ax=ax2, cbar_kws={'label': 'Δ MARD (%) [Positive = Improvement]'})
    ax2.set_title("Ridge Personalization Gain (Δ MARD %)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Personalization Level P (%)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Patient ID", fontsize=10, fontweight='bold')

    plt.suptitle("Patient-Level Personalization Gain relative to Population Base (P = 0%)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(outputs_dir, "patient_adaptation_heatmap.png")
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 3/4] -> {heatmap_path}")

    # 4. Standard Clarke Error Grid Comparison: P=0 vs P=50%
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Subset LightGBM predictions on evaluation mask
    lgb_p0 = predictions_df[(predictions_df["model"] == "LightGBM") & (predictions_df["p_level"] == 0) & (predictions_df["in_eval_mask"] == True)]
    lgb_p50 = predictions_df[(predictions_df["model"] == "LightGBM") & (predictions_df["p_level"] == 50) & (predictions_df["in_eval_mask"] == True)]

    # Subsample up to 15,000 points if large
    n_sample = min(15000, len(lgb_p0))
    sample_p0 = lgb_p0.sample(n=n_sample, random_state=random_state).copy()
    sample_p50 = lgb_p50.sample(n=n_sample, random_state=random_state).copy()

    _, _, _, zones_p0 = evaluate_clarke_ega(sample_p0["y_true"].values, sample_p0["y_pred"].values)
    _, _, _, zones_p50 = evaluate_clarke_ega(sample_p50["y_true"].values, sample_p50["y_pred"].values)
    sample_p0["zone"] = zones_p0
    sample_p50["zone"] = zones_p50

    zone_colors = {'A': '#2ca02c', 'B': '#1f77b4', 'C': '#ff7f0e', 'D': '#d62728', 'E': '#9467bd'}

    for ax, s_data, p_val in zip([ax1, ax2], [sample_p0, sample_p50], [0, 50]):
        # Boundaries
        ax.plot([0, 450], [0, 450], 'k:', linewidth=1)
        ax.plot([0, 450], [0, 450 * 1.2], 'k--', alpha=0.6, linewidth=1)
        ax.plot([0, 450], [0, 450 * 0.8], 'k--', alpha=0.6, linewidth=1)
        ax.axvline(70, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(70, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(180, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(180, color='gray', linestyle=':', alpha=0.5)

        for z in ['A', 'B', 'C', 'D', 'E']:
            z_pts = s_data[s_data["zone"] == z]
            if len(z_pts) > 0:
                ax.scatter(z_pts["y_true"], z_pts["y_pred"], c=zone_colors.get(z, 'gray'), alpha=0.35, s=14, label=f"Zone {z} ({100*len(z_pts)/len(s_data):.1f}%)")

        a_pct = 100.0 * np.sum(s_data["zone"] == 'A') / len(s_data)
        b_pct = 100.0 * np.sum(s_data["zone"] == 'B') / len(s_data)
        ab_pct = a_pct + b_pct

        lbl = "Population Base (P = 0%)" if p_val == 0 else "Fully Adapted (P = 50%)"
        ax.set_title(
            f"LightGBM {lbl}\n"
            f"Zone A: {a_pct:.1f}% | Zone B: {b_pct:.1f}% | Zone A+B: {ab_pct:.1f}%",
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=11)
        ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=11)
        ax.set_xlim(0, 420)
        ax.set_ylim(0, 420)
        ax.legend(title="Clarke Zones", loc="upper left")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    clarke_path = os.path.join(outputs_dir, "clarke_comparison_p0_vs_p50.png")
    fig.savefig(clarke_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot 4/4] -> {clarke_path}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Full 12-Patient Personalization Evaluation
# ---------------------------------------------------------------------------
def run_full_experiment(data_dir="data/OhioT1DM", results_dir="results_objective2", outputs_dir="outputs_objective2", random_state=42):
    t_start = time.time()
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    patient_ids = ["540", "544", "552", "559", "563", "567", "570", "575", "584", "588", "591", "596"]
    p_levels = [0, 5, 10, 20, 30, 50]
    models = ["Persistence", "Ridge", "LightGBM"]
    total_expected_combinations = len(patient_ids) * len(p_levels) * len(models)  # 216

    checkpoint_path = os.path.join(results_dir, "personalization_checkpoint.json")
    results_csv_path = os.path.join(results_dir, "personalization_patient_results.csv")
    preds_csv_path = os.path.join(results_dir, "personalization_predictions.csv")
    summary_csv_path = os.path.join(results_dir, "personalization_summary.csv")

    completed_keys = load_checkpoint(checkpoint_path)

    # Load existing patient results if resuming
    if os.path.exists(results_csv_path):
        try:
            existing_results_df = pd.read_csv(results_csv_path, dtype={"patient": str})
            all_metric_rows = existing_results_df.to_dict('records')
            # Ensure completed_keys only contains keys that actually exist in the saved metric rows
            existing_keys = set(f"{str(r['patient'])}_p{r['p_level']}_{r['model']}" for r in all_metric_rows)
            completed_keys = completed_keys.intersection(existing_keys)
            print(f"[Resume] Loaded {len(all_metric_rows)} existing metric records from {results_csv_path}")
        except Exception as e:
            print(f"[Warning] Could not read existing results: {e}")
            all_metric_rows = []
    else:
        all_metric_rows = []

    # Clean any predictions that do not belong to completed combinations
    if os.path.exists(preds_csv_path):
        try:
            df_p = pd.read_csv(preds_csv_path)
            p_keys = df_p["patient"].astype(str) + "_p" + df_p["p_level"].astype(str) + "_" + df_p["model"].astype(str)
            mask_valid = p_keys.isin(completed_keys)
            if not mask_valid.all():
                df_p = df_p[mask_valid]
                df_p.to_csv(preds_csv_path, index=False)
                print(f"[Resume] Cleaned incomplete predictions from {preds_csv_path} ({len(df_p):,} rows retained)")
        except Exception as e:
            print(f"[Warning] Could not filter predictions: {e}")

    print("=" * 80)
    print("OBJECTIVE 2: FULL 12-PATIENT PERSONALIZATION EXPERIMENT")
    print("=" * 80)
    print(f"Number of patients: {len(patient_ids)} ({', '.join(patient_ids)})")
    print(f"Personalization levels: {p_levels}")
    print(f"Models: {models}")
    print(f"Expected total combinations: {total_expected_combinations}")
    print(f"Results directory: {results_dir}")
    print(f"Outputs directory: {outputs_dir}")
    print(f"Checkpoint status: {len(completed_keys)} / {total_expected_combinations} completed")
    print("=" * 80)

    # Load patient data
    cache_path = os.path.join(results_dir, "patient_features_causal.pkl")
    patient_data = load_all_patients(patient_ids, data_dir=data_dir, cache_file=cache_path)
    sample_df = next(iter(patient_data.values()))
    feature_cols = [c for c in sample_df.columns if c not in ["target", "ts"]]

    # Evaluate each patient
    for p_idx, held_out in enumerate(patient_ids, 1):
        p_keys = [f"{held_out}_p{p}_{m}" for p in p_levels for m in models]
        if all(k in completed_keys for k in p_keys):
            print(f"\n[{p_idx:02d}/12] Patient {held_out}: All 18 combinations already completed. Skipping.")
            continue

        print(f"\n[{p_idx:02d}/12] Patient {held_out}: Starting evaluation...")
        patient_df = patient_data[held_out].copy()
        patient_df = patient_df.sort_values("ts").reset_index(drop=True)
        N_total = len(patient_df)

        # 1. Fixed Chronological 50% Evaluation Set [50%, 100%]
        split_idx = int(np.floor(0.50 * N_total))
        test_df = patient_df.iloc[split_idx:].copy().reset_index(drop=True)
        N_test_rows = len(test_df)

        # 2. Build Population Training Set from other 11 patients
        train_dfs = [patient_data[p] for p in patient_ids if p != held_out]
        train_df = pd.concat(train_dfs, ignore_index=True)

        X_pop_raw = train_df[feature_cols].values
        y_pop = train_df["target"].values

        # 3. Fit Population Preprocessing (FROZEN for all P levels)
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        X_pop_imp = np.nan_to_num(imputer.fit_transform(X_pop_raw), nan=0.0)

        scaler = StandardScaler()
        X_pop = scaler.fit_transform(X_pop_imp)

        # Transform Fixed Test Set once
        X_test_raw = test_df[feature_cols].values
        y_test = test_df["target"].values
        X_test_imp = np.nan_to_num(imputer.transform(X_test_raw), nan=0.0)
        X_test = scaler.transform(X_test_imp)

        # 4. Fit Population Base Models
        t_base = time.time()
        base_lgbm = lgb.LGBMRegressor(
            objective='regression',
            learning_rate=0.05,
            n_estimators=150,
            num_leaves=31,
            min_child_samples=20,
            random_state=random_state,
            verbose=-1
        )
        base_lgbm.fit(X_pop, y_pop)
        base_booster = base_lgbm.booster_

        base_ridge = Ridge(alpha=1.0, random_state=random_state)
        base_ridge.fit(X_pop, y_pop)
        w_base = base_ridge.coef_
        b_base = float(base_ridge.intercept_)

        # Persistence Reference on test set
        pred_persistence = test_df["glucose"].values

        # 5. Define Common Evaluation Mask (Fixed across ALL models and P levels)
        common_mask = np.isfinite(y_test) & np.isfinite(pred_persistence)
        y_eval = y_test[common_mask]
        n_eval = int(np.sum(common_mask))
        n_excluded = N_test_rows - n_eval

        print(f"  Population models trained on {len(y_pop):,} samples ({time.time() - t_base:.1f}s)")
        print(f"  Test rows: {N_test_rows} | Evaluated: {n_eval} | Excluded (dropout NaN): {n_excluded}")

        # 6. Evaluate Across Personalization Levels P
        for p in p_levels:
            # Check if all models for this P are already completed
            level_keys = [f"{held_out}_p{p}_{m}" for m in models]
            if all(k in completed_keys for k in level_keys):
                continue

            n_personal = int(np.floor((p / 100.0) * N_total))
            personal_df = patient_df.iloc[:n_personal].copy().reset_index(drop=True) if p > 0 else None

            # Verify no overlap with test set
            if p > 0:
                assert personal_df["ts"].max() < test_df["ts"].min(), f"Leakage: Personal buffer overlaps test set for {held_out} P={p}!"

            # A. PERSISTENCE
            pred_p = pred_persistence

            # B. RIDGE
            if p == 0:
                pred_r = base_ridge.predict(X_test)
            else:
                X_pers_raw = personal_df[feature_cols].values
                y_pers = personal_df["target"].values
                X_pers_imp = np.nan_to_num(imputer.transform(X_pers_raw), nan=0.0)
                X_pers = scaler.transform(X_pers_imp)
                w_adapt, b_adapt = fit_prior_regularized_ridge(X_pers, y_pers, w_base, b_base, lam=50.0)
                pred_r = X_test @ w_adapt + b_adapt

            # C. LIGHTGBM
            if p == 0:
                pred_lgb = base_booster.predict(X_test)
            else:
                X_pers_raw = personal_df[feature_cols].values
                y_pers = personal_df["target"].values
                X_pers_imp = np.nan_to_num(imputer.transform(X_pers_raw), nan=0.0)
                X_pers = scaler.transform(X_pers_imp)

                adapt_params = {
                    'objective': 'regression',
                    'learning_rate': 0.02,
                    'num_leaves': 15,
                    'max_depth': 6,
                    'min_child_samples': 20,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': random_state,
                    'verbose': -1
                }

                if p in [5, 10]:
                    dtrain_adapt = lgb.Dataset(X_pers, label=y_pers)
                    adapted_booster = lgb.train(
                        adapt_params,
                        dtrain_adapt,
                        num_boost_round=20,
                        init_model=base_booster
                    )
                else:  # P in [20, 30, 50]
                    n_p_total = len(X_pers)
                    n_tr = int(np.floor(0.80 * n_p_total))
                    X_tr, y_tr = X_pers[:n_tr], y_pers[:n_tr]
                    X_val, y_val = X_pers[n_tr:], y_pers[n_tr:]

                    dtrain_adapt = lgb.Dataset(X_tr, label=y_tr)
                    dval_adapt = lgb.Dataset(X_val, label=y_val, reference=dtrain_adapt)
                    callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]

                    adapted_booster = lgb.train(
                        adapt_params,
                        dtrain_adapt,
                        num_boost_round=30,
                        valid_sets=[dval_adapt],
                        callbacks=callbacks,
                        init_model=base_booster
                    )

                pred_lgb = adapted_booster.predict(X_test)

            model_preds = {
                "Persistence": pred_p,
                "Ridge": pred_r,
                "LightGBM": pred_lgb
            }

            # Predictions to export for this (patient, p)
            level_preds = []

            for m_name in models:
                comb_key = f"{held_out}_p{p}_{m_name}"
                if comb_key in completed_keys:
                    continue

                preds_all = model_preds[m_name]
                preds_eval = preds_all[common_mask]
                assert np.all(np.isfinite(preds_eval)), f"NaN in predictions for {comb_key}!"

                rmse = calculate_rmse(y_eval, preds_eval)
                mae = calculate_mae(y_eval, preds_eval)
                r2 = calculate_r2(y_eval, preds_eval)
                mard = calculate_mard(y_eval, preds_eval)
                ega_a, ega_b, ega_ab, _ = evaluate_clarke_ega(y_eval, preds_eval)

                metric_entry = {
                    "patient": held_out,
                    "p_level": p,
                    "model": m_name,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                    "MARD": mard,
                    "EGA_A": ega_a,
                    "EGA_B": ega_b,
                    "EGA_A+B": ega_ab,
                    "n_eval_samples": n_eval,
                    "n_personal_samples": n_personal
                }
                all_metric_rows.append(metric_entry)

                # Collect predictions
                for ts, yt, yp, is_eval in zip(test_df["ts"], y_test, preds_all, common_mask):
                    level_preds.append({
                        "patient": held_out,
                        "p_level": p,
                        "model": m_name,
                        "timestamp": ts,
                        "y_true": float(yt),
                        "y_pred": float(yp) if np.isfinite(yp) else np.nan,
                        "in_eval_mask": bool(is_eval)
                    })

                completed_keys.add(comb_key)

            # Append predictions to CSV
            if level_preds:
                df_lp = pd.DataFrame(level_preds)
                file_exists = os.path.exists(preds_csv_path)
                df_lp.to_csv(preds_csv_path, mode='a', header=not file_exists, index=False)
                del level_preds, df_lp

            # Save checkpoint after each P level
            save_checkpoint(checkpoint_path, completed_keys, random_state=random_state)

            # Immediate interim progress report
            lgb_res = next(r for r in all_metric_rows if r["patient"] == held_out and r["model"] == "LightGBM" and r["p_level"] == p)
            ridge_res = next(r for r in all_metric_rows if r["patient"] == held_out and r["model"] == "Ridge" and r["p_level"] == p)
            print(f"  [P={p:>2d}% (N_buf={n_personal:5d})] LightGBM MARD: {lgb_res['MARD']:5.2f}% (RMSE: {lgb_res['RMSE']:5.2f}) | Ridge MARD: {ridge_res['MARD']:5.2f}% (RMSE: {ridge_res['RMSE']:5.2f})")

        # Save cumulative patient results CSV after each patient
        df_interim = pd.DataFrame(all_metric_rows)
        df_interim.to_csv(results_csv_path, index=False)
        gc.collect()

    print("\n" + "=" * 80)
    print("ALL 12 PATIENTS EVALUATION COMPLETED")
    print(f"Elapsed Time: {time.time() - t_start:.1f}s")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Calculate Personalization Gain Delta_MARD and Delta_RMSE
    # -----------------------------------------------------------------------
    patient_results_df = pd.DataFrame(all_metric_rows)
    patient_results_df["patient"] = patient_results_df["patient"].astype(str)

    # Sort deterministically
    patient_results_df = patient_results_df.sort_values(["patient", "model", "p_level"]).reset_index(drop=True)

    delta_mards = []
    delta_rmses = []
    for _, row in patient_results_df.iterrows():
        pid = str(row["patient"])
        m = str(row["model"])
        base_row = patient_results_df[(patient_results_df["patient"] == pid) & (patient_results_df["model"] == m) & (patient_results_df["p_level"] == 0)]
        base_mard = base_row["MARD"].values[0]
        base_rmse = base_row["RMSE"].values[0]
        delta_mards.append(float(base_mard - row["MARD"]))
        delta_rmses.append(float(base_rmse - row["RMSE"]))

    patient_results_df["Delta_MARD"] = delta_mards
    patient_results_df["Delta_RMSE"] = delta_rmses
    patient_results_df.to_csv(results_csv_path, index=False)
    print(f"[Results Saved] Patient results -> {results_csv_path}")

    # -----------------------------------------------------------------------
    # Aggregate Summary Table (Mean ± SD across 12 patients)
    # -----------------------------------------------------------------------
    summary_rows = []
    for m in models:
        for p in p_levels:
            sub = patient_results_df[(patient_results_df["model"] == m) & (patient_results_df["p_level"] == p)]
            summary_rows.append({
                "model": m,
                "p_level": p,
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
                "Delta_MARD_mean": float(sub["Delta_MARD"].mean()),
                "Delta_MARD_std": float(sub["Delta_MARD"].std(ddof=1)),
                "Delta_RMSE_mean": float(sub["Delta_RMSE"].mean()),
                "Delta_RMSE_std": float(sub["Delta_RMSE"].std(ddof=1)),
                "n_patients": len(sub)
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Results Saved] Summary -> {summary_csv_path}")

    # -----------------------------------------------------------------------
    # Generate Plots
    # -----------------------------------------------------------------------
    print("\nLoading predictions for Clarke Error Grid plot...")
    predictions_df = pd.read_csv(preds_csv_path)
    generate_plots(patient_results_df, summary_df, predictions_df, outputs_dir=outputs_dir, random_state=random_state)

    # -----------------------------------------------------------------------
    # Automated Verification Checks (10 Points)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING AUTOMATED VERIFICATION CHECKS")
    print("=" * 80)

    # Check 1: 216 combinations completed
    assert len(completed_keys) == 216, f"Verification Failed: {len(completed_keys)} combinations in checkpoint, expected 216!"
    assert len(patient_results_df) == 216, f"Verification Failed: {len(patient_results_df)} rows in results, expected 216!"
    print(f"  [PASS] 1. Total combinations completed: {len(patient_results_df)} / 216")

    # Check 2: No missing combinations
    expected_combos = set((str(pid), int(p), str(m)) for pid in patient_ids for p in p_levels for m in models)
    actual_combos = set((str(r["patient"]), int(r["p_level"]), str(r["model"])) for _, r in patient_results_df.iterrows())
    missing = expected_combos - actual_combos
    assert len(missing) == 0, f"Verification Failed: Missing combinations: {missing}"
    print(f"  [PASS] 2. No missing combinations (0 missing)")

    # Check 3: No duplicate combinations
    dup_count = int(patient_results_df.duplicated(subset=["patient", "p_level", "model"]).sum())
    assert dup_count == 0, f"Verification Failed: Found {dup_count} duplicate rows!"
    print(f"  [PASS] 3. No duplicate combinations (0 duplicates)")

    # Check 4: All 12 patients present at all 6 personalization levels
    for pid in patient_ids:
        for p in p_levels:
            for m in models:
                assert (str(pid), int(p), str(m)) in actual_combos, f"Missing combination: {pid}, P={p}, {m}"
    print(f"  [PASS] 4. All 12 patients present at all 6 personalization levels")

    # Check 5: Persistence metrics invariant across P for each patient
    for pid in patient_ids:
        p_sub = patient_results_df[(patient_results_df["patient"] == str(pid)) & (patient_results_df["model"] == "Persistence")]
        mard_spread = p_sub["MARD"].max() - p_sub["MARD"].min()
        rmse_spread = p_sub["RMSE"].max() - p_sub["RMSE"].min()
        assert mard_spread < 1e-6, f"Verification Failed: Persistence MARD not invariant for patient {pid} (spread={mard_spread})"
        assert rmse_spread < 1e-6, f"Verification Failed: Persistence RMSE not invariant for patient {pid} (spread={rmse_spread})"
    print(f"  [PASS] 5. Persistence metrics strictly invariant across P for all 12 patients (spread < 1e-6)")

    # Check 6: Every P level uses the same fixed chronological test rows within each patient
    for pid in patient_ids:
        pid_sub = patient_results_df[patient_results_df["patient"] == str(pid)]
        assert pid_sub["n_eval_samples"].nunique() == 1, f"Verification Failed: Variable n_eval_samples for patient {pid}!"
    print(f"  [PASS] 6. Fixed chronological evaluation set invariant across all P levels and models")

    # Check 7: Aggregate summary calculated as mean ± SD across patients
    for m in models:
        for p in p_levels:
            sub = patient_results_df[(patient_results_df["model"] == m) & (patient_results_df["p_level"] == p)]
            sum_row = summary_df[(summary_df["model"] == m) & (summary_df["p_level"] == p)].iloc[0]
            assert abs(sub["MARD"].mean() - sum_row["MARD_mean"]) < 1e-6, "Summary MARD mean mismatch!"
            assert abs(sub["MARD"].std(ddof=1) - sum_row["MARD_std"]) < 1e-6, "Summary MARD std mismatch!"
    print(f"  [PASS] 7. Aggregate summary strictly calculated as mean ± SD across 12 patients")

    # Check 8: All expected output files exist and are non-empty
    expected_files = [
        checkpoint_path,
        results_csv_path,
        summary_csv_path,
        preds_csv_path,
        os.path.join(outputs_dir, "personalization_curve_mard.png"),
        os.path.join(outputs_dir, "personalization_curve_rmse.png"),
        os.path.join(outputs_dir, "patient_adaptation_heatmap.png"),
        os.path.join(outputs_dir, "clarke_comparison_p0_vs_p50.png"),
    ]
    for ef in expected_files:
        assert os.path.exists(ef) and os.path.getsize(ef) > 0, f"Verification Failed: Missing or empty file {ef}!"
    print(f"  [PASS] 8. All {len(expected_files)} expected output files exist and are non-empty")

    # Check 9: Plots generated
    print(f"  [PASS] 9. All 4 publication-quality plots successfully rendered")

    # Check 10: Final summary printout
    print("\n" + "=" * 105)
    print("OBJECTIVE 2: FINAL AGGREGATE PERSONALIZATION SUMMARY (MEAN +/- SD ACROSS 12 PATIENTS)")
    print("=" * 105)
    header = f"{'Model':<12} {'P Level':>8} {'MARD (%)':>16} {'RMSE (mg/dL)':>18} {'Delta MARD (%)':>18} {'Delta RMSE':>18} {'Clarke A+B (%)':>16}"
    print(header)
    print("-" * 105)
    for _, row in summary_df.iterrows():
        m = row["model"]
        p = int(row["p_level"])
        mard_str = f"{row['MARD_mean']:.2f} +/- {row['MARD_std']:.2f}"
        rmse_str = f"{row['RMSE_mean']:.2f} +/- {row['RMSE_std']:.2f}"
        d_mard_str = f"{row['Delta_MARD_mean']:+.2f} +/- {row['Delta_MARD_std']:.2f}"
        d_rmse_str = f"{row['Delta_RMSE_mean']:+.2f} +/- {row['Delta_RMSE_std']:.2f}"
        clarke_str = f"{row['EGA_A+B_mean']:.1f} +/- {row['EGA_A+B_std']:.1f}%"
        print(f"{m:<12} {p:>7d}% {mard_str:>16} {rmse_str:>18} {d_mard_str:>18} {d_rmse_str:>18} {clarke_str:>16}")
    print("=" * 105)

    return patient_results_df, summary_df


# ---------------------------------------------------------------------------
# Short Validation Mode (Patient 540)
# ---------------------------------------------------------------------------
def run_validation_mode(data_dir="data/OhioT1DM", results_dir="results_objective2", random_state=42):
    print("=" * 80)
    print("STARTING OBJECTIVE 2 SHORT VALIDATION MODE (PATIENT 540)")
    print("=" * 80)

    os.makedirs(results_dir, exist_ok=True)
    patient_ids = discover_patient_ids(data_dir)
    cache_path = os.path.join(results_dir, "patient_features_causal.pkl")
    patient_data = load_all_patients(patient_ids, data_dir=data_dir, cache_file=cache_path)

    sample_df = next(iter(patient_data.values()))
    feature_cols = [c for c in sample_df.columns if c not in ["target", "ts"]]

    test_patient = "540"
    p_levels = [0, 5, 10, 20, 30, 50]
    models = ["Persistence", "Ridge", "LightGBM"]

    checkpoint_path = os.path.join(results_dir, "personalization_checkpoint_validation.json")
    completed_keys = set()

    patient_df = patient_data[test_patient].copy().sort_values("ts").reset_index(drop=True)
    N_total = len(patient_df)
    split_idx = int(np.floor(0.50 * N_total))
    test_df = patient_df.iloc[split_idx:].copy().reset_index(drop=True)
    N_test_rows = len(test_df)

    train_dfs = [patient_data[p] for p in patient_ids if p != test_patient]
    train_df = pd.concat(train_dfs, ignore_index=True)
    X_pop_raw = train_df[feature_cols].values
    y_pop = train_df["target"].values

    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X_pop_imp = np.nan_to_num(imputer.fit_transform(X_pop_raw), nan=0.0)
    scaler = StandardScaler()
    X_pop = scaler.fit_transform(X_pop_imp)

    X_test_raw = test_df[feature_cols].values
    y_test = test_df["target"].values
    X_test_imp = np.nan_to_num(imputer.transform(X_test_raw), nan=0.0)
    X_test = scaler.transform(X_test_imp)

    base_lgbm = lgb.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=150, num_leaves=31, min_child_samples=20, random_state=random_state, verbose=-1)
    base_lgbm.fit(X_pop, y_pop)
    base_booster = base_lgbm.booster_

    base_ridge = Ridge(alpha=1.0, random_state=random_state)
    base_ridge.fit(X_pop, y_pop)
    w_base = base_ridge.coef_
    b_base = float(base_ridge.intercept_)

    pred_persistence = test_df["glucose"].values
    common_mask = np.isfinite(y_test) & np.isfinite(pred_persistence)
    y_eval = y_test[common_mask]
    n_eval = int(np.sum(common_mask))

    val_metric_rows = []
    for p in p_levels:
        n_personal = int(np.floor((p / 100.0) * N_total))
        personal_df = patient_df.iloc[:n_personal].copy().reset_index(drop=True) if p > 0 else None

        pred_p = pred_persistence
        if p == 0:
            pred_r = base_ridge.predict(X_test)
            pred_lgb = base_booster.predict(X_test)
        else:
            X_pers_raw = personal_df[feature_cols].values
            y_pers = personal_df["target"].values
            X_pers_imp = np.nan_to_num(imputer.transform(X_pers_raw), nan=0.0)
            X_pers = scaler.transform(X_pers_imp)
            w_adapt, b_adapt = fit_prior_regularized_ridge(X_pers, y_pers, w_base, b_base, lam=50.0)
            pred_r = X_test @ w_adapt + b_adapt

            adapt_params = {'objective': 'regression', 'learning_rate': 0.02, 'num_leaves': 15, 'max_depth': 6, 'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': random_state, 'verbose': -1}
            if p in [5, 10]:
                dtrain_adapt = lgb.Dataset(X_pers, label=y_pers)
                adapted_booster = lgb.train(adapt_params, dtrain_adapt, num_boost_round=20, init_model=base_booster)
            else:
                n_tr = int(np.floor(0.80 * len(X_pers)))
                dtrain_adapt = lgb.Dataset(X_pers[:n_tr], label=y_pers[:n_tr])
                dval_adapt = lgb.Dataset(X_pers[n_tr:], label=y_pers[n_tr:], reference=dtrain_adapt)
                adapted_booster = lgb.train(adapt_params, dtrain_adapt, num_boost_round=30, valid_sets=[dval_adapt], callbacks=[lgb.early_stopping(10, verbose=False)], init_model=base_booster)
            pred_lgb = adapted_booster.predict(X_test)

        preds_dict = {"Persistence": pred_p, "Ridge": pred_r, "LightGBM": pred_lgb}
        for m_name in models:
            preds_eval = preds_dict[m_name][common_mask]
            val_metric_rows.append({
                "patient": test_patient,
                "p_level": p,
                "model": m_name,
                "RMSE": calculate_rmse(y_eval, preds_eval),
                "MAE": calculate_mae(y_eval, preds_eval),
                "R2": calculate_r2(y_eval, preds_eval),
                "MARD": calculate_mard(y_eval, preds_eval),
                "EGA_A": evaluate_clarke_ega(y_eval, preds_eval)[0],
                "EGA_B": evaluate_clarke_ega(y_eval, preds_eval)[1],
                "EGA_A+B": evaluate_clarke_ega(y_eval, preds_eval)[2],
                "n_eval_samples": n_eval,
                "n_personal_samples": n_personal
            })

    val_metrics_df = pd.DataFrame(val_metric_rows)
    val_csv_path = os.path.join(results_dir, "validation_patient_540_results.csv")
    val_metrics_df.to_csv(val_csv_path, index=False)
    print(f"[Validation Saved] Results -> {val_csv_path}")
    return val_metrics_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Objective 2 Personalization Evaluation")
    parser.add_argument("--validate-patient", type=str, default=None, help="Run validation mode on single patient (e.g. 540)")
    parser.add_argument("--results-dir", type=str, default="results_objective2", help="Results directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs_objective2", help="Outputs directory")
    parser.add_argument("--run-all", action="store_true", help="Run full 12-patient experiment")
    args = parser.parse_args()

    if args.validate_patient:
        run_validation_mode(results_dir=args.results_dir)
    else:
        run_full_experiment(results_dir=args.results_dir, outputs_dir=args.outputs_dir)
