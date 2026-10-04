import pandas as pd
import numpy as np
from pathlib import Path

RESULTS = Path("results_objective2")

patient_file = RESULTS / "personalization_patient_results.csv"
summary_file = RESULTS / "personalization_summary.csv"

df = pd.read_csv(patient_file)
summary = pd.read_csv(summary_file)

EXPECTED_PATIENTS = {540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596}
EXPECTED_P_LEVELS = {0, 5, 10, 20, 30, 50}
EXPECTED_MODELS = {"LightGBM", "Ridge", "Persistence"}

errors = []

print("=" * 70)
print("OBJECTIVE 2 — FINAL 216-RECORD INTEGRITY AUDIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. Basic row count
# ------------------------------------------------------------
print("\n[1] ROW COUNT")

print(f"Patient-level rows: {len(df)}")
print(f"Expected: 216")

if len(df) != 216:
    errors.append(f"Expected 216 rows, found {len(df)}")
    print("FAIL")
else:
    print("PASS")

# ------------------------------------------------------------
# 2. Required columns
# ------------------------------------------------------------
print("\n[2] REQUIRED COLUMNS")

required = {
    "patient", "p_level", "model",
    "RMSE", "MAE", "R2", "MARD",
    "EGA_A", "EGA_B", "EGA_A+B",
    "n_eval_samples", "n_personal_samples",
    "Delta_MARD", "Delta_RMSE"
}

missing = required - set(df.columns)

if missing:
    errors.append(f"Missing columns: {missing}")
    print("FAIL:", missing)
else:
    print("PASS")

# ------------------------------------------------------------
# 3. Patient coverage
# ------------------------------------------------------------
print("\n[3] PATIENT COVERAGE")

patients = set(df["patient"].unique())

print("Patients:", sorted(patients))
print("Expected:", sorted(EXPECTED_PATIENTS))

if patients != EXPECTED_PATIENTS:
    errors.append("Patient coverage mismatch")
    print("FAIL")
else:
    print("PASS")

# ------------------------------------------------------------
# 4. Personalization levels
# ------------------------------------------------------------
print("\n[4] PERSONALIZATION LEVELS")

levels = set(df["p_level"].unique())

print("Levels:", sorted(levels))
print("Expected:", sorted(EXPECTED_P_LEVELS))

if levels != EXPECTED_P_LEVELS:
    errors.append("Personalization-level mismatch")
    print("FAIL")
else:
    print("PASS")

# ------------------------------------------------------------
# 5. Model coverage
# ------------------------------------------------------------
print("\n[5] MODEL COVERAGE")

models = set(df["model"].unique())

print("Models:", sorted(models))
print("Expected:", sorted(EXPECTED_MODELS))

if models != EXPECTED_MODELS:
    errors.append("Model coverage mismatch")
    print("FAIL")
else:
    print("PASS")

# ------------------------------------------------------------
# 6. Duplicate combinations
# ------------------------------------------------------------
print("\n[6] DUPLICATE PATIENT/P/MODEL COMBINATIONS")

key = ["patient", "p_level", "model"]
duplicates = df[df.duplicated(key, keep=False)]

print("Duplicate rows:", len(duplicates))

if len(duplicates) > 0:
    errors.append("Duplicate patient/p_level/model combinations found")
    print("FAIL")
    print(duplicates[key].to_string(index=False))
else:
    print("PASS")

# ------------------------------------------------------------
# 7. Every combination exists exactly once
# ------------------------------------------------------------
print("\n[7] COMPLETE 12 × 6 × 3 DESIGN")

expected_combinations = 12 * 6 * 3
actual_combinations = df[key].drop_duplicates().shape[0]

print(f"Unique combinations: {actual_combinations}")
print(f"Expected: {expected_combinations}")

if actual_combinations != expected_combinations:
    errors.append("Incomplete 12x6x3 experimental design")
    print("FAIL")
else:
    print("PASS")

# ------------------------------------------------------------
# 8. NaN / infinite values
# ------------------------------------------------------------
print("\n[8] NUMERICAL INTEGRITY")

numeric_cols = [
    "RMSE", "MAE", "R2", "MARD",
    "EGA_A", "EGA_B", "EGA_A+B",
    "n_eval_samples", "n_personal_samples",
    "Delta_MARD", "Delta_RMSE"
]

nan_counts = df[numeric_cols].isna().sum()
inf_count = np.isinf(df[numeric_cols].to_numpy(dtype=float)).sum()

print("NaN values:", int(nan_counts.sum()))
print("Infinite values:", int(inf_count))

if nan_counts.sum() > 0 or inf_count > 0:
    errors.append("NaN or infinite numerical values found")
    print("FAIL")
else:
    print("PASS")

# ------------------------------------------------------------
# 9. Persistence invariance
# ------------------------------------------------------------
print("\n[9] PERSISTENCE INVARIANCE")

pers = df[df["model"] == "Persistence"]

mard_spread = pers.groupby("patient")["MARD"].agg(lambda x: x.max() - x.min()).max()
rmse_spread = pers.groupby("patient")["RMSE"].agg(lambda x: x.max() - x.min()).max()
mae_spread = pers.groupby("patient")["MAE"].agg(lambda x: x.max() - x.min()).max()
r2_spread = pers.groupby("patient")["R2"].agg(lambda x: x.max() - x.min()).max()

print(f"Maximum MARD spread: {mard_spread:.12g}")
print(f"Maximum RMSE spread: {rmse_spread:.12g}")
print(f"Maximum MAE spread: {mae_spread:.12g}")
print(f"Maximum R2 spread: {r2_spread:.12g}")

if max(mard_spread, rmse_spread, mae_spread, r2_spread) > 1e-10:
    errors.append("Persistence is not invariant across personalization levels")
    print("FAIL")
else:
    print("PASS")

# ------------------------------------------------------------
# 10. Evaluation sample invariance
# ------------------------------------------------------------
print("\n[10] FIXED TEST EVALUATION SIZE")

eval_spread = (
    df.groupby(["patient", "model"])["n_eval_samples"]
    .agg(lambda x: x.max() - x.min())
    .max()
)

print(f"Maximum n_eval_samples spread: {eval_spread}")

if eval_spread != 0:
    errors.append("Evaluation sample count changes across personalization levels")
    print("FAIL")
else:
    print("PASS")

# ------------------------------------------------------------
# 11. Personalization sample sizes
# ------------------------------------------------------------
print("\n[11] PERSONALIZATION SAMPLE COUNTS")

print(
    df.groupby(["patient", "p_level"])["n_personal_samples"]
    .first()
    .unstack()
    .to_string()
)

# ------------------------------------------------------------
# 12. Recompute summary statistics
# ------------------------------------------------------------
print("\n[12] RECOMPUTE AGGREGATE MEANS AND SAMPLE SD")

metric_cols = [
    "MARD", "RMSE", "MAE", "R2",
    "EGA_A", "EGA_B", "EGA_A+B",
    "Delta_MARD", "Delta_RMSE"
]

recomputed = (
    df.groupby(["model", "p_level"])[metric_cols]
    .agg(["mean", "std"])
    .reset_index()
)

# Flatten columns
recomputed.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in recomputed.columns
]

# Compare with supplied summary
summary_lookup = summary.set_index(["model", "p_level"])
recomputed_lookup = recomputed.set_index(["model", "p_level"])

mapping = {
    "MARD_mean": "MARD_mean",
    "MARD_std": "MARD_std",
    "RMSE_mean": "RMSE_mean",
    "RMSE_std": "RMSE_std",
    "MAE_mean": "MAE_mean",
    "MAE_std": "MAE_std",
    "R2_mean": "R2_mean",
    "R2_std": "R2_std",
    "EGA_A_mean": "EGA_A_mean",
    "EGA_A_std": "EGA_A_std",
    "EGA_B_mean": "EGA_B_mean",
    "EGA_B_std": "EGA_B_std",
    "EGA_A+B_mean": "EGA_A+B_mean",
    "EGA_A+B_std": "EGA_A+B_std",
    "Delta_MARD_mean": "Delta_MARD_mean",
    "Delta_MARD_std": "Delta_MARD_std",
    "Delta_RMSE_mean": "Delta_RMSE_mean",
    "Delta_RMSE_std": "Delta_RMSE_std",
}

max_difference = 0.0
summary_failures = []

for col in mapping:
    a = recomputed_lookup[col]
    b = summary_lookup[col]

    diff = (a - b).abs()
    current_max = diff.max()

    if current_max > max_difference:
        max_difference = current_max

    if current_max > 1e-9:
        summary_failures.append((col, current_max))

print(f"Maximum difference from supplied summary: {max_difference:.12g}")

if summary_failures:
    errors.append("Supplied summary does not match recomputed patient-level statistics")
    print("FAIL")
    for col, diff in summary_failures:
        print(f"  {col}: max difference = {diff}")
else:
    print("PASS")

# ------------------------------------------------------------
# 13. Verify Delta_MARD and Delta_RMSE
# ------------------------------------------------------------
print("\n[13] VERIFY PERSONALIZATION DELTAS")

delta_failures = []

for (patient, model), group in df.groupby(["patient", "model"]):

    baseline = group[group["p_level"] == 0]

    if len(baseline) != 1:
        delta_failures.append(
            f"{patient}/{model}: invalid P=0 baseline count"
        )
        continue

    baseline_mard = baseline["MARD"].iloc[0]
    baseline_rmse = baseline["RMSE"].iloc[0]

    expected_mard = baseline_mard - group["MARD"]
    expected_rmse = baseline_rmse - group["RMSE"]

    mard_error = np.max(
        np.abs(expected_mard.to_numpy() - group["Delta_MARD"].to_numpy())
    )

    rmse_error = np.max(
        np.abs(expected_rmse.to_numpy() - group["Delta_RMSE"].to_numpy())
    )

    if mard_error > 1e-9 or rmse_error > 1e-9:
        delta_failures.append(
            f"{patient}/{model}: "
            f"MARD error={mard_error}, RMSE error={rmse_error}"
        )

if delta_failures:
    errors.append("Delta calculations are incorrect")
    print("FAIL")
    for x in delta_failures:
        print(" ", x)
else:
    print("PASS")

# ------------------------------------------------------------
# 14. Count LightGBM P=50 improvements
# ------------------------------------------------------------
print("\n[14] LIGHTGBM P=50 PATIENT-LEVEL IMPROVEMENT")

lgbm50 = df[
    (df["model"] == "LightGBM") &
    (df["p_level"] == 50)
]

positive_mard = (lgbm50["Delta_MARD"] > 0).sum()
negative_mard = (lgbm50["Delta_MARD"] < 0).sum()
zero_mard = (lgbm50["Delta_MARD"].abs() <= 1e-12).sum()

positive_rmse = (lgbm50["Delta_RMSE"] > 0).sum()
negative_rmse = (lgbm50["Delta_RMSE"] < 0).sum()
zero_rmse = (lgbm50["Delta_RMSE"].abs() <= 1e-12).sum()

print(f"MARD improvement: {positive_mard}/12")
print(f"MARD worse:       {negative_mard}/12")
print(f"MARD unchanged:   {zero_mard}/12")

print(f"RMSE improvement: {positive_rmse}/12")
print(f"RMSE worse:       {negative_rmse}/12")
print(f"RMSE unchanged:   {zero_rmse}/12")

print("\nPatient-level LightGBM P=50 deltas:")
print(
    lgbm50[
        ["patient", "Delta_MARD", "Delta_RMSE"]
    ].sort_values("patient").to_string(index=False)
)

# ------------------------------------------------------------
# 15. Check chronological personalization sizes
# ------------------------------------------------------------
print("\n[15] EXPECTED PERSONALIZATION SAMPLE SIZES")

expected_fractions = {
    0: 0.00,
    5: 0.05,
    10: 0.10,
    20: 0.20,
    30: 0.30,
    50: 0.50,
}

size_failures = []

for patient in EXPECTED_PATIENTS:
    patient_rows = df[df["patient"] == patient]

    # Use LightGBM records because all models should share the same buffer.
    lgb = patient_rows[patient_rows["model"] == "LightGBM"]

    # Recover total usable rows from the 50% buffer.
    p50 = lgb[lgb["p_level"] == 50]["n_personal_samples"]

    if len(p50) != 1:
        size_failures.append(f"{patient}: missing P=50")
        continue

    total = int(p50.iloc[0] * 2)

    for p, frac in expected_fractions.items():
        actual = int(
            lgb[lgb["p_level"] == p]["n_personal_samples"].iloc[0]
        )

        expected = int(total * frac)

        # Allow floor/truncation behavior.
        if abs(actual - expected) > 1:
            size_failures.append(
                f"{patient} P={p}: actual={actual}, expected≈{expected}"
            )

if size_failures:
    errors.append("Personalization sample-size mismatch")
    print("FAIL")
    for x in size_failures:
        print(" ", x)
else:
    print("PASS")

# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------
print("\n" + "=" * 70)

if errors:
    print("AUDIT RESULT: FAIL")
    print("=" * 70)

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")

    print("\nDO NOT LOCK OBJECTIVE 2 YET.")
else:
    print("AUDIT RESULT: PASS")
    print("=" * 70)
    print("All major Objective 2 integrity checks passed.")
    print("216/216 experimental combinations verified.")
    print("Objective 2 is ready to be locked pending final methodology review.")

print("=" * 70)