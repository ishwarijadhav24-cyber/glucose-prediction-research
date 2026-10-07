"""Build frontend/src/data/research.json from the verified research result files (run from the repo root).

Read only. Uses the current causal results only (results_causal/, results_objective2/, results_objective3/,
results_evidence/); the historical pre-causal folders (results/, results_pre_causal/) are never read.
Every value on the website's research pages comes from this file, so nothing is typed in by hand.

    python frontend/scripts/build_research_data.py
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "src" / "data" / "research.json"
SOURCES = {
    "loso_summary": "results_causal/loso_summary.csv",
    "loso_patients": "results_causal/loso_patient_results.csv",
    "loso_vs_persistence": "results_evidence/A2_obj1_vs_persistence.csv",
    "loso_lightgbm_pairs": "results_evidence/A3_obj1_lightgbm_pairs.csv",
    "pers_summary": "results_objective2/personalization_summary.csv",
    "pers_patients": "results_objective2/personalization_patient_results.csv",
    "pers_vs_p0": "results_evidence/B2_obj2_vs_P0.csv",
    "pers_days": "results_evidence/B4_obj2_levels_in_days.csv",
    "hupa_main": "results_objective3/hupa_summary_main.csv",
    "hupa_all": "results_objective3/hupa_summary_all.csv",
    "hupa_patients": "results_objective3/hupa_patient_results.csv",
    "hupa_vs_persistence": "results_evidence/C2_hupa_vs_persistence.csv",
    "gap": "results_evidence/C4_gap_decomposition_ci.csv",
    "ranges": "results_evidence/C5_metrics_by_glucose_range.csv",
    "hypo_episodes": "results_evidence/hypo_alerts/hypo_episode_level.csv",
    "bucketing": "results_evidence/event_bucketing/bucketing_totals.csv",
    "bucketing_diff": "results_evidence/event_bucketing/bucketing_prediction_diff.csv",
    "counts": "results_evidence/F3_patient_counts.csv",
    "features": "models_objective3/feature_list.json",
}
MODEL_ORDER = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"]


def csv(key: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / SOURCES[key])


def records(df: pd.DataFrame, cols: list[str], rename: dict | None = None) -> list[dict]:
    out = df[cols].rename(columns=rename or {})
    return json.loads(out.round(4).to_json(orient="records"))


def sha(key: str) -> str:
    return hashlib.sha256((ROOT / SOURCES[key]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def summary(df: pd.DataFrame) -> list[dict]:
    cols = ["model", "MARD_mean", "MARD_std", "RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean",
            "EGA_A_mean", "EGA_A+B_mean"]
    df = df.set_index("model").loc[[m for m in MODEL_ORDER if m in set(df["model"])]].reset_index()
    return records(df, cols, {"EGA_A+B_mean": "EGA_AB_mean"})


def build() -> dict:
    loso_p = csv("loso_patients")
    loso_p["patient"] = loso_p["patient"].astype(str)
    pers_p = csv("pers_patients")
    pers_p["patient"] = pers_p["patient"].astype(str)
    hupa_p = csv("hupa_patients")
    c2 = csv("hupa_vs_persistence")
    gap = csv("gap")
    hypo = csv("hypo_episodes")
    hypo = hypo[hypo["threshold"] == 70]
    bucket = csv("bucketing").iloc[0]
    bdiff = csv("bucketing_diff").iloc[0]
    features = json.loads((ROOT / SOURCES["features"]).read_text(encoding="utf-8"))
    features = features["features"] if isinstance(features, dict) else features

    return {
        "provenance": {k: {"path": v, "sha256_12": sha(k)} for k, v in SOURCES.items()},
        "models": MODEL_ORDER,
        "features": features,
        "counts": records(csv("counts"), ["objective", "patients", "rows", "note"]),
        "objective1": {
            "summary": summary(csv("loso_summary")),
            "patients": records(loso_p, ["patient", "model", "MARD", "RMSE", "MAE", "R2", "EGA_A", "EGA_A+B",
                                         "n_test_samples"], {"EGA_A+B": "EGA_AB", "n_test_samples": "n"}),
            "vs_persistence": records(csv("loso_vs_persistence"),
                                      ["model", "n_beats_persistence", "n_patients", "mean_diff", "diff_ci_lo",
                                       "diff_ci_hi", "p"]),
            "lightgbm_pairs": records(csv("loso_lightgbm_pairs"), ["comparison", "mean_diff", "p"]),
        },
        "objective2": {
            "summary": records(csv("pers_summary"), ["model", "p_level", "MARD_mean", "MARD_std", "RMSE_mean",
                                                     "EGA_A+B_mean", "Delta_MARD_mean", "n_patients"],
                               {"EGA_A+B_mean": "EGA_AB_mean"}),
            "patients": records(pers_p, ["patient", "p_level", "model", "MARD", "RMSE", "Delta_MARD",
                                         "n_personal_samples", "n_eval_samples"]),
            "vs_p0": records(csv("pers_vs_p0"), ["model", "p_level", "n_improved", "n_patients", "mean_diff",
                                                 "diff_ci_lo", "diff_ci_hi", "p"]),
            "days": records(csv("pers_days"), ["p_level", "median", "min", "max"]),
        },
        "objective3": {
            "main": summary(csv("hupa_main")),
            "all": summary(csv("hupa_all")),
            "patients": records(hupa_p, ["patient", "in_main", "model", "MARD", "RMSE", "EGA_A+B", "n_samples"],
                                {"EGA_A+B": "EGA_AB", "n_samples": "n"}),
            "vs_persistence": records(c2[c2["analysis"] == "MAIN"],
                                      ["model", "n_beats_persistence", "n_patients", "mean_diff", "diff_ci_lo",
                                       "diff_ci_hi", "p"]),
            "gap": records(gap[gap["metric"] == "MARD"],
                           ["model", "ohio_A", "hupa_main", "eval_rule_effect", "sampling_effect",
                            "sampling_ci_lo", "sampling_ci_hi", "remaining_effect", "remaining_ci_lo",
                            "remaining_ci_hi", "total_gap"]),
        },
        "analyses": {
            "ranges": records(csv("ranges"), ["dataset", "model", "range", "n_points", "MARD", "MARD_ci_lo",
                                              "MARD_ci_hi", "RMSE"]),
            "hypo": records(hypo, ["dataset", "model", "n_low_episodes", "detected", "detection_rate",
                                   "detection_15min_rate", "median_lead_min", "false_alarms_per_day"]),
            "bucketing": {
                "total_eligible": int(bucket["total_eligible"]), "affected_any": int(bucket["affected_any"]),
                "affected_bolus": int(bucket["affected_bolus"]), "affected_meal": int(bucket["affected_meal"]),
                "training_rows": int(bucket["training_rows"]),
                "training_rows_affected": int(bucket["training_rows_affected"]),
                "MARD_research": round(float(bucket["MARD_research_mean"]), 3),
                "MARD_causal": round(float(bucket["MARD_causal_mean"]), 3),
                "RMSE_research": round(float(bucket["RMSE_research_mean"]), 2),
                "RMSE_causal": round(float(bucket["RMSE_causal_mean"]), 2),
                "mean_abs_diff_affected": round(float(bdiff["mean_abs_diff"]), 2),
            },
        },
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
