"""Build backend/app/model_card.json from the verified research results (run from the repo root).

Sources (read only, current causal pipeline only):
  results_causal/loso_summary.csv            Objective 1 LOSO summary (12 patients)
  results_causal/loso_patient_results.csv    per-patient results (patient count)
  results_objective3/hupa_summary_main.csv   external validation, HUPA-UCM MAIN
  results_objective3/hupa_summary_all.csv    external validation, HUPA-UCM ALL
  results_evidence/C2_hupa_vs_persistence.csv             HUPA-UCM: LightGBM vs Persistence (paired)
  results_evidence/C5_metrics_by_glucose_range.csv        accuracy below 70 mg/dL
  results_evidence/event_bucketing/bucketing_totals.csv   research event-bucketing sensitivity analysis
  models_objective3/settings.json, model_hashes.json
The older non-causal results (results/, results_pre_causal/) and train.py's
model_artifact.pkl are deliberately NOT used.

The card is generated at build time because results files are not shipped in the container.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "backend" / "app" / "model_card.json"
MODEL = "LightGBM"
SOURCES = {
    "loso_summary": "results_causal/loso_summary.csv",
    "loso_patient_results": "results_causal/loso_patient_results.csv",
    "hupa_summary_main": "results_objective3/hupa_summary_main.csv",
    "hupa_summary_all": "results_objective3/hupa_summary_all.csv",
    "hupa_vs_persistence": "results_evidence/C2_hupa_vs_persistence.csv",
    "metrics_by_glucose_range": "results_evidence/C5_metrics_by_glucose_range.csv",
    "event_bucketing": "results_evidence/event_bucketing/bucketing_totals.csv",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(path: str, model: str = MODEL) -> pd.Series:
    df = pd.read_csv(ROOT / path)
    return df[df["model"] == model].iloc[0]


def low_range_mard(model: str) -> float:
    df = pd.read_csv(ROOT / SOURCES["metrics_by_glucose_range"])
    r = df[(df["dataset"] == "Ohio LOSO") & (df["model"] == model) & (df["range"] == "<70")].iloc[0]
    return round(float(r["MARD"]), 1)


def metric(r, name, digits):
    return {"mean": round(float(r[f"{name}_mean"]), digits), "sd": round(float(r[f"{name}_std"]), digits)}


def build() -> dict:
    loso = row(SOURCES["loso_summary"])
    loso_persist = row(SOURCES["loso_summary"], "Persistence")
    hupa_persist = row(SOURCES["hupa_summary_main"], "Persistence")
    c2 = pd.read_csv(ROOT / SOURCES["hupa_vs_persistence"])
    c2 = c2[(c2["analysis"] == "MAIN") & (c2["model"] == MODEL)].iloc[0]
    bucket = pd.read_csv(ROOT / SOURCES["event_bucketing"]).iloc[0]
    pct_bucketed = round(100.0 * bucket["affected_any"] / bucket["total_eligible"], 1)
    low_model, low_persist = low_range_mard(MODEL), low_range_mard("Persistence")
    per_patient = pd.read_csv(ROOT / SOURCES["loso_patient_results"])
    n_patients = int(per_patient.loc[per_patient["model"] == MODEL, "patient"].nunique())
    settings = json.loads((ROOT / "models_objective3/settings.json").read_text(encoding="utf-8"))
    hashes = json.loads((ROOT / "models_objective3/model_hashes.json").read_text(encoding="utf-8"))
    ext = {}
    for key, n_label in (("hupa_summary_main", "MAIN"), ("hupa_summary_all", "ALL")):
        r = row(SOURCES[key])
        ext[n_label] = {"n_patients": int(r["n_patients"]), "MARD_percent": metric(r, "MARD", 2),
                        "RMSE_mg_dl": metric(r, "RMSE", 2), "clarke_A_plus_B_percent": metric(r, "EGA_A+B", 2)}
    return {
        "model_name": MODEL,
        "model_version": hashes[f"{MODEL}.joblib"][:12],
        "feature_version": hashes["feature_list.json"][:12],
        "n_features": settings["n_features"],
        "prediction_horizon_minutes": 30,
        "sampling_interval_minutes": 5,
        "training_data": f"OhioT1DM, {len(settings['patients'])} patients, {settings['n_training_samples']} feature rows "
                         "(causal 5-minute grid; training + testing XML files)",
        "library_versions": settings["versions"],
        "research_evaluation": {
            "label": "Research evaluation results (population averages). Not a guarantee for any single prediction.",
            "methodology": "Leave-one-subject-out cross-validation on OhioT1DM (train on 11 patients, test on the "
                           "held-out patient; repeated for all 12). The deployed model uses the same pipeline and "
                           "settings, trained on all 12 patients.",
            "n_patients": n_patients,
            "aggregation": "metric computed per patient, then mean and SD across patients",
            "MARD_percent": metric(loso, "MARD", 2),
            "RMSE_mg_dl": metric(loso, "RMSE", 2),
            "MAE_mg_dl": metric(loso, "MAE", 2),
            "R2": metric(loso, "R2", 3),
            "clarke_A_plus_B_percent": metric(loso, "EGA_A+B", 2),
            "clarke_note": "Clarke A+B is the share of research predictions in clinically acceptable zones A or B, "
                           "averaged over patients. It does not mean each individual prediction is that accurate. "
                           "The Persistence baseline also reaches a high Clarke A+B, so A+B alone does not show "
                           "that the model is better than Persistence.",
            "persistence_reference": {"MARD_percent": metric(loso_persist, "MARD", 2),
                                      "RMSE_mg_dl": metric(loso_persist, "RMSE", 2),
                                      "R2": metric(loso_persist, "R2", 3),
                                      "clarke_A_plus_B_percent": metric(loso_persist, "EGA_A+B", 2)},
            "model_selection_note": "LightGBM is not the single best model by MARD: Stacking and MLP are essentially "
                                    "tied with it. It was selected for deployment because its MARD is essentially "
                                    "tied with the best-performing models while it gives strong RMSE/R2 and a "
                                    "simpler single-model deployment.",
        },
        "external_validation": {
            "label": "Research external transfer evaluation on a different dataset (no retraining).",
            "dataset": "HUPA-UCM (FreeStyle Libre, ~15-minute readings)",
            **ext,
            "persistence_reference": {"MAIN": {"MARD_percent": metric(hupa_persist, "MARD", 2),
                                               "clarke_A_plus_B_percent": metric(hupa_persist, "EGA_A+B", 2)}},
            "lightgbm_vs_persistence_MAIN": {"n_patients_lightgbm_better": int(c2["n_beats_persistence"]),
                                             "n_patients": int(c2["n_patients"]),
                                             "wilcoxon_p": round(float(c2["p"]), 2)},
            "note": "HUPA-UCM serves as an external transfer evaluation without retraining. Performance decreased "
                    "relative to OhioT1DM, and LightGBM did not significantly outperform Persistence on HUPA-UCM. "
                    "This demonstrates transferability of the pipeline, but does not establish clinical "
                    "generalization or superiority on a new population. Part of the drop is explained by the "
                    "~15-minute sampling, which is not equivalent to native 5-minute CGM.",
        },
        "limitations": [
            "Research demonstration only. Not a medical device.",
            "Not for diagnosis, treatment decisions or insulin dosing. Does not replace a CGM.",
            "Trained on 5-minute CGM data from 12 OhioT1DM patients only.",
            f"Least accurate below 70 mg/dL (research MARD {low_model}% vs {low_persist}% for Persistence); "
            "not suitable for low-glucose alerts.",
            "Heart rate is not effectively used: the available OhioT1DM heart-rate signal was not correctly mapped "
            "by the current research pipeline, so this feature was always missing and imputed to a constant. "
            "Known limitation and future work item.",
            "On HUPA-UCM (~15-minute FreeStyle Libre data) LightGBM did not significantly outperform Persistence.",
            f"Research-time insulin and meal event bucketing uses 5-minute floor aggregation, which can expose up to "
            f"4:59 of event-time look-ahead on about {pct_bucketed}% of rows. A sensitivity analysis showed only a "
            f"small aggregate effect (MARD {bucket['MARD_research_mean']:.3f} vs {bucket['MARD_causal_mean']:.3f}). "
            "The deployed inference pipeline excludes events after the prediction time and is strictly causal.",
        ],
        "sources": {k: {"path": v, "sha256": sha(ROOT / v)} for k, v in SOURCES.items()},
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
