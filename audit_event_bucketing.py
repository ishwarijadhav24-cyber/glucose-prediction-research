"""
Audit (read-only): how often does the research grid's event bucketing use information
from after the prediction time t, and does it matter?

parse_xml_file assigns each bolus/meal to bucket floor(event_time, 5 min), so grid row t
also contains events from (t, t + 5 min). Those events are in the future at time t.

Eligible timestamps: Objective 1 evaluation rows (create_features(load_subject_data(pid)),
target and glucose at t finite), all 12 OhioT1DM patients, all files.
Affected: an eligible t with a bolus (ts_begin) or meal (ts) strictly inside (t, t + 5 min).

Predictions use the deployed LightGBM artifact (models_objective3, trained on all 12
patients, so these are IN-SAMPLE predictions):
  A  research pipeline: features of row t as produced by create_features
  B  strictly causal backend: InferenceService.predict(events, t) (events after t removed)
Nothing is retrained and no research file is modified.

Outputs: results_evidence/event_bucketing/*.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.errors import BackendError
from backend.app.services.artifacts import load_artifacts
from backend.app.services.inference import InferenceService
from backend.tests.helpers import ohio_events
from data_processing import create_features, load_subject_data
from evaluate_loso import calculate_mard, calculate_rmse, discover_patient_ids
from objective3_common import apply_preprocessing

OHIO = "data/OhioT1DM"
OUT = "results_evidence/event_bucketing"
FIVE = pd.Timedelta("5min")


def patient_events(pid):
    files = sorted(f for f in os.listdir(OHIO) if f.startswith(f"{pid}-ws-") and f.endswith(".xml"))
    ev = pd.concat([ohio_events(os.path.join(OHIO, f)) for f in files], ignore_index=True)
    return ev.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def affected_times(ev, col):
    t = ev.loc[ev[col].notna(), "timestamp"]
    t = t[t != t.dt.floor("5min")]            # event strictly after its bucket start
    return set(t.dt.floor("5min"))


def main():
    os.makedirs(OUT, exist_ok=True)
    art = load_artifacts("models_objective3")
    svc = InferenceService(art)
    fl = list(art.feature_list)
    per_patient, rows = [], []
    t_start = time.time()
    for pid in discover_patient_ids(OHIO):
        ev = patient_events(pid)
        feats = create_features(load_subject_data(pid, data_dir=OHIO))
        all_ts = pd.DatetimeIndex(feats["ts"])
        eval_mask = np.isfinite(feats["target"].values) & np.isfinite(feats["glucose"].values)
        ab, am = affected_times(ev, "bolus_units"), affected_times(ev, "carbs_g")
        is_b, is_m = all_ts.isin(list(ab)), all_ts.isin(list(am))
        X = apply_preprocessing(art.imputer, art.scaler, feats[fl].values)
        pred_a = art.model.predict(X)
        y = feats["target"].values
        pred_b = pred_a.copy()
        idx = np.flatnonzero(eval_mask & (is_b | is_m))
        n_refused = 0
        for i in idx:
            t = all_ts[i]
            try:
                r = svc.predict(ev, t)
                pred_b[i] = r.predicted_glucose_mg_dl
                ok = True
            except BackendError as e:
                n_refused += 1
                ok, r = False, e
            rows.append({"patient": pid, "ts": t, "bolus_after_t": bool(is_b[i]), "meal_after_t": bool(is_m[i]),
                         "y_true": y[i], "pred_research": pred_a[i],
                         "pred_causal": pred_b[i] if ok else np.nan,
                         "causal_status": "ok" if ok else r.code.value})
        e_idx = np.flatnonzero(eval_mask)
        per_patient.append({
            "patient": pid, "n_feature_rows_training": len(feats),
            "n_training_rows_affected": int((is_b | is_m).sum()),
            "n_eligible_eval": int(eval_mask.sum()),
            "n_affected_bolus": int((eval_mask & is_b).sum()), "n_affected_meal": int((eval_mask & is_m).sum()),
            "n_affected_any": int(len(idx)), "n_causal_refused": n_refused,
            "MARD_research": calculate_mard(y[e_idx], pred_a[e_idx]), "MARD_causal": calculate_mard(y[e_idx], pred_b[e_idx]),
            "RMSE_research": calculate_rmse(y[e_idx], pred_a[e_idx]), "RMSE_causal": calculate_rmse(y[e_idx], pred_b[e_idx]),
            "MARD_affected_research": calculate_mard(y[idx], pred_a[idx]) if len(idx) else np.nan,
            "MARD_affected_causal": calculate_mard(y[idx], pred_b[idx]) if len(idx) else np.nan,
        })
        print(f"{pid}: {len(idx)} affected of {int(eval_mask.sum())} ({time.time() - t_start:.0f}s)", flush=True)

    pp = pd.DataFrame(per_patient)
    for k in ("bolus", "meal", "any"):
        pp[f"pct_affected_{k}"] = 100 * pp[f"n_affected_{k}"] / pp["n_eligible_eval"]
    pp["pct_training_rows_affected"] = 100 * pp["n_training_rows_affected"] / pp["n_feature_rows_training"]
    pp.to_csv(os.path.join(OUT, "bucketing_per_patient.csv"), index=False)
    pts = pd.DataFrame(rows)
    pts.to_csv(os.path.join(OUT, "bucketing_affected_points.csv"), index=False)

    ok = pts[pts["causal_status"] == "ok"]
    d = (ok["pred_causal"] - ok["pred_research"]).abs()
    diff = {"n_affected": len(pts), "n_compared": len(ok), "n_causal_refused": int((pts["causal_status"] != "ok").sum()),
            "mean_abs_diff": d.mean(), "median_abs_diff": d.median(), "p95_abs_diff": d.quantile(0.95),
            "max_abs_diff": d.max(), "pct_diff_over_1_mg_dl": 100 * (d > 1).mean(),
            "pct_diff_over_5_mg_dl": 100 * (d > 5).mean()}
    pd.DataFrame([diff]).to_csv(os.path.join(OUT, "bucketing_prediction_diff.csv"), index=False)
    tot = {"total_eligible": int(pp["n_eligible_eval"].sum()), "affected_bolus": int(pp["n_affected_bolus"].sum()),
           "affected_meal": int(pp["n_affected_meal"].sum()), "affected_any": int(pp["n_affected_any"].sum()),
           "training_rows": int(pp["n_feature_rows_training"].sum()),
           "training_rows_affected": int(pp["n_training_rows_affected"].sum()),
           "MARD_research_mean": pp["MARD_research"].mean(), "MARD_causal_mean": pp["MARD_causal"].mean(),
           "RMSE_research_mean": pp["RMSE_research"].mean(), "RMSE_causal_mean": pp["RMSE_causal"].mean()}
    pd.DataFrame([tot]).to_csv(os.path.join(OUT, "bucketing_totals.csv"), index=False)
    print(pd.Series(tot).to_string())
    print(pd.Series(diff).to_string())


if __name__ == "__main__":
    main()
