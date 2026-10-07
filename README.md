# 30-minute blood glucose forecasting for Type 1 diabetes

Research project: forecast CGM glucose **30 minutes ahead** from past CGM glucose, insulin boluses and
carbohydrate intake, evaluated **subject-independently**, with a personalization study, an external
transfer evaluation on a second dataset, and a reproducible local backend API.

> **Research demonstration only. Not a medical device. Not for diagnosis, treatment decisions or insulin
> dosing. Does not replace a CGM.**

The strongest contribution is the combination of subject-independent evaluation, causal inference design,
external-dataset transfer evaluation, personalization analysis, and reproducible deployment — not a claim
of clinical readiness or state-of-the-art performance.

---

## 1. What the model does

- Input: CGM glucose (mg/dL), bolus insulin (U) and carbohydrates (g), up to the prediction time *t*.
- Output: one forecast of CGM glucose at *t* + 30 min. It is a **direct** 30-minute regression (not
  recursive) and uses only data at or before *t*.
- Data are placed on a 5-minute grid. Glucose at grid time *t* is the most recent reading at most 5 minutes
  old; short gaps are forward-filled for at most 30 minutes (never interpolated).
- 47 features (`models_objective3/feature_list.json`): current glucose, bolus, carbs, heart rate (see
  limitations), a decaying insulin-activity feature (`iob`), first and second glucose differences, 30-minute
  rolling mean and SD, hour of day (sin/cos), and 12 five-minute lags of glucose, glucose difference and `iob`.
- Target (training only): glucose 6 grid steps (30 min) later.
- Deployed model: **LightGBM** (`models_objective3/LightGBM.joblib`, version `cd7e2d54f5b4`), trained on all
  12 OhioT1DM patients with the Objective 1 settings, plus the saved mean imputer and standard scaler.

## 2. Objective 1 — subject-independent evaluation (OhioT1DM, LOSO)

Leave-one-subject-out over 12 OhioT1DM patients: each fold trains on 11 patients and tests on the held-out
patient; imputer and scaler are fitted on the training patients only. Metrics are computed per patient,
then reported as mean ± SD across patients. Source: `results_causal/loso_summary.csv`.

| Model | MARD % | RMSE mg/dL | R² | Clarke A+B % |
|---|---|---|---|---|
| Stacking | 10.03 ± 1.54 | 20.42 ± 2.67 | 0.874 | 98.1 |
| MLP | 10.03 ± 1.45 | 20.53 ± 2.78 | 0.873 | 98.2 |
| **LightGBM (deployed)** | **10.03 ± 1.52** | **20.33 ± 2.66** | **0.875** | **98.0** |
| RandomForest | 10.13 ± 1.54 | 20.58 ± 2.66 | 0.872 | 98.0 |
| Ridge | 10.70 ± 1.55 | 21.66 ± 2.71 | 0.859 | 98.4 |
| Persistence (glucose at *t*) | 12.11 ± 1.92 | 24.82 ± 3.10 | 0.814 | 98.6 |

- LightGBM beats Persistence on MARD for all 12 patients (mean −2.08 points; Wilcoxon p = 0.0005,
  `results_evidence/A2_obj1_vs_persistence.csv`).
- LightGBM is **not** the single best model by MARD: Stacking and MLP are essentially tied with it
  (LightGBM vs Stacking: difference 0.004 points, p = 0.97). LightGBM was selected for deployment because
  it achieved essentially tied MARD with the best-performing models while giving strong RMSE/R² performance
  and a simpler single-model deployment.
- Persistence has the highest Clarke A+B and already reaches R² 0.81, so Clarke A+B and R² alone do not
  show that a model is better than Persistence; MARD and RMSE do.
- XGBoost appeared in early planning materials but was never implemented or evaluated in the final
  repository. LightGBM is the actual implemented gradient-boosted model. No XGBoost result exists here.

## 3. Objective 2 — personalization (OhioT1DM)

The population model (trained on the other 11 patients) is adapted with the first P % of the target
patient's own data (P = 5, 10, 20, 30, 50; roughly 2.4 to 24 days) and always tested on the same last 50 %
of that patient's data. LightGBM is adapted by continued boosting, Ridge by prior-regularized adaptation.
Sources: `results_objective2/`, `results_evidence/B2_obj2_vs_P0.csv`.

- LightGBM: no significant change at P = 5 % or 10 %; small, statistically significant MARD reductions at
  P = 20 % (−0.18 points, p = 0.042), 30 % (−0.16, p = 0.002) and 50 % (−0.20, p = 0.012).
- Conclusion: patient-specific history gives a **small but consistent** improvement (about 2 % relative)
  once roughly 10 days of data are available. It is not a large accuracy gain.
- Note: the LightGBM adaptation procedure differs between P ≤ 10 % (fixed 20 rounds) and P ≥ 20 %
  (early stopping on a chronological validation split).

## 4. Objective 3 — external transfer evaluation (HUPA-UCM)

The OhioT1DM-trained models were applied to HUPA-UCM **without retraining** (imputer and scaler: transform
only). HUPA-UCM glucose comes only from raw FreeStyle Libre type-0 readings (~15-minute sampling); the
interpolated Preprocessed glucose is not used. MAIN = 19 patients. Sources: `results_objective3/`,
`results_evidence/C2_hupa_vs_persistence.csv`, `results_objective3/followup/`.

| | OhioT1DM LOSO | HUPA-UCM MAIN |
|---|---|---|
| LightGBM MARD % | 10.03 ± 1.52 | 13.38 ± 1.92 |
| Persistence MARD % | 12.11 ± 1.92 | 13.03 ± 2.00 |
| LightGBM Clarke A+B % | 98.0 | 95.7 |
| Persistence Clarke A+B % | 98.6 | 97.9 |

HUPA-UCM serves as an external transfer evaluation without retraining. Performance decreased relative to
OhioT1DM, and LightGBM did not significantly outperform Persistence on HUPA-UCM (better for 10 of 19
patients, Wilcoxon p = 0.95). This demonstrates transferability of the pipeline, but does not establish
clinical generalization or superiority on a new population.

- The ~15-minute Libre sampling is not equivalent to native 5-minute CGM. Simulating 15-minute sampling on
  OhioT1DM explains about 1.6 of the 3.35-point MARD gap; about 1.8 points remain unexplained
  (population/device/dataset differences).
- On HUPA-UCM the simple Ridge model transfers best (MARD 12.22 %, the only model significantly better than
  Persistence).

## 5. Important scientific limitations

1. **Heart rate is not effectively used.** The feature exists in the 47-feature schema, but the research
   parser looks for a `heart_rate` tag while OhioT1DM stores `basis_heart_rate` events. The parsed heart
   rate is 100 % missing and becomes a constant (0) after imputation, so it contributes no information.
   Known limitation and future work item. The project makes no wearable or heart-rate claims.
2. **Research event bucketing.** Research-time insulin and meal event bucketing uses 5-minute floor
   aggregation, which can expose up to 4:59 of event-time look-ahead on approximately 2.3 % of rows
   (3,882 of 167,376 eligible rows; 4,055 of 169,221 training rows). A sensitivity analysis showed only a
   small aggregate effect on overall metrics (MARD 9.628 vs 9.631; RMSE 19.30 vs 19.34;
   `results_evidence/event_bucketing/`). The deployed inference pipeline excludes events after the
   prediction time and is strictly causal.
3. **Low glucose.** The model is least accurate below 70 mg/dL (LightGBM MARD 33.1 % vs 24.0 % for
   Persistence, `results_evidence/C5_metrics_by_glucose_range.csv`). It must not be used for low-glucose
   alerts.
4. **External data.** On HUPA-UCM, LightGBM did not significantly outperform Persistence (section 4).
5. **Small training cohort**: 12 OhioT1DM patients, one CGM type.
6. The `iob` feature is an exponentially decaying sum of past boluses (time constant 60 min, 4-hour window,
   normalized weights). It is a causal insulin-activity proxy, not a pharmacokinetic insulin-on-board model.
7. Missing insulin or meal records are treated as "no event", as in the research data.

## 6. Repository layout

| Path | Contents |
|---|---|
| `data_processing.py`, `evaluate_loso.py`, `evaluate_personalization.py`, `objective3_common.py`, `parsers/hupa_ucm.py` | Research pipeline (locked; hash-guarded by the backend tests) |
| `results_causal/` | **Current** Objective 1 results (causal pipeline) |
| `results_objective2/`, `results_objective3/`, `results_evidence/` | Current Objective 2, Objective 3 and evidence results |
| `results/`, `results_pre_causal/`, `outputs/`, `outputs_pre_causal/` | **Historical / superseded** pre-causal runs. Do not cite. |
| `results_ohio_official/` | Official OhioT1DM test-split run; not re-verified against the causal pipeline in the final audit. Do not cite. |
| `models_objective3/` | Trained artifacts. Deployment uses only the six files in `backend/deploy/artifact_manifest.json` |
| `model_artifact.pkl` | Legacy artifact; not used by the backend |
| `backend/` | FastAPI backend, tests, synthetic demo data, model card |
| `data/` | Private datasets (git-ignored, never committed) |

## 7. Supported input formats

- **OhioT1DM XML** (1–2 files of one patient).
- **FreeStyle Libre / HUPA-UCM export** (+ optional HUPA-UCM Preprocessed CSV for insulin and carbs).
  ~15-minute data is accepted with an explicit warning that it is not equivalent to native 5-minute CGM.
- **Normalized CSV**: columns `timestamp,glucose_mg_dl,bolus_units,carbs_g`, timestamps
  `YYYY-MM-DD HH:MM[:SS]` local time, one row per event, empty cell = no event
  (example: `backend/demo_data/synthetic_01.csv`, synthetic data).

This research system accepts structured glucose/insulin/carbohydrate data. It does not process arbitrary
medical-record PDFs, images, scanned reports or EHR/FHIR files.

## 8. API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/api/v1/health` | Model loaded, model version |
| GET | `/api/v1/model` | Model card: features, research results, limitations, input format |
| GET | `/api/v1/demo/patients` | List synthetic demo patients |
| GET | `/api/v1/demo/patients/{id}` | Synthetic demo patient history |
| POST | `/api/v1/predict/demo/{id}` | Forecast for a demo patient (optional `prediction_time`) |
| POST | `/api/v1/predict/upload?mode=latest\|all` | Forecast(s) from uploaded files (multipart field `file`) |

Full request/response details for frontend work: [`FRONTEND_HANDOFF.md`](FRONTEND_HANDOFF.md).
Interactive docs: `/docs` while the server runs.

## 9. Running the backend locally

**Easiest (Windows):** double-click `start-local.bat` in the repository root. It starts the backend (port 7860)
and the website (port 3000) in their own windows, restarts the backend automatically if it stops, waits until both
answer and opens http://localhost:3000. Running it again when everything is already up does nothing. Close the two
windows to stop. The website runs as a pre-built production server (pages open instantly); it is rebuilt
automatically after code changes. While editing the website code, use `start-local.bat -Dev` for live reload
(slower: each page compiles on its first visit).

To start the backend by hand instead: requires Python 3.13. The backend refuses to start unless lightgbm 4.7.0, scikit-learn 1.7.2, numpy 2.3.5
and pandas 2.3.3 are installed exactly (they must match the trained artifacts).

```bash
python -m pip install -r backend/requirements.txt
```

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 7860
```

Run from the repository root. In development, CORS allows `http://localhost:3000` by default.

| Variable | Default | Meaning |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` requires explicit `CORS_ORIGINS` (no `*`) |
| `CORS_ORIGINS` | `http://localhost:3000` (development) | Comma-separated allowed frontend origins |
| `MODEL_DIR` | `models_objective3` | Directory with the six deployment artifacts |
| `MAX_UPLOAD_SIZE_MB` | 25 | Total upload size limit |
| `RATE_LIMIT_PER_MINUTE` | 30 | Prediction requests per client per minute |
| `LOG_LEVEL` | `INFO` | Structured JSON logs (no uploaded data is logged) |

Tests (from the repository root; research-data tests are skipped automatically when `data/` is absent):

```bash
python -m pip install -r backend/requirements-dev.txt
```

```bash
python -m pytest backend/tests
```

A `Dockerfile` (production image, allow-listed build context) is included; it has not yet been built or run
on Linux.

## 10. Frontend integration

See [`FRONTEND_HANDOFF.md`](FRONTEND_HANDOFF.md). In short: label the upload action **"Upload CGM Data"**
(not "Upload Medical Record"), list the supported formats, always show the disclaimer, present results as a
research forecast of CGM glucose 30 minutes ahead, and never present them as alerts, dosing advice or
safe/unsafe classifications.
