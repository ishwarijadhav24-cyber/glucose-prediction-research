# Backend Architecture — Phase 0 Repository Audit

> **Historical document.** This is the pre-implementation audit, kept for the record. The decisions it asks
> for were made: the deployed model is **LightGBM** (XGBoost appeared in early planning materials but was
> never implemented or evaluated in this repository; the planning numbers in B1/B2 are unsupported and must
> not be cited). The backend has since been built and tested. For the current system see `README.md` and
> `FRONTEND_HANDOFF.md`; for current results and limitations see `backend/app/model_card.json`.

Status (at the time of writing): **audit only, no backend code written.** Everything below was checked against the
repository on branch `objective3` (HEAD `353dca8`). Items marked **DECISION** need approval
before Phase 1.

---

## 0. Blocking findings (read first)

| # | Brief says | Repository shows | Consequence |
|---|---|---|---|
| B1 | Primary deployment model: **XGBoost** | No XGBoost anywhere: no import, no training code, no artifact; `xgboost` is not installed. The gradient-boosted model in the project is **LightGBM**. | Deploying XGBoost would mean a new model with no LOSO evaluation. **DECISION:** deploy LightGBM (recommended), or approve a new XGBoost experiment first. |
| B2 | MARD ≈ 10.48 %, RMSE ≈ 20.92, R² ≈ 0.867, Clarke A+B ≈ 96.51 % | None of these numbers appears in any results file. | They cannot be shown by the API. The verified Objective 1 numbers are in §4. |
| B3 | Upload a CSV | The only OhioT1DM reader is `parse_xml_file` (XML). There is no CSV schema in the repo. | A CSV schema must be defined (§6). Parity comes from feeding it through `parse_xml_file`, not by rewriting it. |
| B4 | Demo with "existing/project demo data" | There is no demo data. OhioT1DM is distributed under a Data Use Agreement that forbids redistribution, and `data/` is git-ignored. | OhioT1DM cannot be shown on a public site. **DECISION:** see §9.4. |
| B5 | "Existing frontend, deployment files, README, tests" | None exist: no frontend, Dockerfile, README, requirements file or test suite. | Everything for serving is new. Research code is untouched. |
| B6 | Reuse `create_features` for inference | `create_features` builds the target (`glucose.shift(-6)`) and **drops rows with a NaN target**, so the newest row (the one we predict from) is always dropped. `data_processing.py` is locked (CLAUDE.md). | Needs a thin adapter, not a rewrite (§8). |

---

## 1. Current architecture

```
data/ (git-ignored, read-only)
  OhioT1DM/*.xml            12 patients x (training + testing) XML
  HUPA-UCM/Raw_Data, Preprocessed
data_processing.py          parse_xml_file, load_subject_data, create_features   [LOCKED]
evaluate_loso.py            Objective 1 LOSO; models built inline; metrics; Clarke grid [LOCKED]
evaluate_personalization.py Objective 2                                          [LOCKED]
evaluate_ohio_official_test.py official train/test split evaluation             [LOCKED]
train.py                    early prototype: one patient (559), 80/20 split      [LOCKED]
objective3_common.py        Objective 1 model settings + preprocessing (importable)
train_final_ohio.py         final models on all 12 OhioT1DM patients -> models_objective3/
parsers/hupa_ucm.py         HUPA-UCM raw Libre parser (validated, 13 checks)
evaluate_hupa_external.py, audit_objective3.py, followup_objective3.py, hypo_alert_analysis.py
research_evidence.py        statistics report
models_objective3/          Ridge, RandomForest, MLP, LightGBM, Stacking + imputer, scaler,
                            feature_list.json, settings.json, model_hashes.json
model_artifact.pkl          143 MB legacy artifact from train.py (see §4)
results*/, outputs*/        research outputs
```

## 2. Existing ML pipeline (traced in code)

| Stage | Implementation |
|---|---|
| Raw data | OhioT1DM XML: `glucose_level/event(ts, value)`, `bolus/event(ts_begin, dose)`, `meal/event(ts, carbs)` |
| Parsing | `parse_xml_file`: `pd.to_datetime(ts, dayfirst=True)`; non-numeric or unparseable events skipped |
| Alignment | 5-min grid from `floor(first glucose)` to `ceil(last glucose)`; index name `ts` |
| Glucose, causal | Step A `reindex(grid, method="ffill", tolerance=5min)` (duplicates keep last), Step B `ffill(limit=6)`; longer gaps stay NaN |
| Bolus / carbs | Summed into the bucket `floor(event_time, 5min)`; events outside the grid are dropped; no events → 0.0 |
| Heart rate | `findall('.//heart_rate')` finds **0** elements (the tag is `basis_heart_rate`), so the column is all NaN |
| Multiple files | `load_subject_data`: concatenate training + testing files, sort, keep first duplicate timestamp |
| Features | `create_features` (§5): IOB, diffs, rolling, hour sin/cos, 12 lags, `target = glucose.shift(-6)` |
| Row filter | `dropna(subset=["target"])` |
| Preprocessing | `SimpleImputer(mean, keep_empty_features=True)` → `np.nan_to_num(nan=0)` → `StandardScaler`, fitted on training rows only (`objective3_common.fit_preprocessing` / `apply_preprocessing`) |
| Training | `objective3_common.build_model` (settings copied exactly from `evaluate_loso.py`); final models in `train_final_ohio.py` (all 12 patients, 169,221 rows, 47 features) |
| Serialization | joblib files in `models_objective3/` plus SHA-256 hashes in `model_hashes.json` |
| Inference (research) | `evaluate_hupa_external.py`: `create_features` → saved imputer/scaler `.transform` → `model.predict` |
| Evaluation | `evaluate_loso.py` metrics (`calculate_mard/rmse/mae/r2`, `evaluate_clarke_ega`). Rows are scored only where the target and the glucose at t are finite. |

Causality: every feature op is backward-looking (`diff`, `rolling(6)`, positive `shift`, IOB via
`np.convolve(...)[:len]`, which is causal, and `heart_rate.ffill`). The only forward op is the target.
This was verified for Objective 3 by checks 3, 4 and 12 and by `audit_objective3.py`.

## 3. Reusable modules (import, do not copy)

| Need | Reuse |
|---|---|
| Event → 5-min causal grid | `data_processing.parse_xml_file` (accepts a file-like object: verified with `io.BytesIO`) |
| Features | `data_processing.create_features` (through the adapter in §8) |
| Imputer → nan_to_num → scaler | `objective3_common.apply_preprocessing` |
| Feature order | `models_objective3/feature_list.json` (verified identical to `create_features` output order) |
| Model settings and versions | `models_objective3/settings.json` |
| Artifact integrity | `models_objective3/model_hashes.json` |
| Metrics (tests only) | `evaluate_loso.calculate_*`, `evaluate_clarke_ega` |

`objective3_common.py` imports `evaluate_loso`, which imports matplotlib, seaborn and lightgbm at
import time. The backend should import only `apply_preprocessing`, or re-export it through a tiny
shim (§8) to avoid pulling plotting libraries into the container.

## 4. Model artifact strategy

| Candidate | Trained on | Evaluation that matches it | Use? |
|---|---|---|---|
| `models_objective3/LightGBM.joblib` + `imputer.joblib` + `scaler.joblib` | All 12 OhioT1DM patients; Objective 1 settings; LightGBM 4.7.0, scikit-learn 1.7.2 | Objective 1 LOSO (same pipeline). Reproducibility check on patient 559: within 0.012 MARD | **Recommended** |
| `models_objective3/Stacking.joblib` | same | LOSO MARD 10.03 (equal to LightGBM, p=0.97) | Possible, but 7.9 MB and slower |
| `model_artifact.pkl` (Stacking) | **One patient (559)**, 80/20 time split, scikit-learn 1.5.0 (version-mismatch warnings when loaded), **no scaler**, `data/OhioT1DM_2018` path that no longer exists | None | **Do not use** |
| XGBoost | does not exist | none | needs approval + a new LOSO run |

Research metrics the API may expose for LightGBM (`results_causal/loso_summary.csv`, LOSO, 12
patients, mean ± SD across patients): MARD **10.03 ± 1.52 %**, RMSE **20.33 ± 2.66 mg/dL**,
R² **0.875**, Clarke A+B **98.00 %**.
External (HUPA-UCM MAIN, 19 patients, 15-min Libre): MARD **13.38 %**.
Low-glucose alerts (§9.3): at a threshold of 70 mg/dL, LightGBM warns ≥15 min ahead for only **6 %** of lows.
Do not cite `results/` or `results_pre_causal/` (older, non-causal pipeline) or `train.py` output.

Startup loading: load once in the FastAPI lifespan; verify the SHA-256 of each file against
`model_hashes.json`; check `settings.json` versions against the installed ones; refuse to start on
any mismatch. joblib loads pickles, which can run code, so only these hashed, repo-owned files are ever loaded.

## 5. Exact feature schema (47 features, order = `feature_list.json`)

```
 1 glucose            2 bolus              3 carbs              4 heart_rate (always NaN -> imputed 0.0)
 5 iob                6 glucose_diff1      7 glucose_diff2      8 glucose_roll_mean_30
 9 glucose_roll_std_30 (NaN -> 0)         10 hour_sin          11 hour_cos
12-23 glucose_lag_1..12   24-35 glucose_diff1_lag_1..12   36-47 iob_lag_1..12
```
- IOB: kernel `exp(-k*5/60)`, k = 0..47 (DIA 240 min, t_peak 60 min), normalised to sum 1, causal convolution of bolus.
- hour_sin/cos: from the **local clock time** of `ts` (no time zone).
- Target (training only): `glucose.shift(-6)` = 30 min ahead on the 5-min grid.
- **Minimum history, measured** (not guessed): the first fully populated row needs **14 consecutive grid
  rows** of glucose (t and 13 earlier rows = 65 min), because `glucose_diff1_lag_12` uses glucose at t-13.
  IOB needs 240 min of bolus history to be complete. With less, it is underestimated but not NaN, so the API returns a warning.

## 6. Proposed inference input schema (CSV, event-level)

One row per recorded event, using the same raw semantics as the OhioT1DM XML:

| column | required | meaning | maps to |
|---|---|---|---|
| `timestamp` | yes | local time, `YYYY-MM-DD HH:MM[:SS]`, no time zone | event `ts` |
| `glucose_mg_dl` | one of the three per row | CGM sensor reading | `glucose_level/event value` |
| `bolus_units` | | bolus insulin delivered | `bolus/event dose` |
| `carbs_g` | | carbohydrates in grams | `meal/event carbs` |

The backend validates the CSV, then writes the events into an **in-memory OhioT1DM-style XML** and calls
`parse_xml_file`. The grid, the causal rule and the bucketing are therefore byte-for-byte the research code.

Validation, all mapped to structured errors:
- extension `.csv`, MIME type, size ≤ `MAX_UPLOAD_SIZE_MB`, UTF-8, exact column names;
- parseable timestamps (strict format);
- no duplicate timestamp for the same signal, because `parse_xml_file` would sum duplicate boluses;
- numeric values; glucose 40–400 (sensor range), bolus 0–25 U, carbs 0–300 g, no negatives;
- sampling interval about 5 min (median 4–6 min), because the model was trained on 5-min CGM. 15-min Libre data is **rejected in V1**: research showed it costs +1.6 MARD for LightGBM;
- a real reading at the prediction time t (Step A, ≤ 5 min old) and 14 consecutive non-NaN glucose grid rows ending at t, else **422 INSUFFICIENT_HISTORY**;
- after preprocessing: 47 columns in `feature_list.json` order, all finite.

Heart rate is not accepted, because the model never received it (all NaN in training). This is stated in the docs.

## 7. Proposed backend architecture (stateless V1)

```
backend/
  app/main.py            FastAPI app, lifespan: load + verify artifacts once
  app/config.py          pydantic-settings: MODEL_DIR, MAX_UPLOAD_SIZE_MB, CORS_ORIGINS, ENVIRONMENT, LOG_LEVEL
  app/errors.py          error codes -> HTTP status, JSON body {status, error:{code,message}}
  app/schemas.py         Pydantic request/response models
  app/routers/health.py  GET /health, GET /api/v1/health
  app/routers/model.py   GET /api/v1/model
  app/routers/demo.py    GET /api/v1/demo/patients[/{id}], POST /api/v1/predict/demo/{id}
  app/routers/predict.py POST /api/v1/predict/upload
  app/services/artifacts.py  hash check + load (LightGBM, imputer, scaler, feature_list, settings)
  app/services/ingest.py     CSV validation -> in-memory XML -> parse_xml_file
  app/services/features.py   adapter around create_features (§8) + schema/finite checks
  app/services/predict.py    apply_preprocessing -> model.predict, one 30-min forecast
  tests/                     pytest (causality + parity mandatory)
  Dockerfile, requirements.txt (pinned to settings.json versions + fastapi, uvicorn, lxml)
```
- Single forecast per request from real observations only: no recursive forecasting.
- A demo "actual vs predicted" series returns predictions made at past times t, each computed from data ≤ t. Predictions are never fed back as inputs.
- No database, auth, queue or cache. Uploads are processed in memory and discarded.
- Response: `predicted_glucose_mg_dl`, `prediction_time` (grid t), `latest_observation_time` (last raw reading), `forecast_time` (t + 30 min), `horizon_minutes` = 30, model `{name, version = model_hashes sha256 prefix, feature_version}`, `warnings`, and a fixed `disclaimer`.

## 8. Required refactors (minimal; no research file is modified)

1. **Feature adapter** (`backend/app/services/features.py`): to keep the newest row, append 6 future grid rows with a placeholder glucose value, call `create_features` unchanged, select the row at t, and drop `target`/`ts`.
   - Because every feature is causal, the placeholder can only affect `target`.
   - This is exactly what the mandatory causality test checks: change everything after t → identical prediction.
   - Alternative: a new `features_inference()` in research code. That would duplicate logic, so it is not recommended.
2. **Preprocessing shim**: import `apply_preprocessing` without pulling `evaluate_loso`'s plotting imports.
   - Either a lazy import guard in the backend, or one small new module `pipeline_common.py` holding `apply_preprocessing`, which `objective3_common.py` would import.
   - The second option changes an Objective 3 file. **DECISION**; the default is the backend guard.
3. **Container paths**: the backend references the research modules through the package path (`PYTHONPATH=/app`) and copies only `data_processing.py`, `objective3_common.py` (or the shim) and the 5 artifact files. No `data/` goes into the image.

## 9. Risks

**9.1 Security:** joblib pickles (mitigated: hash-verified repo files only); upload DoS (size limit, row limit, timeout); CSV injection is irrelevant (never re-exported); permissive CORS (env-restricted in prod); stack-trace leakage (global exception handler).

**9.2 Privacy:** glucose data is health data. No storage, no request-body logging, no values in error messages, no analytics on payloads, in-memory processing only.

**9.3 Clinical / misinterpretation:**
- The model is weakest in the low range: <70 mg/dL MARD 33 % on OhioT1DM vs 24 % for Persistence.
- At a threshold of 70 it gives ≥15 min warning for only 6 % of lows.
- So V1 returns **no action advice and no hypo alert**, only the forecast and the disclaimer. Clarke A+B 98 % is a population research result, not a per-prediction guarantee.

**9.4 Demo data licence:** OhioT1DM must not be published.
- **DECISION**, options:
  - (a) HUPA-UCM. Public dataset; I believe it is CC BY 4.0, which must be verified. But it is 15-min Libre data, so the 5-min model is out of domain.
  - (b) Synthetic 5-min traces from a documented simulator, labelled as synthetic.
  - (c) Demo only locally with OhioT1DM, nothing public.

**9.5 Data leakage:**
- The target column must never reach the model (the adapter drops it; a test asserts it).
- The placeholder future rows must not influence features (causality test).
- No fitting on request data (only `.transform`; a test asserts the scaler and imputer are unchanged).

**9.6 Training-serving skew:**
- Same parser (via XML), same features (unchanged function), same imputer/scaler, same order (`feature_list.json`).
- Pinned library versions.
- Local-time requirement for hour features.
- 5-min data only.
- Heart rate is always NaN, as in training.
- Remaining residual risk: event semantics in user CSVs (e.g. bolus logged at a different time than delivery).

## 10. Deployment strategy

- Docker `python:3.13-slim` plus `libgomp1` (needed by LightGBM). Pinned: lightgbm 4.7.0, scikit-learn 1.7.2, numpy 2.3.5, pandas 2.3.3, lxml, fastapi, uvicorn, pydantic-settings, python-multipart.
- Hugging Face Spaces, Docker SDK, port 7860, non-root user, health check `/health`.
- Image contains about 0.5 MB of model files (LightGBM + imputer + scaler), no data. Frontend (Next.js, later) on Vercel; `CORS_ORIGINS` set to its domain.

## 11. Recommended implementation order

Phase 1 (artifacts) → 2 (inference service and adapter) → 3 (API) → 4 (validation/security) → 5 (tests)
→ 6 (demo, after the 9.4 decision) → 7 (Docker) → 8 (HF Spaces) → 9 (frontend) → 10 (audit).
Mandatory causality and feature-parity tests are written in Phase 2 together with the adapter,
not deferred to Phase 5.

## 12. Phase 1 plan (after approval)

1. Create `backend/` skeleton (config only; no endpoints).
2. `artifacts.py`: load LightGBM + imputer + scaler + feature_list + settings from `MODEL_DIR`, verify SHA-256 against `model_hashes.json`, verify installed library versions, and expose read-only metadata (name, version = sha256 prefix, 47 features, horizon 30 min, 5-min sampling, LOSO metrics read from `results_causal/loso_summary.csv` at build time into a small `model_card.json`).
3. Tests: hashes match; a tampered file fails to load; version mismatch fails; feature list equals `create_features` output; predictions of the loaded model on a fixed OhioT1DM sample equal `train_final_ohio.py`'s model (bitwise/tolerance).
4. No changes to any research file; no retraining.
