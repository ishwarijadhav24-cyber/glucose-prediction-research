# Frontend handoff — Glucose Forecast API

For the frontend developer. Everything here describes the backend as it is implemented now
(`backend/`). The backend is not changed for the frontend; if something is missing, ask before adding it.

> **Disclaimer — show it on every page that displays a forecast (the API also returns it in every response,
> field `disclaimer`):**
> "Research demonstration only. Not a medical device. Not for diagnosis, treatment decisions or insulin
> dosing. Does not replace a CGM."

---

## A. Base URL

| Environment | Base URL |
|---|---|
| Local development | `http://127.0.0.1:7860` |
| Deployed backend | `<BACKEND_URL — to be inserted when a backend deployment is decided>` |

Keep the base URL in one configuration value (for example `NEXT_PUBLIC_API_BASE_URL`). All paths below are
relative to it. Interactive API docs: `<base>/docs`.

Start the backend locally (repository root, Python 3.13):

```bash
python -m pip install -r backend/requirements.txt
```

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 7860
```

## B/C. Endpoints

| # | Method | Path | Rate limited | Purpose |
|---|---|---|---|---|
| 1 | GET | `/health` | no | Liveness: `{"status": "ok"}` |
| 2 | GET | `/api/v1/health` | no | Model loaded + version |
| 3 | GET | `/api/v1/model` | no | Model card: features, research results, limitations, accepted input formats |
| 4 | GET | `/api/v1/demo/patients` | no | List synthetic demo patients |
| 5 | GET | `/api/v1/demo/patients/{patient_id}` | no | Full history of one demo patient (for charts) |
| 6 | POST | `/api/v1/predict/demo/{patient_id}` | yes | Forecast for a demo patient |
| 7 | POST | `/api/v1/predict/upload?mode=latest\|all` | yes | Forecast(s) from uploaded files |

Only GET and POST are allowed. Wrong methods return a structured 405.

## D. Request formats

**Demo prediction (6).** The body is optional. With no body, the forecast is made at the latest demo
reading. To forecast at a past time, send JSON (`Content-Type: application/json`, body ≤ 16 KB):

```json
{"prediction_time": "2027-03-02 12:00:00"}
```

`prediction_time` is local time **without** a time zone (`2027-03-02T12:00:00` also works). A time-zone
offset returns `422 INVALID_TIMESTAMP`. The backend uses only demo data at or before that time.

**Upload prediction (7).** `multipart/form-data`, one or more parts named **`file`** (1–4 files of ONE
dataset), plus the query parameter `mode=latest` (default) or `mode=all`. Total upload ≤ 25 MB. No other form
fields are accepted.

## E. Response formats

All timestamps are ISO 8601 local times without a time zone (`2027-03-02T12:00:00`). Glucose is mg/dL.

`prediction` object (used by endpoints 6 and 7):

| Field | Meaning |
|---|---|
| `predicted_glucose_mg_dl` | Forecast CGM glucose at `forecast_time` |
| `prediction_time` | Grid time *t* the forecast is made at |
| `latest_observation_time` | Last real glucose reading used (≤ `prediction_time`, at most 5 min earlier) |
| `forecast_time` | `prediction_time` + 30 min |
| `horizon_minutes` | Always 30 |

Endpoint 6 also returns `actual_glucose_mg_dl_at_forecast_time`: the recorded demo glucose at the forecast
time, **for display/comparison only** (never a model input; `null` if not recorded).

Endpoint 7 also returns `dataset` (detected type, sampling profile, counts, coverage), `validation`
(checks passed), `mode`, `predictions` (list), `prediction` (= last element of `predictions`),
`n_predictions`, `truncated`, `skipped` (counts of times without a forecast, by reason), `preprocessing`
(what was done to the data) and `warnings`.

Every success and error response contains `disclaimer`. Success responses contain `model`
(`name`, `version`, `feature_version`) and `warnings` (list of strings; **display them** when non-empty).

## F. Error format

```json
{
  "status": "error",
  "error": {"code": "NOT_FOUND", "message": "Unknown demo patient."},
  "disclaimer": "Research demonstration only. Not a medical device. Not for diagnosis, treatment decisions or insulin dosing. Does not replace a CGM."
}
```

Branch on `error.code`, show `error.message` (it never contains uploaded values).

| HTTP | Codes |
|---|---|
| 400 | `INVALID_FILE` (wrong extension, not text, unsupported content type) |
| 404 | `NOT_FOUND` |
| 405 | wrong method (code `INVALID_FORMAT`) |
| 413 | `FILE_TOO_LARGE` (uploads > 25 MB, JSON > 16 KB) |
| 415 | `UNSUPPORTED_ENCODING` |
| 422 | `INVALID_FORMAT`, `UNSUPPORTED_FORMAT`, `MISSING_COLUMNS`, `INVALID_TIMESTAMP`, `DUPLICATE_TIMESTAMP`, `INVALID_NUMERIC_VALUE`, `INSUFFICIENT_HISTORY`, `MISSING_CURRENT_GLUCOSE`, `UNSUPPORTED_SAMPLING`, `INFINITE_FEATURE` |
| 429 | `RATE_LIMITED` — header `Retry-After` (seconds): wait, then retry |
| 503 | `SERVER_BUSY` (too many predictions at once), `MODEL_NOT_AVAILABLE` |
| 504 | `PREDICTION_TIMEOUT` |
| 500 | `INTERNAL_ERROR`, `PREDICTION_FAILED`, `NAN_AFTER_PREPROCESSING`, `FEATURE_SCHEMA_MISMATCH` |

Useful user messages: `INSUFFICIENT_HISTORY` → "Not enough recent glucose history (about 1 hour of CGM
data is needed)". `MISSING_CURRENT_GLUCOSE` → "No glucose reading in the 5 minutes before this time".

## G. Supported upload types

**UI label: "Upload CGM Data"** — do **not** use "Upload Medical Record".

Supporting text:

> Supported formats: OhioT1DM XML, FreeStyle Libre / HUPA-UCM exports, or the provided CSV format.
> This research system accepts structured glucose/insulin/carbohydrate data. It does not currently process
> arbitrary medical-record PDFs, images, or EHR files.

| Format | Files | Notes |
|---|---|---|
| OhioT1DM XML | 1–2 `.xml` files of the same patient | `glucose_level`, `bolus`, `meal` events |
| FreeStyle Libre / HUPA-UCM | Libre export `.csv` (+ optional HUPA-UCM Preprocessed `.csv`) | ~15-minute data; without the Preprocessed file there are no insulin/carb records (warning returned) |
| Normalized CSV | 1 `.csv` | Header `timestamp,glucose_mg_dl,bolus_units,carbs_g`; timestamps `YYYY-MM-DD HH:MM` or `YYYY-MM-DD HH:MM:SS` (local, no zone); one row per event; empty cell = no event; ranges glucose 40–400, bolus 0–25 U, carbs 0–500 g |

Template / example file: `backend/demo_data/synthetic_01.csv` (synthetic, safe to ship). Accepted part
content types: `text/csv`, `application/csv`, `text/plain`, `application/vnd.ms-excel`, `text/xml`,
`application/xml`, `application/octet-stream` (browsers normally set one of these).

Everything else (PDF, images, Excel `.xlsx`, EHR/FHIR, heart-rate or other signals) is rejected.

## H. `mode=latest` vs `mode=all`

- `latest` (default): one forecast at the most recent eligible time. `predictions` has one element.
- `all`: an independent forecast at every eligible 5-minute time, in chronological order, capped at the
  latest 2016 forecasts (7 days; `truncated: true` when capped). Use it for a forecast-vs-actual chart.
  Times without enough history are not forecast and are counted in `skipped`.
- `prediction` always equals the last element of `predictions`, so `latest` and the end of `all` agree.

## I. Dataset detection

The backend detects the format from the file content and extension (no format parameter):
XML starting with `<` and named `.xml` → OhioT1DM; CSV with the normalized header → normalized CSV;
CSV with the HUPA-UCM Preprocessed columns or a Libre export header → HUPA-UCM. Mixing formats in one
request, or XML files of different patients, returns 422. The result is in `dataset.detected_type`
(`ohiot1dm_xml`, `hupa_ucm`, `normalized_csv`) and `dataset.sampling_profile` (`5min` or `15min`).

## J. Warnings to display

Show every string in `warnings` near the forecast. Examples the backend can return:
- ~15-minute data (see section S).
- "No HUPA-UCM Preprocessed file: no insulin or carbohydrate records are available."
- "The earliest forecasts have less than 4 hours of history: insulin on board may be underestimated for them."
- "Only the latest N forecasts are returned (MAX_PREDICTIONS_PER_REQUEST)."

## K. Wording rules for the UI

Use: **"Research forecast of CGM glucose 30 minutes ahead."**

Do not use or imply: hypoglycemia prediction or early warning, safe/unsafe or green/red risk labels,
clinical decision support, insulin or dosing recommendations, CGM replacement, diagnosis. The model is least
accurate below 70 mg/dL (research MARD 33.1 % vs 24.0 % for Persistence) — do not highlight low forecasts
as alerts. Research accuracy figures, if shown, come from `GET /api/v1/model` (`research_evaluation`,
`external_validation`, `limitations`) — do not hard-code numbers or quote any XGBoost result.

## L. Example successful prediction response (demo, synthetic data)

`POST /api/v1/predict/demo/synthetic_01` with `{"prediction_time": "2027-03-02 12:00:00"}`:

```json
{
  "status": "success",
  "prediction": {
    "predicted_glucose_mg_dl": 122.1,
    "prediction_time": "2027-03-02T12:00:00",
    "latest_observation_time": "2027-03-02T11:56:37",
    "forecast_time": "2027-03-02T12:30:00",
    "horizon_minutes": 30
  },
  "model": {"name": "LightGBM", "version": "cd7e2d54f5b4", "feature_version": "1e58d04b100d"},
  "warnings": [],
  "disclaimer": "Research demonstration only. Not a medical device. Not for diagnosis, treatment decisions or insulin dosing. Does not replace a CGM.",
  "actual_glucose_mg_dl_at_forecast_time": 105.0
}
```

Upload of `synthetic_01.csv` with `mode=latest` (abbreviated):

```json
{
  "status": "success",
  "dataset": {"detected_type": "normalized_csv", "file_types": ["normalized_csv"], "sampling_profile": "5min",
              "median_interval_minutes": 5.0, "n_glucose_readings": 854, "n_bolus_events": 9, "n_carb_events": 9,
              "start": "2027-03-01T06:01:37", "end": "2027-03-04T05:56:37", "glucose_coverage_percent": 98.8},
  "validation": {"status": "valid", "checks": ["format detected", "..."]},
  "mode": "latest",
  "predictions": [{"predicted_glucose_mg_dl": 121.1, "prediction_time": "2027-03-04T06:00:00",
                   "latest_observation_time": "2027-03-04T05:56:37", "forecast_time": "2027-03-04T06:30:00",
                   "horizon_minutes": 30}],
  "prediction": {"predicted_glucose_mg_dl": 121.1, "prediction_time": "2027-03-04T06:00:00", "...": "..."},
  "n_predictions": 1, "truncated": false, "skipped": {}, "preprocessing": [],
  "model": {"name": "LightGBM", "version": "cd7e2d54f5b4", "feature_version": "1e58d04b100d"},
  "warnings": [],
  "disclaimer": "Research demonstration only. Not a medical device. Not for diagnosis, treatment decisions or insulin dosing. Does not replace a CGM."
}
```

The same file with `mode=all` returns 828 chronological forecasts and
`"skipped": {"INSUFFICIENT_HISTORY": 26, "recomputed_causal_bucket": 15}`.

## M. Example upload request

```js
const form = new FormData();
form.append("file", fileInput.files[0]);          // repeat append("file", ...) for 2-4 files
const res = await fetch(`${API_BASE}/api/v1/predict/upload?mode=all`, { method: "POST", body: form });
const body = await res.json();
if (body.status === "error") showError(body.error.code, body.error.message);
else render(body.predictions, body.warnings, body.disclaimer);
```

Do not set the `Content-Type` header yourself for `FormData`; the browser adds the multipart boundary.

```bash
curl -F "file=@backend/demo_data/synthetic_01.csv;type=text/csv" "http://127.0.0.1:7860/api/v1/predict/upload?mode=latest"
```

## N. CORS

- Development: `http://localhost:3000` is allowed by default.
- Any other frontend origin (for example the Vercel URL) must be added on the backend:
  `CORS_ORIGINS=https://<your-app>.vercel.app` (comma-separated for several). In `ENVIRONMENT=production` the
  backend refuses to start without explicit origins and never allows `*`.
- Allowed methods: GET, POST. Allowed request header: `Content-Type`. No cookies/credentials.

## O. Authentication

None. There are no accounts, API keys or sessions. Do not add login flows that suggest data is stored.

## P. Data persistence

None. Uploaded files are processed in memory and discarded after the response; they are not stored, not
logged and never added to the demo list. If the UI needs to keep results, keep them client-side only.

## Q. Statelessness

Every request is independent: identical requests return identical forecasts, and one user's upload never
affects another request. Re-send the file to recompute.

Limits to handle in the UI: 30 prediction requests per minute per client (429 + `Retry-After`), at most
4 predictions running at once (503 `SERVER_BUSY`), 30-second prediction timeout (504).

## R. Forecast horizon

Always **30 minutes** after `prediction_time`. One forecast per time; the model is not recursive and does
not forecast further ahead. The minimum history is about 65 minutes of CGM data (`minimum_history_minutes`
in `GET /api/v1/model`).

## S. HUPA-UCM / FreeStyle Libre 15-minute warning

Libre historic data is sampled about every 15 minutes, while the model was trained on 5-minute CGM. Such
uploads are accepted with `dataset.sampling_profile = "15min"` and this warning (show it prominently):

> "Readings are ~15 minutes apart, but the model was trained on 5-minute CGM data. They are placed on the
> 5-minute grid with the research forward-fill rule (no interpolation), so trend and lag features contain
> repeated values. This is NOT equivalent to native 5-minute CGM. Research accuracy on such data
> (HUPA-UCM): MARD 13.38%."

Note for any explanatory text: on HUPA-UCM, LightGBM did not significantly outperform the Persistence
baseline. Other sampling intervals are rejected with `422 UNSUPPORTED_SAMPLING`.
