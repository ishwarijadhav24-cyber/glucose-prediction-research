from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from backend.app.errors import BackendError, ErrorCode
from backend.app.routers.common import service
from backend.app.schemas import DatasetSummary, ModelRef, Prediction, UploadPredictionResponse, UploadEvaluationResponse, Validation
from backend.app.security import rate_limit, run_bounded
from backend.app.services.datasets import MAX_FILES, UploadedFile, ingest_upload
from backend.app.services.features import build_grid

router = APIRouter(prefix="/api/v1", tags=["predict"])
ALLOWED_TYPES = {"text/csv", "application/csv", "text/plain", "application/vnd.ms-excel", "text/xml",
                 "application/xml", "application/octet-stream"}
CHUNK = 64 * 1024
CHECKS = ["format detected", "file type and encoding", "required signals present", "timestamps parsed",
          "numeric values finite and in range", "duplicates handled", "chronological order", "sampling interval",
          "history sufficient for every returned prediction", "47 features finite and in model order"]

# The multipart form is parsed here (not through File(...)) so that only 'file' parts are accepted
# and files stay in memory (see main.py: spool size = upload limit).
UPLOAD_OPENAPI = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["file"],
    "properties": {"file": {"type": "array", "items": {"type": "string", "format": "binary"},
                            "description": "1-4 files of one dataset: OhioT1DM XML (1-2), HUPA-UCM FreeStyle Libre "
                                           "export(s) + optional Preprocessed CSV, or one normalized CSV. "
                                           "See GET /api/v1/model input_format."}}}}}}}


async def _read_limited(upload: UploadFile, budget: list) -> bytes:
    data = bytearray()
    while chunk := await upload.read(CHUNK):
        data += chunk
        budget[0] -= len(chunk)
        if budget[0] < 0:
            raise BackendError(ErrorCode.FILE_TOO_LARGE, "The uploaded files exceed the size limit.")
    return bytes(data)


def _process(svc, files, mode, settings, external):
    ds = ingest_upload(files, external)
    if mode == "latest":
        results, skipped, truncated = [svc.predict(ds.events)], {}, False
    else:
        results, skipped, truncated = svc.predict_many(ds.events, settings.max_predictions_per_request)
    return ds, results, skipped, truncated


def _summary(ds) -> DatasetSummary:
    ev = ds.events
    g = ev.loc[ev["glucose_mg_dl"].notna(), "timestamp"]
    span = (g.max() - g.min()) / pd.Timedelta("1min")
    step = {"5min": 5.0, "15min": 15.0}[ds.sampling_profile]
    expected = span / step + 1
    return DatasetSummary(detected_type=ds.dataset_type, file_types=ds.file_types, sampling_profile=ds.sampling_profile,
                          median_interval_minutes=ds.median_interval_minutes, n_glucose_readings=len(g),
                          n_bolus_events=int(ev["bolus_units"].notna().sum()),
                          n_carb_events=int(ev["carbs_g"].notna().sum()),
                          start=g.min().to_pydatetime(), end=g.max().to_pydatetime(),
                          glucose_coverage_percent=round(float(min(100.0, 100.0 * len(g) / expected)), 1))


@router.post("/predict/upload", response_model=UploadPredictionResponse, dependencies=[Depends(rate_limit)],
             openapi_extra=UPLOAD_OPENAPI,
             summary="30-minute forecast(s) from uploaded dataset files (processed in memory, never stored)")
async def predict_upload(request: Request,
                         mode: Literal["latest", "all"] = Query("latest", description=(
                             "latest: one forecast at the last reading; all: a forecast at every eligible time "
                             "(latest MAX_PREDICTIONS_PER_REQUEST)"))):
    svc = service(request)
    settings = request.app.state.settings
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise BackendError(ErrorCode.INVALID_FORMAT, "Send the files as multipart/form-data in 'file' field(s).")
    try:
        form = await request.form(max_files=MAX_FILES, max_fields=0, max_part_size=1024)
    except (MultiPartException, StarletteHTTPException):
        raise BackendError(ErrorCode.INVALID_FORMAT, f"Malformed multipart request (1-{MAX_FILES} 'file' parts).") \
            from None
    try:
        items = form.multi_items()
        if not items or any(k != "file" or not isinstance(v, UploadFile) for k, v in items):
            raise BackendError(ErrorCode.INVALID_FORMAT, "Only 'file' parts are accepted.")
        budget = [settings.max_upload_bytes]
        files = []
        for _, up in items:
            ftype = (up.content_type or "").split(";")[0].strip().lower()
            if ftype not in ALLOWED_TYPES:
                raise BackendError(ErrorCode.INVALID_FILE, "Unsupported content type; upload CSV or XML files.")
            files.append(UploadedFile(data=await _read_limited(up, budget), filename=up.filename or ""))
    finally:
        await form.close()
    external = request.app.state.model_card.get("external_validation")
    ds, results, skipped, truncated = await run_bounded(request, _process, svc, files, mode, settings, external)
    del files
    art = request.app.state.artifacts
    preds = [Prediction(predicted_glucose_mg_dl=round(r.predicted_glucose_mg_dl, 1),
                        prediction_time=r.prediction_time.to_pydatetime(),
                        latest_observation_time=r.latest_observation_time.to_pydatetime(),
                        forecast_time=r.forecast_time.to_pydatetime(), horizon_minutes=r.horizon_minutes)
             for r in results]
    per_result = list(dict.fromkeys(w for r in results for w in r.warnings))
    if any(w.startswith("No insulin or carbohydrate records") for w in ds.warnings):   # already stated once
        per_result = [w for w in per_result if not w.startswith("No insulin or carbohydrate records")]
    warnings = list(ds.warnings) + per_result
    if mode == "all":
        if truncated:
            warnings.append(f"Only the latest {len(preds)} forecasts are returned (MAX_PREDICTIONS_PER_REQUEST).")
        if len(ds.events) and ds.events["timestamp"].min() > results[0].prediction_time - pd.Timedelta("240min"):
            warnings.append("The earliest forecasts have less than 4 hours of history: insulin on board may be "
                            "underestimated for them.")
    return UploadPredictionResponse(
        dataset=_summary(ds), validation=Validation(checks=CHECKS), mode=mode, predictions=preds,
        prediction=preds[-1], n_predictions=len(preds), truncated=truncated, skipped=skipped,
        preprocessing=ds.transformations, model=ModelRef(name=art.model_name, version=art.model_version,
                                                         feature_version=art.feature_version),
        warnings=warnings)


def _process_evaluate(svc, files, history_days, external):
    ds = ingest_upload(files, external)
    ev = ds.events
    g = ev.loc[ev["glucose_mg_dl"].notna(), "timestamp"]
    if g.empty:
        raise BackendError(ErrorCode.INVALID_FORMAT, "No glucose readings found in the dataset.")
    start = g.min()
    end = g.max()
    eval_start = start + pd.Timedelta(days=history_days)

    # We want to evaluate strictly unseen data.
    # We will pass the full events to predict_many so it has history, but we will filter the returned predictions.
    results, skipped, truncated = svc.predict_many(ev, 100000) # High limit to ensure we get all available evaluation points
    return ds, results, start, end, eval_start


@router.post("/evaluate/upload", response_model=UploadEvaluationResponse, dependencies=[Depends(rate_limit)],
             openapi_extra=UPLOAD_OPENAPI,
             summary="Evaluate 30-minute forecasts over a specific history/evaluation split")
async def evaluate_upload(request: Request, history_days: float = Query(0.0)):
    svc = service(request)
    settings = request.app.state.settings
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise BackendError(ErrorCode.INVALID_FORMAT, "Send the files as multipart/form-data in 'file' field(s).")
    try:
        form = await request.form(max_files=MAX_FILES, max_fields=0, max_part_size=1024)
    except (MultiPartException, StarletteHTTPException):
        raise BackendError(ErrorCode.INVALID_FORMAT, f"Malformed multipart request (1-{MAX_FILES} 'file' parts).") \
            from None
    try:
        items = form.multi_items()
        if not items or any(k != "file" or not isinstance(v, UploadFile) for k, v in items):
            raise BackendError(ErrorCode.INVALID_FORMAT, "Only 'file' parts are accepted.")
        budget = [settings.max_upload_bytes]
        files = []
        for _, up in items:
            ftype = (up.content_type or "").split(";")[0].strip().lower()
            if ftype not in ALLOWED_TYPES:
                raise BackendError(ErrorCode.INVALID_FILE, "Unsupported content type; upload CSV or XML files.")
            files.append(UploadedFile(data=await _read_limited(up, budget), filename=up.filename or ""))
    finally:
        await form.close()

    external = request.app.state.model_card.get("external_validation")
    ds, results, start, end, eval_start = await run_bounded(request, _process_evaluate, svc, files, history_days, external)
    del files

    # Filter results to evaluation window (prediction_time >= eval_start)
    results = [r for r in results if r.prediction_time >= eval_start]

    # Fast lookup for actual glucose using the 5-minute research grid
    grid = build_grid(ds.events)
    g_lookup = dict(zip(grid.index, grid["glucose"]))

    preds = []
    for r in results:
        actual = g_lookup.get(r.forecast_time)
        preds.append(Prediction(
            predicted_glucose_mg_dl=round(r.predicted_glucose_mg_dl, 1),
            prediction_time=r.prediction_time.to_pydatetime(),
            latest_observation_time=r.latest_observation_time.to_pydatetime(),
            forecast_time=r.forecast_time.to_pydatetime(),
            horizon_minutes=r.horizon_minutes,
            actual_glucose_mg_dl_at_forecast_time=round(actual, 1) if pd.notna(actual) else None
        ))

    eval_days = max(0.0, (end - eval_start) / pd.Timedelta(days=1))

    warnings = list(ds.warnings)
    if not preds:
        warnings.append("No eligible predictions found in the evaluation window.")

    return UploadEvaluationResponse(
        dataset=_summary(ds),
        history_days=history_days,
        evaluation_days=eval_days,
        predictions=preds,
        warnings=warnings
    )
