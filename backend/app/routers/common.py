"""Shared helpers for routers."""

from fastapi import Request

from backend.app.errors import BackendError, ErrorCode
from backend.app.schemas import ModelRef, Prediction, PredictionResponse
from backend.app.services.inference import PredictionResult


def service(request: Request):
    svc = getattr(request.app.state, "service", None)
    if svc is None:
        raise BackendError(ErrorCode.MODEL_NOT_AVAILABLE, "The model is not loaded.")
    return svc


def to_response(request: Request, result: PredictionResult, extra_warnings=(), cls=PredictionResponse, **extra):
    art = request.app.state.artifacts
    return cls(
        prediction=Prediction(predicted_glucose_mg_dl=round(result.predicted_glucose_mg_dl, 1),
                              prediction_time=result.prediction_time.to_pydatetime(),
                              latest_observation_time=result.latest_observation_time.to_pydatetime(),
                              forecast_time=result.forecast_time.to_pydatetime(),
                              horizon_minutes=result.horizon_minutes),
        model=ModelRef(name=art.model_name, version=art.model_version, feature_version=art.feature_version),
        warnings=list(extra_warnings) + list(result.warnings),
        **extra,
    )
