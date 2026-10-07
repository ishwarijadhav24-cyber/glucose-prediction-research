"""Pydantic request/response models (the public API contract)."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

DISCLAIMER = ("Research demonstration only. Not a medical device. Not for diagnosis, treatment decisions "
              "or insulin dosing. Does not replace a CGM.")


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: ErrorBody
    disclaimer: str = DISCLAIMER


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class ApiHealth(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_name: str
    model_version: str
    environment: str


class ModelRef(BaseModel):
    name: str
    version: str = Field(description="First 12 hex characters of the model file's SHA-256")
    feature_version: str = Field(description="First 12 hex characters of feature_list.json's SHA-256")


class Prediction(BaseModel):
    predicted_glucose_mg_dl: float = Field(description="Forecast glucose at forecast_time (mg/dL)")
    prediction_time: datetime = Field(description="5-minute grid time t the forecast is made at (local time)")
    latest_observation_time: datetime = Field(description="Last real glucose reading used (<= prediction_time)")
    forecast_time: datetime = Field(description="prediction_time + horizon")
    horizon_minutes: int
    actual_glucose_mg_dl_at_forecast_time: Optional[float] = Field(
        default=None, description="Recorded glucose at forecast_time, if available (for evaluation). Never used as input."
    )


class PredictionResponse(BaseModel):
    status: Literal["success"] = "success"
    prediction: Prediction
    model: ModelRef
    warnings: list[str]
    disclaimer: str = DISCLAIMER


class DatasetSummary(BaseModel):
    detected_type: Literal["ohiot1dm_xml", "hupa_ucm", "normalized_csv"]
    file_types: list[str]
    sampling_profile: Literal["5min", "15min"] = Field(
        description="5min = native CGM as in training; 15min = FreeStyle Libre historic data, external-validation only")
    median_interval_minutes: float
    n_glucose_readings: int
    n_bolus_events: int
    n_carb_events: int
    start: datetime
    end: datetime
    glucose_coverage_percent: float = Field(description="Readings received / readings expected at the detected interval")


class Validation(BaseModel):
    status: Literal["valid"] = "valid"
    checks: list[str]


class UploadPredictionResponse(BaseModel):
    status: Literal["success"] = "success"
    dataset: DatasetSummary
    validation: Validation
    mode: Literal["latest", "all"]
    predictions: list[Prediction] = Field(description="Independent forecasts, each from data at or before its "
                                                      "prediction_time; chronological")
    prediction: Prediction = Field(description="The latest forecast (last element of predictions)")
    n_predictions: int
    truncated: bool = Field(description="True if only the latest max_predictions are returned")
    skipped: dict[str, int]
    preprocessing: list[str] = Field(description="Exactly what was done to the uploaded data")
    model: ModelRef
    warnings: list[str]
    disclaimer: str = DISCLAIMER


class UploadEvaluationResponse(BaseModel):
    status: Literal["success"] = "success"
    dataset: DatasetSummary
    history_days: float
    evaluation_days: float
    predictions: list[Prediction]
    warnings: list[str]
    disclaimer: str = DISCLAIMER


class DemoPrediction(PredictionResponse):
    actual_glucose_mg_dl_at_forecast_time: Optional[float] = Field(
        default=None, description="Recorded glucose at forecast_time, for display only. Never used as model "
                                  "input; null if not recorded.")


class DemoPredictRequest(BaseModel):
    prediction_time: Optional[datetime] = Field(
        default=None, description="Grid time (multiple of 5 minutes). Default: the latest possible time.")


class GlucosePoint(BaseModel):
    timestamp: datetime
    glucose_mg_dl: float


class EventPoint(BaseModel):
    timestamp: datetime
    value: float


class DemoPatientSummary(BaseModel):
    patient_id: str
    n_glucose_readings: int
    start: datetime
    end: datetime


class DemoPatientData(DemoPatientSummary):
    glucose: list[GlucosePoint]
    bolus_units: list[EventPoint]
    carbs_g: list[EventPoint]


class DemoPatientList(BaseModel):
    patients: list[DemoPatientSummary]
    note: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    feature_version: str
    n_features: int
    features: list[str]
    prediction_horizon_minutes: int
    sampling_interval_minutes: int
    minimum_history_minutes: int
    training_data: str
    library_versions: dict
    research_evaluation: dict
    external_validation: dict
    limitations: list[str]
    input_format: dict
    disclaimer: str = DISCLAIMER
