"""Phase 5: guards against training-serving skew, unreachable-in-practice error paths,
response-schema conformance and the public endpoint inventory."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.app.services.inference as inference_mod
from backend.app.config import Settings
from backend.app.errors import HTTP_STATUS, BackendError, ErrorCode
from backend.app.main import create_app
from backend.app.schemas import (ApiHealth, DemoPatientData, DemoPatientList, DemoPrediction, ErrorResponse,
                                 Health, ModelInfo, PredictionResponse)
from backend.app.services.inference import InferenceService
from backend.tests.conftest import MODEL_DIR, REPO_ROOT
from backend.tests.helpers import synthetic_events
from backend.tests.test_api import to_csv, upload
from objective3_common import apply_preprocessing

HASHES = json.loads((REPO_ROOT / "backend" / "tests" / "research_code_hashes.json").read_text())["files"]
PUBLIC_PATHS = {"/health", "/api/v1/health", "/api/v1/model", "/api/v1/demo/patients",
                "/api/v1/demo/patients/{patient_id}", "/api/v1/predict/demo/{patient_id}", "/api/v1/predict/upload",
                "/api/v1/evaluate/upload"}


@pytest.fixture(scope="module")
def events():
    return synthetic_events(hours=30, seed=11)


@pytest.fixture(scope="module")
def service(artifacts):
    return InferenceService(artifacts)


@pytest.fixture(scope="module")
def t(events):
    return events.loc[events["glucose_mg_dl"].notna(), "timestamp"].iloc[150].ceil("5min")


@pytest.fixture(scope="module")
def client(tmp_path_factory, events):
    d = tmp_path_factory.mktemp("demo_g")
    (d / "synthetic_01.csv").write_bytes(to_csv(events))
    with TestClient(create_app(Settings(MODEL_DIR=MODEL_DIR, DEMO_DATA_DIR=d, RATE_LIMIT_PER_MINUTE=10_000)),
                    raise_server_exceptions=False) as c:
        yield c


# ------------------------------------------------------------------ training-serving skew guards
@pytest.mark.parametrize("name", sorted(HASHES))
def test_research_code_unchanged(name):
    """The backend imports these research modules; any change requires re-verifying parity."""
    data = (REPO_ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == HASHES[name], f"{name} changed since parity was verified"


def test_heart_rate_is_never_available_and_imputed_like_training(service, artifacts, events, t):
    res = service.predict(events, t)
    assert np.isnan(res.features["heart_rate"])                       # as in every training row
    X = apply_preprocessing(artifacts.imputer, artifacts.scaler, res.features.values.reshape(1, -1))
    j = list(artifacts.feature_list).index("heart_rate")
    expected = (artifacts.imputer.statistics_[j] - artifacts.scaler.mean_[j]) / artifacts.scaler.scale_[j]
    assert X[0, j] == pytest.approx(expected) and artifacts.imputer.statistics_[j] == 0.0


def test_imputer_statistics_match_training_settings(artifacts):
    """Only the always-NaN heart_rate column has an imputed constant of 0; others are training means."""
    stats = dict(zip(artifacts.feature_list, artifacts.imputer.statistics_))
    assert stats["heart_rate"] == 0.0
    assert 80 < stats["glucose"] < 250            # mean glucose of the OhioT1DM training rows


# ------------------------------------------------------------------ error paths
def _raises(code, fn):
    with pytest.raises(BackendError) as e:
        fn()
    assert e.value.code == code


def test_feature_schema_mismatch(service, events, t, monkeypatch):
    real = inference_mod.features_at
    monkeypatch.setattr(inference_mod, "features_at", lambda g, tt, fl: real(g, tt, fl).iloc[::-1])
    _raises(ErrorCode.FEATURE_SCHEMA_MISMATCH, lambda: service.predict(events, t))


def test_unexpected_nan_feature(service, events, t, monkeypatch):
    real = inference_mod.features_at

    def with_nan(g, tt, fl):
        f = real(g, tt, fl).copy()
        f["glucose_lag_3"] = np.nan
        return f
    monkeypatch.setattr(inference_mod, "features_at", with_nan)
    _raises(ErrorCode.INSUFFICIENT_HISTORY, lambda: service.predict(events, t))


def test_infinite_feature(service, events, t, monkeypatch):
    real = inference_mod.features_at

    def with_inf(g, tt, fl):
        f = real(g, tt, fl).copy()
        f["iob"] = np.inf
        return f
    monkeypatch.setattr(inference_mod, "features_at", with_inf)
    _raises(ErrorCode.INFINITE_FEATURE, lambda: service.predict(events, t))


def test_nan_after_preprocessing(service, events, t, monkeypatch):
    monkeypatch.setattr(inference_mod, "apply_preprocessing", lambda *a: np.full((1, 47), np.nan))
    _raises(ErrorCode.NAN_AFTER_PREPROCESSING, lambda: service.predict(events, t))


@pytest.mark.parametrize("behaviour", ["raise", "nan", "inf"])
def test_model_failure(service, artifacts, events, t, monkeypatch, behaviour):
    def bad(X):
        if behaviour == "raise":
            raise RuntimeError("boom")
        return np.array([np.nan if behaviour == "nan" else np.inf])
    monkeypatch.setattr(artifacts.model, "predict", bad)
    _raises(ErrorCode.PREDICTION_FAILED, lambda: service.predict(events, t))


def test_model_failure_via_api(client, events, monkeypatch):
    monkeypatch.setattr(client.app.state.artifacts.model, "predict", lambda X: np.array([np.nan]))
    r = client.post("/api/v1/predict/demo/synthetic_01")
    assert r.status_code == 500 and r.json()["error"]["code"] == "PREDICTION_FAILED"
    ErrorResponse.model_validate(r.json())


def test_every_error_code_has_an_http_status():
    assert set(HTTP_STATUS) == set(ErrorCode)
    assert all(400 <= s < 600 for s in HTTP_STATUS.values())


# ------------------------------------------------------------------ API contract
def test_endpoint_inventory(client):
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) == PUBLIC_PATHS                      # nothing else is exposed
    methods = {p: set(v) for p, v in spec["paths"].items()}
    assert methods["/api/v1/predict/upload"] == {"post"} and methods["/api/v1/model"] == {"get"}
    assert methods["/api/v1/evaluate/upload"] == {"post"}
    assert "multipart/form-data" in spec["paths"]["/api/v1/predict/upload"]["post"]["requestBody"]["content"]
    assert "multipart/form-data" in spec["paths"]["/api/v1/evaluate/upload"]["post"]["requestBody"]["content"]


def test_wrong_method_is_structured(client):
    r = client.get("/api/v1/predict/upload")
    assert r.status_code == 405
    ErrorResponse.model_validate(r.json())


def test_responses_match_schemas(client, events):
    Health.model_validate(client.get("/health").json())
    ApiHealth.model_validate(client.get("/api/v1/health").json())
    ModelInfo.model_validate(client.get("/api/v1/model").json())
    DemoPatientList.model_validate(client.get("/api/v1/demo/patients").json())
    DemoPatientData.model_validate(client.get("/api/v1/demo/patients/synthetic_01").json())
    DemoPrediction.model_validate(client.post("/api/v1/predict/demo/synthetic_01").json())
    cut = events[events["timestamp"] <= events["timestamp"].iloc[200]]
    PredictionResponse.model_validate(upload(client, to_csv(cut)).json())
    ErrorResponse.model_validate(client.get("/api/v1/demo/patients/nope").json())


def test_timestamps_are_naive_iso8601(client):
    p = client.post("/api/v1/predict/demo/synthetic_01").json()["prediction"]
    for k in ("prediction_time", "latest_observation_time", "forecast_time"):
        ts = pd.Timestamp(p[k])
        assert ts.tzinfo is None and "+" not in p[k] and not p[k].endswith("Z")
    assert pd.Timestamp(p["forecast_time"]) - pd.Timestamp(p["prediction_time"]) == pd.Timedelta("30min")
    assert pd.Timestamp(p["latest_observation_time"]) <= pd.Timestamp(p["prediction_time"])


def test_single_forecast_no_recursion(client, events):
    """One request -> exactly one 30-min value; repeating requests gives the same value (no state)."""
    a = client.post("/api/v1/predict/demo/synthetic_01").json()
    b = client.post("/api/v1/predict/demo/synthetic_01").json()
    assert isinstance(a["prediction"]["predicted_glucose_mg_dl"], float)
    assert a["prediction"] == b["prediction"]
    assert set(a["prediction"]) == {"predicted_glucose_mg_dl", "prediction_time", "latest_observation_time",
                                    "forecast_time", "horizon_minutes", "actual_glucose_mg_dl_at_forecast_time"}


def test_no_medical_claims_in_public_text(client):
    text = json.dumps(client.get("/openapi.json").json()).lower() + json.dumps(client.get("/api/v1/model").json()).lower()
    for phrase in ("diagnos", "insulin dose recommendation", "recommended dose", "risk of diabetes", "xgboost"):
        if phrase == "diagnos":
            assert "not for diagnosis" in text            # only as a disclaimer
            continue
        assert phrase not in text
    assert "not a medical device" in text
