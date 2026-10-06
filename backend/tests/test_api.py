"""Phase 3: FastAPI endpoints, schemas, structured errors and startup behaviour."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.artifacts import ArtifactError
from backend.tests.conftest import MODEL_DIR
from backend.tests.helpers import COLS, corrupt_after, synthetic_events


def to_csv(events: pd.DataFrame) -> bytes:
    df = events.reindex(columns=COLS).copy()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.to_csv(index=False, na_rep="").encode("utf-8")


def CAUSAL_T(events):
    return pd.Timestamp(events["timestamp"].iloc[260]).floor("5min")


@pytest.fixture(scope="module")
def demo_events():
    return synthetic_events(hours=30, seed=3, gaps=((600, 50),))


@pytest.fixture(scope="module")
def client(tmp_path_factory, demo_events):
    d = tmp_path_factory.mktemp("demo")
    (d / "synthetic_01.csv").write_bytes(to_csv(demo_events))
    (d / "synthetic_02_future_changed.csv").write_bytes(to_csv(corrupt_after(demo_events, CAUSAL_T(demo_events))))
    (d / "not a valid id!.csv").write_bytes(to_csv(demo_events))         # ignored: invalid id
    # rate limiting is tested separately (test_hardening.py); this module makes many requests
    settings = Settings(MODEL_DIR=MODEL_DIR, DEMO_DATA_DIR=d, MAX_UPLOAD_SIZE_MB=1, RATE_LIMIT_PER_MINUTE=10_000)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        yield c


def upload(client, data: bytes, name="data.csv", ctype="text/csv"):
    return client.post("/api/v1/predict/upload", files={"file": (name, data, ctype)})


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["status"] == "error" and body["error"]["code"] == code
    assert "Traceback" not in r.text and "disclaimer" in body


# ------------------------------------------------------------------ health / model / docs
def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/api/v1/health").json()
    assert r["model_loaded"] and r["model_name"] == "LightGBM" and len(r["model_version"]) == 12


def test_model_endpoint(client):
    r = client.get("/api/v1/model").json()
    assert r["model_name"] == "LightGBM" and r["n_features"] == 47 and len(r["features"]) == 47
    assert r["prediction_horizon_minutes"] == 30 and r["sampling_interval_minutes"] == 5
    assert r["minimum_history_minutes"] == 65
    ev = r["research_evaluation"]
    assert ev["MARD_percent"] == {"mean": 10.03, "sd": 1.52} and ev["RMSE_mg_dl"] == {"mean": 20.33, "sd": 2.66}
    assert ev["R2"]["mean"] == 0.875 and ev["clarke_A_plus_B_percent"]["mean"] == 98.0
    assert "not mean each individual prediction" in ev["clarke_note"]
    assert "Research evaluation" in ev["label"]
    text = json.dumps(r).lower()
    assert "xgboost" not in text and "10.478" not in text
    assert "not a medical device" in r["disclaimer"].lower()


def test_openapi_docs(client):
    spec = client.get("/openapi.json").json()
    for path in ("/health", "/api/v1/health", "/api/v1/model", "/api/v1/demo/patients",
                 "/api/v1/demo/patients/{patient_id}", "/api/v1/predict/demo/{patient_id}", "/api/v1/predict/upload"):
        assert path in spec["paths"], path
    assert "Not a medical device" in spec["info"]["description"]
    assert client.get("/docs").status_code == 200


# ------------------------------------------------------------------ demo
def test_demo_list_and_get(client, demo_events):
    lst = client.get("/api/v1/demo/patients").json()
    assert [p["patient_id"] for p in lst["patients"]] == ["synthetic_01", "synthetic_02_future_changed"]
    d = client.get("/api/v1/demo/patients/synthetic_01").json()
    assert d["n_glucose_readings"] == int(demo_events["glucose_mg_dl"].notna().sum()) == len(d["glucose"])
    assert len(d["carbs_g"]) == int(demo_events["carbs_g"].notna().sum())


@pytest.mark.parametrize("pid", ["unknown", "..%2F..%2Fsecret", "not a valid id!"])
def test_demo_unknown_patient(client, pid):
    assert_error(client.get(f"/api/v1/demo/patients/{pid}"), 404, "NOT_FOUND")
    assert_error(client.post(f"/api/v1/predict/demo/{pid}"), 404, "NOT_FOUND")


def test_demo_prediction_matches_service(client, demo_events):
    svc = client.app.state.service
    t = pd.Timestamp(demo_events["timestamp"].iloc[200]).floor("5min")
    r = client.post("/api/v1/predict/demo/synthetic_01", json={"prediction_time": t.isoformat()})
    assert r.status_code == 200, r.text
    body = r.json()
    direct = svc.predict(demo_events, t)
    assert body["prediction"]["predicted_glucose_mg_dl"] == round(direct.predicted_glucose_mg_dl, 1)
    assert pd.Timestamp(body["prediction"]["prediction_time"]) == t
    assert pd.Timestamp(body["prediction"]["forecast_time"]) == t + pd.Timedelta("30min")
    assert body["actual_glucose_mg_dl_at_forecast_time"] is not None
    assert body["model"]["name"] == "LightGBM"


def test_demo_default_time_is_latest(client, demo_events):
    body = client.post("/api/v1/predict/demo/synthetic_01").json()
    last = demo_events.loc[demo_events["glucose_mg_dl"].notna(), "timestamp"].max()
    assert pd.Timestamp(body["prediction"]["prediction_time"]) == last.ceil("5min")
    assert body["actual_glucose_mg_dl_at_forecast_time"] is None      # no data 30 min after the end


def test_demo_errors(client, demo_events):
    t0 = demo_events["timestamp"].min().ceil("5min")
    assert_error(client.post("/api/v1/predict/demo/synthetic_01", json={"prediction_time": t0.isoformat()}),
                 422, "INSUFFICIENT_HISTORY")
    off = (t0 + pd.Timedelta("3h 2min")).isoformat()
    assert_error(client.post("/api/v1/predict/demo/synthetic_01", json={"prediction_time": off}),
                 422, "INVALID_TIMESTAMP")
    assert_error(client.post("/api/v1/predict/demo/synthetic_01", json={"prediction_time": "not-a-date"}),
                 422, "INVALID_FORMAT")


# ------------------------------------------------------------------ upload: success + causality via API
def test_upload_success_matches_service(client, demo_events):
    cut = demo_events[demo_events["timestamp"] <= demo_events["timestamp"].iloc[250]]
    r = upload(client, to_csv(cut))
    assert r.status_code == 200, r.text
    body = r.json()
    direct = client.app.state.service.predict(cut)
    assert body["status"] == "success"
    assert body["prediction"]["predicted_glucose_mg_dl"] == round(direct.predicted_glucose_mg_dl, 1)
    assert body["prediction"]["horizon_minutes"] == 30
    assert set(body) == {"status", "dataset", "validation", "mode", "predictions", "prediction", "n_predictions",
                         "truncated", "skipped", "preprocessing", "model", "warnings", "disclaimer"}
    assert body["dataset"]["detected_type"] == "normalized_csv" and body["dataset"]["sampling_profile"] == "5min"
    assert body["predictions"] == [body["prediction"]] and body["n_predictions"] == 1


def test_causality_end_to_end_via_api(client, demo_events):
    """Two demo files identical up to t, radically different after t -> identical forecast at t."""
    t = CAUSAL_T(demo_events)
    body = {"prediction_time": t.isoformat()}
    a = client.post("/api/v1/predict/demo/synthetic_01", json=body).json()
    b = client.post("/api/v1/predict/demo/synthetic_02_future_changed", json=body).json()
    assert a["status"] == b["status"] == "success"
    assert a["prediction"] == b["prediction"]
    assert a["actual_glucose_mg_dl_at_forecast_time"] != b["actual_glucose_mg_dl_at_forecast_time"]


def test_unsorted_upload_is_sorted_with_warning(client, demo_events):
    cut = demo_events[demo_events["timestamp"] <= demo_events["timestamp"].iloc[250]]
    shuffled = cut.sample(frac=1, random_state=0)
    r = upload(client, to_csv(shuffled))
    assert r.status_code == 200
    assert any("chronological" in w for w in r.json()["warnings"])
    assert r.json()["prediction"] == upload(client, to_csv(cut)).json()["prediction"]


# ------------------------------------------------------------------ upload: validation errors
def _good(demo_events, n=60):
    return demo_events[demo_events["glucose_mg_dl"].notna()].iloc[:n].copy()


def test_upload_file_checks(client, demo_events):
    data = to_csv(_good(demo_events))
    assert_error(upload(client, data, name="data.txt"), 400, "INVALID_FILE")
    assert_error(upload(client, data, ctype="application/json"), 400, "INVALID_FILE")
    assert_error(upload(client, b"\xff\xfe\x00bad"), 400, "INVALID_FILE")
    assert_error(upload(client, b""), 422, "INVALID_FORMAT")
    assert_error(upload(client, b"x" * (1024 * 1024 + 10)), 413, "FILE_TOO_LARGE")
    assert_error(upload(client, data, name="../../etc/passwd"), 400, "INVALID_FILE")


def test_upload_schema_errors(client, demo_events):
    ev = _good(demo_events)
    no_carbs = to_csv(ev).decode().splitlines()
    no_carbs = "\n".join(line.rsplit(",", 1)[0] for line in no_carbs).encode()   # drop last column (carbs_g)
    assert no_carbs.splitlines()[0] == b"timestamp,glucose_mg_dl,bolus_units"
    assert_error(upload(client, no_carbs), 422, "MISSING_COLUMNS")
    # heart_rate is not accepted: the model never received heart-rate data
    hr = b"timestamp,glucose_mg_dl,bolus_units,carbs_g,heart_rate\n2027-03-01 06:00,100,,,70\n"
    assert_error(upload(client, hr), 422, "INVALID_FORMAT")
    repeated = b"timestamp,glucose_mg_dl,glucose_mg_dl,bolus_units,carbs_g\n2027-03-01 06:00,100,100,,\n"
    assert_error(upload(client, repeated), 422, "INVALID_FORMAT")


@pytest.mark.parametrize("ts", ["2027-03-01T06:00:00+01:00", "01/03/2027 06:00", "2027-13-01 06:00", ""])
def test_upload_bad_timestamp(client, ts):
    data = f"timestamp,glucose_mg_dl,bolus_units,carbs_g\n{ts},120,,\n".encode()
    assert_error(upload(client, data), 422, "INVALID_TIMESTAMP")


def test_upload_duplicate_timestamp(client, demo_events):
    ev = _good(demo_events)
    dup = pd.concat([ev, ev.iloc[[10]]]).reindex(columns=COLS)
    assert_error(upload(client, to_csv(dup)), 422, "DUPLICATE_TIMESTAMP")


@pytest.mark.parametrize("col,val", [("glucose_mg_dl", "abc123secret"), ("glucose_mg_dl", "inf"),
                                     ("glucose_mg_dl", "900"), ("bolus_units", "-1"), ("carbs_g", "1000")])
def test_upload_bad_numbers_without_echoing_values(client, demo_events, col, val):
    ev = _good(demo_events)
    csv = to_csv(ev).decode().splitlines()
    row = ev.iloc[5]
    fields = [row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), "", "", ""]
    fields[COLS.index(col)] = val
    csv.insert(7, ",".join(fields))
    r = upload(client, "\n".join(csv).encode())
    assert_error(r, 422, "INVALID_NUMERIC_VALUE")
    assert val not in r.text


def test_upload_15_minute_data_accepted_with_explicit_warning(client, demo_events):
    """Phase 6: ~15-minute data is accepted only with the external-validation warning (never as 5-min)."""
    ev = _good(demo_events, 90).iloc[::3]
    r = upload(client, to_csv(ev))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["sampling_profile"] == "15min"
    assert any("NOT equivalent to native 5-minute CGM" in w for w in body["warnings"])
    assert any("nothing interpolated" in p for p in body["preprocessing"])


def test_upload_other_sampling_rejected(client, demo_events):
    ev = _good(demo_events, 60).iloc[::2]                                   # ~10-minute data
    assert_error(upload(client, to_csv(ev)), 422, "UNSUPPORTED_SAMPLING")


def test_upload_insufficient_history(client, demo_events):
    assert_error(upload(client, to_csv(_good(demo_events, 13))), 422, "INSUFFICIENT_HISTORY")


def test_upload_row_without_value(client):
    data = b"timestamp,glucose_mg_dl,bolus_units,carbs_g\n2027-03-01 06:00,120,,\n2027-03-01 06:05,,,\n"
    assert_error(upload(client, data), 422, "INVALID_FORMAT")


# ------------------------------------------------------------------ robustness
def test_unexpected_error_is_generic(client, demo_events, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(client.app.state.service, "predict", boom)
    r = upload(client, to_csv(_good(demo_events)))
    assert_error(r, 500, "INTERNAL_ERROR")
    assert "secret" not in r.text


def test_startup_fails_without_valid_model(tmp_path):
    app = create_app(Settings(MODEL_DIR=tmp_path))
    with pytest.raises(ArtifactError):
        with TestClient(app):
            pass


def test_cors(client):
    ok = client.options("/api/v1/model", headers={"Origin": "http://localhost:3000",
                                                  "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    bad = client.get("/api/v1/model", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_production_rejects_wildcard_cors():
    with pytest.raises(ValueError):
        create_app(Settings(MODEL_DIR=MODEL_DIR, ENVIRONMENT="production", CORS_ORIGINS="*"))


def test_no_upload_persisted(client, demo_events, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    upload(client, to_csv(_good(demo_events)))
    assert list(tmp_path.iterdir()) == []
