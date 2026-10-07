import pytest
import pandas as pd
from backend.tests.test_api import client, demo_events, to_csv

def evaluate_upload(client, data: bytes, history_days: float = 0.0, name="data.csv", ctype="text/csv"):
    return client.post(f"/api/v1/evaluate/upload?history_days={history_days}", files={"file": (name, data, ctype)})

def test_evaluate_endpoint_success(client, demo_events):
    cut = demo_events[demo_events["timestamp"] <= demo_events["timestamp"].iloc[250]]
    r = evaluate_upload(client, to_csv(cut), history_days=0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert "predictions" in body
    assert body["history_days"] == 0
    assert "evaluation_days" in body

def test_prediction_actual_timestamp_alignment(client, demo_events):
    cut = demo_events[demo_events["timestamp"] <= demo_events["timestamp"].iloc[250]]
    r = evaluate_upload(client, to_csv(cut), history_days=0)
    body = r.json()
    predictions = body["predictions"]
    valid = [p for p in predictions if p["actual_glucose_mg_dl_at_forecast_time"] is not None]
    assert len(valid) > 0
    
    # Verify alignment
    for p in valid:
        pt = pd.Timestamp(p["prediction_time"])
        ft = pd.Timestamp(p["forecast_time"])
        assert ft == pt + pd.Timedelta(minutes=p["horizon_minutes"])

def test_history_window_selection(client, demo_events):
    cut = demo_events[demo_events["timestamp"] <= demo_events["timestamp"].iloc[250]]
    r1 = evaluate_upload(client, to_csv(cut), history_days=0.5)
    body1 = r1.json()
    assert body1["history_days"] == 0.5
    
    r0 = evaluate_upload(client, to_csv(cut), history_days=0)
    body0 = r0.json()
    assert body0["history_days"] == 0
    assert body1["evaluation_days"] < body0["evaluation_days"]
