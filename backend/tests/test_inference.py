"""Phase 2: causality, feature parity and inference-service behaviour.

Causality: two inputs identical up to and including t, radically different after t,
must give identical features (all 47) and identical predictions at t.
Parity: backend features == research pipeline features (parse_xml_file on the full
input + create_features on the full grid, row t) for identical input data.
"""

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from backend.app.errors import BackendError, ErrorCode
from backend.app.services.features import (CONTEXT, build_grid, events_to_xml, features_at,
                                           pipeline_constants)
from backend.app.services.inference import InferenceService
from backend.tests.conftest import OHIO_XML, requires
from backend.tests.helpers import COLS, corrupt_after, ohio_events, synthetic_events
from data_processing import create_features, parse_xml_file
from objective3_common import apply_preprocessing

needs_ohio = requires(OHIO_XML)
FIVE = pd.Timedelta("5min")
# gap of 50 min (> 30-min forward-fill limit) and a gap of 25 min (bridged by forward fill)
GAPS = ((600, 50), (1000, 25))


@pytest.fixture(scope="module")
def service(artifacts):
    return InferenceService(artifacts)


@pytest.fixture(scope="module")
def base_events():
    return synthetic_events(hours=30, seed=0, gaps=GAPS)


def _try(service, events, t):
    try:
        return service.predict(events, t)
    except BackendError as e:
        return e


def _grid_times(events):
    return build_grid(events).index


def _boundary_times(events):
    """Grid times around gaps, history and rolling-window edges, meals and boluses."""
    t0 = events["timestamp"].min().ceil("5min")
    idx = _grid_times(events)
    out = set()
    for gs, gl in GAPS:
        start = (events["timestamp"].min() + pd.Timedelta(minutes=gs)).floor("5min")
        end = start + pd.Timedelta(minutes=gl)
        for d in (-10, -5, 0, 5, 25, 30, 35, 40, 55, 60, 65, 70, 75, 80, 90):
            out.add(start + pd.Timedelta(minutes=d))
            out.add(end + pd.Timedelta(minutes=d))
    for d in range(0, 100, 5):                      # start of data: history 1..20 rows
        out.add(t0 + pd.Timedelta(minutes=d))
    for col in ("bolus_units", "carbs_g"):
        for ts in events.loc[events[col].notna(), "timestamp"]:
            for d in (-5, 0, 5, 25, 30):            # IOB / lag windows around events
                out.add(ts.floor("5min") + pd.Timedelta(minutes=d))
    return sorted(t for t in out if idx[0] <= t <= idx[-1])


# ------------------------------------------------------------------ constants
def test_pipeline_constants_measured_from_create_features():
    c = pipeline_constants()
    assert c["horizon_rows"] == 6 and c["horizon_minutes"] == 30
    assert c["min_history_rows"] == 14          # glucose at t and the 13 rows before (65 min)


# ------------------------------------------------------------------ causality (mandatory)
def _assert_same(a, b, t):
    if isinstance(a, BackendError):
        assert isinstance(b, BackendError) and a.code == b.code, f"t={t}: {a!r} vs {b!r}"
        return False
    assert not isinstance(b, BackendError), f"t={t}: A ok, B {b!r}"
    assert list(a.features.index) == list(b.features.index)
    np.testing.assert_array_equal(a.features.values, b.features.values, err_msg=f"features differ at {t}")
    assert a.predicted_glucose_mg_dl == b.predicted_glucose_mg_dl, f"prediction differs at {t}"
    assert a.latest_observation_time == b.latest_observation_time
    return True


def test_causality_boundary_and_random_times(service, base_events):
    times = _boundary_times(base_events)
    rng = np.random.RandomState(0)
    grid_times = _grid_times(base_events)
    times += list(rng.choice(grid_times, 40, replace=False))
    n_ok = 0
    for i, t in enumerate(sorted(set(pd.DatetimeIndex(times)))):
        a = _try(service, base_events, t)
        b = _try(service, corrupt_after(base_events, t, seed=i), t)
        n_ok += _assert_same(a, b, t)
    assert n_ok > 60   # most times produce a prediction; the rest must fail identically


@needs_ohio
def test_causality_real_ohio_data(service):
    ev = ohio_events(OHIO_XML)
    grid_times = build_grid(ev).index
    rng = np.random.RandomState(1)
    n_ok = 0
    for i, t in enumerate(rng.choice(grid_times[300:], 25, replace=False)):
        t = pd.Timestamp(t)
        n_ok += _assert_same(_try(service, ev, t), _try(service, corrupt_after(ev, t, seed=i), t), t)
    assert n_ok >= 15


def test_placeholder_rows_affect_only_target(artifacts, base_events):
    grid = build_grid(base_events)
    fl = artifacts.feature_list
    checked = 0
    for t in grid.index[grid["glucose"].notna()][20::45]:
        f1, tgt1 = features_at(grid, t, fl, placeholder_glucose=40.0, return_target=True)
        f2, tgt2 = features_at(grid, t, fl, placeholder_glucose=400.0, return_target=True)
        np.testing.assert_array_equal(f1.values, f2.values)
        assert tgt1 == 40.0 and tgt2 == 400.0          # only the target sees the placeholders
        checked += 1
    assert checked >= 5


def test_adapter_ignores_grid_rows_after_t(artifacts, base_events):
    grid = build_grid(base_events)
    t = grid.index[300]
    full = features_at(grid, t, artifacts.feature_list)
    cut = features_at(grid.loc[:t], t, artifacts.feature_list)
    np.testing.assert_array_equal(full.values, cut.values)


# ------------------------------------------------------------------ model inputs
def test_target_never_passed_to_model(service, artifacts, base_events, monkeypatch):
    seen = []
    real_predict = artifacts.model.predict

    def spy(X, *a, **k):
        seen.append(np.array(X, copy=True))
        return real_predict(X, *a, **k)

    monkeypatch.setattr(artifacts.model, "predict", spy)
    t = _grid_times(base_events)[300]
    res = service.predict(base_events, t)
    assert "target" not in res.features.index and "ts" not in res.features.index
    assert len(seen) == 1 and seen[0].shape == (1, 47)
    expected = apply_preprocessing(artifacts.imputer, artifacts.scaler, res.features.values.reshape(1, -1))
    np.testing.assert_array_equal(seen[0], expected)


def test_preprocessing_uses_transform_only(service, artifacts, base_events, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("fit called during inference")
    for obj in (artifacts.imputer, artifacts.scaler):
        for name in ("fit", "fit_transform", "partial_fit"):
            if hasattr(obj, name):
                monkeypatch.setattr(obj, name, forbidden)
    for name in ("fit", "partial_fit"):
        monkeypatch.setattr(artifacts.model, name, forbidden, raising=False)
    t = _grid_times(base_events)[300]
    assert np.isfinite(service.predict(base_events, t).predicted_glucose_mg_dl)


def test_artifacts_unchanged_by_predictions(service, artifacts, base_events):
    before = [joblib.hash(o) for o in (artifacts.model, artifacts.imputer, artifacts.scaler)]
    for t in _grid_times(base_events)[200:260]:
        _try(service, base_events, t)
    after = [joblib.hash(o) for o in (artifacts.model, artifacts.imputer, artifacts.scaler)]
    assert before == after


def test_input_not_mutated_and_no_state(service, base_events):
    ev = base_events.copy()
    times = _grid_times(ev)
    t1, t2 = times[300], times[350]
    p1 = service.predict(ev, t1)
    service.predict(ev, t2)                          # a later prediction must not influence t1
    p1b = service.predict(ev, t1)
    pd.testing.assert_frame_equal(ev, base_events)
    assert p1.predicted_glucose_mg_dl == p1b.predicted_glucose_mg_dl


# ------------------------------------------------------------------ feature parity (mandatory)
def _parity(service, artifacts, events, research_grid, times):
    """Compare backend vs research features and predictions at the given times."""
    research = create_features(research_grid).set_index("ts")
    fl = list(artifacts.feature_list)
    bolus_carb_times = events.loc[events["bolus_units"].notna() | events["carbs_g"].notna(), "timestamp"].values
    compared, skipped_bucket = 0, 0
    b_rows, r_rows = [], []
    for t in times:
        t = pd.Timestamp(t)
        if t not in research.index:
            continue
        res = _try(service, events, t)
        if isinstance(res, BackendError):
            continue
        # research grid row t also holds events from (t, t+5 min): documented difference, excluded here
        if ((bolus_carb_times > np.datetime64(t)) & (bolus_carb_times < np.datetime64(t + FIVE))).any():
            skipped_bucket += 1
            continue
        r = research.loc[t, fl]
        assert list(res.features.index) == list(r.index) == fl
        assert res.features.dtype == np.float64 and r.astype(float).dtype == np.float64
        np.testing.assert_allclose(res.features.values, r.values.astype(float), rtol=0, atol=1e-9,
                                   equal_nan=True, err_msg=f"feature parity failed at {t}")
        b_rows.append(res.predicted_glucose_mg_dl)
        r_rows.append(r.values.astype(float))
        compared += 1
    r_pred = artifacts.model.predict(apply_preprocessing(artifacts.imputer, artifacts.scaler, np.array(r_rows)))
    np.testing.assert_allclose(np.array(b_rows), r_pred, rtol=0, atol=1e-9)
    return compared, skipped_bucket


def test_feature_parity_synthetic(service, artifacts, base_events):
    research_grid = parse_xml_file(io.BytesIO(events_to_xml(base_events)))   # full input, research path
    compared, _ = _parity(service, artifacts, base_events, research_grid, research_grid.index)
    assert compared > 250


@needs_ohio
def test_feature_parity_real_ohio_file(service, artifacts):
    ev = ohio_events(OHIO_XML)
    research_grid = parse_xml_file(str(OHIO_XML))      # research code on the original XML file
    rng = np.random.RandomState(2)
    times = rng.choice(research_grid.index[300:], 400, replace=False)
    compared, skipped = _parity(service, artifacts, ev, research_grid, times)
    assert compared > 250


def test_documented_event_bucket_difference(service, artifacts, base_events):
    """A meal 1 minute after t is counted in research row t, but never by the backend."""
    times = _grid_times(base_events)
    t = times[300]
    ev = pd.concat([base_events, pd.DataFrame([{"timestamp": t + pd.Timedelta(minutes=1), "carbs_g": 50.0}])],
                   ignore_index=True).reindex(columns=COLS).sort_values("timestamp", kind="mergesort")
    research = create_features(parse_xml_file(io.BytesIO(events_to_xml(ev)))).set_index("ts")
    backend = service.predict(ev, t)
    without = service.predict(base_events, t)
    assert research.loc[t, "carbs"] == without.features["carbs"] + 50.0
    assert backend.features["carbs"] == without.features["carbs"]
    assert backend.predicted_glucose_mg_dl == without.predicted_glucose_mg_dl


def test_context_window_does_not_change_features(service, artifacts, base_events):
    """The backend keeps 24 h before t; features equal the research pipeline on the full history."""
    t = _grid_times(base_events)[-50]
    assert (t - base_events["timestamp"].min()) > CONTEXT          # the window really crops here
    res = service.predict(base_events, t)
    # research reference: full input (all history and later data), row t
    ev_times = base_events.loc[base_events["bolus_units"].notna() | base_events["carbs_g"].notna(), "timestamp"]
    assert not ((ev_times > t) & (ev_times < t + FIVE)).any()
    full = create_features(parse_xml_file(io.BytesIO(events_to_xml(base_events)))).set_index("ts")
    assert t in full.index
    np.testing.assert_allclose(res.features.values, full.loc[t, list(artifacts.feature_list)].values.astype(float),
                               rtol=0, atol=1e-9, equal_nan=True)


# ------------------------------------------------------------------ history / current glucose
def _aligned(n, end="2027-03-02 12:00:00", offset_s=0):
    end = pd.Timestamp(end)
    ts = end - pd.to_timedelta(np.arange(n)[::-1] * 5, unit="min") + pd.Timedelta(seconds=offset_s)
    return pd.DataFrame({"timestamp": ts, "glucose_mg_dl": 120.0 + np.arange(n)}).reindex(columns=COLS), end


def test_exact_minimum_history(service):
    ev, t = _aligned(14)
    assert np.isfinite(service.predict(ev, t).predicted_glucose_mg_dl)
    ev13, t = _aligned(13)
    with pytest.raises(BackendError) as e:
        service.predict(ev13, t)
    assert e.value.code == ErrorCode.INSUFFICIENT_HISTORY


def test_long_gap_inside_window_is_insufficient(service):
    ev, t = _aligned(40)
    ev = ev[~ev["timestamp"].between(t - pd.Timedelta(minutes=50), t - pd.Timedelta(minutes=10))]
    with pytest.raises(BackendError) as e:                       # 40-min gap > 30-min forward fill
        service.predict(ev, t)
    assert e.value.code == ErrorCode.INSUFFICIENT_HISTORY


def test_short_gap_bridged_by_forward_fill(service):
    ev, t = _aligned(40)
    ev = ev[~ev["timestamp"].between(t - pd.Timedelta(minutes=35), t - pd.Timedelta(minutes=15))]
    assert np.isfinite(service.predict(ev, t).predicted_glucose_mg_dl)   # 25-min gap, research ffill


@pytest.mark.parametrize("age,ok", [("0s", True), ("2min", True), ("4min 59s", True), ("5min", False),
                                    ("6min", False), ("20min", False)])
def test_missing_current_glucose(service, age, ok):
    """A real reading must lie in (t - 5 min, t]. With a reading exactly 5 min old the research
    grid (parse_xml_file, ends at ceil(last reading)) has no row t unless a LATER reading exists."""
    t = pd.Timestamp("2027-03-02 12:00:00")
    ev, _ = _aligned(30, end=t - pd.Timedelta(age))
    if ok:
        assert np.isfinite(service.predict(ev, t).predicted_glucose_mg_dl)
    else:
        with pytest.raises(BackendError) as e:
            service.predict(ev, t)
        assert e.value.code == ErrorCode.MISSING_CURRENT_GLUCOSE


def test_default_prediction_time_is_last_grid_row(service):
    ev, end = _aligned(20, offset_s=97)
    res = service.predict(ev)
    assert res.prediction_time == ev["timestamp"].max().ceil("5min")
    assert res.latest_observation_time == ev["timestamp"].max()
    assert res.forecast_time == res.prediction_time + pd.Timedelta("30min")
    assert res.horizon_minutes == 30


# ------------------------------------------------------------------ other errors and warnings
def test_infinite_feature_rejected(service):
    ev, t = _aligned(20)
    ev.loc[ev.index[-3], "glucose_mg_dl"] = np.inf
    with pytest.raises(BackendError) as e:
        service.predict(ev, t)
    assert e.value.code == ErrorCode.INFINITE_FEATURE


def test_missing_columns(service):
    ev, t = _aligned(20)
    with pytest.raises(BackendError) as e:
        service.predict(ev.drop(columns=["carbs_g"]), t)
    assert e.value.code == ErrorCode.MISSING_COLUMNS


def test_prediction_time_must_be_on_grid(service):
    ev, t = _aligned(20)
    with pytest.raises(BackendError) as e:
        service.predict(ev, t + pd.Timedelta(minutes=2))
    assert e.value.code == ErrorCode.INVALID_TIMESTAMP


def test_warnings(service):
    ev, t = _aligned(20)
    res = service.predict(ev, t)
    assert any("insulin on board" in w for w in res.warnings)
    assert any("No insulin or carbohydrate" in w for w in res.warnings)
