"""Phase 6: dataset detection and adapters (OhioT1DM XML, HUPA-UCM, normalized CSV),
sampling handling, multiple predictions, privacy and research-data isolation."""

import io
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import parsers.hupa_ucm as hupa
from backend.app.config import Settings
from backend.app.errors import BackendError, ErrorCode
from backend.app.main import create_app
from backend.app.schemas import UploadPredictionResponse
from backend.app.services.datasets import (HUPA, NORMALIZED, OHIO, UploadedFile, _libre_readings, detect_file_type,
                                           ingest_upload)
from backend.app.services.features import build_grid, events_to_xml
from backend.app.services.inference import InferenceService
from backend.tests.conftest import HUPA_DATA, HUPA_PREDICTIONS, MODEL_DIR, OHIO_XML, REPO_ROOT, requires
from backend.tests.helpers import COLS, corrupt_after, synthetic_events
from backend.tests.test_api import assert_error, to_csv
from data_processing import create_features, parse_xml_file
from objective3_common import apply_preprocessing

FIVE = pd.Timedelta("5min")


# ================================================================== builders (exact on-disk formats)
def ohio_xml(events, pid="999"):
    return events_to_xml(events).replace(b'<patient id="upload">', f'<patient id="{pid}">'.encode())


def readings_15min(hours=20, start="2027-05-01 06:00:00", seed=0, offset_s=127):
    rng = np.random.RandomState(seed)
    t = pd.Timestamp(start) + pd.Timedelta(seconds=offset_s) + pd.to_timedelta(np.arange(0, hours * 60, 15), unit="min")
    g = np.clip(np.round(140 + 50 * np.sin(np.arange(len(t)) / 9) + rng.normal(0, 4, len(t))), 45, 395)
    return pd.Series(g, index=t)


def libre_old(readings, scans=()):
    lines = ["HUPA9999P;;;;;;;;;;;;;;;;;;",
             "ID;Hora;Tipo de registro;Histórico glucosa (mg/dL);Glucosa leída (mg/dL);Insulina de acción rápida "
             "sin valor numérico;Insulina de acción rápida (unidades);Alimentos sin valor numérico;"
             "Carbohidratos (raciones);Notas"]
    for i, (t, v) in enumerate(readings.items()):
        lines.append(f"{1000 + i};{t:%Y/%m/%d %H:%M};0;{v:g};;;;;;")
    for t, v in scans:                                                   # type 1 = scan: must be ignored
        lines.append(f"5000;{t:%Y/%m/%d %H:%M};1;;{v:g};;;;;")
    return ("\n".join(lines) + "\n").encode("utf-8-sig")


def libre_new(readings):
    lines = ["Informe del paciente HUPA9999P,Generado el,05-11-2020 00:45 UTC,Generado por,UCM", "HUPA9999P,",
             "Dispositivo,Número de serial,Sello de tiempo del dispositivo,Tipo de registro,Historial de glucosa mg/dL,"
             "Escaneo de glucosa mg/dL,Insulina de acción rápida no numérica,Notas"]
    lines += [f"FreeStyle LibreLink,,{t:%d-%m-%Y %H:%M},0,{v:g},,," for t, v in readings.items()]
    return ("\n".join(lines) + "\n").encode("utf-8")


def preprocessed(start, end, bolus=(), carbs_servings=(), glucose_value=100.0):
    idx = pd.date_range(start, end, freq="5min")
    df = pd.DataFrame({"time": idx.strftime("%Y-%m-%dT%H:%M:%S"), "glucose": glucose_value, "calories": 1.0,
                       "heart_rate": 70.0, "steps": 0.0, "basal_rate": 0.1, "bolus_volume_delivered": 0.0,
                       "carb_input": 0.0})
    for t, v in bolus:
        df.loc[idx == t, "bolus_volume_delivered"] = v
    for t, v in carbs_servings:
        df.loc[idx == t, "carb_input"] = v
    return df.to_csv(sep=";", index=False).encode()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo_ds")
    (d / "synthetic_01.csv").write_bytes(to_csv(synthetic_events(hours=30, seed=3)))
    with TestClient(create_app(Settings(MODEL_DIR=MODEL_DIR, DEMO_DATA_DIR=d, RATE_LIMIT_PER_MINUTE=10_000,
                                        MAX_UPLOAD_SIZE_MB=25)), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def service(artifacts):
    return InferenceService(artifacts)


def post(client, files, mode="latest"):
    return client.post(f"/api/v1/predict/upload?mode={mode}",
                       files=[("file", (n, d, ct)) for n, d, ct in files])


# ================================================================== detection
@pytest.mark.parametrize("name,data,kind", [
    ("a.xml", b'<patient id="1"><glucose_level/></patient>', OHIO),
    ("a.csv", b"timestamp,glucose_mg_dl,bolus_units,carbs_g\n", NORMALIZED),
    ("a.csv", b"carbs_g,timestamp,bolus_units,glucose_mg_dl\n", NORMALIZED),
    ("a.csv", b"time;glucose;bolus_volume_delivered;carb_input\n", "hupa_preprocessed_csv"),
])
def test_detect_file_type(name, data, kind):
    assert detect_file_type(UploadedFile(data=data, filename=name)) == kind


def test_detect_libre_layouts():
    r = readings_15min()
    assert detect_file_type(UploadedFile(libre_old(r), "x.csv")) == "libre_csv"
    assert detect_file_type(UploadedFile(libre_new(r), "x.csv")) == "libre_csv"


@pytest.mark.parametrize("name,data,code", [
    ("a.csv", b"foo;bar\n1;2\n", ErrorCode.UNSUPPORTED_FORMAT),
    ("a.csv", b'<patient id="1"/>', ErrorCode.INVALID_FILE),          # XML must be .xml
    ("a.json", b'{"a": 1}', ErrorCode.INVALID_FILE),
    ("a.csv", b"\x1f\x8b\x08\x00\x00binary", ErrorCode.INVALID_FILE),
    ("a.csv", b"   \n\n", ErrorCode.INVALID_FORMAT),
])
def test_detect_rejects(name, data, code):
    with pytest.raises(BackendError) as e:
        detect_file_type(UploadedFile(data=data, filename=name))
    assert e.value.code == code


# ================================================================== OhioT1DM adapter
@pytest.fixture(scope="module")
def ohio_events_syn():
    return synthetic_events(hours=30, seed=21)


def test_ohio_synthetic_upload_equals_normalized(client, ohio_events_syn):
    a = post(client, [("p.xml", ohio_xml(ohio_events_syn), "application/xml")], mode="all")
    b = post(client, [("p.csv", to_csv(ohio_events_syn), "text/csv")], mode="all")
    assert a.status_code == b.status_code == 200, a.text
    assert a.json()["dataset"]["detected_type"] == "ohiot1dm_xml"
    assert a.json()["dataset"]["sampling_profile"] == "5min"
    assert a.json()["predictions"] == b.json()["predictions"] and a.json()["n_predictions"] > 200


def test_ohio_grid_equals_research_parser(ohio_events_syn):
    """Canonical events -> grid is exactly parse_xml_file on the same XML (single file)."""
    xml = ohio_xml(ohio_events_syn)
    ds = ingest_upload([UploadedFile(xml, "p.xml")])
    pd.testing.assert_frame_equal(build_grid(ds.events), parse_xml_file(io.BytesIO(xml)))


@requires(OHIO_XML)
def test_ohio_real_file_grid_and_predictions_match_research(service, artifacts):
    data = OHIO_XML.read_bytes()
    ds = ingest_upload([UploadedFile(data, "559-ws-training.xml")])
    research_grid = parse_xml_file(str(OHIO_XML))
    pd.testing.assert_frame_equal(build_grid(ds.events), research_grid)       # identical canonical grid
    results, _, _ = service.predict_many(ds.events, 400)
    research = create_features(research_grid).set_index("ts")
    fl = list(artifacts.feature_list)
    ev_t = ds.events.loc[ds.events["bolus_units"].notna() | ds.events["carbs_g"].notna(), "timestamp"].values
    compared = 0
    for r in results:
        t = r.prediction_time
        if t not in research.index or ((ev_t > np.datetime64(t)) & (ev_t < np.datetime64(t + FIVE))).any():
            continue                                  # research row t also holds events from (t, t+5): documented
        np.testing.assert_allclose(r.features.values, research.loc[t, fl].values.astype(float), atol=1e-9,
                                   equal_nan=True)
        ref = artifacts.model.predict(apply_preprocessing(artifacts.imputer, artifacts.scaler,
                                                          research.loc[[t], fl].values.astype(float)))[0]
        assert abs(r.predicted_glucose_mg_dl - ref) < 1e-9
        compared += 1
    assert compared > 300


def test_ohio_two_files_same_patient(client, ohio_events_syn):
    cut = ohio_events_syn["timestamp"].iloc[len(ohio_events_syn) // 2]
    a, b = ohio_events_syn[ohio_events_syn["timestamp"] <= cut], ohio_events_syn[ohio_events_syn["timestamp"] > cut]
    r = post(client, [("tr.xml", ohio_xml(a), "text/xml"), ("te.xml", ohio_xml(b), "text/xml")])
    assert r.status_code == 200 and any("combined" in p for p in r.json()["preprocessing"])
    assert_error(post(client, [("tr.xml", ohio_xml(a, "1"), "text/xml"), ("te.xml", ohio_xml(b, "2"), "text/xml")]),
                 422, "UNSUPPORTED_FORMAT")
    assert_error(post(client, [(f"{i}.xml", ohio_xml(a), "text/xml") for i in range(3)]), 422, "UNSUPPORTED_FORMAT")


@pytest.mark.parametrize("xml,code", [
    (b'<?xml version="1.0"?><!DOCTYPE p [<!ENTITY x SYSTEM "file:///etc/passwd">]><patient id="1">'
     b'<glucose_level><event ts="01-01-2027 00:00:00" value="&x;"/></glucose_level></patient>', "INVALID_FORMAT"),
    (b'<patient id="1"><glucose_level><event ts="01-01-2027 00:00:00" value="100"/>', "INVALID_FORMAT"),
    (b'<other><glucose_level/></other>', "UNSUPPORTED_FORMAT"),
    (b'<patient id="1"><glucose_level><event ts="2027-01-01 00:00" value="100"/></glucose_level></patient>',
     "INVALID_TIMESTAMP"),
    (b'<patient id="1"><glucose_level><event ts="01-01-2027 00:00:00" value="abc"/></glucose_level></patient>',
     "INVALID_NUMERIC_VALUE"),
    (b'<patient id="1"><glucose_level></glucose_level></patient>', "MISSING_COLUMNS"),
])
def test_ohio_rejections(client, xml, code):
    assert_error(post(client, [("p.xml", xml, "application/xml")]), 422, code)


# ================================================================== HUPA-UCM adapter
@requires(HUPA_DATA)
def test_libre_reader_equals_research_reader_on_every_real_file():
    files = sorted((HUPA_DATA / "Raw_Data").glob("*/free_style_sensor/*.csv"))
    checked = 0
    for path in files:
        try:
            ref, _ = hupa._read_libre_file(str(path))
        except hupa.LibreFormatError:
            continue
        got = _libre_readings(path.read_bytes())
        pd.testing.assert_frame_equal(got.reset_index(drop=True), ref.reset_index(drop=True))
        checked += 1
    assert checked >= 25


@requires(HUPA_DATA, HUPA_PREDICTIONS)
def test_hupa_real_patient_reproduces_objective3(service):
    """Libre export + Preprocessed CSV upload -> the canonical data and predictions of Objective 3."""
    pid = "HUPA0002P"
    files = [UploadedFile(p.read_bytes(), p.name) for p in sorted((HUPA_DATA / "Raw_Data" / pid / "free_style_sensor")
                                                                 .glob("*.csv"))]
    files.append(UploadedFile((HUPA_DATA / "Preprocessed" / f"{pid}.csv").read_bytes(), f"{pid}.csv"))
    ds = ingest_upload(files)
    assert ds.dataset_type == HUPA and ds.sampling_profile == "15min"
    raw, _ = hupa.load_libre_glucose(pid)
    g = ds.events[ds.events["glucose_mg_dl"].notna()].set_index("timestamp")["glucose_mg_dl"]
    np.testing.assert_array_equal(g.values, raw.clip(40, 400).values)
    np.testing.assert_array_equal(g.index.values, raw.index.values)
    df, _, _ = hupa.parse_hupa_patient(pid)
    pd.testing.assert_frame_equal(build_grid(ds.events), df, check_freq=False)   # identical research grid
    results, _, _ = service.predict_many(ds.events, 5000)
    rec = pd.read_csv(HUPA_PREDICTIONS, parse_dates=["ts"])
    rec = rec[rec["patient"] == pid].set_index("ts")["LightGBM"]
    common = [r for r in results if r.prediction_time in rec.index]
    assert len(common) > 500
    np.testing.assert_allclose([r.predicted_glucose_mg_dl for r in common],
                               rec.loc[[r.prediction_time for r in common]].values, atol=1e-9)


def _hupa_upload(readings, with_prep=True, layout="old", **prep_kw):
    files = [("libre.csv", libre_old(readings) if layout == "old" else libre_new(readings), "text/csv")]
    if with_prep:
        start = readings.index.min().floor("5min") - pd.Timedelta("10min")
        end = readings.index.max().ceil("5min")
        files.append(("HUPA9999P.csv", preprocessed(start, end, **prep_kw), "text/csv"))
    return files


@pytest.mark.parametrize("layout", ["old", "new"])
def test_hupa_synthetic_upload(client, layout):
    r = readings_15min()
    t_meal = (r.index[30] + pd.Timedelta("2min")).floor("5min")
    files = _hupa_upload(r, layout=layout, bolus=[(t_meal, 3.5), (t_meal + FIVE * 12, -1.0)],
                         carbs_servings=[(t_meal, 4.0)])
    res = post(client, files, mode="all")
    assert res.status_code == 200, res.text
    b = res.json()
    UploadPredictionResponse.model_validate(b)
    d = b["dataset"]
    assert d["detected_type"] == "hupa_ucm" and d["sampling_profile"] == "15min"
    assert d["n_glucose_readings"] == len(r) and d["n_bolus_events"] == 1 and d["n_carb_events"] == 1
    pre = " ".join(b["preprocessing"])
    assert "carb_input x 10" in pre and "<= 0 treated as no event" in pre and "nothing interpolated" in pre
    assert any("NOT equivalent to native 5-minute CGM" in w for w in b["warnings"])
    # predictions only right after real readings (every ~15 min), never on forward-filled rows
    times = pd.to_datetime([p["prediction_time"] for p in b["predictions"]])
    assert (times.to_series().diff().dropna() >= pd.Timedelta("10min")).all()


def test_hupa_conversion_is_forward_fill_not_interpolation():
    """Exactly how 15-min readings enter the 5-min grid: Step A (<=5 min) then forward fill <=30 min."""
    r = readings_15min(hours=4, start="2027-05-01 08:00:00", offset_s=120)      # readings at hh:02, :17, :32, :47
    gap = (r.index > pd.Timestamp("2027-05-01 10:20")) & (r.index < pd.Timestamp("2027-05-01 11:30"))
    r = r[~gap]                                                                  # no reading 10:32 ... 11:17
    ds = ingest_upload([UploadedFile(libre_old(r), "l.csv")])
    grid = build_grid(ds.events)["glucose"]
    v = r.to_dict()
    at = lambda s: pd.Timestamp(f"2027-05-01 {s}")
    assert grid[at("10:05")] == v[at("10:02")]                                   # Step A: reading 10:02
    assert grid[at("10:10")] == grid[at("10:15")] == v[at("10:02")]              # repeated, NOT interpolated
    assert grid[at("10:20")] == v[at("10:17")]
    assert grid[at("10:50")] == v[at("10:17")]                                   # forward fill: 30 min after 10:20
    assert np.isnan(grid[at("10:55")])                                           # longer gaps stay missing
    assert grid[at("11:35")] == v[at("11:32")]


def test_hupa_preprocessed_glucose_is_never_used(client):
    r = readings_15min()
    a = post(client, _hupa_upload(r, glucose_value=100.0), mode="all").json()
    b = post(client, _hupa_upload(r, glucose_value=399.0), mode="all").json()
    assert a["predictions"] == b["predictions"]


def test_research_formats_keep_recorded_values_with_warning(client):
    r = readings_15min()
    t = r.index[20].floor("5min")
    b = post(client, _hupa_upload(r, carbs_servings=[(t, 130.0)])).json()      # 1300 g, present in HUPA-UCM
    assert b["status"] == "success"
    assert any("outside the usual range" in w for w in b["warnings"])
    ev = synthetic_events(hours=6, seed=1)
    ev.loc[ev["carbs_g"].notna(), "carbs_g"] = 600.0
    assert_error(post(client, [("a.csv", to_csv(ev), "text/csv")]), 422, "INVALID_NUMERIC_VALUE")   # strict CSV


def test_hupa_window_clip_and_duplicates(client):
    r = readings_15min()
    r.iloc[5] = 455.0                                            # above sensor range -> clipped (research rule)
    files = _hupa_upload(r)
    extra = libre_old(pd.concat([r.iloc[:3], pd.Series([150.0], index=[r.index[0] - pd.Timedelta("1D")])]))
    files.append(("libre2.csv", extra, "text/csv"))              # duplicates + a reading outside the window
    b = post(client, files).json()
    pre = " ".join(b["preprocessing"])
    assert "clipped to [40, 400]" in pre and "duplicate reading" in pre and "outside the Preprocessed study window" in pre


def test_hupa_scans_ignored(client):
    r = readings_15min()
    scans = [(t + pd.Timedelta("7min"), 300.0) for t in r.index[::4]]
    b = post(client, [("l.csv", libre_old(r, scans), "text/csv")]).json()
    assert b["dataset"]["n_glucose_readings"] == len(r)


def test_hupa_without_preprocessed_warns(client):
    b = post(client, _hupa_upload(readings_15min(), with_prep=False)).json()
    assert any("No HUPA-UCM Preprocessed file" in w for w in b["warnings"])


def test_hupa_preprocessed_only_rejected(client):
    r = readings_15min()
    prep = preprocessed(r.index.min().floor("5min"), r.index.max().ceil("5min"))
    res = post(client, [("HUPA9999P.csv", prep, "text/csv")])
    assert_error(res, 422, "UNSUPPORTED_FORMAT")
    assert "never used" in res.json()["error"]["message"]


# ================================================================== sampling / signals / history
@pytest.mark.parametrize("step,profile", [(5, "5min"), (15, "15min")])
def test_sampling_detection(client, step, profile):
    t = pd.date_range("2027-03-01 06:00", periods=60, freq=f"{step}min")
    ev = pd.DataFrame({"timestamp": t, "glucose_mg_dl": 120.0 + np.arange(60) % 7}).reindex(columns=COLS)
    b = post(client, [("a.csv", to_csv(ev), "text/csv")]).json()
    assert b["dataset"]["sampling_profile"] == profile and b["dataset"]["median_interval_minutes"] == step


@pytest.mark.parametrize("step", [1, 2, 10, 30, 60])
def test_unsupported_sampling(client, step):
    t = pd.date_range("2027-03-01 06:00", periods=200, freq=f"{step}min")
    ev = pd.DataFrame({"timestamp": t, "glucose_mg_dl": 120.0}).reindex(columns=COLS)
    assert_error(post(client, [("a.csv", to_csv(ev), "text/csv")]), 422, "UNSUPPORTED_SAMPLING")


def test_missing_glucose_signal(client):
    ev = pd.DataFrame({"timestamp": pd.date_range("2027-03-01", periods=5, freq="1h"), "bolus_units": 1.0})
    assert_error(post(client, [("a.csv", to_csv(ev.reindex(columns=COLS)), "text/csv")]), 422, "MISSING_COLUMNS")


def test_insufficient_history_all_formats(client):
    t = pd.date_range("2027-03-01 06:00", periods=10, freq="5min")
    ev = pd.DataFrame({"timestamp": t, "glucose_mg_dl": 120.0}).reindex(columns=COLS)
    assert_error(post(client, [("a.csv", to_csv(ev), "text/csv")]), 422, "INSUFFICIENT_HISTORY")
    assert_error(post(client, [("a.xml", ohio_xml(ev), "text/xml")], mode="all"), 422, "INSUFFICIENT_HISTORY")
    # 15-min data: 4 readings span 45 min (< 65 min of grid history needed)
    assert_error(post(client, [("l.csv", libre_old(readings_15min(hours=1)), "text/csv")]), 422,
                 "INSUFFICIENT_HISTORY")


def test_mixed_dataset_types_rejected(client, ohio_events_syn):
    assert_error(post(client, [("a.xml", ohio_xml(ohio_events_syn), "text/xml"),
                               ("b.csv", to_csv(ohio_events_syn), "text/csv")]), 422, "UNSUPPORTED_FORMAT")


# ================================================================== multiple predictions
def test_batch_equals_single_predictions(service):
    ev = synthetic_events(hours=30, seed=4, gaps=((600, 50),))
    results, skipped, truncated = service.predict_many(ev, 10_000)
    assert not truncated and skipped["recomputed_causal_bucket"] > 0
    for r in results[::7] + [x for x in results if x.prediction_time.minute % 30 == 0][:20]:
        single = service.predict(ev, r.prediction_time)
        np.testing.assert_allclose(r.features.values, single.features.values, atol=1e-9, equal_nan=True)
        assert abs(r.predicted_glucose_mg_dl - single.predicted_glucose_mg_dl) < 1e-9
        assert r.latest_observation_time == single.latest_observation_time
    # every eligible time is returned (single path succeeds exactly at these times)
    grid_times = build_grid(ev).index
    ok = {t for t in grid_times if not isinstance(_safe(service, ev, t), BackendError)}
    assert {r.prediction_time for r in results} == ok


def _safe(service, ev, t):
    try:
        return service.predict(ev, t)
    except BackendError as e:
        return e


def test_batch_causality(service):
    ev = synthetic_events(hours=30, seed=8)
    results, _, _ = service.predict_many(ev, 10_000)
    t = results[len(results) // 2].prediction_time
    changed, _, _ = service.predict_many(corrupt_after(ev, t), 10_000)
    before = {r.prediction_time: r.predicted_glucose_mg_dl for r in results if r.prediction_time <= t}
    after = {r.prediction_time: r.predicted_glucose_mg_dl for r in changed if r.prediction_time <= t}
    assert before.keys() == after.keys()
    assert all(abs(before[k] - after[k]) < 1e-9 for k in before)


def test_mode_all_list_and_truncation(make_client=None):
    d = tempfile.mkdtemp()
    ev = synthetic_events(hours=30, seed=9)
    with TestClient(create_app(Settings(MODEL_DIR=MODEL_DIR, DEMO_DATA_DIR=d, RATE_LIMIT_PER_MINUTE=10_000,
                                        MAX_PREDICTIONS_PER_REQUEST=10))) as c:
        b = post(c, [("a.csv", to_csv(ev), "text/csv")], mode="all").json()
    assert b["mode"] == "all" and b["n_predictions"] == 10 and b["truncated"] is True
    times = [p["prediction_time"] for p in b["predictions"]]
    assert times == sorted(times) and b["prediction"] == b["predictions"][-1]
    assert any("Only the latest 10" in w for w in b["warnings"])


def test_invalid_mode(client, ohio_events_syn):
    r = client.post("/api/v1/predict/upload?mode=everything",
                    files={"file": ("a.csv", to_csv(ohio_events_syn), "text/csv")})
    assert_error(r, 422, "INVALID_FORMAT")


# ================================================================== privacy / isolation
def test_upload_not_persisted_and_not_exposed(client, ohio_events_syn):
    before = set(client.get("/api/v1/demo/patients").json()["patients"][0].items())
    tmp = tempfile.gettempdir()
    start = time.time()
    marker = ohio_events_syn.copy()
    big = to_csv(marker)
    assert post(client, [("secret_patient.csv", big, "text/csv")], mode="all").status_code == 200
    new = [os.path.join(tmp, f) for f in os.listdir(tmp)
           if os.path.isfile(os.path.join(tmp, f)) and os.path.getmtime(os.path.join(tmp, f)) >= start]
    for f in new:
        try:
            with open(f, "rb") as fh:
                assert big[:200] not in fh.read(), "upload content written to a temporary file"
        except PermissionError:
            pass
    lst = client.get("/api/v1/demo/patients").json()["patients"]
    assert len(lst) == 1 and set(lst[0].items()) == before          # uploads never become demo data


def test_demo_data_is_only_the_synthetic_file():
    demo = REPO_ROOT / "backend" / "demo_data"
    assert sorted(p.name for p in demo.iterdir()) == ["synthetic_01.csv"]
    text = (demo / "synthetic_01.csv").read_text(encoding="utf-8")
    assert text.startswith("timestamp,glucose_mg_dl,bolus_units,carbs_g")
    for marker in ("HUPA", "FreeStyle", "glucose_level", "<patient", "Dispositivo", "Hora;"):
        assert marker not in text


def test_demo_file_regenerates_identically(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mk", REPO_ROOT / "backend" / "scripts" / "make_synthetic_demo.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    out = tmp_path / "synthetic_01.csv"
    monkeypatch.setattr(mk, "OUT", out)
    ev = mk.synthetic_events(start="2027-03-01 06:00:00", hours=72, seed=2027, gaps=((1500, 50),)).reindex(columns=COLS)
    ev["timestamp"] = ev["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.write_text(ev.to_csv(index=False, na_rep="", lineterminator="\n"), encoding="utf-8")
    assert out.read_bytes() == (REPO_ROOT / "backend" / "demo_data" / "synthetic_01.csv").read_bytes()


def test_no_research_data_inside_backend():
    """Nothing from data/ (OhioT1DM XML, HUPA files) is copied into the backend tree."""
    for p in (REPO_ROOT / "backend").rglob("*"):
        if p.is_file() and p.suffix.lower() in (".xml", ".pkl", ".joblib"):
            raise AssertionError(f"unexpected data/model file in backend/: {p}")
