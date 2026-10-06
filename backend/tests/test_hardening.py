"""Phase 4: rate limiting, bounded execution, upload hardening, edge cases, logging, CORS."""

import gzip
import io
import json
import logging
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.formparsers import MultiPartParser

from backend.app.config import Settings
from backend.app.logging_setup import LOGGER_NAME, SAFE_FIELDS, JsonFormatter
from backend.app.main import create_app
from backend.app.security import RateLimiter
from backend.tests.conftest import MODEL_DIR
from backend.tests.helpers import COLS, synthetic_events
from backend.tests.test_api import assert_error, to_csv, upload

HEADER = b"timestamp,glucose_mg_dl,bolus_units,carbs_g\n"


@pytest.fixture(scope="module")
def events():
    return synthetic_events(hours=30, seed=5, gaps=((600, 50),))


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory, events):
    d = tmp_path_factory.mktemp("demo_h")
    (d / "synthetic_01.csv").write_bytes(to_csv(events))
    return d


@pytest.fixture
def make_client(demo_dir):
    opened = []

    def _make(**env):
        env = {"MODEL_DIR": MODEL_DIR, "DEMO_DATA_DIR": demo_dir, "MAX_UPLOAD_SIZE_MB": 1,
               "RATE_LIMIT_PER_MINUTE": 10_000, **env}
        c = TestClient(create_app(Settings(**env)), raise_server_exceptions=False)
        c.__enter__()
        opened.append(c)
        return c
    yield _make
    for c in opened:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def good_csv(events, n=60):
    return to_csv(events[events["glucose_mg_dl"].notna()].iloc[:n])


def rows_csv(*rows):
    return HEADER + "".join(r + "\n" for r in rows).encode()


def series_csv(n=30, last=None, value=None, extra=()):
    """n glucose rows 5 min apart; optionally replace the last glucose value (string) and add rows."""
    t = pd.Timestamp("2027-03-02 10:00:00") - pd.to_timedelta([5 * k for k in range(n - 1, -1, -1)], unit="min")
    lines = [f"{ts:%Y-%m-%d %H:%M:%S},{120 + i % 5},," for i, ts in enumerate(t)]
    if value is not None:
        lines[-1] = f"{t[-1]:%Y-%m-%d %H:%M:%S},{value},,"
    return rows_csv(*lines, *extra)


# ================================================================== 1. rate limiting
def test_rate_limiter_unit():
    rl = RateLimiter(3, window_s=60)
    assert [rl.check("a", now=t) for t in (0, 1, 2)] == [None, None, None]
    assert rl.check("a", now=3) == pytest.approx(57)
    assert rl.check("b", now=3) is None                 # separate key
    assert rl.check("a", now=60.5) is None              # oldest hit left the window
    assert RateLimiter(0).check("a") is None            # 0 disables


def test_rate_limiter_bounded_memory():
    rl = RateLimiter(5, max_keys=100)
    for i in range(1000):
        rl.check(f"ip{i}", now=0)
    assert len(rl._hits) == 100


def test_rate_limit_api(make_client):
    c = make_client(RATE_LIMIT_PER_MINUTE=2)
    for _ in range(2):
        assert c.post("/api/v1/predict/demo/synthetic_01").status_code == 200
    r = c.post("/api/v1/predict/demo/synthetic_01")
    assert_error(r, 429, "RATE_LIMITED")
    assert int(r.headers["Retry-After"]) >= 1
    assert_error(c.post("/api/v1/predict/upload", files={"file": ("a.csv", b"x", "text/csv")}), 429, "RATE_LIMITED")
    for _ in range(5):                                   # read-only endpoints are not limited
        assert c.get("/api/v1/model").status_code == 200


def test_forwarded_for_ignored_unless_trusted(make_client):
    c = make_client(RATE_LIMIT_PER_MINUTE=1)
    assert c.post("/api/v1/predict/demo/synthetic_01", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    r = c.post("/api/v1/predict/demo/synthetic_01", headers={"X-Forwarded-For": "2.2.2.2"})
    assert_error(r, 429, "RATE_LIMITED")                 # spoofed header does not bypass the limit


def test_forwarded_for_used_when_trusted(make_client):
    c = make_client(RATE_LIMIT_PER_MINUTE=1, TRUST_PROXY_HEADERS=True)
    assert c.post("/api/v1/predict/demo/synthetic_01", headers={"X-Forwarded-For": "9.9.9.9, 1.1.1.1"}).status_code == 200
    assert c.post("/api/v1/predict/demo/synthetic_01", headers={"X-Forwarded-For": "9.9.9.9, 2.2.2.2"}).status_code == 200
    assert_error(c.post("/api/v1/predict/demo/synthetic_01", headers={"X-Forwarded-For": "8.8.8.8, 1.1.1.1"}),
                 429, "RATE_LIMITED")                    # keyed on the last (proxy-added) hop


# ================================================================== 2. bounded execution
def test_prediction_timeout(make_client, monkeypatch):
    c = make_client(PREDICTION_TIMEOUT_SECONDS=0.2)
    monkeypatch.setattr(c.app.state.service, "predict", lambda *a, **k: time.sleep(1.0))
    t0 = time.perf_counter()
    assert_error(c.post("/api/v1/predict/demo/synthetic_01"), 504, "PREDICTION_TIMEOUT")
    assert time.perf_counter() - t0 < 0.9


def test_server_busy(client, events, monkeypatch):
    class Full:
        def locked(self):
            return True
    monkeypatch.setattr(client.app.state, "prediction_slots", Full())
    assert_error(client.post("/api/v1/predict/demo/synthetic_01"), 503, "SERVER_BUSY")
    assert_error(upload(client, good_csv(events)), 503, "SERVER_BUSY")


# ================================================================== 3. upload hardening
def test_body_larger_than_limit_rejected_by_content_length(client):
    r = upload(client, b"a" * (2 * 1024 * 1024))
    assert_error(r, 413, "FILE_TOO_LARGE")


def test_streamed_body_without_length_is_capped(client):
    boundary = "xyz"
    def body():
        yield f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.csv\"\r\n" \
              f"Content-Type: text/csv\r\n\r\n".encode()
        for _ in range(40):                              # 40 x 64 KiB = 2.5 MiB, no Content-Length
            yield b"b" * 65536
        yield f"\r\n--{boundary}--\r\n".encode()
    r = client.post("/api/v1/predict/upload", content=body(),
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert "content-length" not in {k.lower() for k in r.request.headers}
    assert_error(r, 413, "FILE_TOO_LARGE")


def test_file_just_over_limit_rejected_while_reading(client):
    data = HEADER + b"2027-03-01 06:00:00,120,,\n" * 40330              # ~1.0 MiB + a little
    assert len(data) > 1024 * 1024
    assert_error(upload(client, data), 413, "FILE_TOO_LARGE")


def test_compressed_request_body_rejected(client, events):
    gz = gzip.compress(good_csv(events))
    r = client.post("/api/v1/predict/upload", files={"file": ("a.csv", gz, "text/csv")},
                    headers={"Content-Encoding": "gzip"})
    assert_error(r, 415, "UNSUPPORTED_ENCODING")


def test_compressed_file_contents_rejected(client, events):
    gz = gzip.compress(good_csv(events))
    assert_error(upload(client, gz, name="a.csv"), 400, "INVALID_FILE")       # not UTF-8 text
    assert_error(upload(client, gz, name="a.csv.gz"), 400, "INVALID_FILE")


def test_json_body_limit_on_demo_endpoint(client):
    r = client.post("/api/v1/predict/demo/synthetic_01", content=b'{"prediction_time": "' + b"9" * 40000 + b'"}',
                    headers={"Content-Type": "application/json"})
    assert_error(r, 413, "FILE_TOO_LARGE")


def test_uploads_are_kept_in_memory(client):
    assert MultiPartParser.spool_max_size > client.app.state.settings.max_upload_bytes


# ================================================================== 4. edge cases
@pytest.mark.parametrize("data,status,code", [
    (b"", 422, "INVALID_FORMAT"),                                       # empty file
    (HEADER, 422, "INVALID_FORMAT"),                                    # header only
    (b"\n\n\n", 422, "INVALID_FORMAT"),                                 # blank lines = empty file
    (rows_csv("2027-03-01 06:00:00,120,,"), 422, "INSUFFICIENT_HISTORY"),   # one row
    (HEADER + b"2027-03-01 06:00:00,120,,\x00\n", 400, "INVALID_FILE"),   # NUL byte: not a text file (Phase 6)
    (rows_csv("2027-03-01 06:00:00,120,,,,"), 422, "INVALID_FORMAT"),        # too many fields
    (rows_csv("2027-03-01 06:00:00,120"), 422, "INVALID_FORMAT"),            # too few fields
    (rows_csv('2027-03-01 06:00:00,"' + "1" * 200_000 + '",,'), 422, "INVALID_FORMAT"),   # huge field
    (rows_csv("2027-03-01 06:00:00," + "," * 50_000), 422, "INVALID_FORMAT"),            # very wide row
], ids=["empty", "header_only", "blank_lines", "one_row", "nul_byte", "too_many_fields", "too_few_fields",
        "huge_field", "very_wide_row"])
def test_malformed_csv(client, data, status, code):
    assert_error(upload(client, data), status, code)


def test_duplicate_timestamps_per_signal(client):
    assert_error(upload(client, series_csv(extra=("2027-03-02 09:00:00,,1.5,", "2027-03-02 09:00:00,,2.0,"))),
                 422, "DUPLICATE_TIMESTAMP")
    # same time for different signals is fine (one glucose reading and one bolus)
    assert upload(client, series_csv(extra=("2027-03-02 09:00:00,,1.5,",))).status_code == 200


@pytest.mark.parametrize("v", ["nan", "NaN", "inf", "-inf", "Infinity", "1e400", "0x10", "12,5"])
def test_nan_inf_and_non_numbers(client, v):
    data = series_csv(value=f'"{v}"' if "," in v else v)
    assert_error(upload(client, data), 422, "INVALID_NUMERIC_VALUE")


@pytest.mark.parametrize("value,ok", [("40", True), ("400", True), ("39.99", False), ("400.01", False)])
def test_glucose_boundaries(client, value, ok):
    r = upload(client, series_csv(value=value))
    assert r.status_code == 200 if ok else r.status_code == 422, r.text


@pytest.mark.parametrize("row,ok", [("2027-03-02 09:00:00,,0,", True), ("2027-03-02 09:00:00,,25,", True),
                                    ("2027-03-02 09:00:00,,25.01,", False), ("2027-03-02 09:00:00,,,0", True),
                                    ("2027-03-02 09:00:00,,,500", True), ("2027-03-02 09:00:00,,,500.5", False),
                                    ("2027-03-02 09:00:00,,-0.1,", False)])
def test_bolus_and_carb_boundaries(client, row, ok):
    r = upload(client, series_csv(extra=(row,)))
    if ok:
        assert r.status_code == 200, r.text
    else:
        assert_error(r, 422, "INVALID_NUMERIC_VALUE")


@pytest.mark.parametrize("ts,ok", [("2028-02-29 06:00", True), ("2027-02-29 06:00", False),
                                   ("2027-03-01 24:00", False), ("2027-03-01 06:00:60", False),
                                   ("2027-03-01 06:00:59", True), ("2027-03-01T06:00:00", False),
                                   ("2027-03-01 06:00:00Z", False), (" 2027-03-01 06:00 ", True)])
def test_timestamp_boundaries(client, ts, ok):
    r = upload(client, series_csv(extra=(f"{ts},,1.0,",)))
    if ok:
        assert r.status_code == 200, r.text
    else:
        assert_error(r, 422, "INVALID_TIMESTAMP")


def test_insufficient_history_and_gaps(client):
    assert_error(upload(client, series_csv(n=13)), 422, "INSUFFICIENT_HISTORY")
    assert upload(client, series_csv(n=14)).status_code == 200


def test_missing_current_glucose_via_demo(client, events):
    late = (events["timestamp"].max() + pd.Timedelta("1h")).ceil("5min")
    assert_error(client.post("/api/v1/predict/demo/synthetic_01", json={"prediction_time": late.isoformat()}),
                 422, "MISSING_CURRENT_GLUCOSE")


def test_large_valid_input(make_client):
    c = make_client(MAX_UPLOAD_SIZE_MB=5)
    ev = synthetic_events(hours=24 * 150, seed=7)                       # 150 days of 5-min data
    data = to_csv(ev)
    assert len(data) > 1024 * 1024
    t0 = time.perf_counter()
    r = upload(c, data)
    assert r.status_code == 200, r.text
    assert time.perf_counter() - t0 < 10


@pytest.mark.parametrize("kwargs", [
    {"content": b"garbage", "headers": {"Content-Type": "multipart/form-data"}},                  # no boundary
    {"content": b"--b\r\nnot a part\r\n--b--\r\n", "headers": {"Content-Type": "multipart/form-data; boundary=b"}},
    {"files": [("file", ("a.csv", b"x", "text/csv")) for _ in range(5)]},                     # > 4 files
    {"files": {"other": ("a.csv", b"x", "text/csv")}},                                          # wrong field
    {"data": {"file": "not a file"}},                                                           # text field
    {"json": {"file": "x"}},                                                                    # not multipart
])
def test_malformed_multipart(client, kwargs):
    r = client.post("/api/v1/predict/upload", **kwargs)
    assert_error(r, 422, "INVALID_FORMAT")


@pytest.mark.parametrize("pid", ["nope", "synthetic_02", "a" * 100, "%00", "synthetic_01.csv"])
def test_nonexistent_demo_patient(client, pid):
    assert_error(client.post(f"/api/v1/predict/demo/{pid}"), 404, "NOT_FOUND")


@pytest.mark.parametrize("value,code", [("2027-03-01T12:02:00", "INVALID_TIMESTAMP"),
                                        ("2027-03-01T12:00:00+01:00", "INVALID_TIMESTAMP"),
                                        ("yesterday", "INVALID_FORMAT"), (12345678901234567890, "INVALID_FORMAT"),
                                        ("2027-03-01T06:00:00", "MISSING_CURRENT_GLUCOSE"),   # before 1st reading
                                        ("2027-03-01T06:30:00", "INSUFFICIENT_HISTORY")])
def test_invalid_prediction_time(client, value, code):
    r = client.post("/api/v1/predict/demo/synthetic_01", json={"prediction_time": value})
    assert_error(r, 422, code)


def test_null_prediction_time_uses_latest(client):
    assert client.post("/api/v1/predict/demo/synthetic_01", json={"prediction_time": None}).status_code == 200


def test_unexpected_exception_hidden(client, events, monkeypatch):
    def boom(*a, **k):
        raise ValueError("glucose=123.4 secret-detail")
    monkeypatch.setattr(client.app.state.service, "predict", boom)
    r = upload(client, good_csv(events))
    assert_error(r, 500, "INTERNAL_ERROR")
    assert "secret" not in r.text and "123.4" not in r.text


# ================================================================== 5. logging
@pytest.fixture
def log_buffer():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(h)
    yield buf
    logger.removeHandler(h)


def test_logs_are_structured_and_private(client, events, log_buffer, monkeypatch):
    upload(client, good_csv(events), name="patient_john_doe_secret.csv")
    client.get("/api/v1/demo/patients/synthetic_01")
    upload(client, series_csv(value="999"), name="x.csv")                  # validation error
    monkeypatch.setattr(client.app.state.service, "predict",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("leak 321.5")))
    upload(client, good_csv(events))
    lines = [json.loads(line) for line in log_buffer.getvalue().splitlines()]
    assert lines, "no log lines captured"
    allowed = set(SAFE_FIELDS) | {"ts", "level"}
    for rec in lines:
        assert set(rec) <= allowed, rec
    text = log_buffer.getvalue()
    for secret in ("john_doe", "synthetic_01", "321.5", "leak", "999", "2027-03"):
        assert secret not in text, secret
    routes = {r.get("route") for r in lines}
    assert "/api/v1/demo/patients/{patient_id}" in routes and "/api/v1/predict/upload" in routes
    codes = {r.get("error_code") for r in lines}
    assert {"INVALID_NUMERIC_VALUE", "INTERNAL_ERROR"} <= codes
    assert any(r.get("event") == "unhandled_exception" and r.get("exception_type") == "RuntimeError" for r in lines)
    req = next(r for r in lines if r.get("event") == "request")
    assert {"method", "route", "status", "duration_ms", "rid"} <= set(req)


def test_log_level_configurable(make_client, log_buffer):
    c = make_client(LOG_LEVEL="WARNING")
    c.get("/api/v1/model")
    assert '"event":"request"' not in log_buffer.getvalue()
    assert logging.getLogger(LOGGER_NAME).level == logging.WARNING
    make_client(LOG_LEVEL="INFO")                         # restore for other tests
    assert logging.getLogger("uvicorn.access").disabled


# ================================================================== 6. CORS
@pytest.mark.parametrize("origins", ["*", "https://a.example,*", "", " , "])
def test_production_cors_must_be_explicit(origins):
    with pytest.raises(ValueError):
        create_app(Settings(MODEL_DIR=MODEL_DIR, ENVIRONMENT="production", CORS_ORIGINS=origins))


def test_production_cors_explicit_origin(make_client):
    c = make_client(ENVIRONMENT="production", CORS_ORIGINS="https://app.example")
    ok = c.get("/health", headers={"Origin": "https://app.example"})
    assert ok.headers.get("access-control-allow-origin") == "https://app.example"
    bad = c.get("/health", headers={"Origin": "https://other.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_development_wildcard_is_logged(make_client, log_buffer):
    make_client(CORS_ORIGINS="*")
    assert "cors_wildcard_development_only" in log_buffer.getvalue()
