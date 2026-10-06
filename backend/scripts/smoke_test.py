"""HTTP smoke test against a RUNNING server (standard library only; works against the container).

python backend/scripts/smoke_test.py http://127.0.0.1:7860 [--origin https://allowed.example]

Uses only the public synthetic demo file. Exit code 0 = every check passed.
The rate-limit check is last because it uses up the per-minute budget of this client.
"""

import http.client
import json
import sys
import urllib.parse
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = (ROOT / "backend" / "demo_data" / "synthetic_01.csv").read_bytes()
results = []


def call(base, method, path, body=None, ctype=None, headers=None):
    req = urllib.request.Request(base + path, data=body, method=method, headers=dict(headers or {}))
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def multipart(files):
    b = uuid.uuid4().hex
    parts = []
    for name, data, ctype in files:
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                     f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    return b"".join(parts) + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def err_code(body):
    try:
        return json.loads(body)["error"]["code"]
    except Exception:   # noqa: BLE001
        return None


def main(base, origin):
    s, _, b = call(base, "GET", "/health")
    check("health", s == 200 and json.loads(b) == {"status": "ok"})
    s, _, b = call(base, "GET", "/api/v1/health")
    check("api health / model loaded", s == 200 and json.loads(b)["model_loaded"], json.loads(b).get("model_version"))
    s, _, b = call(base, "GET", "/api/v1/model")
    m = json.loads(b)
    check("model endpoint", s == 200 and m["model_name"] == "LightGBM" and m["n_features"] == 47
          and m["research_evaluation"]["MARD_percent"]["mean"] == 10.03)
    s, _, b = call(base, "GET", "/api/v1/demo/patients")
    check("demo list", s == 200 and [p["patient_id"] for p in json.loads(b)["patients"]] == ["synthetic_01"])
    s, _, b = call(base, "POST", "/api/v1/predict/demo/synthetic_01")
    check("demo prediction", s == 200 and json.loads(b)["prediction"]["horizon_minutes"] == 30)
    body, ct = multipart([("synthetic_01.csv", DEMO, "text/csv")])
    s, _, b = call(base, "POST", "/api/v1/predict/upload?mode=all", body, ct)
    r = json.loads(b)
    check("CSV upload (mode=all)", s == 200 and r["n_predictions"] == 828 and r["dataset"]["sampling_profile"] == "5min",
          f"{r.get('n_predictions')} predictions")
    lines = DEMO.decode().splitlines()[1:]
    xml = ['<patient id="smoke"><glucose_level>']
    for row in lines:
        ts, g, bo, c = row.split(",")
        if g:
            t = ts.split(" ")
            d = t[0].split("-")
            xml.append(f'<event ts="{d[2]}-{d[1]}-{d[0]} {t[1]}" value="{g}"/>')
    xml.append("</glucose_level></patient>")
    body, ct = multipart([("p.xml", "".join(xml).encode(), "application/xml")])
    s, _, b = call(base, "POST", "/api/v1/predict/upload", body, ct)
    check("OhioT1DM XML upload", s == 200 and json.loads(b)["dataset"]["detected_type"] == "ohiot1dm_xml")
    cases = [
        ("malformed CSV", [("a.csv", b"timestamp,glucose_mg_dl,bolus_units,carbs_g\n2027-03-01 06:00,abc,,\n", "text/csv")],
         422, "INVALID_NUMERIC_VALUE"),
        ("insufficient history", [("a.csv", "\n".join(DEMO.decode().splitlines()[:14]).encode() + b"\n", "text/csv")],
         422, "INSUFFICIENT_HISTORY"),
        ("invalid file type", [("a.exe", b"MZ\x90\x00", "application/octet-stream")], 400, "INVALID_FILE"),
        ("unsupported sampling", [("a.csv", ("\n".join([DEMO.decode().splitlines()[0]] +
                                                      [l for l in lines if l.split(',')[1]][::2]) + "\n").encode(),
                                   "text/csv")], 422, "UNSUPPORTED_SAMPLING"),
    ]
    for name, files, status, code in cases:
        body, ct = multipart(files)
        s, _, b = call(base, "POST", "/api/v1/predict/upload", body, ct)
        check(name, s == status and err_code(b) == code, f"{s} {err_code(b)}")
    late = json.dumps({"prediction_time": "2027-03-10T12:00:00"}).encode()
    s, _, b = call(base, "POST", "/api/v1/predict/demo/synthetic_01", late, "application/json")
    check("missing current glucose", s == 422 and err_code(b) == "MISSING_CURRENT_GLUCOSE")
    # Announce a 26 MB body (default limit 25 MB) and read the answer before sending it.
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=30)
    conn.putrequest("POST", "/api/v1/predict/upload")
    conn.putheader("Content-Type", "multipart/form-data; boundary=x")
    conn.putheader("Content-Length", str(26 * 1024 * 1024))
    conn.endheaders()
    resp = conn.getresponse()
    s, b = resp.status, resp.read()
    conn.close()
    check("oversized upload rejected before the body is read", s == 413 and err_code(b) == "FILE_TOO_LARGE",
          f"{s} {err_code(b)}")
    s, h, _ = call(base, "GET", "/api/v1/model", headers={"Origin": "https://not-allowed.example"})
    check("CORS: unknown origin not allowed", "access-control-allow-origin" not in {k.lower() for k in h})
    if origin:
        s, h, _ = call(base, "GET", "/api/v1/model", headers={"Origin": origin})
        check("CORS: configured origin allowed", {k.lower(): v for k, v in h.items()}.get("access-control-allow-origin")
              == origin)
    statuses = [call(base, "POST", "/api/v1/predict/demo/synthetic_01")[0] for _ in range(40)]
    check("rate limiting returns 429", 429 in statuses, f"first 429 after {statuses.index(429) if 429 in statuses else '-'}")
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 0 if not failed else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: smoke_test.py BASE_URL [--origin ORIGIN]")
    origin = args[args.index("--origin") + 1] if "--origin" in args else None
    sys.exit(main(args[0].rstrip("/"), origin))
