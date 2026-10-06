"""Phase 7B: cross-environment reference, timezone independence, deployment artifact packaging."""

import json
import os
import shutil
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.tests.conftest import MODEL_DIR, REPO_ROOT, requires

sys.path.insert(0, str(REPO_ROOT / "backend" / "scripts"))
import reference          # noqa: E402
import verify_artifacts   # noqa: E402

MANIFEST = json.loads((REPO_ROOT / "backend" / "deploy" / "artifact_manifest.json").read_text())["files"]


# ------------------------------------------------------------------ 1. reference predictions
def test_matches_validated_reference():
    """Features, scaled values, predictions, ordering and error codes equal the validated environment."""
    rep = reference.compare(MODEL_DIR)
    assert rep["passed"], json.dumps(rep["checks"], indent=1)
    assert rep["n_predictions"] == 828


# ------------------------------------------------------------------ 7. timezone independence
TZ_SNIPPET = r"""
import hashlib, json, os, sys, time
if hasattr(time, "tzset"):
    time.tzset()
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, os.path.join(sys.argv[1], "backend", "scripts"))
import reference
a, m = reference.compute()
h = lambda x: hashlib.sha256(x.tobytes()).hexdigest()
print(json.dumps({"utc_offset_s": -time.timezone, "localtime_hour": time.localtime(0).tm_hour,
                  "times": h(a["prediction_time_ns"]), "obs": h(a["latest_observation_ns"]),
                  "features": h(a["features"]), "preprocessed": h(a["preprocessed"]),
                  "predictions": h(a["predictions"]), "errors": m["error_codes"]}))
"""


def _run_tz(tz):
    env = dict(os.environ, TZ=tz, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", TZ_SNIPPET, str(REPO_ROOT)], env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_results_independent_of_timezone():
    runs = {tz: _run_tz(tz) for tz in ("UTC0", "EST5EDT", "JST-9")}
    offsets = {r["localtime_hour"] for r in runs.values()}
    assert len(offsets) == 3, f"TZ setting had no effect: {offsets}"            # the setting really changed
    keys = ("times", "obs", "features", "preprocessed", "predictions", "errors")
    first = next(iter(runs.values()))
    for tz, r in runs.items():
        assert {k: r[k] for k in keys} == {k: first[k] for k in keys}, f"results differ under TZ={tz}"


# ------------------------------------------------------------------ 4. artifact packaging
@pytest.fixture
def deploy_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    for name in MANIFEST:
        shutil.copy2(MODEL_DIR / name, d / name)
    return d


def test_manifest_matches_research_artifacts():
    recorded = json.loads((MODEL_DIR / "model_hashes.json").read_text())
    for name, digest in MANIFEST.items():
        assert verify_artifacts.sha256(MODEL_DIR / name) == digest
        if name != "model_hashes.json":
            assert recorded[name] == digest


def test_deploy_dir_with_six_files_is_sufficient(deploy_dir):
    assert verify_artifacts.verify(deploy_dir) == []
    assert sorted(p.name for p in deploy_dir.iterdir()) == sorted(MANIFEST)
    with TestClient(create_app(Settings(MODEL_DIR=deploy_dir))) as c:
        assert c.get("/api/v1/health").json()["model_version"] == MANIFEST["LightGBM.joblib"][:12]


@requires(MODEL_DIR / "RandomForest.joblib")       # only the research environment has the unused models
def test_full_research_dir_is_not_a_deploy_dir():
    assert any("unexpected files" in p for p in verify_artifacts.verify(MODEL_DIR))


@pytest.mark.parametrize("victim", sorted(MANIFEST))
def test_tampered_deploy_file_detected(deploy_dir, victim):
    p = deploy_dir / victim
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0x01
    p.write_bytes(bytes(data))
    assert any("hash mismatch" in x for x in verify_artifacts.verify(deploy_dir))


def test_legacy_or_unused_artifacts_rejected(deploy_dir):
    (deploy_dir / "model_artifact.pkl").write_bytes(b"legacy")
    assert any("unexpected files" in x for x in verify_artifacts.verify(deploy_dir))


def test_consistent_tampering_of_hash_file_detected(deploy_dir):
    """Replacing a model AND updating model_hashes.json is still caught by the fixed manifest."""
    (deploy_dir / "scaler.joblib").write_bytes(b"evil")
    hashes = json.loads((deploy_dir / "model_hashes.json").read_text())
    hashes["scaler.joblib"] = verify_artifacts.sha256(deploy_dir / "scaler.joblib")
    (deploy_dir / "model_hashes.json").write_text(json.dumps(hashes))
    problems = verify_artifacts.verify(deploy_dir)
    assert any("scaler.joblib" in p for p in problems) and any("model_hashes.json" in p for p in problems)


# ------------------------------------------------------------------ 5/6. container settings in the Dockerfile
def test_dockerfile_settings():
    text = (REPO_ROOT / "Dockerfile").read_text()
    cmd = json.loads(next(l for l in text.splitlines() if l.startswith("CMD ["))[4:])
    assert cmd[:2] == ["uvicorn", "backend.app.main:app"]
    flags = dict(zip(cmd[2::2], cmd[3::2])) if "--no-access-log" not in cmd else None
    assert "--no-access-log" in cmd
    rest = [c for c in cmd[2:] if c != "--no-access-log"]
    flags = dict(zip(rest[::2], rest[1::2]))
    assert flags == {"--host": "0.0.0.0", "--port": "7860", "--workers": "1"}
    assert "--proxy-headers" not in cmd and "--forwarded-allow-ips" not in text
    for needle in ("MPLCONFIGDIR=/tmp/matplotlib", "MPLBACKEND=Agg", "ENVIRONMENT=production", "libgomp1",
                   "verify_artifacts.py /app/models", "USER app", "--no-deps"):
        assert needle in text, needle
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "TRUST_PROXY_HEADERS" not in code and "CORS_ORIGINS=" not in code      # both set only at deploy time
    assert "model_artifact.pkl" not in code
    copied = " ".join(l for l in text.splitlines() if l.strip().startswith(("COPY", "models_objective3/")))
    for forbidden in ("results", "outputs", "data/", "RandomForest", "Stacking", "MLP.joblib", "Ridge.joblib"):
        assert forbidden not in copied, forbidden


def test_dockerignore_excludes_private_and_research_files():
    lines = [l.strip() for l in (REPO_ROOT / ".dockerignore").read_text().splitlines() if l.strip()
             and not l.startswith("#")]
    assert lines[0] == "**", "allow-list: everything excluded by default"
    allowed = [l[1:] for l in lines if l.startswith("!")]
    def segments(path):
        return path.rstrip("*").rstrip("/").split("/")
    for forbidden in ("data", "results", "results_causal", "outputs", "model_artifact.pkl", ".git", "backend/tests",
                      "models_objective3/RandomForest.joblib", "models_objective3/Stacking.joblib",
                      "models_objective3/MLP.joblib", "models_objective3/Ridge.joblib", ".env"):
        f = forbidden.split("/")
        assert not any(segments(a)[:len(f)] == f for a in allowed), forbidden
    assert not any(a.startswith(("results", "outputs")) for a in allowed)
