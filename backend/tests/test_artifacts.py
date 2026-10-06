"""Phase 1: artifact integrity, loading, feature order and prediction reproducibility."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from backend.app.services.artifacts import (ArtifactError, check_versions, installed_versions,
                                            load_artifacts, sha256)
from backend.tests.conftest import (HUPA_DATA, HUPA_PREDICTIONS, LOSO_SUMMARY, MODEL_DIR, requires,
                                   research_available)


# ---------------------------------------------------------------- integrity
def test_all_recorded_hashes_match():
    hashes = json.loads((MODEL_DIR / "model_hashes.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        assert sha256(MODEL_DIR / name) == expected, name


def test_loads_lightgbm(artifacts):
    assert artifacts.model_name == "LightGBM"
    assert type(artifacts.model).__name__ == "LGBMRegressor"
    assert artifacts.n_features == 47
    assert artifacts.model_version == artifacts.hashes["LightGBM.joblib"][:12]


@pytest.mark.parametrize("victim", ["LightGBM.joblib", "imputer.joblib", "scaler.joblib",
                                    "feature_list.json", "settings.json"])
def test_tampered_file_refused(model_dir_copy, victim):
    path = model_dir_copy / victim
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ArtifactError, match="Hash mismatch"):
        load_artifacts(model_dir_copy)


def test_missing_file_refused(model_dir_copy):
    (model_dir_copy / "scaler.joblib").unlink()
    with pytest.raises(ArtifactError, match="Missing artifact"):
        load_artifacts(model_dir_copy)


def test_missing_hash_file_refused(model_dir_copy):
    (model_dir_copy / "model_hashes.json").unlink()
    with pytest.raises(ArtifactError, match="model_hashes.json"):
        load_artifacts(model_dir_copy)


def test_unhashed_file_refused(model_dir_copy):
    hashes = json.loads((model_dir_copy / "model_hashes.json").read_text())
    del hashes["imputer.joblib"]
    (model_dir_copy / "model_hashes.json").write_text(json.dumps(hashes))
    with pytest.raises(ArtifactError, match="No recorded hash"):
        load_artifacts(model_dir_copy)


def test_no_fallback_to_other_model(model_dir_copy):
    (model_dir_copy / "LightGBM.joblib").unlink()
    with pytest.raises(ArtifactError):
        load_artifacts(model_dir_copy)   # must not silently use Ridge/Stacking


# ---------------------------------------------------------------- versions
def test_installed_versions_match_settings(artifacts):
    check_versions(artifacts.settings["versions"], installed_versions())


@pytest.mark.parametrize("lib", ["lightgbm", "scikit-learn", "numpy", "pandas"])
def test_library_version_mismatch_refused(lib):
    expected = json.loads((MODEL_DIR / "settings.json").read_text())["versions"]
    installed = dict(installed_versions(), **{lib: "0.0.0"})
    with pytest.raises(ArtifactError, match=lib):
        check_versions(expected, installed)
    with pytest.raises(ArtifactError):
        load_artifacts(MODEL_DIR, installed=installed)


def test_python_minor_mismatch_refused():
    expected = json.loads((MODEL_DIR / "settings.json").read_text())["versions"]
    with pytest.raises(ArtifactError, match="python"):
        check_versions(expected, dict(installed_versions(), python="3.12.1"))


# ---------------------------------------------------------------- feature order
def _synthetic_xml(n=40):
    times = pd.date_range("2027-01-01 08:02", periods=n, freq="5min")
    g = "".join(f'<event ts="{t:%d-%m-%Y %H:%M:%S}" value="{100 + (i % 9)}"/>' for i, t in enumerate(times))
    return (f'<patient id="0"><glucose_level>{g}</glucose_level>'
            '<bolus><event ts_begin="01-01-2027 08:40:00" dose="2.5"/></bolus>'
            '<meal><event ts="01-01-2027 08:41:00" carbs="30"/></meal></patient>').encode()


def test_feature_list_equals_research_feature_order(artifacts):
    from data_processing import create_features, parse_xml_file
    feats = create_features(parse_xml_file(io.BytesIO(_synthetic_xml())))
    research_cols = [c for c in feats.columns if c not in ("target", "ts")]
    assert tuple(research_cols) == artifacts.feature_list


def test_feature_list_matches_training_settings(artifacts):
    assert artifacts.settings["n_features"] == len(artifacts.feature_list)
    assert "target" not in artifacts.feature_list and "ts" not in artifacts.feature_list


# ---------------------------------------------------------------- reproducibility
def test_deterministic_predictions(artifacts):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, artifacts.n_features))
    again = load_artifacts(MODEL_DIR)
    assert np.array_equal(artifacts.model.predict(X), again.model.predict(X))


@requires(HUPA_DATA, HUPA_PREDICTIONS)
def test_reproduces_recorded_research_predictions(artifacts):
    """Backend-loaded artifacts reproduce the LightGBM predictions stored by evaluate_hupa_external.py."""
    from data_processing import create_features
    from evaluate_hupa_external import evaluable_mask
    from objective3_common import apply_preprocessing
    from parsers.hupa_ucm import parse_hupa_patient

    pid = "HUPA0002P"
    df, is_real, _ = parse_hupa_patient(pid)
    feats = create_features(df)
    ev = feats[evaluable_mask(df, is_real, feats)]
    X = apply_preprocessing(artifacts.imputer, artifacts.scaler, ev[list(artifacts.feature_list)].values)
    pred = artifacts.model.predict(X)

    rec = pd.read_csv(HUPA_PREDICTIONS, parse_dates=["ts"])
    rec = rec[rec["patient"] == pid].set_index("ts")["LightGBM"]
    assert len(rec) == len(pred) > 500
    np.testing.assert_allclose(pred, rec.reindex(pd.DatetimeIndex(ev["ts"])).values, rtol=0, atol=1e-9)


# ---------------------------------------------------------------- model card
def test_model_card_has_only_verified_causal_metrics():
    card = json.loads((MODEL_DIR.parent / "backend" / "app" / "model_card.json").read_text(encoding="utf-8"))
    text = json.dumps(card).lower()
    assert "xgboost" not in text and "10.478" not in text and "10.48" not in text
    assert "results_pre_causal" not in text and '"results/' not in text and "model_artifact" not in text
    r = card["research_evaluation"]
    assert r["n_patients"] == 12
    if research_available(LOSO_SUMMARY):
        s = pd.read_csv(LOSO_SUMMARY).set_index("model").loc["LightGBM"]
        assert r["MARD_percent"] == {"mean": round(s["MARD_mean"], 2), "sd": round(s["MARD_std"], 2)}
        assert r["RMSE_mg_dl"]["mean"] == round(s["RMSE_mean"], 2)
        assert r["clarke_A_plus_B_percent"]["mean"] == round(s["EGA_A+B_mean"], 2)
    assert card["model_version"] == json.loads((MODEL_DIR / "model_hashes.json").read_text())["LightGBM.joblib"][:12]
