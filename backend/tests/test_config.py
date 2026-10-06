import pytest
from pydantic import ValidationError

from backend.app.config import DEPLOYED_MODEL, Settings


def test_defaults():
    s = Settings()
    assert s.environment == "development"
    assert s.max_upload_bytes == 25 * 1024 * 1024        # OhioT1DM training+testing XML ~12.5 MB
    assert s.cors_origin_list == ["http://localhost:3000"]
    assert DEPLOYED_MODEL == "LightGBM"


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "2")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    s = Settings()
    assert s.model_dir == tmp_path
    assert s.max_upload_bytes == 2 * 1024 * 1024
    assert s.cors_origin_list == ["https://a.example", "https://b.example"]


def test_invalid_environment(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    with pytest.raises(ValidationError):
        Settings()


def test_production_rejects_wildcard_cors(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(ValueError):
        Settings().cors_origin_list


def test_upload_limit_bounds(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_production_requires_explicit_cors(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(ValueError, match="explicitly"):
        Settings().cors_origin_list
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example")
    assert Settings().cors_origin_list == ["https://app.example"]


def test_proxy_headers_off_by_default():
    assert Settings().trust_proxy_headers is False
