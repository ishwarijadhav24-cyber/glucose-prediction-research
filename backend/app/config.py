"""Centralised configuration, read from environment variables.

MODEL_DIR            directory holding the research artifacts (default: <repo>/models_objective3)
MAX_UPLOAD_SIZE_MB   total upload limit in megabytes (default 25; OhioT1DM XML files are up to ~10 MB)
CORS_ORIGINS         comma-separated allowed origins; REQUIRED in production (development default http://localhost:3000)
ENVIRONMENT          development | production
LOG_LEVEL            DEBUG | INFO | WARNING | ERROR
DEMO_DATA_DIR        directory of demo CSV files (upload schema); default backend/demo_data
RATE_LIMIT_PER_MINUTE        prediction requests per client per minute (default 30; 0 disables)
TRUST_PROXY_HEADERS          use the last X-Forwarded-For hop as client address (default false)
PREDICTION_TIMEOUT_SECONDS   maximum time a prediction request may take (default 30)
MAX_PREDICTIONS_PER_REQUEST  cap for mode=all (default 2016 = 7 days of 5-minute times; latest kept)
MAX_CONCURRENT_PREDICTIONS   predictions running at once; more get 503 (default 4)
MAX_JSON_BODY_KB             body limit for non-upload requests (default 16)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# The deployed model is fixed by decision (LightGBM, causal Objective 1 pipeline).
# It is not configurable, so the API can never silently switch models.
DEPLOYED_MODEL = "LightGBM"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    model_dir: Path = Field(default=REPO_ROOT / "models_objective3", alias="MODEL_DIR")
    model_card_path: Path = Field(default=BACKEND_ROOT / "app" / "model_card.json", alias="MODEL_CARD_PATH")
    max_upload_size_mb: float = Field(default=25.0, gt=0, le=100, alias="MAX_UPLOAD_SIZE_MB")
    max_predictions_per_request: int = Field(default=2016, ge=1, le=20_000, alias="MAX_PREDICTIONS_PER_REQUEST")
    cors_origins: str | None = Field(default=None, alias="CORS_ORIGINS")
    demo_data_dir: Path = Field(default=BACKEND_ROOT / "demo_data", alias="DEMO_DATA_DIR")
    rate_limit_per_minute: int = Field(default=30, ge=0, le=10_000, alias="RATE_LIMIT_PER_MINUTE")
    trust_proxy_headers: bool = Field(default=False, alias="TRUST_PROXY_HEADERS")
    prediction_timeout_seconds: float = Field(default=30.0, gt=0, le=300, alias="PREDICTION_TIMEOUT_SECONDS")
    max_concurrent_predictions: int = Field(default=4, ge=1, le=64, alias="MAX_CONCURRENT_PREDICTIONS")
    max_json_body_kb: int = Field(default=16, ge=1, le=1024, alias="MAX_JSON_BODY_KB")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("environment")
    @classmethod
    def _env(cls, v: str) -> str:
        v = v.lower()
        if v not in ("development", "production"):
            raise ValueError("ENVIRONMENT must be 'development' or 'production'")
        return v

    @field_validator("log_level")
    @classmethod
    def _level(cls, v: str) -> str:
        v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins is None:
            if self.environment == "production":
                raise ValueError("CORS_ORIGINS must be set explicitly in production")
            return ["http://localhost:3000"]
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if self.environment == "production" and (not origins or "*" in origins):
            raise ValueError("CORS_ORIGINS must list explicit origins in production (no '*')")
        return origins

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
