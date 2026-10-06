# syntax=docker/dockerfile:1
# Production image: Glucose Forecast API (research demonstration only; not a medical device).
# Build context = repository root; .dockerignore is an allow-list (only the files below can enter).
# Build:  docker build -t glucose-api .
# Run:    docker run -p 7860:7860 -e CORS_ORIGINS=https://your-frontend.example glucose-api
FROM python:3.13.9-slim-bookworm

# MPLBACKEND/MPLCONFIGDIR: evaluate_loso.py (research code, unchanged) imports matplotlib/seaborn.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    ENVIRONMENT=production \
    MODEL_DIR=/app/models \
    DEMO_DATA_DIR=/app/backend/demo_data \
    LOG_LEVEL=INFO

# libgomp1: OpenMP runtime required by the LightGBM wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Exact, complete pins (no resolver); binary wheels only.
COPY backend/requirements-lock.txt backend/requirements-lock.txt
RUN pip install --no-deps --only-binary=:all: -r backend/requirements-lock.txt && pip check

# Research modules imported by the backend (unchanged; hashes guarded by the test suite).
COPY data_processing.py objective3_common.py evaluate_loso.py ./
COPY parsers/__init__.py parsers/hupa_ucm.py parsers/

# Backend application, synthetic demo data, artifact manifest, reference predictions.
COPY backend/ backend/

# Exactly the six deployment artifacts (no legacy model_artifact.pkl, no unused models).
COPY models_objective3/LightGBM.joblib models_objective3/imputer.joblib models_objective3/scaler.joblib \
     models_objective3/feature_list.json models_objective3/settings.json models_objective3/model_hashes.json \
     /app/models/

# Fail the build if the artifacts differ from the validated manifest, if anything else is in
# /app/models, or if the model cannot be loaded with the installed library versions.
RUN python backend/scripts/verify_artifacts.py /app/models \
 && python -c "from backend.app.services.artifacts import load_artifacts; a = load_artifacts('/app/models'); print('model', a.model_name, a.model_version)"

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /tmp/matplotlib && chown app:app /tmp/matplotlib
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3).status == 200 else 1)"

# One worker: rate-limit counters are per process. Access log off: it would record raw paths and IPs.
# CORS_ORIGINS must be provided at run time (production refuses to start without it).
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--no-access-log"]
