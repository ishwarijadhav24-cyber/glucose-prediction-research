"""FastAPI application. Run: uvicorn backend.app.main:app (from the repository root).

Research demonstration only. Not a medical device. Not for diagnosis, treatment decisions
or insulin dosing. Does not replace a CGM.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartParser

from backend.app.config import Settings, get_settings
from backend.app.errors import BackendError, ErrorCode
from backend.app.logging_setup import RequestLogMiddleware, configure_logging, log_event
from backend.app.routers import demo, health, model, predict
from backend.app.schemas import DISCLAIMER, ErrorBody, ErrorResponse
from backend.app.security import BodyLimitMiddleware, BodyTooLarge, RateLimiter
from backend.app.services.artifacts import load_artifacts
from backend.app.services.demo import DemoStore
from backend.app.services.inference import InferenceService


def _error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    log = configure_logging(settings.log_level)
    origins = settings.cors_origin_list   # raises in production if '*' or empty
    if "*" in origins:
        log_event(log, logging.WARNING, "cors_wildcard_development_only")
    # Keep uploads (<= limit) in memory instead of spooling them to a temporary file on disk.
    MultiPartParser.spool_max_size = max(MultiPartParser.spool_max_size, settings.max_upload_bytes + 1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once. Any failure stops start-up: there is no fallback model.
        artifacts = load_artifacts(settings.model_dir)
        app.state.artifacts = artifacts
        app.state.service = InferenceService(artifacts)
        app.state.model_card = json.loads(settings.model_card_path.read_text(encoding="utf-8"))
        if app.state.model_card.get("model_version") != artifacts.model_version:
            raise RuntimeError("model_card.json does not describe the loaded model")
        app.state.demo = DemoStore(settings.demo_data_dir)
        log_event(log, logging.INFO, "startup", model=artifacts.model_name, model_version=artifacts.model_version,
                  n_demo_patients=len(app.state.demo.ids()))
        yield

    app = FastAPI(
        title="Glucose Forecast API",
        version="1.0.0",
        description=("30-minute-ahead glucose forecast from CGM, insulin and carbohydrate records, using the "
                     "LightGBM model of a leave-one-subject-out OhioT1DM study.\n\n**" + DISCLAIMER + "**"),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.prediction_slots = asyncio.Semaphore(settings.max_concurrent_predictions)

    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], allow_credentials=False)
    app.add_middleware(BodyLimitMiddleware, upload_limit=settings.max_upload_bytes,
                       default_limit=settings.max_json_body_kb * 1024, error_response=_error)
    app.add_middleware(RequestLogMiddleware, logger=log)     # outermost of the user middleware

    @app.exception_handler(BackendError)
    async def backend_error(request: Request, exc: BackendError):
        request.state.error_code = exc.code.value
        headers = None
        if exc.code == ErrorCode.RATE_LIMITED:
            headers = {"Retry-After": str(getattr(request.state, "retry_after", 60))}
        return _error(exc.http_status, exc.code.value, exc.message, headers)

    @app.exception_handler(BodyTooLarge)
    async def body_too_large(request: Request, exc: BodyTooLarge):
        request.state.error_code = ErrorCode.FILE_TOO_LARGE.value
        return _error(413, ErrorCode.FILE_TOO_LARGE.value, "The request body is too large.")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        request.state.error_code = ErrorCode.INVALID_FORMAT.value
        fields = sorted({".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()})
        return _error(422, ErrorCode.INVALID_FORMAT.value, f"Invalid request fields: {fields}")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = ErrorCode.NOT_FOUND.value if exc.status_code == 404 else ErrorCode.INVALID_FORMAT.value
        request.state.error_code = code
        return _error(exc.status_code, code, "Not found." if exc.status_code == 404 else "Invalid request.")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        # Only the exception TYPE is logged: messages/tracebacks can contain uploaded values.
        request.state.error_code = ErrorCode.INTERNAL_ERROR.value
        log_event(log, logging.ERROR, "unhandled_exception", exception_type=type(exc).__name__,
                  route=getattr(request.scope.get("route"), "path", "unmatched"))
        return _error(500, ErrorCode.INTERNAL_ERROR.value, "Internal error.")

    for r in (health.router, model.router, demo.router, predict.router):
        app.include_router(r)
    return app


app = create_app()
