"""Privacy-safe structured (JSON) logging.

Only whitelisted fields are emitted: method, route template (never the raw path, which can
contain user input), status, duration, request id and error code. No bodies, values,
file names, client addresses or exception messages.
"""

import json
import logging
import sys
import time
import uuid

LOGGER_NAME = "glucose_api"
SAFE_FIELDS = ("event", "rid", "method", "route", "status", "duration_ms", "error_code", "exception_type",
               "model", "model_version", "n_demo_patients")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
               "level": record.levelname}
        for f in SAFE_FIELDS:
            v = getattr(record, f, None)
            if v is not None:
                out[f] = v
        return json.dumps(out, separators=(",", ":"))


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(h, "_glucose_api", False) for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        h._glucose_api = True
        logger.addHandler(h)
    for h in logger.handlers:
        h.setLevel(level)
    # uvicorn's access log prints raw paths and client addresses: disable it.
    logging.getLogger("uvicorn.access").disabled = True
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields) -> None:
    logger.log(level, event, extra={"event": event, **{k: v for k, v in fields.items() if k in SAFE_FIELDS}})


class RequestLogMiddleware:
    """Pure ASGI request logger (no BaseHTTPMiddleware, so exceptions are not wrapped).

    Logs method, route template, status, duration and the error code set by the error
    handlers; adds an X-Request-ID header. Unhandled exceptions are logged as status 500
    and re-raised to the server-error handler.
    """

    def __init__(self, app, logger: logging.Logger):
        self.app = app
        self.logger = logger

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        status = {"code": None}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + [(b"x-request-id", rid.encode())]
            await send(message)

        def done(code, error_code):
            route = getattr(scope.get("route"), "path", "unmatched")
            log_event(self.logger, logging.INFO, "request", rid=rid, method=scope.get("method"), route=route,
                      status=code, duration_ms=round((time.perf_counter() - start) * 1000, 1), error_code=error_code)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            done(500, "INTERNAL_ERROR")
            raise
        done(status["code"], (scope.get("state") or {}).get("error_code"))
