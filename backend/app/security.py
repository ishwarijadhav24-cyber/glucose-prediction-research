"""Request hardening: rate limiting, body-size limits, content-encoding checks, bounded execution.

All in-process (no external store). State is per process: with several workers each
worker has its own rate-limit counters.
"""

import asyncio
import threading
import time
from collections import OrderedDict, deque

from fastapi import Request
from fastapi.concurrency import run_in_threadpool

from backend.app.errors import BackendError, ErrorCode

UPLOAD_PATH = "/api/v1/predict/upload"
MULTIPART_OVERHEAD = 64 * 1024     # boundaries + part headers on top of the file itself


class BodyTooLarge(Exception):
    """Raised from the wrapped ASGI receive() when the request body exceeds its limit."""


class RateLimiter:
    """Sliding 60-second window per client key, bounded number of tracked keys."""

    def __init__(self, per_minute: int, window_s: float = 60.0, max_keys: int = 10_000):
        self.per_minute = per_minute
        self.window = window_s
        self.max_keys = max_keys
        self._hits: OrderedDict[str, deque] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> float | None:
        """None if allowed, otherwise seconds until the next request is allowed."""
        if self.per_minute <= 0:
            return None
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits.get(key)
            if q is None:
                if len(self._hits) >= self.max_keys:
                    self._hits.popitem(last=False)     # evict least recently used key
                q = self._hits[key] = deque()
            self._hits.move_to_end(key)
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(0.0, self.window - (now - q[0]))
            q.append(now)
            return None


def client_key(request: Request) -> str:
    settings = request.app.state.settings
    if settings.trust_proxy_headers:
        xff = request.headers.get("x-forwarded-for", "")
        hops = [h.strip() for h in xff.split(",") if h.strip()]
        if hops:
            return hops[-1]          # entry appended by the nearest (trusted) proxy
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request) -> None:
    """FastAPI dependency for prediction endpoints."""
    retry = request.app.state.rate_limiter.check(client_key(request))
    if retry is not None:
        request.state.retry_after = int(retry) + 1
        raise BackendError(ErrorCode.RATE_LIMITED, "Too many prediction requests; please retry later.")


async def run_bounded(request: Request, fn, *args):
    """Run blocking work in the thread pool with a concurrency cap and a timeout.

    The timeout bounds the time a client waits; Python cannot kill the worker thread,
    but the concurrency cap keeps stuck work from accumulating without limit.
    """
    slots: asyncio.Semaphore = request.app.state.prediction_slots
    if slots.locked():
        raise BackendError(ErrorCode.SERVER_BUSY, "The server is busy; please retry shortly.")
    async with slots:
        try:
            return await asyncio.wait_for(run_in_threadpool(fn, *args),
                                          timeout=request.app.state.settings.prediction_timeout_seconds)
        except asyncio.TimeoutError:
            raise BackendError(ErrorCode.PREDICTION_TIMEOUT, "The request took too long.") from None


class BodyLimitMiddleware:
    """Pure ASGI middleware: rejects compressed request bodies and caps body size while streaming."""

    def __init__(self, app, upload_limit: int, default_limit: int, error_response):
        self.app = app
        self.upload_limit = upload_limit + MULTIPART_OVERHEAD
        self.default_limit = default_limit
        self.error_response = error_response     # callable(status, code, message) -> ASGI response

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.upload_limit if scope.get("path") == UPLOAD_PATH else self.default_limit
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        encoding = headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):
            resp = self.error_response(415, ErrorCode.UNSUPPORTED_ENCODING.value,
                                       "Compressed request bodies are not accepted.")
            return await resp(scope, receive, send)
        length = headers.get("content-length")
        if length is not None:
            try:
                too_big = int(length) > limit
            except ValueError:
                resp = self.error_response(400, ErrorCode.INVALID_FORMAT.value, "Invalid Content-Length.")
                return await resp(scope, receive, send)
            if too_big:
                resp = self.error_response(413, ErrorCode.FILE_TOO_LARGE.value, "The request body is too large.")
                return await resp(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge()
            return message

        await self.app(scope, limited_receive, send)
