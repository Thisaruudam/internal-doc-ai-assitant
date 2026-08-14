"""Correlation-id middleware.

Runs first in the chain so that every log line produced downstream — including
those from a failing auth check — carries the same id. The id is echoed back in
``X-Correlation-ID`` so a user reporting a problem can quote one string that
pins the request in the logs and in the LangSmith trace.

An inbound ``X-Correlation-ID`` is honoured (the Streamlit client sets one per
turn) but validated first: it lands in log output, so an unbounded caller-
supplied string is a log-injection vector.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.logging import bind_request_context, clear_request_context, get_logger

log = get_logger(__name__)

HEADER = "X-Correlation-ID"

#: Conservative: hex, dashes, underscores, bounded length.
_SAFE_ID = re.compile(r"\A[A-Za-z0-9_-]{8,64}\Z")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get(HEADER, "")
        correlation_id = supplied if _SAFE_ID.match(supplied) else uuid.uuid4().hex

        request.state.correlation_id = correlation_id
        clear_request_context()
        bind_request_context(
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
        )

        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        log.info("request_completed", status_code=response.status_code, duration_ms=elapsed_ms)
        response.headers[HEADER] = correlation_id
        return response
