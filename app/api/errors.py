"""Uniform error responses (RFC 7807 ``application/problem+json``).

One shape for every failure, from a bad password to an exhausted rate limit to
an unhandled exception. Two properties matter for this system in particular:

* **Every problem carries the correlation id.** A user reporting "it broke" hands
  over one string that pins the exact request in the logs and in LangSmith.
* **Unexpected exceptions never leak internals.** The traceback goes to the log;
  the caller gets a generic detail plus the correlation id. Stack traces in HTTP
  responses are an information-disclosure finding, and this is a bank persona.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.observability.logging import get_logger

log = get_logger(__name__)

PROBLEM_CONTENT_TYPE = "application/problem+json"

#: Base for the ``type`` URI. Points at the repository's error documentation
#: rather than a dereferenceable service endpoint.
_TYPE_BASE = "https://github.com/atrium/docs/errors"


class AtriumError(Exception):
    """Base for errors that map to a deliberate HTTP response.

    Anything not deriving from this is treated as a bug and reported as a 500
    with no detail.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal-error"
    title: str = "Internal server error"

    def __init__(self, detail: str, *, extra: dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra = extra or {}


class AuthenticationError(AtriumError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "authentication-failed"
    title = "Authentication failed"


class AuthorizationError(AtriumError):
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "authorization-denied"
    title = "Not permitted"


class RateLimitError(AtriumError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "rate-limited"
    title = "Rate limit exceeded"

    def __init__(self, detail: str, *, retry_after_seconds: int) -> None:
        super().__init__(detail, extra={"retry_after_seconds": retry_after_seconds})
        self.retry_after_seconds = retry_after_seconds


class ValidationError(AtriumError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    error_code = "invalid-request"
    title = "Request failed validation"


class GuardrailError(AtriumError):
    """The request was understood and refused on policy grounds.

    Distinct from an authorization failure: the caller *may* ask, but the
    content violated an ingress guard or a brand policy.
    """

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "guardrail-blocked"
    title = "Request blocked by policy"


class DependencyError(AtriumError):
    """An upstream dependency failed and no degraded path was available."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "dependency-unavailable"
    title = "A required service is unavailable"


def _problem(
    *,
    status_code: int,
    error_code: str,
    title: str,
    detail: str,
    request: Request,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"{_TYPE_BASE}/{error_code}",
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": str(request.url.path),
        "correlation_id": getattr(request.state, "correlation_id", None),
    }
    if extra:
        body.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type=PROBLEM_CONTENT_TYPE,
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the handlers. Called once during app construction."""

    @app.exception_handler(AtriumError)
    async def _handle_atrium_error(request: Request, exc: AtriumError) -> JSONResponse:
        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitError):
            headers["Retry-After"] = str(exc.retry_after_seconds)
        if isinstance(exc, AuthenticationError):
            headers["WWW-Authenticate"] = "Bearer"

        log.warning(
            "request_rejected",
            error_code=exc.error_code,
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
        )
        return _problem(
            status_code=exc.status_code,
            error_code=exc.error_code,
            title=exc.title,
            detail=exc.detail,
            request=request,
            extra=exc.extra,
            headers=headers or None,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _problem(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            error_code="invalid-request",
            title="Request failed validation",
            detail="One or more fields were missing or malformed.",
            request=request,
            # Field-level errors are safe to return: they describe the caller's
            # own payload, not server internals.
            extra={"errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _problem(
            status_code=exc.status_code,
            error_code="http-error",
            title=str(exc.detail),
            detail=str(exc.detail),
            request=request,
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # exc_info goes to the log; the caller sees only the correlation id.
        log.exception("unhandled_exception", path=request.url.path, error=str(exc))
        return _problem(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="internal-error",
            title="Internal server error",
            detail=(
                "The request could not be completed. Quote the correlation id when reporting this."
            ),
            request=request,
        )
