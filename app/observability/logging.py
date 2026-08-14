"""Structured logging.

Every log line is JSON and carries the correlation id of the request that
produced it, bound once by middleware into a context variable rather than
threaded through every call site. That makes a single conversation's whole
execution — API, graph nodes, retrieval, tool calls — greppable by one key.

Two rules the processors enforce:

* **No secrets.** ``_redact`` strips values whose key looks credential-shaped,
  so an accidental ``log.info("config", **settings)`` cannot leak an API key.
* **No raw user identifiers in shipped logs.** ``bind_request_context`` records
  a stable hash of the user id. Traces in LangSmith carry the readable name for
  debugging; the log stream carries the pseudonym.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import Any

import structlog

from app.config import ObservabilitySettings

#: Keys whose values are replaced before a record is emitted.
_SECRET_KEY_MARKERS = ("password", "secret", "token", "api_key", "authorization", "credential")

_REDACTED = "[redacted]"


def _redact(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Replace credential-shaped values anywhere in the event."""
    for key in list(event_dict):
        lowered = key.lower()
        if any(marker in lowered for marker in _SECRET_KEY_MARKERS):
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(settings: ObservabilitySettings) -> None:
    """Install the processor chain. Idempotent — safe to call from tests."""
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _redact,
    ]

    renderer: Any
    if settings.log_format == "json":
        shared.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Route stdlib loggers (uvicorn, httpx, langchain) through the same sink so
    # the stream stays uniformly parseable.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=logging.getLevelNamesMapping()[settings.log_level],
        force=True,
    )
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def pseudonymise(user_id: str) -> str:
    """Stable, non-reversible identifier for the log stream."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def bind_request_context(*, correlation_id: str, user_id: str | None = None, **extra: Any) -> None:
    """Bind per-request fields for the remainder of the task's execution."""
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id, **extra)
    if user_id is not None:
        structlog.contextvars.bind_contextvars(user=pseudonymise(user_id))


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
