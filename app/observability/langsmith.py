"""LangSmith tracing.

Tracing is mandatory in the brief, and the reason is worth stating: this system
makes several model calls per turn across nested sub-agents, and when an answer
is wrong the useful question is *which step* went wrong. Logs tell you what
happened; a trace tells you what the model saw when it decided.

LangChain reads its tracing configuration from the process environment rather
than from an injected client, so ``configure_tracing`` sets those variables from
typed settings. That keeps the credential in one place — ``app.config`` — instead
of having callers reach for ``os.environ`` directly.

What gets attached to every run:

* **The correlation id**, so a trace and a log line can be joined.
* **The role**, not the user id. Traces are a debugging surface that a wider
  group can see than the audit log, and the role is what explains a retrieval
  filter; the identity is not needed to understand the run.
"""

from __future__ import annotations

import os
from typing import Any

from app.config import ObservabilitySettings
from app.observability.logging import get_logger, pseudonymise

log = get_logger(__name__)


def configure_tracing(settings: ObservabilitySettings) -> bool:
    """Enable LangSmith if it is configured. Returns whether tracing is on.

    Missing credentials disable tracing rather than failing startup: an
    observability backend being unreachable must not stop the system answering
    questions.
    """
    if not settings.langsmith_tracing or settings.langsmith_api_key is None:
        os.environ["LANGSMITH_TRACING"] = "false"
        log.info("tracing_disabled", reason="not configured")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    log.info("tracing_enabled", project=settings.langsmith_project)
    return True


def run_metadata(
    *,
    correlation_id: str,
    role: str,
    user_id: str,
    thread_id: str,
    **extra: Any,
) -> dict[str, Any]:
    """Metadata attached to a graph run.

    The user is pseudonymised for the same reason it is in the log stream: the
    trace explains the decision, and the role is the part of identity that
    actually shaped it.
    """
    return {
        "correlation_id": correlation_id,
        "role": role,
        "user": pseudonymise(user_id),
        "thread_id": thread_id,
        **extra,
    }


def run_tags(*, role: str, route: str | None = None) -> list[str]:
    """Tags for filtering runs in the LangSmith UI.

    Role and route are the two axes worth slicing by when comparing behaviour —
    "show me every research-agent run for a viewer" is a question that gets
    asked while debugging.
    """
    tags = [f"role:{role}"]
    if route:
        tags.append(f"route:{route}")
    return tags
