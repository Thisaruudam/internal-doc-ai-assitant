"""Model routing.

One model does not fit every node, and pretending otherwise is how an agent
system ends up either expensive or unreliable. The guard and validator run on
*every* turn and only classify; the response agent writes prose a person reads;
the research agent reasons over long horizons where a cheap model compounds its
mistakes across recursion levels.

Routing therefore lives in one table, keyed by the job rather than by the node
name, so a reviewer can see every cost/quality decision in one place and change
one without hunting through node code.

Temperatures are part of the routing decision, not an afterthought. Anything
that produces structured output for the system to act on runs at 0.0 — a
supervisor that routes differently on identical input is untestable. Only the
final prose is allowed any variation, and not much.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GeminiSettings
from app.observability.logging import get_logger

log = get_logger(__name__)


class ModelRole(StrEnum):
    """What a model is being asked to do."""

    #: Classification on every turn: injection scanning, validation verdicts.
    GUARD = "guard"
    #: Planning, routing, tool selection.
    AGENT = "agent"
    #: Long-horizon reasoning: RLM reduce steps, hard analysis.
    DEEP = "deep"
    #: The answer the user reads.
    RESPONSE = "response"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model: str
    temperature: float
    #: Rationale, kept next to the choice so it stays true when the choice changes.
    reason: str


def profiles(settings: GeminiSettings) -> dict[ModelRole, ModelProfile]:
    return {
        ModelRole.GUARD: ModelProfile(
            model=settings.model_fast,
            temperature=0.0,
            reason="pure classification on every turn; cheapest and lowest latency",
        ),
        ModelRole.AGENT: ModelProfile(
            model=settings.model_agent,
            temperature=0.0,
            reason="planning and tool selection; deterministic so routing is testable",
        ),
        ModelRole.DEEP: ModelProfile(
            model=settings.model_deep,
            temperature=0.1,
            reason="long-horizon reasoning where quality dominates token cost",
        ),
        ModelRole.RESPONSE: ModelProfile(
            model=settings.model_response,
            temperature=0.2,
            reason="prose a person reads; slight variation reads better than none",
        ),
    }


@lru_cache(maxsize=8)
def _build(
    model: str, temperature: float, api_key: str, timeout_s: float, max_retries: int
) -> ChatGoogleGenerativeAI:
    """Construct and cache a client.

    Cached per configuration: building a client per node per turn would rebuild
    the HTTP transport on every question for no benefit.
    """
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        timeout=timeout_s,
        # Retries here cover transient 5xx and rate limits. Application-level
        # fallback to a cheaper model lives in the nodes, which know whether a
        # degraded answer is acceptable for that step.
        max_retries=max_retries,
    )


def get_model(role: ModelRole, settings: GeminiSettings) -> ChatGoogleGenerativeAI:
    """The model for a given job."""
    profile = profiles(settings)[role]
    return _build(
        profile.model,
        profile.temperature,
        settings.api_key.get_secret_value(),
        settings.request_timeout_s,
        settings.max_retries,
    )


def get_fallback_model(settings: GeminiSettings) -> ChatGoogleGenerativeAI:
    """The model used when the primary one fails.

    Always the cheapest and most available option: the point of a fallback is
    that it works, not that it is good.
    """
    return get_model(ModelRole.GUARD, settings)


def describe_routing(settings: GeminiSettings) -> list[dict[str, str]]:
    """Routing table, for the UI and the demo.

    Exposed rather than buried so the cost/quality trade-off can be shown
    directly instead of described.
    """
    return [
        {
            "role": role.value,
            "model": profile.model,
            "temperature": str(profile.temperature),
            "reason": profile.reason,
        }
        for role, profile in profiles(settings).items()
    ]
