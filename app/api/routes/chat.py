"""Chat streaming.

One SSE connection carries both the answer tokens and the agent's internal
activity. That is the whole point of the design: the panel is not polling a
debug endpoint and reconstructing a story afterwards — it is watching the same
stream the answer arrives on, in the order things actually happened.

The endpoint never raises mid-stream. Once the response has started, an HTTP
status code is no longer available to report failure, so errors become an
``error`` event and the stream closes cleanly. A client that has already rendered
half an answer must not be left waiting on a socket that will never close.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import PrincipalDep, SettingsDep
from app.api.errors import RateLimitError
from app.auth.principal import Principal
from app.graph.events import EventType
from app.graph.state import Budget, initial_state
from app.observability.langsmith import run_metadata, run_tags
from app.observability.logging import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])
log = get_logger(__name__)

#: Reserved against the caller's token budget before a turn starts. Refunded
#: down to actual usage afterwards.
_ESTIMATED_TURN_TOKENS = 12_000


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    #: Continues an existing conversation when supplied. The checkpointer keys
    #: session memory on this.
    thread_id: str | None = None


def _frame(event_type: str, payload: dict[str, Any]) -> dict[str, str]:
    """Encode one SSE frame.

    sse-starlette passes non-string ``data`` through ``str()``, which emits
    Python dict repr — single-quoted and not parseable by any standard SSE
    client. Every frame is therefore JSON-encoded here rather than relying on
    the default.
    """
    return {"event": event_type, "data": json.dumps(payload, default=str)}


@router.post("/stream")
async def stream_chat(
    payload: ChatRequest,
    request: Request,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> EventSourceResponse:
    """Answer a question, streaming activity and tokens as they happen."""
    limiter = request.app.state.limiter
    graph = request.app.state.graph

    decision = limiter.check_request(principal)
    if not decision.allowed:
        raise RateLimitError(decision.reason, retry_after_seconds=decision.retry_after_seconds)

    token_decision = limiter.check_tokens(principal, _ESTIMATED_TURN_TOKENS)
    if not token_decision.allowed:
        raise RateLimitError(
            token_decision.reason, retry_after_seconds=token_decision.retry_after_seconds
        )

    thread_id = payload.thread_id or uuid.uuid4().hex
    correlation_id = getattr(request.state, "correlation_id", uuid.uuid4().hex)

    return EventSourceResponse(
        _run_turn(
            graph=graph,
            limiter=limiter,
            question=payload.question,
            principal=principal,
            settings=settings,
            thread_id=thread_id,
            correlation_id=correlation_id,
        ),
        media_type="text/event-stream",
    )


async def _run_turn(
    *,
    graph: Any,
    limiter: Any,
    question: str,
    principal: Principal,
    settings: Any,
    thread_id: str,
    correlation_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """Drive the graph and translate its stream into SSE frames."""
    graph_settings = settings.graph
    state = initial_state(
        principal=principal,
        question=question,
        thread_id=thread_id,
        correlation_id=correlation_id,
        budget=Budget(
            depth=graph_settings.max_recursion_depth,
            tool_calls=graph_settings.max_tool_calls,
            tokens=graph_settings.max_tokens,
            supervisor_steps=graph_settings.max_supervisor_steps,
        ),
    )

    config = {
        "configurable": {"thread_id": thread_id},
        # Bounded so a routing defect cannot spin indefinitely on a user's turn.
        "recursion_limit": 25,
        "metadata": run_metadata(
            correlation_id=correlation_id,
            role=principal.role.value,
            user_id=principal.user_id,
            thread_id=thread_id,
        ),
        "tags": run_tags(role=principal.role.value),
        "run_name": "atrium-turn",
    }

    yield _frame("start", {"thread_id": thread_id, "correlation_id": correlation_id})

    final_state: dict[str, Any] = {}
    answer_chars = 0

    try:
        async for mode, chunk in graph.astream(
            state, config=config, stream_mode=["custom", "values"]
        ):
            if mode == "custom":
                # Activity events, already shaped by app.graph.events.
                yield _frame(str(chunk.get("type", EventType.NODE_ENTER.value)), chunk)
            else:
                final_state = chunk

        messages = final_state.get("messages", [])
        answer = ""
        if messages:
            last = messages[-1]
            answer = getattr(last, "text", None) or str(getattr(last, "content", ""))
            if callable(answer):
                answer = answer()
        answer_chars = len(answer)

        yield _frame(
            "answer",
            {
                "text": answer,
                "citations": [c.model_dump() for c in final_state.get("citations", [])],
                "degraded": bool(final_state.get("degraded_retrieval")),
                "errors": [e.model_dump() for e in final_state.get("errors", [])],
                "thread_id": thread_id,
                "correlation_id": correlation_id,
            },
        )

    except Exception as exc:  # the stream is already open; report, do not raise
        log.exception("chat_stream_failed", correlation_id=correlation_id)
        yield _frame(
            "error",
            {
                "detail": (
                    "The assistant could not complete this turn. Quote the "
                    "correlation id when reporting it."
                ),
                "error_type": type(exc).__name__,
                "correlation_id": correlation_id,
            },
        )
    finally:
        # Refund the unused part of the reservation. Estimating high and
        # refunding keeps concurrent turns from collectively overspending.
        spent = max(1_000, answer_chars // 2)
        limiter.refund_tokens(principal, max(0, _ESTIMATED_TURN_TOKENS - spent))
        yield _frame("done", {"thread_id": thread_id})


@router.get("/limits")
async def limits(request: Request, principal: PrincipalDep) -> dict[str, Any]:
    """Remaining allowance for the signed-in user, for the UI meter."""
    return request.app.state.limiter.snapshot(principal)
