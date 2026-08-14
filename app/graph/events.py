"""Agent activity events.

The brief asks that an evaluator be able to observe what the agent is doing
internally. That is treated here as a product surface rather than a debug log:
events are typed, emitted from inside graph nodes, and multiplexed onto the same
SSE stream as the answer tokens, so what the panel shows is the execution itself
rather than a reconstruction after it.

Emission goes through ``langgraph.config.get_stream_writer()``, which requires
Python 3.11+ in async code — one of the reasons the project pins 3.12.

Two rules the emitters enforce:

* **Never emit raw retrieved text.** Retrieved content is untrusted, and the
  panel renders in a browser. Only identifiers, scores, and counts go out.
* **Never emit tool arguments verbatim.** They can carry user data. A digest and
  the argument names are enough to follow what happened.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.observability.logging import get_logger

log = get_logger(__name__)


class EventType(StrEnum):
    """Every activity event the panel understands."""

    NODE_ENTER = "node.enter"
    NODE_EXIT = "node.exit"
    PLAN_UPDATE = "plan.update"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    RETRIEVAL_STAGE = "retrieval.stage"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    VALIDATION_RESULT = "validation.result"
    GUARD_RESULT = "guard.result"
    RECURSION = "recursion"
    DEGRADATION = "degradation"
    BUDGET = "budget"
    #: A streamed answer token, not a credential.
    TOKEN = "token"  # noqa: S105
    DONE = "done"
    ERROR = "error"


class ActivityEvent(BaseModel):
    """One observable moment in a turn."""

    type: EventType
    #: Graph node that produced this, for grouping in the panel.
    node: str
    #: Free-form, type-specific payload. Kept a plain dict so adding a field to
    #: one event type does not require a schema migration across the client.
    data: dict[str, Any] = Field(default_factory=dict)
    #: Nesting level — sub-agents emit at depth 1+, so the panel can indent
    #: recursion rather than presenting a flat list.
    depth: int = 0

    def to_sse(self) -> str:
        """Render as a Server-Sent Event frame."""
        payload = json.dumps(self.model_dump(mode="json"), separators=(",", ":"))
        return f"event: {self.type.value}\ndata: {payload}\n\n"


def _writer() -> Any | None:
    """The active stream writer, or ``None`` outside a graph run.

    Nodes are unit-tested by calling them directly, with no LangGraph runtime
    present. Emission must be a no-op there rather than an error — an
    observability call should never be able to fail the thing it observes.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except (ImportError, RuntimeError):
        return None


def emit(event: ActivityEvent) -> None:
    """Publish one event to the active stream, if there is one."""
    writer = _writer()
    if writer is None:
        return
    try:
        writer(event.model_dump(mode="json"))
    except Exception as exc:  # observability must never break the run it observes
        log.warning("activity_event_dropped", event_type=event.type.value, error=str(exc))


# ── Typed emitters ──────────────────────────────────────────────────────
# Helpers rather than raw dicts, so the payload a node sends and the payload the
# panel renders cannot drift apart silently.


def node_enter(node: str, *, depth: int = 0, **data: Any) -> None:
    emit(ActivityEvent(type=EventType.NODE_ENTER, node=node, depth=depth, data=data))


def node_exit(node: str, *, depth: int = 0, **data: Any) -> None:
    emit(ActivityEvent(type=EventType.NODE_EXIT, node=node, depth=depth, data=data))


def plan_update(node: str, plan: list[Any], *, depth: int = 0) -> None:
    emit(
        ActivityEvent(
            type=EventType.PLAN_UPDATE,
            node=node,
            depth=depth,
            data={
                "steps": [
                    {
                        "id": item.id,
                        "description": item.description,
                        "agent": item.agent,
                        "status": str(item.status),
                    }
                    for item in plan
                ]
            },
        )
    )


def digest_arguments(arguments: dict[str, Any]) -> str:
    """Short stable hash of tool arguments.

    Arguments can carry user data, so the panel gets a fingerprint instead. Two
    identical calls still look identical, which is what makes a retry loop
    visible.
    """
    canonical = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def tool_call(
    node: str, tool: str, arguments: dict[str, Any], *, allowed: bool, depth: int = 0
) -> None:
    emit(
        ActivityEvent(
            type=EventType.TOOL_CALL,
            node=node,
            depth=depth,
            data={
                "tool": tool,
                # Names but not values: enough to follow the call, not enough to
                # leak its contents.
                "argument_names": sorted(arguments),
                "argument_digest": digest_arguments(arguments),
                "allowed": allowed,
            },
        )
    )


def tool_result(
    node: str, tool: str, *, ok: bool, duration_ms: float, summary: str = "", depth: int = 0
) -> None:
    emit(
        ActivityEvent(
            type=EventType.TOOL_RESULT,
            node=node,
            depth=depth,
            data={"tool": tool, "ok": ok, "duration_ms": round(duration_ms, 1), "summary": summary},
        )
    )


def retrieval_stage(
    node: str,
    stage: str,
    *,
    count: int,
    depth: int = 0,
    **details: Any,
) -> None:
    """Report one step of the retrieval pipeline.

    Counts and identifiers only — never chunk text.
    """
    emit(
        ActivityEvent(
            type=EventType.RETRIEVAL_STAGE,
            node=node,
            depth=depth,
            data={"stage": stage, "count": count, **details},
        )
    )


def memory_read(node: str, *, recalled: int, depth: int = 0) -> None:
    emit(
        ActivityEvent(
            type=EventType.MEMORY_READ, node=node, depth=depth, data={"recalled": recalled}
        )
    )


def memory_write(node: str, *, written: int, depth: int = 0) -> None:
    emit(
        ActivityEvent(
            type=EventType.MEMORY_WRITE, node=node, depth=depth, data={"written": written}
        )
    )


def validation_result(
    node: str,
    *,
    passed: bool,
    grounded_claims: int,
    ungrounded_claims: int,
    attempt: int,
    depth: int = 0,
) -> None:
    emit(
        ActivityEvent(
            type=EventType.VALIDATION_RESULT,
            node=node,
            depth=depth,
            data={
                "passed": passed,
                "grounded_claims": grounded_claims,
                "ungrounded_claims": ungrounded_claims,
                "attempt": attempt,
            },
        )
    )


def guard_result(node: str, *, verdict: str, score: float, signals: list[str]) -> None:
    emit(
        ActivityEvent(
            type=EventType.GUARD_RESULT,
            node=node,
            data={"verdict": verdict, "score": round(score, 3), "signals": signals},
        )
    )


def recursion(node: str, *, depth: int, batches: int, task: str) -> None:
    """Announce an RLM map step, so the panel can render the tree."""
    emit(
        ActivityEvent(
            type=EventType.RECURSION,
            node=node,
            depth=depth,
            data={"batches": batches, "task": task[:200]},
        )
    )


def degradation(node: str, *, component: str, reason: str, fallback: str) -> None:
    """Announce a rung of the degradation ladder.

    The panel highlights these: the system is allowed to return a worse answer,
    but never a silently worse one.
    """
    emit(
        ActivityEvent(
            type=EventType.DEGRADATION,
            node=node,
            data={"component": component, "reason": reason, "fallback": fallback},
        )
    )
    log.warning("degraded", component=component, reason=reason, fallback=fallback)


def budget_update(node: str, budget: Any, *, depth: int = 0) -> None:
    emit(
        ActivityEvent(
            type=EventType.BUDGET,
            node=node,
            depth=depth,
            data={
                "tool_calls": budget.tool_calls,
                "tokens": budget.tokens,
                "supervisor_steps": budget.supervisor_steps,
                "depth": budget.depth,
                "exhausted": budget.exhausted,
            },
        )
    )
