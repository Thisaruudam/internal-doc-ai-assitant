"""Memory nodes.

``memory_load`` runs before the supervisor and recalls durable facts about the
user. ``memory_write`` runs after validation and decides what, if anything, from
this turn is worth keeping.

The write step is deliberately conservative. A memory system that stores
everything degrades into a second, worse retrieval index — noisier than the real
one, unauthorised by construction, and impossible to correct. So a turn produces
at most one fact, and only when it carries something durable about the *user*:
what they work on, a constraint they have stated, a preference. Answers are not
remembered; they can be re-derived from documents, and a stale remembered answer
is worse than no memory at all.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.models import ModelRole, get_model
from app.config import Settings
from app.graph import events
from app.graph.state import AgentState
from app.memory.longterm import LongTermMemory, format_recalled
from app.memory.summarizer import SUMMARY_PREFIX, compact
from app.observability.logging import get_logger

log = get_logger(__name__)

EXTRACT_SYSTEM = """\
You decide whether a conversation turn contains something worth remembering \
about the user for future conversations.

Remember only durable facts about the person or their work:
- what they are responsible for, or which systems and departments they work on
- a standing constraint they stated ("I only care about the payments side")
- a stated preference about how they want answers

Do NOT remember:
- the answer to their question — it can be re-derived from documents, and a \
stale remembered answer is worse than none
- anything about document contents
- one-off details with no bearing on a future conversation

Most turns contain nothing worth remembering. Returning nothing is the common \
and correct outcome. If there is something, write one short sentence in the \
third person."""


class MemoryCandidate(BaseModel):
    worth_remembering: bool = Field(description="true only for durable facts about the user")
    fact: str = Field(default="", description="one short sentence, third person")


def _question(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            text = getattr(message, "text", None)
            return text if isinstance(text, str) else str(message.content)
    return ""


async def memory_load(
    state: AgentState, settings: Settings, memory: LongTermMemory
) -> dict[str, Any]:
    """Recall durable facts, and compact the history if it has grown."""
    events.node_enter("memory_load")
    principal = state["principal"]
    question = _question(state)

    facts = memory.recall(principal, question)
    events.memory_read("memory_load", recalled=len(facts))

    update: dict[str, Any] = {}
    if facts:
        update["files"] = {"memory.md": format_recalled(facts)}

    # Compaction happens on load rather than on write so the cost lands before
    # the expensive part of the turn, and a compacted history is what the
    # specialists actually see.
    compacted = await compact(state.get("messages", []), settings.gemini, settings.graph)
    if compacted is not None:
        summary, keep = compacted
        keep_ids = {message.id for message in keep if message.id}
        removals = [
            RemoveMessage(id=message.id)
            for message in state.get("messages", [])
            if message.id and message.id not in keep_ids
        ]
        update["messages"] = [
            *removals,
            SystemMessage(content=f"{SUMMARY_PREFIX} {summary}"),
        ]
        events.memory_write("memory_load", written=len(removals))

    events.node_exit("memory_load", recalled=len(facts))
    return update


async def memory_write(
    state: AgentState, settings: Settings, memory: LongTermMemory
) -> dict[str, Any]:
    """Decide whether this turn produced anything worth keeping."""
    events.node_enter("memory_write")
    principal = state["principal"]
    question = _question(state)

    if not question.strip():
        events.node_exit("memory_write", written=0)
        return {}

    try:
        model = get_model(ModelRole.GUARD, settings.gemini).with_structured_output(MemoryCandidate)
        candidate = cast(
            MemoryCandidate,
            await model.ainvoke(
                [
                    SystemMessage(content=EXTRACT_SYSTEM),
                    HumanMessage(content=f"The user asked: {question}"),
                ]
            ),
        )
    except Exception as exc:
        # Memory is an enhancement. Failing to write a fact must never fail the
        # turn the user actually asked for.
        log.warning("memory_extraction_failed", error_type=type(exc).__name__)
        events.node_exit("memory_write", written=0)
        return {}

    if not candidate.worth_remembering or not candidate.fact.strip():
        events.memory_write("memory_write", written=0)
        events.node_exit("memory_write", written=0)
        return {}

    memory.remember(principal, candidate.fact, thread_id=state.get("thread_id", ""))
    events.memory_write("memory_write", written=1)
    events.node_exit("memory_write", written=1)
    return {}
