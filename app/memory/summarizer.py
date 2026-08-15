"""Working-memory compaction.

Multi-turn conversations grow their own cost. Every turn resends the whole
history, so an unmanaged thread is quadratic in tokens: turn twenty pays for
turns one through nineteen again.

Two ways to bound it, and the choice matters:

* **Truncation** — drop the oldest turns. Cheap, and it silently discards
  exactly the material that shapes a long conversation. A constraint stated in
  turn two ("only the payments department, and only last year") falls out of the
  window and the assistant starts answering a different question without anyone
  noticing.

* **Compaction** — summarise the old turns and keep the recent ones verbatim.
  Costs one model call when it triggers, and preserves standing constraints.

This module compacts. The summary is regenerated from the previous summary plus
the turns being retired, so it stays bounded rather than growing with the
conversation.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from app.agents.models import ModelRole, get_model
from app.config import GeminiSettings, GraphSettings
from app.observability.logging import get_logger

log = get_logger(__name__)

SUMMARY_SYSTEM = """\
You maintain a running summary of a conversation between an employee and an \
internal knowledge assistant.

Preserve, in this order of priority:
1. Standing constraints the user has stated — a department, a date range, a \
system they care about. These shape every later answer and must survive.
2. What has already been established, so the assistant does not re-answer it.
3. Open threads the user seems to be working towards.

Drop: pleasantries, the assistant's phrasing, and detail already superseded.

Write compact prose in the third person, under 200 words. Output only the \
summary."""

#: Marks the summary message so it can be found and replaced on the next
#: compaction rather than accumulating one summary per compaction.
SUMMARY_PREFIX = "[conversation so far]"


def _text_of(message: AnyMessage) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return str(getattr(message, "content", ""))


def estimate_tokens(messages: list[AnyMessage]) -> int:
    """Rough token count for the history. Four characters per token."""
    return sum(len(_text_of(message)) for message in messages) // 4


def should_compact(messages: list[AnyMessage], settings: GraphSettings) -> bool:
    """Whether the history has grown enough to be worth a summarisation call."""
    conversational = [
        message for message in messages if isinstance(message, HumanMessage | AIMessage)
    ]
    if len(conversational) <= settings.memory_verbatim_turns:
        return False
    return estimate_tokens(messages) >= settings.memory_compact_threshold


def split_history(
    messages: list[AnyMessage], settings: GraphSettings
) -> tuple[str, list[AnyMessage], list[AnyMessage]]:
    """Separate the existing summary, the turns to retire, and the ones to keep.

    Returns ``(existing_summary, to_retire, to_keep)``.
    """
    existing_summary = ""
    body: list[AnyMessage] = []

    for message in messages:
        text = _text_of(message)
        if isinstance(message, SystemMessage) and text.startswith(SUMMARY_PREFIX):
            existing_summary = text[len(SUMMARY_PREFIX) :].strip()
            continue
        body.append(message)

    keep_count = settings.memory_verbatim_turns
    return existing_summary, body[:-keep_count], body[-keep_count:]


async def compact(
    messages: list[AnyMessage],
    gemini: GeminiSettings,
    graph: GraphSettings,
) -> tuple[str, list[AnyMessage]] | None:
    """Compact the history if it has grown too large.

    Returns ``(summary, messages_to_keep)``, or ``None`` when no compaction was
    needed or the summarisation call failed. A failed compaction is not an
    error: the conversation continues uncompacted, costing tokens rather than
    correctness.
    """
    if not should_compact(messages, graph):
        return None

    existing, to_retire, to_keep = split_history(messages, graph)
    if not to_retire:
        return None

    transcript = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {_text_of(m)[:1500]}"
        for m in to_retire
        if isinstance(m, HumanMessage | AIMessage)
    )
    if not transcript.strip():
        return None

    prompt = (
        f"Summary so far:\n{existing}\n\n" if existing else ""
    ) + f"Turns to fold into the summary:\n{transcript}"

    try:
        model = get_model(ModelRole.GUARD, gemini)
        reply = await model.ainvoke(
            [SystemMessage(content=SUMMARY_SYSTEM), HumanMessage(content=prompt)]
        )
        summary = _text_of(reply).strip()
    except Exception as exc:
        log.warning("memory_compaction_failed", error_type=type(exc).__name__)
        return None

    if not summary:
        return None

    log.info(
        "memory_compacted",
        retired=len(to_retire),
        kept=len(to_keep),
        summary_chars=len(summary),
    )
    return summary, to_keep
