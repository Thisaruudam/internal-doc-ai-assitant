"""Long-term, cross-session memory.

Durable facts about a user — what they work on, what they have asked before,
preferences they have stated — recalled into later conversations so the
assistant does not start from zero every morning.

**The security property that shapes this module.** Every fact records the access
ceiling of the role that was active when it was written, and recall drops facts
written above the reader's *current* ceiling.

That is not hypothetical tidiness. Memory is the one path that carries content
from one turn into another while bypassing retrieval entirely — so all three
authorization layers, which live in the retrieval and tool paths, simply do not
apply to it. Without this check, an administrator's question about restricted
compensation bands becomes a stored fact, and after that person moves to a
viewer role the same fact is replayed into their context. The document is still
unreachable; a summary of it is not.

**Scope.** Namespaced per user. Facts never cross users, and there is no shared
or organisational memory — one user's question is not another's context.

**Recall is lexical, not semantic.** A deliberate limit: the fact set per user is
tens of entries, where term overlap plus recency is as good as an embedding and
costs nothing. If a user accumulated thousands, an embedding index would earn
its keep — ``GeminiEmbedder`` is already available for that, and this is the one
place to change.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.auth.principal import AccessLevel, Principal
from app.observability.logging import get_logger

log = get_logger(__name__)

#: Facts recalled into a single turn. Enough to be useful, small enough that
#: memory never crowds out the retrieved evidence it is meant to complement.
MAX_RECALLED = 4

#: Ceiling on stored facts per user. Oldest are dropped first.
MAX_FACTS_PER_USER = 60

_WORD = re.compile(r"[a-z][a-z\-]{2,}")

_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "был",
        "about",
        "into",
        "your",
        "our",
        "are",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "how",
        "why",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
    ]
)


@dataclass(frozen=True, slots=True)
class MemoryFact:
    """One durable thing worth remembering about a user."""

    text: str
    #: The access ceiling in force when this was written. Recall compares the
    #: reader's *current* ceiling against this.
    written_at_ceiling: int
    created_at: float
    thread_id: str = ""

    def readable_by(self, principal: Principal) -> bool:
        return self.written_at_ceiling <= int(principal.access_ceiling)

    def to_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "written_at_ceiling": self.written_at_ceiling,
            "created_at": self.created_at,
            "thread_id": self.thread_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> MemoryFact:
        return cls(
            text=str(payload.get("text", "")),
            written_at_ceiling=int(payload.get("written_at_ceiling", int(AccessLevel.RESTRICTED))),
            created_at=float(payload.get("created_at", 0.0)),
            thread_id=str(payload.get("thread_id", "")),
        )


def _terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


@dataclass
class LongTermMemory:
    """Per-user fact store.

    Backed by a plain dict here. The LangGraph ``Store`` interface (Postgres +
    pgvector) is the durable swap; this keeps the system runnable without a
    database and keeps the authorization logic — the part worth testing — free
    of storage concerns.
    """

    _facts: dict[str, list[MemoryFact]] = field(default_factory=dict)

    def remember(self, principal: Principal, text: str, *, thread_id: str = "") -> MemoryFact:
        """Store a fact, stamped with the writer's current ceiling."""
        fact = MemoryFact(
            text=text.strip(),
            written_at_ceiling=int(principal.access_ceiling),
            created_at=time.time(),
            thread_id=thread_id,
        )
        facts = self._facts.setdefault(principal.user_id, [])
        facts.append(fact)

        if len(facts) > MAX_FACTS_PER_USER:
            del facts[: len(facts) - MAX_FACTS_PER_USER]

        log.info(
            "memory_fact_written",
            role=principal.role.value,
            ceiling=principal.access_ceiling.name.lower(),
        )
        return fact

    def recall(
        self, principal: Principal, query: str, *, limit: int = MAX_RECALLED
    ) -> list[MemoryFact]:
        """Recall facts relevant to a query that this caller may still read."""
        stored = self._facts.get(principal.user_id, [])
        if not stored:
            return []

        readable = [fact for fact in stored if fact.readable_by(principal)]
        withheld = len(stored) - len(readable)
        if withheld:
            # The signal that a downgrade actually took effect.
            log.info(
                "memory_facts_withheld",
                withheld=withheld,
                role=principal.role.value,
                reason="written above the caller's current access ceiling",
            )

        query_terms = _terms(query)
        if not query_terms:
            return sorted(readable, key=lambda f: -f.created_at)[:limit]

        def relevance(fact: MemoryFact) -> tuple[float, float]:
            overlap = len(query_terms & _terms(fact.text)) / max(len(query_terms), 1)
            return (overlap, fact.created_at)

        ranked = sorted(readable, key=relevance, reverse=True)
        # Anything with no term in common is noise; recency alone is not a
        # reason to inject a fact into an unrelated question.
        return [fact for fact in ranked if _terms(fact.text) & query_terms][:limit]

    def forget(self, user_id: str) -> int:
        """Delete everything stored for a user. Returns how many facts went."""
        removed = len(self._facts.pop(user_id, []))
        if removed:
            log.info("memory_purged", facts=removed)
        return removed

    def count(self, user_id: str) -> int:
        return len(self._facts.get(user_id, []))


def format_recalled(facts: list[MemoryFact]) -> str:
    """Render recalled facts for a prompt.

    Labelled as context rather than evidence, and explicitly not citable: a fact
    is a compressed memory of an earlier turn, not a passage, so an answer must
    never cite one as a source.
    """
    if not facts:
        return ""
    lines = "\n".join(f"- {fact.text}" for fact in facts)
    return (
        "Context remembered from this user's earlier conversations. Use it to "
        "interpret the question — not as evidence, and never cite it as a "
        "source:\n" + lines
    )
