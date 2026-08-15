"""The corpus environment the RLM plans against.

The point of a Recursive Language Model is that the corpus is an *environment
the model queries*, not a payload it swallows. So the research agent never sees
document bodies while it is deciding what to look at. It sees a **manifest** —
one row per chunk, metadata only — and writes Python that selects and partitions
those rows.

**Where the LLM calls happen, and why it matters.** The sandbox has no network,
so generated code cannot call a model. That constraint turns out to be the right
architecture rather than a limitation to work around:

* **Code decides what to look at.** Selection and partitioning are deterministic,
  auditable, cheap, and reproducible. The plan is a visible artifact.
* **The model decides what it means.** Map and reduce run outside the sandbox,
  in the orchestrator, over batches the plan produced.

A design that let generated code call a model inline would also make cost
unbounded and untraceable: a loop in the plan becomes a loop of billed
inference. Here the fan-out is a data structure the orchestrator can count and
cap *before* spending anything.

**Authorization.** The manifest is built from chunks the principal may read, and
the orchestrator re-checks every id the plan returns. A plan cannot reach a
document by naming its id, because the id was never in its manifest and is
refused on the way back regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.auth.principal import Principal
from app.observability.logging import get_logger
from app.retrieval.schema import Chunk

log = get_logger(__name__)

#: Manifest fields exposed to the plan. Metadata only — deliberately no text.
#: Adding the body here would defeat the entire purpose: the plan would then be
#: reasoning over the corpus in context rather than selecting from it.
MANIFEST_FIELDS = (
    "chunk_id",
    "doc_id",
    "title",
    "department",
    "document_type",
    "created_date",
    "owner",
    "heading_path",
    "token_estimate",
    "tags",
)

#: Ceiling on manifest rows handed to a plan. A manifest is small per row, but
#: it crosses a pipe as JSON and is held in the sandbox's memory.
MAX_MANIFEST_ROWS = 4_000


@dataclass
class RlmCorpus:
    """A principal-scoped view of the corpus, addressable by chunk id."""

    principal: Principal
    chunks: dict[str, Chunk] = field(default_factory=dict)

    @classmethod
    def for_principal(cls, principal: Principal, chunks: list[Chunk]) -> RlmCorpus:
        """Build the view, dropping anything this caller may not read."""
        visible = {
            chunk.chunk_id: chunk
            for chunk in chunks
            if principal.may_read(chunk.metadata.access_level, chunk.metadata.department)
        }
        withheld = len(chunks) - len(visible)
        if withheld:
            log.info(
                "rlm_corpus_scoped",
                visible=len(visible),
                withheld=withheld,
                role=principal.role.value,
            )
        return cls(principal=principal, chunks=visible)

    def manifest(self, *, limit: int = MAX_MANIFEST_ROWS) -> list[dict[str, Any]]:
        """Metadata rows for the plan to work over, oldest first."""
        rows = [_manifest_row(chunk) for chunk in self.chunks.values()]
        rows.sort(key=lambda row: (str(row["created_date"]), str(row["chunk_id"])))
        return rows[:limit]

    def summarise(self) -> dict[str, Any]:
        """A description of the manifest, for the planning prompt.

        The model is told the shape and the vocabulary of the corpus rather than
        being asked to infer it, which is the difference between a plan that
        filters on a real tag and one that filters on a plausible guess.
        """
        departments: dict[str, int] = {}
        document_types: dict[str, int] = {}
        tags: dict[str, int] = {}
        dates: list[str] = []

        for chunk in self.chunks.values():
            metadata = chunk.metadata
            departments[metadata.department] = departments.get(metadata.department, 0) + 1
            document_types[metadata.document_type] = (
                document_types.get(metadata.document_type, 0) + 1
            )
            dates.append(metadata.created_date)
            for tag in metadata.tags:
                tags[tag] = tags.get(tag, 0) + 1

        return {
            "total_chunks": len(self.chunks),
            "departments": dict(sorted(departments.items(), key=lambda kv: -kv[1])),
            "document_types": dict(sorted(document_types.items(), key=lambda kv: -kv[1])),
            "common_tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])[:24]),
            "date_range": [min(dates), max(dates)] if dates else [],
        }

    def resolve(self, chunk_ids: list[str]) -> tuple[list[Chunk], list[str]]:
        """Turn plan-selected ids into chunks, refusing anything unauthorized.

        The manifest already excluded unreadable chunks, so a refusal here means
        the plan invented an id or the corpus changed underneath it. Either way
        the answer is no, and unknown and forbidden are not distinguished.
        """
        resolved: list[Chunk] = []
        refused: list[str] = []

        for chunk_id in chunk_ids:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                refused.append(chunk_id)
                continue
            if not self.principal.may_read(chunk.metadata.access_level, chunk.metadata.department):
                refused.append(chunk_id)
                continue
            resolved.append(chunk)

        if refused:
            log.warning("rlm_plan_refused_ids", refused=len(refused), sample=sorted(refused)[:5])
        return resolved, refused


def _manifest_row(chunk: Chunk) -> dict[str, Any]:
    metadata = chunk.metadata
    row: dict[str, Any] = {"chunk_id": chunk.chunk_id}
    for name in MANIFEST_FIELDS[1:]:
        value = getattr(metadata, name)
        row[name] = list(value) if isinstance(value, list) else value
    return row


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """What a plan produced: groups of chunk ids, and why."""

    batches: list[list[str]]
    rationale: str = ""

    @property
    def total_ids(self) -> int:
        return sum(len(batch) for batch in self.batches)


def parse_plan_result(result: Any, *, max_batches: int, max_per_batch: int) -> BatchPlan:
    """Validate what the sandbox returned.

    The plan is model-authored code, so its output is untrusted input like any
    other. Anything malformed yields an empty plan rather than an exception —
    the research agent then degrades to plain retrieval, which is a worse answer
    but still an answer.
    """
    if not isinstance(result, dict):
        return BatchPlan(batches=[], rationale="plan did not return a mapping")

    raw_batches = result.get("batches")
    if not isinstance(raw_batches, list):
        return BatchPlan(batches=[], rationale="plan returned no batches list")

    batches: list[list[str]] = []
    for raw in raw_batches[:max_batches]:
        if not isinstance(raw, list):
            continue
        ids = [str(item) for item in raw[:max_per_batch] if isinstance(item, str | int)]
        if ids:
            batches.append(ids)

    rationale = str(result.get("rationale", ""))[:400]
    return BatchPlan(batches=batches, rationale=rationale)
