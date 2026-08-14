"""Retrieval domain types.

These are the shapes that flow from the corpus, through chunking and indexing,
into Pinecone metadata, and back out as citations. Defining them once — rather
than passing dictionaries around — means the metadata contract is checkable, and
a typo in a field name fails at the boundary instead of silently producing an
empty filter.

``ChunkMetadata`` in particular is the security-relevant type: ``access_level``
and ``department`` on every chunk are what the authorization filter matches
against.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.auth.principal import AccessLevel


class Department(StrEnum):
    """Corpus partitions. Each maps to one Pinecone namespace."""

    PAYMENTS = "payments"
    PLATFORM = "platform"
    SECURITY = "security"
    RISK = "risk"
    RETAIL_BANKING = "retail-banking"
    PEOPLE = "people"


class DocumentType(StrEnum):
    """The document classes named in the brief."""

    POLICY = "policy"
    ARCHITECTURE = "architecture"
    RUNBOOK = "runbook"
    INCIDENT = "incident"
    PRODUCT_SPEC = "product_spec"
    MEETING_NOTES = "meeting_notes"


class DocumentFrontmatter(BaseModel):
    """The YAML header on every corpus file.

    ``extra="forbid"`` is deliberate: an unrecognised key in a document header is
    a mistake worth surfacing at ingest time, not silently dropping.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=3, max_length=64)
    title: str = Field(min_length=3, max_length=300)
    department: Department
    document_type: DocumentType
    access_level: str
    created_date: date
    owner: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("access_level")
    @classmethod
    def _known_access_level(cls, value: str) -> str:
        # Parse-and-discard: rejects an unknown level at ingest rather than
        # letting it reach the index, where it would fail closed and make the
        # document silently unreachable for everyone.
        AccessLevel.parse(value)
        return value.lower()


class Document(BaseModel):
    """A parsed corpus file: header plus body."""

    frontmatter: DocumentFrontmatter
    body: str
    source_path: str

    @property
    def source_uri(self) -> str:
        return f"corpus://{self.frontmatter.department.value}/{self.frontmatter.doc_id}"


class ChunkMetadata(BaseModel):
    """What is stored alongside each vector.

    Kept flat and primitive because Pinecone metadata supports only strings,
    numbers, booleans, and string lists — and because a flat shape is what the
    filter language can express.
    """

    doc_id: str
    title: str
    department: str
    document_type: str
    access_level: str
    #: ISO date string. Also carried as ``created_ts`` because Pinecone can only
    #: apply range filters ($gte/$lt) to numbers, and the RLM batches by time
    #: window.
    created_date: str
    created_ts: int
    owner: str
    tags: list[str]
    source_uri: str
    #: Position within the document, for ordering and adjacent-chunk expansion.
    chunk_idx: int
    #: Breadcrumb of enclosing headings, so a citation points at a section
    #: rather than an opaque offset.
    heading_path: str
    token_estimate: int

    def to_pinecone(self) -> dict[str, Any]:
        return self.model_dump()


class Chunk(BaseModel):
    """An indexable unit of text with its metadata."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata

    @property
    def department(self) -> str:
        return self.metadata.department

    @property
    def access_level(self) -> str:
        return self.metadata.access_level


class ScoredChunk(BaseModel):
    """A chunk with its retrieval provenance attached.

    Carrying *how* a chunk was found — dense rank, sparse rank, fused score,
    rerank score — is what lets the activity panel explain a result rather than
    just present it.
    """

    chunk: Chunk
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    #: Which retriever(s) surfaced this chunk: "dense", "sparse", "bm25".
    retrievers: list[str] = Field(default_factory=list)

    @property
    def final_score(self) -> float:
        """Rerank score when available, otherwise the fusion score."""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.fused_score or 0.0


class Citation(BaseModel):
    """A reference the response agent is allowed to make.

    The validator checks every claim against this set; a citation naming a
    ``chunk_id`` that was never retrieved is treated as a hallucination.
    """

    chunk_id: str
    doc_id: str
    title: str
    heading_path: str
    source_uri: str
    department: str
    access_level: str

    @classmethod
    def of(cls, chunk: Chunk) -> Self:
        meta = chunk.metadata
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=meta.doc_id,
            title=meta.title,
            heading_path=meta.heading_path,
            source_uri=meta.source_uri,
            department=meta.department,
            access_level=meta.access_level,
        )
