"""Heading-aware chunking.

Fixed-width chunking is the default in most RAG tutorials and it is the wrong
default for this corpus. These documents are strongly sectioned — an incident
report's "Root Cause" section is a self-contained answer to a specific question —
and a fixed window cuts across those boundaries, producing chunks that begin
mid-sentence in one section and end mid-sentence in the next.

This splitter instead walks the heading tree and packs *whole sections*, only
falling back to paragraph splitting when a single section exceeds the target
size. Two consequences matter downstream:

* Every chunk carries a ``heading_path``, so a citation reads
  "INC-2026-0142 → Root Cause" rather than naming a byte offset.
* The heading path is prepended to the indexed text, which measurably helps the
  sparse retriever: the words "root cause" are then lexically present in the
  chunk that actually contains the root cause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.retrieval.schema import Chunk, ChunkMetadata, Document

#: Target chunk size in tokens.
#:
#: Tuned against this corpus rather than copied from a tutorial. Documents here
#: average ~400 tokens across ~8 sections, so the conventional 800-token target
#: packs every document into a single chunk — which silently disables all of the
#: section awareness below and makes every citation document-level. At 250 an
#: incident report yields three chunks split on meaning, and a query about root
#: causes retrieves the root-cause section rather than the whole report.
DEFAULT_TARGET_TOKENS = 250

#: Overlap carried between chunks split out of the same oversized section.
#: Only applies to that case — section boundaries are already semantic
#: boundaries, so overlapping across them would duplicate text for no gain.
DEFAULT_OVERLAP_TOKENS = 60

#: Sections shorter than this are merged forward rather than indexed alone. A
#: lone "## Impact" heading with one line under it retrieves poorly and dilutes
#: the ranking.
MIN_SECTION_TOKENS = 60

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$", re.MULTILINE)

#: Average characters per token for English prose. Deliberately a heuristic:
#: the exact count only drives packing decisions, and calling the Gemini
#: tokenizer per section would turn indexing into thousands of network round
#: trips for no material gain.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass
class Section:
    """A heading and the prose beneath it, before the next heading."""

    heading_path: list[str]
    text: str

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    @property
    def path_label(self) -> str:
        return " → ".join(self.heading_path) if self.heading_path else "(document)"


@dataclass
class _Accumulator:
    """Sections being packed into the current chunk."""

    sections: list[Section] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.sections)

    def is_empty(self) -> bool:
        return not self.sections


def split_into_sections(body: str) -> list[Section]:
    """Walk the markdown heading tree into a flat list of sections.

    The heading *path* is maintained as a stack, so a level-3 heading inherits
    the level-1 and level-2 headings above it. That breadcrumb is what makes a
    citation legible.
    """
    matches = list(_HEADING.finditer(body))

    if not matches:
        stripped = body.strip()
        return [Section(heading_path=[], text=stripped)] if stripped else []

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []

    # Any prose before the first heading belongs to the document root.
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(heading_path=[], text=preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Pop headings at or below this level; this heading replaces them.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[match.end() : end].strip()

        heading_path = [title for _, title in stack]
        # Keep the heading line in the text: it is meaningful content for the
        # sparse retriever, not just structure.
        heading_line = f"{'#' * level} {title}"
        text = f"{heading_line}\n\n{content}" if content else heading_line
        sections.append(Section(heading_path=heading_path, text=text))

    return sections


def _split_oversized(section: Section, target: int, overlap: int) -> list[Section]:
    """Break a section too large for one chunk, on paragraph boundaries.

    Overlap is applied here — and only here — because this is the one place a
    split lands somewhere that is not already a semantic boundary.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
    pieces: list[Section] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)

        if current and current_tokens + paragraph_tokens > target:
            pieces.append(Section(section.heading_path, "\n\n".join(current)))

            # Carry the tail of the previous piece into the next one so a
            # sentence spanning the cut is retrievable from either side.
            carried: list[str] = []
            carried_tokens = 0
            for previous in reversed(current):
                previous_tokens = estimate_tokens(previous)
                if carried_tokens + previous_tokens > overlap:
                    break
                carried.insert(0, previous)
                carried_tokens += previous_tokens

            current = [*carried, paragraph]
            current_tokens = carried_tokens + paragraph_tokens
        else:
            current.append(paragraph)
            current_tokens += paragraph_tokens

    if current:
        pieces.append(Section(section.heading_path, "\n\n".join(current)))

    return pieces


def chunk_document(
    document: Document,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_section_tokens: int = MIN_SECTION_TOKENS,
) -> list[Chunk]:
    """Split one document into indexable chunks."""
    sections = split_into_sections(document.body)

    # Expand any section too large to fit a chunk on its own.
    expanded: list[Section] = []
    for section in sections:
        if section.tokens > target_tokens:
            expanded.extend(_split_oversized(section, target_tokens, overlap_tokens))
        else:
            expanded.append(section)

    # Pack sections greedily, keeping tiny ones attached to their neighbour.
    packed: list[list[Section]] = []
    accumulator = _Accumulator()

    for section in expanded:
        would_be = accumulator.tokens + section.tokens
        too_big = would_be > target_tokens
        holds_enough = accumulator.tokens >= min_section_tokens

        if not accumulator.is_empty() and too_big and holds_enough:
            packed.append(accumulator.sections)
            accumulator = _Accumulator()

        accumulator.sections.append(section)

    if not accumulator.is_empty():
        packed.append(accumulator.sections)

    packed = _absorb_undersized_tail(packed, min_section_tokens)

    return [
        _build_chunk(document, group, index)
        for index, group in enumerate(packed)
        if any(s.text.strip() for s in group)
    ]


def _absorb_undersized_tail(
    packed: list[list[Section]], min_section_tokens: int
) -> list[list[Section]]:
    """Merge a too-small final group back into its predecessor.

    Sections are merged *forward* during packing, which leaves the last group
    with nowhere to go. Documents here commonly end with a short "## Review" or
    "## Verification" section, and indexing that alone produces a 25-token chunk
    that matches many queries weakly and none of them well.
    """
    while len(packed) > 1 and sum(s.tokens for s in packed[-1]) < min_section_tokens:
        tail = packed.pop()
        packed[-1].extend(tail)
    return packed


def _section_span_label(sections: list[Section]) -> str:
    """Describe what a chunk actually covers.

    When several sections are packed together, naming only the first is
    misleading — a citation reading "Timeline" for a chunk that also contains
    the root cause sends the reader to the wrong place. Multi-section chunks are
    therefore labelled with every leaf heading they contain.
    """
    leaves: list[str] = []
    for section in sections:
        leaf = section.heading_path[-1] if section.heading_path else "(preamble)"
        if leaf not in leaves:
            leaves.append(leaf)
    return " / ".join(leaves)


def _build_chunk(document: Document, sections: list[Section], index: int) -> Chunk:
    frontmatter = document.frontmatter
    heading_path = _section_span_label(sections)
    body = "\n\n".join(section.text for section in sections)

    # Prepend the document title and section breadcrumb. This is the single
    # cheapest retrieval improvement available: it puts the document's subject
    # into every chunk, so a chunk about "Remediation" is still findable by a
    # query naming the incident.
    text = f"{frontmatter.title}\n{heading_path}\n\n{body}"

    metadata = ChunkMetadata(
        doc_id=frontmatter.doc_id,
        title=frontmatter.title,
        department=frontmatter.department.value,
        document_type=frontmatter.document_type.value,
        access_level=frontmatter.access_level,
        created_date=frontmatter.created_date.isoformat(),
        # Pinecone range filters ($gte/$lt) apply to numbers only, and the RLM
        # batches incidents by time window — so the date is carried twice.
        created_ts=int(frontmatter.created_date.toordinal() * 86400),
        owner=frontmatter.owner,
        tags=list(frontmatter.tags),
        source_uri=document.source_uri,
        chunk_idx=index,
        heading_path=heading_path,
        token_estimate=estimate_tokens(text),
    )

    return Chunk(
        chunk_id=f"{frontmatter.doc_id}#{index:03d}",
        text=text,
        metadata=metadata,
    )


def chunk_corpus(documents: list[Document], **kwargs: int) -> list[Chunk]:
    """Chunk every document, preserving corpus order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, **kwargs))
    return chunks
