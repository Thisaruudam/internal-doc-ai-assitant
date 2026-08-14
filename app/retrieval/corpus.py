"""Corpus loading.

Reads the markdown corpus off disk and validates every document header against
``DocumentFrontmatter``. Validation is strict and fails loudly: a document with a
malformed ``access_level`` must never reach the index, because a level the
lattice cannot parse fails closed and the document becomes silently unreachable
for every user — a data-loss bug that looks like a retrieval-quality bug.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from app.observability.logging import get_logger
from app.retrieval.schema import Document, DocumentFrontmatter

log = get_logger(__name__)

_FRONTMATTER_DELIMITER = "---"


class CorpusError(Exception):
    """Raised when a document cannot be parsed or fails validation."""


def parse_document(path: Path) -> Document:
    """Parse one markdown file with a YAML frontmatter header."""
    raw = path.read_text(encoding="utf-8")

    if not raw.lstrip().startswith(_FRONTMATTER_DELIMITER):
        raise CorpusError(f"{path}: missing YAML frontmatter header")

    # Split on the first two delimiters only; the body may legitimately contain
    # horizontal rules.
    parts = raw.split(_FRONTMATTER_DELIMITER, 2)
    if len(parts) < 3:
        raise CorpusError(f"{path}: frontmatter header is not terminated")

    _, header_text, body = parts

    try:
        header = yaml.safe_load(header_text) or {}
    except yaml.YAMLError as exc:
        raise CorpusError(f"{path}: frontmatter is not valid YAML: {exc}") from exc

    try:
        frontmatter = DocumentFrontmatter.model_validate(header)
    except ValidationError as exc:
        raise CorpusError(f"{path}: frontmatter failed validation: {exc}") from exc

    return Document(frontmatter=frontmatter, body=body.strip(), source_path=str(path))


def load_documents(corpus_dir: str | Path) -> list[Document]:
    """Load and validate the whole corpus, ordered by document id.

    Deterministic ordering keeps chunk ids stable between runs, so re-indexing
    updates vectors in place rather than orphaning the previous generation.
    """
    directory = Path(corpus_dir)
    if not directory.is_dir():
        raise CorpusError(f"corpus directory {directory} does not exist")

    documents: list[Document] = []
    failures: list[str] = []

    for path in sorted(directory.glob("*.md")):
        try:
            documents.append(parse_document(path))
        except CorpusError as exc:
            failures.append(str(exc))

    if failures:
        raise CorpusError(
            f"{len(failures)} document(s) failed to load:\n  " + "\n  ".join(failures)
        )

    if not documents:
        raise CorpusError(f"corpus directory {directory} contains no documents")

    duplicates = _duplicate_ids(documents)
    if duplicates:
        raise CorpusError(f"duplicate doc_id values in corpus: {sorted(duplicates)}")

    log.info("corpus_loaded", document_count=len(documents), directory=str(directory))
    return sorted(documents, key=lambda d: d.frontmatter.doc_id)


def _duplicate_ids(documents: list[Document]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for document in documents:
        doc_id = document.frontmatter.doc_id
        if doc_id in seen:
            duplicates.add(doc_id)
        seen.add(doc_id)
    return duplicates
