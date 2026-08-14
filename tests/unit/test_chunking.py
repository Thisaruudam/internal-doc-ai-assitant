"""Chunking behaviour.

The properties asserted here are the ones that were actually wrong at first: a
target size copied from convention packed every document into one chunk, and a
trailing short section became a 25-token chunk of its own. Both are silent
failures — retrieval still "works", just worse — so they get explicit tests.
"""

from __future__ import annotations

import itertools
from datetime import date

import pytest

from app.retrieval.chunking import (
    DEFAULT_TARGET_TOKENS,
    MIN_SECTION_TOKENS,
    chunk_document,
    estimate_tokens,
    split_into_sections,
)
from app.retrieval.schema import Document, DocumentFrontmatter

INCIDENT_BODY = """# SEV2: payment failures in settlement-service

## Summary

On 2026-03-14 the settlement service experienced elevated payment authorization
failures for 42 minutes. Roughly 18,000 transactions failed and an estimated
4,200 customers were affected during the peak of the incident window.

## Impact

- Failed authorizations: 18,000
- Customers affected: 4,200
- Duration: 42 minutes
- Customer-facing symptom: card payments declined with a generic error message
- Revenue at risk: LKR 41,200,000

## Timeline

- 09:14 — Automated alert fired on the payment success-rate monitor.
- +4m — On-call engineer acknowledged and opened the incident bridge.
- +11m — Impact confirmed as customer-facing; SEV2 declared.
- +21m — Root cause identified as connection pool exhaustion.
- +42m — Mitigation applied and success rate returned to baseline.

## Root Cause

The settlement service opened a database connection per in-flight authorization
instead of borrowing from the shared pool. Under peak load the pool ceiling of
40 was reached and further authorizations queued until they timed out entirely.

## Detection

Authorization p99 latency crossed eight seconds while CPU utilisation stayed
below thirty percent, which is the signature of queueing rather than of compute
saturation on the affected nodes.

## Remediation

Pool ceiling raised to 160, per-request connection acquisition replaced with a
scoped borrow, and a queue-depth alert added to the service dashboard.

## Prevention

The load test was extended to hold three times peak concurrency for thirty
minutes, and pool saturation was added to the pre-release checklist.

## Review

This incident is reviewed annually.
"""


def make_document(body: str, doc_id: str = "INC-2026-0001") -> Document:
    return Document(
        frontmatter=DocumentFrontmatter(
            doc_id=doc_id,
            title="SEV2: payment failures in settlement-service",
            department="payments",
            document_type="incident",
            access_level="confidential",
            created_date=date(2026, 3, 14),
            owner="Payments Platform Team",
            tags=["outage", "connection-pool-exhaustion"],
        ),
        body=body,
        source_path=f"data/corpus/{doc_id}.md",
    )


class TestSectionSplitting:
    def test_heading_path_accumulates_ancestors(self) -> None:
        sections = split_into_sections("# Top\n\nintro\n\n## Middle\n\nbody\n\n### Leaf\n\ndeep")
        paths = [s.heading_path for s in sections]
        assert paths == [["Top"], ["Top", "Middle"], ["Top", "Middle", "Leaf"]]

    def test_sibling_heading_pops_the_stack(self) -> None:
        sections = split_into_sections("# T\n\n## A\n\na\n\n## B\n\nb")
        assert [s.heading_path for s in sections] == [["T"], ["T", "A"], ["T", "B"]]

    def test_preamble_before_first_heading_is_kept(self) -> None:
        sections = split_into_sections("loose text\n\n# Heading\n\nbody")
        assert sections[0].heading_path == []
        assert "loose text" in sections[0].text

    def test_document_without_headings_yields_one_section(self) -> None:
        sections = split_into_sections("just prose, no structure at all")
        assert len(sections) == 1
        assert sections[0].heading_path == []

    def test_empty_body_yields_nothing(self) -> None:
        assert split_into_sections("   \n\n  ") == []

    def test_heading_text_is_retained_in_the_chunk(self) -> None:
        """The heading is content for the sparse retriever, not just structure."""
        sections = split_into_sections("## Root Cause\n\nthe pool was exhausted")
        assert "Root Cause" in sections[0].text


class TestChunking:
    def test_document_splits_on_meaning_not_only_size(self) -> None:
        chunks = chunk_document(make_document(INCIDENT_BODY))
        assert len(chunks) > 1, "an incident report must not collapse into one chunk"

    def test_root_cause_is_separately_addressable(self) -> None:
        """The regression that motivated retuning the target size.

        With an 800-token target this document became a single chunk, so a query
        about root causes retrieved the whole report — impact figures, timeline
        and all — and the citation could only name the document.
        """
        chunks = chunk_document(make_document(INCIDENT_BODY))
        labelled = [c for c in chunks if "Root Cause" in c.metadata.heading_path]
        assert labelled, "no chunk is labelled as containing the root cause"
        assert "connection per in-flight authorization" in labelled[0].text

    def test_no_chunk_is_undersized(self) -> None:
        """A trailing short section must be absorbed, not indexed alone."""
        chunks = chunk_document(make_document(INCIDENT_BODY))
        assert all(c.metadata.token_estimate >= MIN_SECTION_TOKENS for c in chunks)

    def test_trailing_review_section_is_absorbed_backwards(self) -> None:
        chunks = chunk_document(make_document(INCIDENT_BODY))
        assert "reviewed annually" in chunks[-1].text
        assert chunks[-1].metadata.heading_path != "Review", (
            "the short trailing section became its own chunk"
        )

    def test_chunks_stay_near_the_target_size(self) -> None:
        chunks = chunk_document(make_document(INCIDENT_BODY))
        # Some overshoot is expected: a whole section is never cut just to hit
        # the number. The bound guards against unbounded growth.
        assert all(c.metadata.token_estimate <= DEFAULT_TARGET_TOKENS * 2 for c in chunks)

    def test_label_names_every_section_the_chunk_contains(self) -> None:
        """A citation reading only the first heading would mislead the reader."""
        chunks = chunk_document(make_document(INCIDENT_BODY))
        multi = [c for c in chunks if "/" in c.metadata.heading_path]
        assert multi, "expected at least one chunk spanning several sections"
        for chunk in multi:
            for leaf in chunk.metadata.heading_path.split(" / "):
                assert leaf.strip() in chunk.text

    def test_chunk_ids_are_stable_and_ordered(self) -> None:
        chunks = chunk_document(make_document(INCIDENT_BODY))
        assert [c.chunk_id for c in chunks] == [
            f"INC-2026-0001#{i:03d}" for i in range(len(chunks))
        ]
        assert [c.metadata.chunk_idx for c in chunks] == list(range(len(chunks)))

    def test_regenerating_produces_identical_chunks(self) -> None:
        first = chunk_document(make_document(INCIDENT_BODY))
        second = chunk_document(make_document(INCIDENT_BODY))
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.text for c in first] == [c.text for c in second]

    def test_title_and_breadcrumb_are_prepended(self) -> None:
        """Puts the document's subject into every chunk, so a chunk about
        remediation is still findable by a query naming the incident."""
        for chunk in chunk_document(make_document(INCIDENT_BODY)):
            assert chunk.text.startswith(chunk.metadata.title)


class TestOversizedSections:
    def test_a_single_huge_section_is_split_with_overlap(self) -> None:
        paragraphs = "\n\n".join(f"Paragraph {i} " + "filler text " * 40 for i in range(12))
        chunks = chunk_document(make_document(f"# Doc\n\n## Big\n\n{paragraphs}"))
        assert len(chunks) > 1
        # Overlap means adjacent chunks share some paragraph text.
        assert any(set(a.text.split()) & set(b.text.split()) for a, b in itertools.pairwise(chunks))

    def test_every_section_survives_chunking(self) -> None:
        chunks = chunk_document(make_document(INCIDENT_BODY))
        combined = " ".join(c.text for c in chunks)
        for heading in ("Summary", "Impact", "Root Cause", "Remediation", "Review"):
            assert heading in combined


class TestMetadata:
    def test_security_fields_are_present_on_every_chunk(self) -> None:
        """These two fields are what the authorization filter matches on."""
        for chunk in chunk_document(make_document(INCIDENT_BODY)):
            assert chunk.metadata.access_level == "confidential"
            assert chunk.metadata.department == "payments"

    def test_created_ts_orders_the_same_way_as_created_date(self) -> None:
        early = chunk_document(make_document(INCIDENT_BODY))[0]
        document = make_document(INCIDENT_BODY, doc_id="INC-2026-0002")
        document.frontmatter.created_date = date(2026, 9, 1)
        late = chunk_document(document)[0]
        assert late.metadata.created_ts > early.metadata.created_ts

    @pytest.mark.parametrize(("text", "expected"), [("", 1), ("abcd", 1), ("a" * 400, 100)])
    def test_token_estimate(self, text: str, expected: int) -> None:
        assert estimate_tokens(text) == expected
