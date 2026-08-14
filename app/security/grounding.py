"""Citation grounding.

Enforces the principle that an unsupported claim is a bug, not a stylistic
preference. The validator node runs this against every drafted answer and either
passes it, sends it back for repair, or downgrades it to an explicit
"insufficient evidence" response.

Three checks, in descending order of how badly they fail:

1. **Fabricated citations.** The answer cites a ``chunk_id`` that was never
   retrieved. This is unambiguous and non-negotiable — the model invented a
   source. Always a hard failure.

2. **Unsupported numbers.** A claim states a figure that appears nowhere in the
   passage it cites. Checked separately from prose because in a banking context
   a wrong number is the most expensive thing the system can emit, and because
   numbers are exactly comparable in a way that sentences are not. "18,000
   transactions failed" is either in the evidence or it is not.

3. **Uncited claims.** A factual assertion carries no citation at all.

Prose support is checked lexically rather than semantically. That is a
deliberate limit: a cheap deterministic proxy catches the case that actually
occurs — a citation stapled to an unrelated claim — without the cost and
non-determinism of a second model call on every turn. Genuine paraphrase
detection is left to the LLM validator that runs above this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.observability.logging import get_logger

log = get_logger(__name__)

#: Citations are written as [DOC-ID#000]. Bracketed rather than footnoted so
#: they survive markdown rendering and are trivially machine-checkable.
CITATION = re.compile(r"\[([A-Za-z0-9][\w.\-]*#\d{1,4})\]")

#: Sentence splitter. Deliberately simple — abbreviations and decimals are
#: handled by requiring a following capital or end-of-string, which is enough
#: for the prose these models produce.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}")

#: Numbers worth checking: integers with separators, decimals, percentages,
#: currency amounts, and durations. Bare small integers are excluded — "the 3
#: services" is not the kind of figure worth a grounding failure.
_NUMERIC_CLAIM = re.compile(
    r"(?<![\w.])(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 18,000  1,234,567.89
    r"|\d+\.\d+"  # 99.95
    r"|\d{3,}"  # 1200
    r")(?![\w])"
)

_PERCENTAGE = re.compile(r"\d+(?:\.\d+)?\s*%")

#: Sentences that assert nothing and therefore need no citation.
_NON_CLAIM_PREFIXES = (
    "i could not",
    "i was unable",
    "i do not have",
    "no documents",
    "no evidence",
    "based on the",
    "according to the retrieved",
    "the retrieved documents",
    "here is",
    "here are",
    "in summary",
    "to summarise",
    "to summarize",
    "this answer",
    "note that",
    "you may not have access",
    "your role does not",
)

#: Words that make a sentence a factual assertion worth grounding. A sentence
#: with none of these is usually connective tissue.
_HEDGE_ONLY = re.compile(
    r"^\s*(?:however|additionally|furthermore|in addition|finally|also|"
    r"overall|therefore|as a result)[,\s]",
    re.IGNORECASE,
)

#: Function words carry no evidential weight, so they are excluded before
#: measuring overlap between a claim and the passage it cites.
_STOPWORD_SOURCE = (
    "a an and are as at be been but by for from had has have he her his if in into is it "
    "its of on or that the their there these they this to was were what when where which "
    "who will with would you your our we us"
)
_STOPWORDS = frozenset(_STOPWORD_SOURCE.split())


@dataclass
class ClaimIssue:
    """One problem found in a drafted answer."""

    kind: str
    sentence: str
    detail: str

    def render(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass
class GroundingReport:
    grounded_claims: int = 0
    total_claims: int = 0
    fabricated_citations: list[str] = field(default_factory=list)
    issues: list[ClaimIssue] = field(default_factory=list)
    cited_chunk_ids: set[str] = field(default_factory=set)

    @property
    def ungrounded_claims(self) -> int:
        return len(self.issues)

    @property
    def passed(self) -> bool:
        """A fabricated citation always fails; other issues fail too.

        There is no partial credit here. The repair loop exists precisely so
        that a first draft can fail without the turn failing.
        """
        return not self.fabricated_citations and not self.issues

    def repair_instructions(self) -> str:
        """Feedback for the response agent's next attempt.

        Written as concrete corrections rather than "try again", because a
        general instruction reliably produces the same answer a second time.
        """
        lines: list[str] = []
        if self.fabricated_citations:
            lines.append(
                "You cited passages that were never retrieved: "
                f"{', '.join(sorted(self.fabricated_citations))}. "
                "Only cite ids present in the evidence block."
            )
        for issue in self.issues[:6]:
            lines.append(f"- {issue.render()}")
        lines.append(
            "Rewrite the answer so every factual statement cites a passage that "
            "contains it. If the evidence does not support a statement, remove "
            "the statement rather than the citation."
        )
        return "\n".join(lines)


def extract_citations(text: str) -> list[str]:
    """Every ``[chunk_id]`` marker in order of appearance."""
    return CITATION.findall(text)


def _significant_terms(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z\-]{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _numbers_in(text: str) -> set[str]:
    """Normalised numeric tokens, so 18,000 and 18000 compare equal."""
    found = {match.replace(",", "") for match in _NUMERIC_CLAIM.findall(text)}
    found |= {m.replace(" ", "") for m in _PERCENTAGE.findall(text)}
    return found


def _is_claim(sentence: str) -> bool:
    """Whether a sentence asserts something that needs evidence."""
    stripped = sentence.strip()
    if len(stripped) < 25:
        return False

    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in _NON_CLAIM_PREFIXES):
        return False

    # A bare connective with no content of its own.
    if _HEDGE_ONLY.match(stripped) and len(_significant_terms(stripped)) < 4:
        return False

    # Headings and bullet labels are structure, not assertions.
    return not (stripped.startswith("#") or stripped.endswith(":"))


def check_grounding(
    answer: str,
    *,
    retrieved_chunks: dict[str, str],
    minimum_term_overlap: float = 0.12,
) -> GroundingReport:
    """Validate an answer against the evidence that was actually retrieved.

    ``retrieved_chunks`` maps ``chunk_id`` to that chunk's text. It is built from
    the retrieval result, not from the answer, so the model cannot expand its own
    permitted citation set.
    """
    report = GroundingReport()

    all_citations = set(extract_citations(answer))
    report.cited_chunk_ids = all_citations

    # ── Check 1: fabricated citations ───────────────────────────────────
    report.fabricated_citations = sorted(all_citations - retrieved_chunks.keys())
    if report.fabricated_citations:
        log.warning(
            "fabricated_citations",
            count=len(report.fabricated_citations),
            ids=report.fabricated_citations[:5],
        )

    sentences = [s.strip() for s in _SENTENCE.split(answer) if s.strip()]

    for sentence in sentences:
        if not _is_claim(sentence):
            continue

        report.total_claims += 1
        citations = extract_citations(sentence)

        # Analyse the prose only. Citation markers carry digits of their own —
        # [INC-2026-0014#001] contributes "2026", "0014" and "001" — which would
        # otherwise be read as unsupported figures and fail every cited answer.
        claim_text = CITATION.sub(" ", sentence)

        # ── Check 3: uncited claim ──────────────────────────────────────
        if not citations:
            report.issues.append(
                ClaimIssue(
                    kind="uncited_claim",
                    sentence=sentence,
                    detail=f"no citation for: {_truncate(sentence)}",
                )
            )
            continue

        supporting_text = " ".join(retrieved_chunks.get(chunk_id, "") for chunk_id in citations)
        if not supporting_text.strip():
            # Every citation on this sentence was fabricated; already recorded.
            continue

        # ── Check 2: unsupported numbers ────────────────────────────────
        claim_numbers = _numbers_in(claim_text)
        evidence_numbers = _numbers_in(supporting_text)
        missing = claim_numbers - evidence_numbers
        if missing:
            report.issues.append(
                ClaimIssue(
                    kind="unsupported_number",
                    sentence=sentence,
                    detail=(
                        f"the figure(s) {', '.join(sorted(missing))} do not appear "
                        f"in {', '.join(citations)}"
                    ),
                )
            )
            continue

        # ── Lexical support ─────────────────────────────────────────────
        claim_terms = _significant_terms(claim_text)
        if claim_terms:
            overlap = len(claim_terms & _significant_terms(supporting_text)) / len(claim_terms)
            if overlap < minimum_term_overlap:
                report.issues.append(
                    ClaimIssue(
                        kind="weak_support",
                        sentence=sentence,
                        detail=(
                            f"{', '.join(citations)} shares little wording with: "
                            f"{_truncate(sentence)}"
                        ),
                    )
                )
                continue

        report.grounded_claims += 1

    return report


def _truncate(text: str, limit: int = 90) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


INSUFFICIENT_EVIDENCE = (
    "I could not find enough supporting evidence in the documents available to "
    "you to answer that reliably. This may be because the relevant material sits "
    "outside your access level, or because it is not in the knowledge base."
)


def insufficient_evidence_answer(*, role: str | None = None) -> str:
    """The answer given when repair fails.

    Saying "I don't know" is the correct output, and it is deliberately phrased
    so it does not confirm whether restricted material exists — a viewer being
    told "there is a confidential document you cannot see" is itself a leak.
    """
    if role:
        return f"{INSUFFICIENT_EVIDENCE} You are signed in with the {role} role."
    return INSUFFICIENT_EVIDENCE
