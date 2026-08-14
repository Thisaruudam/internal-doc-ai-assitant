"""Egress scanning — the last gate before an answer reaches the user.

Ingress scanning assumes an attacker is trying to get instructions *in*. This
assumes they have already succeeded and are trying to get data *out*, or that the
model has simply gone wrong in a way that matters for a bank.

Four checks:

* **Exfiltration channels.** A model that has been successfully injected does not
  attack the user directly; it embeds their data in a URL and relies on the
  client to fetch it. A markdown image pointing at an attacker's host renders
  automatically and leaks its query string without anyone clicking anything —
  which is why link stripping alone is not sufficient.

* **Secret material.** Keys, tokens, and connection strings that reached the
  context through a document or a tool result must not be relayed onward.

* **Verbatim bulk reproduction.** Retrieval answers questions; it does not
  license reprinting a document. A response that is mostly one passage copied
  out is an extraction attack whether or not the reader was entitled to that
  passage.

* **Brand policy.** The system speaks as Commercial Bank. Some statements are not
  unsafe in a general sense but are unacceptable from a bank: committing to a
  rate, giving individual financial advice, or disparaging a competitor.

Findings are separated into *blocking* and *advisory* because over-blocking has
a real cost — an assistant that refuses a legitimate answer teaches people not to
use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.observability.logging import get_logger

log = get_logger(__name__)


class EgressFinding(StrEnum):
    EXFILTRATION_URL = "exfiltration_url"
    MARKDOWN_IMAGE_BEACON = "markdown_image_beacon"
    #: Names a finding category, not a credential.
    SECRET_MATERIAL = "secret_material"  # noqa: S105
    BULK_REPRODUCTION = "bulk_reproduction"
    FINANCIAL_ADVICE = "financial_advice"
    RATE_COMMITMENT = "rate_commitment"
    COMPETITOR_DISPARAGEMENT = "competitor_disparagement"
    UNAUTHORISED_UNDERTAKING = "unauthorised_undertaking"


@dataclass(frozen=True, slots=True)
class Rule:
    finding: EgressFinding
    regex: re.Pattern[str]
    blocking: bool
    message: str


def _rule(finding: EgressFinding, pattern: str, *, blocking: bool, message: str) -> Rule:
    return Rule(finding, re.compile(pattern, re.IGNORECASE), blocking, message)


#: A zero-click leak: the client renders the image and the query string travels
#: with the request. Blocking, always.
_IMAGE_BEACON = _rule(
    EgressFinding.MARKDOWN_IMAGE_BEACON,
    r"!\[[^\]]*\]\(\s*https?://[^)]+\)",
    blocking=True,
    message="answer contains a remote image, which would leak data on render",
)

SECRET_RULES: tuple[Rule, ...] = (
    _rule(
        EgressFinding.SECRET_MATERIAL,
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        blocking=True,
        message="answer contains a private key",
    ),
    _rule(
        EgressFinding.SECRET_MATERIAL,
        r"\b(?:sk|pk)-[A-Za-z0-9]{20,}\b",
        blocking=True,
        message="answer contains an API key",
    ),
    _rule(
        EgressFinding.SECRET_MATERIAL,
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
        blocking=True,
        message="answer contains a JSON web token",
    ),
    _rule(
        EgressFinding.SECRET_MATERIAL,
        r"\b(?:postgres|postgresql|mysql|mongodb|redis)://[^\s:]+:[^\s@]+@",
        blocking=True,
        message="answer contains a connection string with credentials",
    ),
    _rule(
        EgressFinding.SECRET_MATERIAL,
        r"\bAKIA[0-9A-Z]{16}\b",
        blocking=True,
        message="answer contains an AWS access key id",
    ),
)

BRAND_RULES: tuple[Rule, ...] = (
    _rule(
        EgressFinding.RATE_COMMITMENT,
        r"\b(?:we\s+(?:will|can)\s+(?:offer|give|guarantee)|"
        r"you\s+(?:will|are\s+guaranteed\s+to)\s+(?:receive|get))\b[^.]{0,60}"
        r"\d+(?:\.\d+)?\s*%",
        blocking=True,
        message="answer commits to a specific rate",
    ),
    _rule(
        EgressFinding.FINANCIAL_ADVICE,
        r"\byou\s+should\s+(?:invest|buy|sell|switch|move\s+your\s+(?:money|savings)|"
        r"take\s+out\s+(?:a\s+)?loan)\b",
        blocking=True,
        message="answer gives individual financial advice",
    ),
    _rule(
        EgressFinding.FINANCIAL_ADVICE,
        r"\b(?:this|it)\s+is\s+(?:a\s+)?(?:guaranteed|risk-free|sure)\s+"
        r"(?:investment|return|profit)\b",
        blocking=True,
        message="answer characterises an investment as risk-free",
    ),
    _rule(
        EgressFinding.COMPETITOR_DISPARAGEMENT,
        r"\b(?:better|safer|cheaper)\s+than\s+(?:any\s+)?other\s+bank",
        blocking=False,
        message="answer makes a comparative claim about competitors",
    ),
    _rule(
        EgressFinding.UNAUTHORISED_UNDERTAKING,
        r"\b(?:i|we)\s+(?:will|have)\s+(?:refund|reverse|waive|credit|cancel)"
        r"(?:ed|\s+your)\b",
        blocking=True,
        message="answer undertakes an action on a customer's account",
    ),
)


@dataclass
class EgressReport:
    findings: list[tuple[EgressFinding, str]] = field(default_factory=list)
    blocking: list[tuple[EgressFinding, str]] = field(default_factory=list)
    #: Fraction of the answer that is a verbatim run from one source passage.
    max_verbatim_ratio: float = 0.0

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def must_block(self) -> bool:
        return bool(self.blocking)

    def summary(self) -> str:
        return "; ".join(message for _, message in self.findings)


def _longest_common_run(answer: str, source: str) -> int:
    """Length of the longest shared word run, in words.

    Compares word sequences rather than characters: character-level matching is
    dominated by shared punctuation and whitespace, and would flag ordinary
    paraphrase.
    """
    answer_words = answer.lower().split()
    source_words = source.lower().split()
    if not answer_words or not source_words:
        return 0

    source_index: dict[str, list[int]] = {}
    for position, word in enumerate(source_words):
        source_index.setdefault(word, []).append(position)

    best = 0
    # Bounded scan: long answers are already truncated upstream, and this keeps
    # the check linear enough to run on every turn.
    for start in range(len(answer_words)):
        if best >= len(answer_words) - start:
            break
        for source_start in source_index.get(answer_words[start], ())[:32]:
            run = 0
            while (
                start + run < len(answer_words)
                and source_start + run < len(source_words)
                and answer_words[start + run] == source_words[source_start + run]
            ):
                run += 1
            best = max(best, run)
    return best


def scan_egress(
    answer: str,
    *,
    source_passages: dict[str, str] | None = None,
    verbatim_threshold: float = 0.5,
) -> EgressReport:
    """Scan a drafted answer before it is released.

    ``source_passages`` is the retrieved evidence. Supplying it enables the bulk
    reproduction check; without it the other three still run.
    """
    report = EgressReport()

    def record(rule: Rule) -> None:
        report.findings.append((rule.finding, rule.message))
        if rule.blocking:
            report.blocking.append((rule.finding, rule.message))

    if _IMAGE_BEACON.regex.search(answer):
        record(_IMAGE_BEACON)

    for rule in (*SECRET_RULES, *BRAND_RULES):
        if rule.regex.search(answer):
            record(rule)

    # ── Bulk reproduction ───────────────────────────────────────────────
    if source_passages:
        answer_word_count = len(answer.split())
        if answer_word_count >= 40:
            longest = max(
                (_longest_common_run(answer, text) for text in source_passages.values()),
                default=0,
            )
            report.max_verbatim_ratio = longest / answer_word_count
            if report.max_verbatim_ratio >= verbatim_threshold:
                report.findings.append(
                    (
                        EgressFinding.BULK_REPRODUCTION,
                        f"{report.max_verbatim_ratio:.0%} of the answer is copied "
                        f"verbatim from a single passage",
                    )
                )
                report.blocking.append(
                    (
                        EgressFinding.BULK_REPRODUCTION,
                        "answer reproduces a source document rather than answering",
                    )
                )

    if report.findings:
        log.warning(
            "egress_findings",
            findings=[f.value for f, _ in report.findings],
            blocking=bool(report.blocking),
        )

    return report


def redact(answer: str) -> str:
    """Neutralise what can be neutralised, for advisory findings.

    Used when an answer is otherwise sound: remote images are defanged in place
    rather than discarding a good response over one bad link.
    """
    return _IMAGE_BEACON.regex.sub("[image removed by policy]", answer)


BLOCKED_ANSWER = (
    "I prepared an answer but it did not pass the outbound safety checks, so I "
    "have withheld it. This is usually caused by content in the source documents "
    "rather than by your question. The correlation id below will let someone "
    "review what happened."
)
