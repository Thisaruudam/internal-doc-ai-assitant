"""Prompt-injection detection.

Two entry points, because the two threats are genuinely different and conflating
them produces a system that is both annoying and unsafe.

**Direct injection** — the user types something hostile. They are authenticated,
their reach is already bounded by their role, and the worst case is that they
talk the assistant into a policy violation. High-confidence hits are refused.

**Indirect injection** — hostile text arrives *inside a retrieved document*. This
is the dangerous one, and the one this corpus plants a fixture for
(``MTG-PLA-999``): the user asks an ordinary question, and a document written by
someone else carries instructions aimed at the model. The user did nothing wrong,
so refusing their turn is the wrong response. Instead the content is quarantined —
wrapped in delimiters that mark it as data, with the injected span reported to the
activity panel — and the turn proceeds.

The layers, cheapest first:

1. **Normalise.** Unicode-tag characters, zero-width joiners, and homoglyphs are
   the standard way to smuggle text past a naive matcher. Folded before matching.
2. **Decode.** Base64 and percent-encoded payloads are decoded and rescanned, so
   an instruction hidden one encoding deep is still seen.
3. **Match.** Pattern families with weights, not one flat blocklist, so the score
   reflects *what kind* of attack this looks like.

An LLM classifier sits above this for ambiguous cases; these heuristics run first
because they are free, deterministic, and catch the overwhelming majority.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

from app.observability.logging import get_logger

log = get_logger(__name__)


class InjectionFamily(StrEnum):
    """What an attack is trying to do. Reported so a refusal can explain itself."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_MANIPULATION = "role_manipulation"
    ACCESS_ESCALATION = "access_escalation"
    EXFILTRATION = "exfiltration"
    TOOL_ABUSE = "tool_abuse"
    CONCEALMENT = "concealment"
    ENCODED_PAYLOAD = "encoded_payload"


@dataclass(frozen=True, slots=True)
class Pattern:
    family: InjectionFamily
    regex: re.Pattern[str]
    weight: float
    label: str


def _p(family: InjectionFamily, pattern: str, weight: float, label: str) -> Pattern:
    # Patterns are kept anchor-free and bounded — no nested quantifiers — so a
    # hostile input cannot turn detection itself into a denial of service.
    return Pattern(family, re.compile(pattern, re.IGNORECASE), weight, label)


PATTERNS: tuple[Pattern, ...] = (
    # ── Instruction override ────────────────────────────────────────────
    _p(
        InjectionFamily.INSTRUCTION_OVERRIDE,
        r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|earlier|above)\s+"
        r"(?:instructions?|prompts?|rules?|directions?)",
        0.85,
        "ignore previous instructions",
    ),
    _p(
        InjectionFamily.INSTRUCTION_OVERRIDE,
        r"\bdisregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|system)",
        0.8,
        "disregard prior context",
    ),
    _p(
        InjectionFamily.INSTRUCTION_OVERRIDE,
        r"\bforget\s+(?:everything|all|your)\s+(?:instructions?|rules?|training|above)",
        0.8,
        "forget instructions",
    ),
    _p(
        InjectionFamily.INSTRUCTION_OVERRIDE,
        r"\b(?:new|updated|revised)\s+(?:system\s+)?(?:instructions?|prompt|directive)s?\s*[:\-]",
        0.7,
        "new instructions header",
    ),
    _p(
        InjectionFamily.INSTRUCTION_OVERRIDE,
        r"\b(?:system|important)\s+(?:notice|message|override|alert)\b",
        0.4,
        "spoofed system notice",
    ),
    # ── Role manipulation ───────────────────────────────────────────────
    _p(
        InjectionFamily.ROLE_MANIPULATION,
        r"\byou\s+are\s+now\s+(?:in\s+|an?\s+)?(?:unrestricted|admin|developer|god|dan\b)",
        0.85,
        "role reassignment",
    ),
    _p(
        InjectionFamily.ROLE_MANIPULATION,
        r"\b(?:administrator|admin|developer|debug|god)\s+mode\b",
        0.7,
        "privileged mode claim",
    ),
    _p(
        InjectionFamily.ROLE_MANIPULATION,
        r"\bact\s+as\s+(?:if\s+you\s+are\s+)?(?:an?\s+)?(?:unrestricted|jailbroken|uncensored)",
        0.8,
        "unrestricted persona",
    ),
    # ── Access escalation ───────────────────────────────────────────────
    _p(
        InjectionFamily.ACCESS_ESCALATION,
        r"\b(?:disregard|ignore|bypass|override|remove)\s+(?:any\s+|all\s+|the\s+)?"
        r"(?:access|permission|authorization|security)\s*(?:level|control|restriction)?s?",
        0.9,
        "bypass access control",
    ),
    _p(
        InjectionFamily.ACCESS_ESCALATION,
        r"\b(?:list|show|reveal|dump|output)\s+(?:me\s+)?(?:every|all)\s+"
        r"(?:restricted|confidential|classified|secret|private)\b",
        0.85,
        "enumerate restricted material",
    ),
    _p(
        InjectionFamily.ACCESS_ESCALATION,
        r"\b(?:salary|compensation|payroll)\s+(?:records?|data|information|bands?)\b",
        0.35,
        "sensitive record request",
    ),
    # ── Exfiltration ────────────────────────────────────────────────────
    _p(
        InjectionFamily.EXFILTRATION,
        r"!\[[^\]]*\]\(\s*https?://[^)]*[?&][^)]*=",
        0.9,
        "markdown image exfiltration",
    ),
    _p(
        InjectionFamily.EXFILTRATION,
        r"\b(?:send|post|upload|transmit|forward|exfiltrate)\s+(?:it\s+|them\s+|the\s+)?"
        r"(?:results?|contents?|data|summary|documents?)?\s*to\s+https?://",
        0.9,
        "send data to external URL",
    ),
    _p(
        InjectionFamily.EXFILTRATION,
        r"https?://[^\s)]{0,200}[?&](?:data|q|payload|content|body)=",
        0.6,
        "URL with data parameter",
    ),
    # ── Tool abuse ──────────────────────────────────────────────────────
    _p(
        InjectionFamily.TOOL_ABUSE,
        r"\b(?:call|invoke|run|execute|use)\s+(?:the\s+)?(?:admin_\w+|admin\s+tool)",
        0.8,
        "administrative tool invocation",
    ),
    _p(
        InjectionFamily.TOOL_ABUSE,
        r"\b(?:delete|purge|drop|wipe|destroy)\s+(?:all\s+|every\s+|the\s+)?"
        r"(?:records?|data|memory|index|documents?)\b",
        0.7,
        "destructive operation",
    ),
    # ── Concealment ─────────────────────────────────────────────────────
    # On its own, weak. Combined with anything else it is a strong signal:
    # legitimate documents do not ask the reader to hide what they just read.
    _p(
        InjectionFamily.CONCEALMENT,
        r"\bdo\s+not\s+(?:mention|tell|inform|reveal|disclose|report)\s+"
        r"(?:this|these|them|it|the\s+user|anyone)",
        0.75,
        "instruction to conceal",
    ),
    _p(
        InjectionFamily.CONCEALMENT,
        r"\bwithout\s+(?:telling|informing|notifying|alerting)\s+(?:the\s+)?(?:user|anyone)",
        0.75,
        "act without informing user",
    ),
)

#: Characters used to smuggle text past naive matching. Unicode tag block
#: (U+E0000–U+E007F) is invisible in most renderers and survives copy-paste.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿\U000e0000-\U000e007f]")

_BASE64_CANDIDATE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_PERCENT_ENCODED = re.compile(r"(?:%[0-9A-Fa-f]{2}){6,}")


@dataclass
class InjectionReport:
    """What the scan found."""

    score: float = 0.0
    families: set[InjectionFamily] = field(default_factory=set)
    signals: list[str] = field(default_factory=list)
    #: Set when a hit was only visible after decoding a nested payload.
    decoded_hit: bool = False

    @property
    def clean(self) -> bool:
        return self.score == 0.0

    def merge(self, other: InjectionReport, *, weight: float = 1.0) -> None:
        self.score = max(self.score, other.score * weight)
        self.families |= other.families
        self.signals.extend(other.signals)
        self.decoded_hit = self.decoded_hit or other.decoded_hit


def normalise(text: str) -> str:
    """Fold away the usual evasion tricks before matching.

    NFKC collapses homoglyphs and full-width forms; invisible control and tag
    characters are removed outright. Without this, ``ｉｇｎｏｒｅ　ａｌｌ`` and
    ``ig​nore all`` both sail past a plain pattern match.
    """
    folded = unicodedata.normalize("NFKC", text)
    return _INVISIBLE.sub("", folded)


def _decode_candidates(text: str) -> list[str]:
    """Extract and decode plausibly-encoded payloads for a second pass."""
    decoded: list[str] = []

    for match in _BASE64_CANDIDATE.findall(text)[:20]:
        try:
            raw = base64.b64decode(match, validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            candidate = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        # Only keep decodings that look like prose; random bytes that happen to
        # decode are noise.
        if sum(c.isprintable() for c in candidate) / max(len(candidate), 1) > 0.9:
            decoded.append(candidate)

    for match in _PERCENT_ENCODED.findall(text)[:20]:
        try:
            from urllib.parse import unquote

            decoded.append(unquote(match))
        except (ValueError, UnicodeDecodeError):
            continue

    return decoded


def _match_patterns(text: str) -> InjectionReport:
    report = InjectionReport()
    for pattern in PATTERNS:
        if pattern.regex.search(text):
            report.families.add(pattern.family)
            report.signals.append(pattern.label)
            report.score = max(report.score, pattern.weight)
    return report


def scan(text: str) -> InjectionReport:
    """Scan one piece of text, including anything it encodes."""
    if not text or not text.strip():
        return InjectionReport()

    normalised = normalise(text)
    report = _match_patterns(normalised)

    for payload in _decode_candidates(normalised):
        nested = _match_patterns(normalise(payload))
        if not nested.clean:
            nested.decoded_hit = True
            nested.signals = [f"{s} (encoded)" for s in nested.signals]
            nested.families.add(InjectionFamily.ENCODED_PAYLOAD)
            # Hiding an instruction inside an encoding is itself evidence of
            # intent, so a decoded hit scores higher than the same text in the
            # clear.
            report.merge(nested, weight=1.1)

    # Concealment alone is weak; concealment plus any other family is a strong
    # signal, because a document with a legitimate reason to say "do not mention
    # this" does not also try to override instructions.
    if InjectionFamily.CONCEALMENT in report.families and len(report.families) > 1:
        report.score = min(1.0, report.score + 0.15)
        report.signals.append("concealment combined with another attack family")

    report.score = min(1.0, report.score)
    return report


@dataclass(frozen=True, slots=True)
class ScanDecision:
    """The outcome of scanning, and what the caller should do about it."""

    report: InjectionReport
    blocked: bool
    quarantine: bool

    @property
    def score(self) -> float:
        return self.report.score

    @property
    def signals(self) -> list[str]:
        return self.report.signals


def scan_user_input(
    text: str, *, block_threshold: float = 0.7, warn_threshold: float = 0.4
) -> ScanDecision:
    """Scan text the authenticated user typed.

    Blocking is reasonable here: the user is the author, and a refusal is
    addressed to the person responsible for the content.
    """
    report = scan(text)
    if report.score >= block_threshold:
        log.warning(
            "injection_blocked",
            source="user_input",
            score=round(report.score, 3),
            families=sorted(f.value for f in report.families),
        )
        return ScanDecision(report=report, blocked=True, quarantine=False)

    return ScanDecision(report=report, blocked=False, quarantine=report.score >= warn_threshold)


def scan_retrieved_content(text: str, *, quarantine_threshold: float = 0.4) -> ScanDecision:
    """Scan text that came out of the corpus.

    Never blocks the turn. The user asked an ordinary question; a hostile
    document is not their fault, and refusing would let anyone who can write to
    the corpus deny service to everyone else. Hostile content is quarantined and
    reported instead.
    """
    report = scan(text)
    quarantine = report.score >= quarantine_threshold

    if quarantine:
        log.warning(
            "injection_in_retrieved_content",
            score=round(report.score, 3),
            families=sorted(f.value for f in report.families),
        )

    return ScanDecision(report=report, blocked=False, quarantine=quarantine)
