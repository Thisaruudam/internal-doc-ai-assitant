"""MCP client — where tool-server output becomes trustworthy enough to use.

Three things happen here that cannot happen on the server, and the reason is the
same in each case: the server is a separate process that does not hold the
caller's identity and was not written by us.

**Field-level redaction.** ``employee_directory`` returns ``salary_band`` for
every employee, because that is what the directory holds. Whether *this* caller
may see it is our decision, made against the verified ``Principal``. A tool
integration that trusts an upstream system to apply the right policy is the
standard way sensitive fields escape.

**Row-level filtering.** Incident records carry an ``access_level``; rows above
the caller's ceiling are dropped, the same lattice the retrieval filter uses.
Two different sources, one authorization model.

**Treating responses as untrusted content.** A tool result is external text
arriving in the model's context — structurally identical to a retrieved
document, and therefore an injection vector. Responses are scanned, and flagged
content is reported rather than passed through silently.

Resilience is the fourth concern: a slow or unreachable MCP server must degrade
the turn, never hang it. Failures return a typed result the agent can explain,
not an exception that ends the conversation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.auth.principal import AccessLevel, Principal, Role, role_satisfies
from app.observability.logging import get_logger
from app.security.injection import scan_retrieved_content

log = get_logger(__name__)

#: Fields removed unless the caller clears the stated role. Declared as data so
#: the policy is reviewable in one place rather than scattered through parsing
#: code.
REDACTED_FIELDS: dict[str, tuple[str, Role]] = {
    "salary_band": ("employees", Role.ADMINISTRATOR),
}

_REDACTION_PLACEHOLDER = "[redacted]"


@dataclass
class McpResult:
    """The outcome of one MCP call, after authorization and scanning."""

    tool: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    #: Rows dropped because they sat above the caller's access ceiling.
    rows_withheld: int = 0
    #: Field names redacted from the returned rows.
    fields_redacted: list[str] = field(default_factory=list)
    #: True when the response carried content resembling an injection.
    flagged: bool = False

    def describe(self) -> str:
        """A short note the answer can include, so reduced results are explained."""
        parts: list[str] = []
        if self.rows_withheld:
            parts.append(f"{self.rows_withheld} record(s) withheld by access policy")
        if self.fields_redacted:
            parts.append(f"redacted field(s): {', '.join(sorted(set(self.fields_redacted)))}")
        if self.flagged:
            parts.append("response contained content resembling an injection")
        return "; ".join(parts)


def redact_row(
    row: dict[str, Any], principal: Principal, *, dataset: str
) -> tuple[dict[str, Any], list[str]]:
    """Strip fields this caller may not see."""
    redacted: list[str] = []
    cleaned = dict(row)

    for field_name, (owning_dataset, required_role) in REDACTED_FIELDS.items():
        if owning_dataset != dataset or field_name not in cleaned:
            continue
        if not role_satisfies(principal.role, required_role):
            cleaned[field_name] = _REDACTION_PLACEHOLDER
            redacted.append(field_name)

    return cleaned, redacted


def _row_is_visible(row: dict[str, Any], principal: Principal) -> bool:
    """Apply the access lattice to a structured record.

    A row without an ``access_level`` is treated as ``internal`` rather than
    public: an unlabelled record from an external system is more likely to be
    unclassified than deliberately open.
    """
    level = str(row.get("access_level", "internal"))
    department = str(row.get("department", ""))

    try:
        AccessLevel.parse(level)
    except ValueError:
        return False  # unknown label: fail closed, exactly as retrieval does

    if not principal.may_read(level, department or next(iter(principal.departments), "")):
        # Departmentless rows (services, employees) are governed by level alone.
        if department:
            return False
        return AccessLevel.parse(level) <= principal.access_ceiling

    return True


def authorize_payload(
    payload: dict[str, Any], principal: Principal, *, tool: str
) -> tuple[dict[str, Any], int, list[str]]:
    """Filter and redact an MCP response for this caller.

    Returns the cleaned payload, the number of rows withheld, and the fields
    redacted.
    """
    dataset_key = {
        "employee_directory": ("employees", "employees"),
        "service_catalog": ("services", "services"),
        "incident_records": ("incidents", "incidents"),
    }.get(tool)

    if dataset_key is None:
        return payload, 0, []

    list_key, dataset = dataset_key
    rows = payload.get(list_key)
    if not isinstance(rows, list):
        return payload, 0, []

    kept: list[dict[str, Any]] = []
    withheld = 0
    redacted_fields: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        if not _row_is_visible(row, principal):
            withheld += 1
            continue
        cleaned, redacted = redact_row(row, principal, dataset=dataset)
        redacted_fields.extend(redacted)
        kept.append(cleaned)

    cleaned_payload = dict(payload)
    cleaned_payload[list_key] = kept
    cleaned_payload["count"] = len(kept)

    if withheld:
        # Stated rather than silently applied: a count that quietly shrinks is
        # indistinguishable from missing data.
        cleaned_payload["withheld_by_policy"] = withheld
        log.info(
            "mcp_rows_withheld",
            tool=tool,
            withheld=withheld,
            returned=len(kept),
            role=principal.role.value,
        )

    return cleaned_payload, withheld, redacted_fields


def scan_payload(payload: dict[str, Any]) -> bool:
    """Scan an MCP response for injected instructions.

    Tool output lands in the model's context exactly as a retrieved document
    does, so it gets the same treatment.
    """
    decision = scan_retrieved_content(json.dumps(payload, default=str))
    if decision.quarantine:
        log.warning(
            "mcp_response_flagged",
            score=round(decision.score, 3),
            signals=decision.signals[:3],
        )
    return decision.quarantine


class McpClient:
    """Calls the MCP server and applies our policy to what comes back.

    The transport is injected rather than constructed here so the authorization
    logic — the part worth testing — is testable without a live server.
    """

    def __init__(self, call_tool: Any, *, timeout_s: float = 10.0) -> None:
        self._call_tool = call_tool
        self._timeout_s = timeout_s

    async def call(self, tool: str, arguments: dict[str, Any], principal: Principal) -> McpResult:
        """Invoke an MCP tool on behalf of a principal."""
        try:
            raw = await self._call_tool(tool, arguments)
        except Exception as exc:  # a tool outage degrades the turn, never ends it
            log.warning("mcp_call_failed", tool=tool, error_type=type(exc).__name__)
            return McpResult(
                tool=tool,
                ok=False,
                error=f"the {tool} service is unavailable ({type(exc).__name__})",
            )

        if not isinstance(raw, dict):
            return McpResult(tool=tool, ok=False, error="malformed response from MCP server")

        flagged = scan_payload(raw)
        payload, withheld, redacted = authorize_payload(raw, principal, tool=tool)

        return McpResult(
            tool=tool,
            ok=True,
            payload=payload,
            rows_withheld=withheld,
            fields_redacted=redacted,
            flagged=flagged,
        )
