"""Tool catalogue.

Every tool declares what it needs from a caller before it can be invoked:
which role, how risky it is, how long it may run, and whether it is safe to
retry. Those declarations are what the three authorization layers act on.

The registry itself implements **layer 1 — binding**. ``tools_for`` returns only
the tools a principal may use, and the graph binds exactly that set to the model.
A tool the caller is not entitled to is not merely refused; it is absent from the
schema the model sees, so there is no name for it to emit. That is a meaningfully
stronger position than refusing afterwards: the model cannot be talked into
calling something it does not know exists.

Layer 2 (``app.tools.guard``) re-checks at invocation, because binding assumes
the graph wired itself correctly. Layer 3 (``app.retrieval.filters``) filters the
data itself and holds even if both of the others are bypassed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from app.auth.principal import Principal, Role, role_satisfies


class RiskLevel(StrEnum):
    """How much damage a tool could do if invoked wrongly.

    Distinct from ``required_role``: seniority says who *may* call a tool, risk
    says whether a human should confirm it first. An administrator calling a
    high-risk tool still goes through the approval node.
    """

    #: Read-only, no side effects, no sensitive egress.
    LOW = "low"
    #: Reads sensitive data or consumes significant budget.
    MEDIUM = "medium"
    #: Mutates state, or acts outside the system. Always human-approved.
    HIGH = "high"


#: A tool handler: validated arguments plus the calling principal, returning a
#: JSON-serialisable result. The principal is passed rather than looked up so a
#: handler cannot accidentally operate under different authority than the
#: guard checked.
ToolHandler = Callable[[BaseModel, Principal], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool and the conditions under which it may run."""

    name: str
    description: str
    args_schema: type[BaseModel]
    handler: ToolHandler

    #: Minimum role. Checked at binding and again at invocation.
    required_role: Role
    risk_level: RiskLevel = RiskLevel.LOW

    #: Wall-clock cap. Every tool has one; a hung dependency must not hold a
    #: turn open indefinitely.
    timeout_s: float = 20.0

    #: Whether a retry is safe. Non-idempotent tools are never auto-retried,
    #: which is why the flag exists rather than being assumed.
    idempotent: bool = True

    @property
    def requires_approval(self) -> bool:
        return self.risk_level is RiskLevel.HIGH

    def permitted_for(self, principal: Principal) -> bool:
        return role_satisfies(principal.role, self.required_role)

    def to_model_schema(self) -> dict[str, Any]:
        """The definition handed to the model when this tool is bound."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }


class ToolRegistry:
    """The set of tools this deployment knows about."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return sorted(self._tools.values(), key=lambda spec: spec.name)

    def tools_for(self, principal: Principal) -> list[ToolSpec]:
        """Layer 1 — the only tools bound to the model for this caller.

        Sorted for determinism: an unstable tool order changes the prompt, which
        changes model behaviour, which would make runs incomparable in the
        evaluation harness.
        """
        return [spec for spec in self.all() if spec.permitted_for(principal)]

    def names_for(self, principal: Principal) -> list[str]:
        return [spec.name for spec in self.tools_for(principal)]

    def denied_for(self, principal: Principal) -> list[ToolSpec]:
        """Tools this caller cannot use.

        Surfaced in the UI so a viewer can see *that* analytics exists and is
        unavailable to them, rather than wondering why the assistant seems less
        capable than a colleague's.
        """
        return [spec for spec in self.all() if not spec.permitted_for(principal)]


# ── Argument schemas ────────────────────────────────────────────────────
# Declared as Pydantic models so arguments are validated before a handler runs.
# Model-produced arguments are untrusted input like any other.


class KnowledgeSearchArgs(BaseModel):
    query: str
    #: Narrowing hints only. app.retrieval.filters intersects these with the
    #: caller's permissions; they can never widen access.
    departments: list[str] | None = None
    document_types: list[str] | None = None
    top_k: int = 8


class PythonAnalysisArgs(BaseModel):
    #: Analysis code, executed in the restricted sandbox.
    code: str
    #: Chunk ids whose content the analysis may read.
    chunk_ids: list[str]


class EmployeeDirectoryArgs(BaseModel):
    name_contains: str | None = None
    department: str | None = None


class ServiceCatalogArgs(BaseModel):
    service_name: str | None = None
    owner_team: str | None = None


class IncidentRecordsArgs(BaseModel):
    service: str | None = None
    since: str | None = None
    severity: str | None = None


class ReindexArgs(BaseModel):
    confirm: bool = False


class PurgeMemoryArgs(BaseModel):
    user_id: str
    confirm: bool = False


def build_default_registry(handlers: dict[str, ToolHandler]) -> ToolRegistry:
    """Assemble the catalogue.

    Handlers are injected rather than imported so the registry stays testable
    without a live Pinecone, MCP server, or sandbox — and so the role and risk
    declarations can be reviewed in one place, separately from the code that
    implements them.
    """
    registry = ToolRegistry()

    specs = [
        # ── Viewer ──────────────────────────────────────────────────────
        ToolSpec(
            name="knowledge_search",
            description=(
                "Search indexed organizational documents. Returns passages with "
                "citations. Results are already restricted to what the calling "
                "user is permitted to read."
            ),
            args_schema=KnowledgeSearchArgs,
            handler=handlers["knowledge_search"],
            required_role=Role.VIEWER,
            risk_level=RiskLevel.LOW,
            timeout_s=15.0,
        ),
        # ── Analyst ─────────────────────────────────────────────────────
        ToolSpec(
            name="python_analysis",
            description=(
                "Run a short Python analysis over already-retrieved passages. "
                "Executes in a sandbox with no network or filesystem access."
            ),
            args_schema=PythonAnalysisArgs,
            handler=handlers["python_analysis"],
            required_role=Role.ANALYST,
            # Sandboxed, but it executes model-authored code and can burn the
            # whole budget, so it is not low risk.
            risk_level=RiskLevel.MEDIUM,
            timeout_s=25.0,
        ),
        ToolSpec(
            name="employee_directory",
            description="Look up employees in the enterprise directory (MCP).",
            args_schema=EmployeeDirectoryArgs,
            handler=handlers["employee_directory"],
            required_role=Role.ANALYST,
            risk_level=RiskLevel.MEDIUM,
            timeout_s=10.0,
        ),
        ToolSpec(
            name="service_catalog",
            description="Look up services, owners, and dependencies (MCP).",
            args_schema=ServiceCatalogArgs,
            handler=handlers["service_catalog"],
            required_role=Role.ANALYST,
            risk_level=RiskLevel.LOW,
            timeout_s=10.0,
        ),
        ToolSpec(
            name="incident_records",
            description="Query structured incident records (MCP).",
            args_schema=IncidentRecordsArgs,
            handler=handlers["incident_records"],
            required_role=Role.ANALYST,
            risk_level=RiskLevel.LOW,
            timeout_s=10.0,
        ),
        # ── Administrator ───────────────────────────────────────────────
        ToolSpec(
            name="admin_reindex",
            description="Rebuild the retrieval indexes from the corpus.",
            args_schema=ReindexArgs,
            handler=handlers["admin_reindex"],
            required_role=Role.ADMINISTRATOR,
            # Mutates shared state — approval required even for an administrator.
            risk_level=RiskLevel.HIGH,
            timeout_s=120.0,
            idempotent=False,
        ),
        ToolSpec(
            name="admin_purge_memory",
            description="Delete a user's stored long-term memory.",
            args_schema=PurgeMemoryArgs,
            handler=handlers["admin_purge_memory"],
            required_role=Role.ADMINISTRATOR,
            risk_level=RiskLevel.HIGH,
            timeout_s=30.0,
            idempotent=False,
        ),
    ]

    for spec in specs:
        registry.register(spec)
    return registry
