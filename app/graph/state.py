"""LangGraph state.

The shape every node reads and writes. Three properties are deliberate and are
what make the rest of the architecture work:

* **``principal`` is written once and never by a node.** The API layer stamps it
  from the verified JWT before the graph starts. Retrieval filters are derived
  from it, so a compromised model cannot widen its own reach.

* **Budgets live in state, not in exception handlers.** Running out of depth,
  tool calls, or tokens is an ordinary transition to the response node with a
  partial answer — not an error. That is what stops a recursive research task
  from becoming an unbounded bill.

* **Sub-agents exchange artifacts, not transcripts.** ``files`` is a virtual
  filesystem: a specialist writes its findings under a name and returns the
  name. The parent context never accumulates raw tool output, which is the
  single biggest driver of context growth in multi-agent systems.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from app.auth.principal import Principal


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class TodoItem(BaseModel):
    """One step of the supervisor's plan.

    Rendered live in the activity panel, which is why the description is written
    for a human rather than for the model.
    """

    id: str
    description: str
    #: Which specialist should handle this step.
    agent: Literal["retrieval", "research", "analysis", "mcp"]
    status: TodoStatus = TodoStatus.PENDING
    #: Name of the artifact in ``files`` holding this step's output.
    result_ref: str | None = None
    note: str | None = None


class Budget(BaseModel):
    """What this turn has left to spend.

    Immutable: ``spend`` returns a new instance rather than mutating, so a
    concurrent sub-agent cannot race another's accounting.
    """

    model_config = {"frozen": True}

    depth: int
    tool_calls: int
    tokens: int
    supervisor_steps: int

    def spend(
        self,
        *,
        depth: int = 0,
        tool_calls: int = 0,
        tokens: int = 0,
        supervisor_steps: int = 0,
    ) -> Self:
        """Deduct, flooring at zero.

        Clamping rather than going negative keeps ``exhausted`` a simple
        predicate and avoids a nonsensical "-4 tool calls remaining" in the UI.
        """
        return type(self)(
            depth=max(0, self.depth - depth),
            tool_calls=max(0, self.tool_calls - tool_calls),
            tokens=max(0, self.tokens - tokens),
            supervisor_steps=max(0, self.supervisor_steps - supervisor_steps),
        )

    @property
    def exhausted(self) -> bool:
        return (
            self.tool_calls <= 0
            or self.tokens <= 0
            or self.supervisor_steps <= 0
            or self.depth <= 0
        )

    @property
    def exhausted_reason(self) -> str | None:
        """Why the turn stopped, phrased for the user.

        The response agent puts this in the answer: a truncated result the user
        cannot account for is worse than a short one they can.
        """
        if self.supervisor_steps <= 0:
            return "the planning step limit for a single question was reached"
        if self.tool_calls <= 0:
            return "the tool-call budget for a single question was reached"
        if self.tokens <= 0:
            return "the token budget for a single question was reached"
        if self.depth <= 0:
            return "the research recursion limit was reached"
        return None


class RiskVerdict(StrEnum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


class RiskAssessment(BaseModel):
    """Output of the ingress guard."""

    verdict: RiskVerdict = RiskVerdict.SAFE
    score: float = 0.0
    #: Human-readable reasons, shown in the activity panel so a refusal is never
    #: unexplained.
    signals: list[str] = Field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.verdict is RiskVerdict.BLOCKED


class ErrorRecord(BaseModel):
    """A degradation that occurred during the turn.

    Not an exception — these are recorded and carried to the response so the
    answer can state what was unavailable. A silently worse answer is the one
    outcome the failure model forbids.
    """

    stage: str
    kind: str
    detail: str
    #: What the system fell back to, if anything.
    degraded_to: str | None = None


def merge_files(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """Reducer for the virtual filesystem.

    Last write wins per key. Concurrent sub-agents write under distinct names
    (``findings/batch_1.md``), so collisions indicate a bug in name assignment
    rather than a legitimate race.
    """
    return {**left, **right}


class AgentState(TypedDict, total=False):
    """The graph's working state.

    ``total=False`` because nodes return partial updates; LangGraph merges them
    through the reducers declared here.
    """

    # Conversation. add_messages appends and de-duplicates by id.
    messages: Annotated[list[AnyMessage], add_messages]

    # Identity. Written once by the API layer; no node may modify it.
    principal: Principal

    # Deep-agent planning surface.
    plan: list[TodoItem]

    # Virtual filesystem — artifacts passed by name, not by value.
    files: Annotated[dict[str, str], merge_files]

    # Retrieval results for this turn, and the citations derived from them.
    # Always ScoredChunk, whichever specialist produced them: the validator and
    # response agent read one shape. Annotated loosely to avoid a circular
    # import between graph state and the retrieval schema.
    retrieved: list[Any]  # list[ScoredChunk]
    citations: list[Any]  # list[Citation]

    # Security and accounting.
    risk: RiskAssessment
    budget: Budget

    # Degradations, appended by whichever stage hit them.
    errors: Annotated[list[ErrorRecord], operator.add]

    # Routing and bookkeeping.
    route: str
    repair_attempts: int
    thread_id: str
    correlation_id: str
    #: Set when retrieval ran without Pinecone, so the answer can say so.
    degraded_retrieval: bool


def initial_state(
    *,
    principal: Principal,
    question: str,
    thread_id: str,
    correlation_id: str,
    budget: Budget,
) -> AgentState:
    """Build the starting state for a turn.

    The only place ``principal`` is set. Every other write to state happens in a
    node, and none of them touch this field.
    """
    from langchain_core.messages import HumanMessage

    return AgentState(
        messages=[HumanMessage(content=question)],
        principal=principal,
        plan=[],
        files={},
        retrieved=[],
        citations=[],
        risk=RiskAssessment(),
        budget=budget,
        errors=[],
        route="",
        repair_attempts=0,
        thread_id=thread_id,
        correlation_id=correlation_id,
        degraded_retrieval=False,
    )
