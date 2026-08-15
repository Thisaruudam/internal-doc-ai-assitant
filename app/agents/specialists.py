"""Analysis and MCP specialist nodes.

Both follow the same shape: the model picks a tool call, the guard authorizes
and executes it, and the result is written to the virtual filesystem for the
response agent. Neither node touches a tool directly — every invocation goes
through ``ToolGuard``, so the second authorization layer applies to agent-driven
calls exactly as it does to anything else.

The analysis agent is the more interesting of the two. It answers questions
whose answer is a number — "which service failed most often", "what was the
average outage duration" — by writing code rather than by counting in prose.
Models are unreliable at arithmetic over a list and reliable at writing the
three lines of Python that do it exactly, and the difference shows up precisely
on the questions where a wrong number is most expensive.
"""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.models import ModelRole, get_model
from app.config import Settings
from app.graph import events
from app.graph.state import AgentState, ErrorRecord, TodoStatus
from app.observability.logging import get_logger
from app.tools.guard import (
    ApprovalRequired,
    ToolArgumentsInvalidError,
    ToolDeniedError,
    ToolGuard,
    ToolTimeoutError,
    ToolUnknownError,
)
from app.tools.python_analysis import ANALYSIS_PROMPT

log = get_logger(__name__)

ANALYSIS_SYSTEM = f"""\
You compute an exact answer from passages that have already been retrieved.

{ANALYSIS_PROMPT}

You are given the passages' metadata. Choose the chunk_ids your analysis needs \
and write the code. Do not attempt to answer in prose — the code's `result` is \
the answer."""

MCP_SYSTEM = """\
You look up structured enterprise records: the employee directory, the service \
catalogue, or incident records.

Choose exactly one tool and the arguments for it. Prefer narrow arguments — an \
unfiltered lookup returns a page of rows that answers nothing in particular.

Incident records carry a report_doc_id linking to the full written report, so a \
count from here can be checked against the narrative later."""


class AnalysisRequest(BaseModel):
    code: str = Field(description="Python that assigns the answer to `result`")
    chunk_ids: list[str] = Field(description="ids of passages the code should read")


class McpRequest(BaseModel):
    tool: str = Field(description="employee_directory, service_catalog, or incident_records")
    arguments: dict[str, Any] = Field(default_factory=dict)


def _mark_done(state: AgentState, agent: str, artifact: str) -> list[Any]:
    return [
        step.model_copy(update={"status": TodoStatus.DONE, "result_ref": artifact})
        if step.agent == agent
        else step
        for step in state.get("plan", [])
    ]


def _question(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            text = getattr(message, "text", None)
            return text if isinstance(text, str) else str(message.content)
    return ""


def _guard_failure(node: str, exc: Exception) -> ErrorRecord:
    """Translate a guard refusal into something the answer can explain."""
    if isinstance(exc, ToolDeniedError):
        kind, detail = "tool_denied", str(exc)
    elif isinstance(exc, ApprovalRequired):
        kind, detail = "approval_required", "this action needs human approval"
    elif isinstance(exc, ToolArgumentsInvalidError):
        kind, detail = "invalid_arguments", str(exc)
    elif isinstance(exc, ToolTimeoutError):
        kind, detail = "tool_timeout", str(exc)
    elif isinstance(exc, ToolUnknownError):
        kind, detail = "unknown_tool", str(exc)
    else:
        kind, detail = "tool_failed", type(exc).__name__

    log.info("specialist_tool_unavailable", node=node, kind=kind)
    return ErrorRecord(stage=node, kind=kind, detail=detail, degraded_to="retrieved evidence")


# ── analysis agent ──────────────────────────────────────────────────────


async def analysis_agent(state: AgentState, settings: Settings, guard: ToolGuard) -> dict[str, Any]:
    """Answer a numeric question by computing it."""
    events.node_enter("analysis_agent")

    retrieved = state.get("retrieved", [])
    if not retrieved:
        # Analysis operates on evidence someone else gathered. With nothing to
        # compute over, the honest move is to hand back to retrieval rather than
        # invent a dataset.
        events.node_exit("analysis_agent", reason="no evidence to analyse")
        return {
            "plan": _mark_done(state, "analysis", "analysis.md"),
            "errors": [
                ErrorRecord(
                    stage="analysis",
                    kind="no_input",
                    detail="no passages had been retrieved to analyse",
                    degraded_to="retrieval",
                )
            ],
        }

    catalogue = [
        {
            "chunk_id": scored.chunk.chunk_id,
            "doc_id": scored.chunk.metadata.doc_id,
            "title": scored.chunk.metadata.title,
            "section": scored.chunk.metadata.heading_path,
            "tags": scored.chunk.metadata.tags,
        }
        for scored in retrieved
    ]

    model = get_model(ModelRole.AGENT, settings.gemini).with_structured_output(AnalysisRequest)
    try:
        request = cast(
            AnalysisRequest,
            await model.ainvoke(
                [
                    SystemMessage(content=ANALYSIS_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Passages available:\n{json.dumps(catalogue, indent=2)}\n\n"
                            f"Question: {_question(state)}"
                        )
                    ),
                ]
            ),
        )
    except Exception as exc:
        log.warning("analysis_codegen_failed", error_type=type(exc).__name__)
        events.node_exit("analysis_agent", reason="code generation failed")
        return {
            "plan": _mark_done(state, "analysis", "analysis.md"),
            "errors": [_guard_failure("analysis", exc)],
        }

    try:
        outcome = await guard.invoke(
            "python_analysis",
            {"code": request.code, "chunk_ids": request.chunk_ids},
            state["principal"],
        )
    except Exception as exc:
        events.node_exit("analysis_agent", reason="tool refused")
        return {
            "plan": _mark_done(state, "analysis", "analysis.md"),
            "errors": [_guard_failure("analysis", exc)],
        }

    artifact = _analysis_artifact(request.code, outcome.result)
    events.node_exit("analysis_agent", ok=outcome.ok)

    return {
        "files": {"analysis.md": artifact},
        "plan": _mark_done(state, "analysis", "analysis.md"),
        "budget": state["budget"].spend(tool_calls=1),
        "errors": []
        if outcome.ok
        else [
            ErrorRecord(
                stage="analysis",
                kind="analysis_failed",
                detail=str(outcome.error or "the analysis did not produce a result"),
                degraded_to="retrieved evidence",
            )
        ],
    }


def _analysis_artifact(code: str, result: Any) -> str:
    """Record the computation and its output.

    The code is included so the number can be checked rather than trusted — an
    answer that says "14" is worth much less than one that shows the three lines
    that produced 14.
    """
    payload = result if isinstance(result, dict) else {"result": result}
    return (
        "# Computed analysis\n\n"
        "```python\n" + code.strip() + "\n```\n\n"
        "## Result\n\n```json\n"
        + json.dumps(payload.get("result", payload), indent=2, default=str)[:4000]
        + "\n```\n"
    )


def _mcp_artifact(tool: str, result: Any) -> str:
    """Render a tool response so the reliable parts are read first.

    Dumping the raw payload put a long ``incidents`` list next to the computed
    rollups, and the model counted rows instead of reading the totals — then
    correctly reported that its count was based on a truncated page. The rollups
    cover every matched record, so they lead; the rows follow, explicitly
    labelled as a sample that must not be counted.
    """
    if not isinstance(result, dict):
        payload = json.dumps(result, default=str)[:4000]
        return f"# Enterprise records: {tool}\n\n```json\n{payload}\n```\n"

    lines = [f"# Enterprise records: {tool}", ""]

    totals = {k: v for k, v in result.items() if k.endswith("_summary") or k.startswith("total")}
    if totals:
        lines += [
            "## Totals over ALL matched records",
            "",
            "These are computed across every matching record, not only the rows "
            "listed below. Use them for any count.",
            "",
            "```json",
            json.dumps(totals, indent=2, default=str)[:2500],
            "```",
            "",
        ]

    rows_key = next(
        (k for k in ("incidents", "employees", "services") if isinstance(result.get(k), list)),
        None,
    )
    if rows_key:
        rows = result[rows_key]
        lines += [
            f"## Sample rows ({len(rows)} shown"
            + (f" of {result['total_matched']} matched" if "total_matched" in result else "")
            + ")",
            "",
            "A sample for detail only. Do NOT count these to answer "
            "'how many' — use the totals above.",
            "",
            "```json",
            json.dumps(rows[:12], indent=2, default=str)[:3500],
            "```",
        ]

    for key in ("error", "note"):
        if result.get(key):
            lines += ["", f"**{key}:** {result[key]}"]

    return "\n".join(lines)


# ── MCP agent ───────────────────────────────────────────────────────────


async def mcp_agent(state: AgentState, settings: Settings, guard: ToolGuard) -> dict[str, Any]:
    """Look up structured enterprise records."""
    events.node_enter("mcp_agent")

    model = get_model(ModelRole.AGENT, settings.gemini).with_structured_output(McpRequest)
    try:
        request = cast(
            McpRequest,
            await model.ainvoke(
                [
                    SystemMessage(content=MCP_SYSTEM),
                    HumanMessage(content=f"Question: {_question(state)}"),
                ]
            ),
        )
    except Exception as exc:
        log.warning("mcp_selection_failed", error_type=type(exc).__name__)
        events.node_exit("mcp_agent", reason="tool selection failed")
        return {
            "plan": _mark_done(state, "mcp", "enterprise.md"),
            "errors": [_guard_failure("mcp", exc)],
        }

    try:
        outcome = await guard.invoke(request.tool, request.arguments, state["principal"])
    except Exception as exc:
        events.node_exit("mcp_agent", reason="tool refused")
        return {
            "plan": _mark_done(state, "mcp", "enterprise.md"),
            "errors": [_guard_failure("mcp", exc)],
        }

    artifact = _mcp_artifact(request.tool, outcome.result)
    events.node_exit("mcp_agent", tool=request.tool, ok=outcome.ok)

    return {
        "files": {"enterprise.md": artifact},
        "plan": _mark_done(state, "mcp", "enterprise.md"),
        "budget": state["budget"].spend(tool_calls=1),
    }
