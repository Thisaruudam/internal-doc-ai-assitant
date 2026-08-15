"""Graph nodes.

Each node is an async function taking state and returning a partial update.
Kept in one module because they share a small set of helpers and because the
sequence is easier to follow read top to bottom than spread across six files.

Every node follows the same shape:

1. Emit ``node.enter`` so the activity panel shows where execution is.
2. Do one thing.
3. Emit what it found, in counts and identifiers — never raw retrieved text.
4. Return a partial state update. Nodes never mutate state in place.

Failures degrade rather than raise. A node that cannot do its job returns an
``ErrorRecord`` and lets the turn continue to the response agent, which is what
lets the system answer partially instead of not at all.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.models import ModelRole, get_model
from app.agents.prompts import (
    RESPONSE_SYSTEM,
    RETRIEVAL_QUERY_SYSTEM,
    SUPERVISOR_SYSTEM,
    evidence_preamble,
    insufficient_evidence,
)
from app.config import Settings
from app.graph import events
from app.graph.state import (
    AgentState,
    ErrorRecord,
    RiskAssessment,
    RiskVerdict,
    TodoItem,
    TodoStatus,
)
from app.observability.logging import get_logger
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.schema import Citation
from app.security.egress import BLOCKED_ANSWER, redact, scan_egress
from app.security.grounding import check_grounding
from app.security.injection import scan_user_input
from app.security.quarantine import build_evidence_block, quarantine_chunks

log = get_logger(__name__)


def _last_user_question(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return _text_of(message)
    return ""


def _text_of(message: Any) -> str:
    """Extract plain text from a model response.

    langchain-core 1.x returns content as a list of typed blocks, so ``.content``
    is not a string. ``.text`` flattens it.
    """
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):  # older langchain exposed .text() as a method
        resolved = text()
        if isinstance(resolved, str):
            return resolved
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


# ── ingress guard ───────────────────────────────────────────────────────


async def ingress_guard(state: AgentState, settings: Settings) -> dict[str, Any]:
    """Scan the user's question before anything else runs.

    Heuristics only — cheap, deterministic, and enough for the direct-injection
    case. Hostile content arriving through *retrieved documents* is handled
    later, by quarantining rather than refusing, because the user is not the
    author there.
    """
    events.node_enter("ingress_guard")
    question = _last_user_question(state)

    decision = scan_user_input(
        question,
        block_threshold=settings.security.injection_block_threshold,
        warn_threshold=settings.security.injection_warn_threshold,
    )
    verdict = (
        RiskVerdict.BLOCKED
        if decision.blocked
        else RiskVerdict.SUSPICIOUS
        if decision.quarantine
        else RiskVerdict.SAFE
    )

    events.guard_result(
        "ingress_guard",
        verdict=verdict.value,
        score=decision.score,
        signals=decision.signals[:4],
    )
    events.node_exit("ingress_guard")

    return {
        "risk": RiskAssessment(verdict=verdict, score=decision.score, signals=decision.signals),
        "route": "refusal" if decision.blocked else "supervisor",
    }


async def refusal(state: AgentState) -> dict[str, Any]:
    """Decline a blocked request, naming why."""
    events.node_enter("refusal")
    risk: RiskAssessment = state["risk"]

    answer = (
        "I can't act on that request. It contains instructions that attempt to "
        "override how this assistant operates"
        + (f" ({', '.join(risk.signals[:2])})" if risk.signals else "")
        + ". If this was a genuine question, please rephrase it."
    )

    events.node_exit("refusal")
    return {"messages": [AIMessage(content=answer)], "citations": []}


# ── supervisor ──────────────────────────────────────────────────────────


class PlanStep(BaseModel):
    """One step the supervisor wants performed."""

    agent: str = Field(description="retrieval, research, analysis, or mcp")
    description: str = Field(description="what this step should achieve, for a human")


class SupervisorPlan(BaseModel):
    plan: list[PlanStep] = Field(description="ordered steps, usually one")
    reasoning: str = Field(description="one sentence on why this route")


_VALID_AGENTS = {"retrieval", "research", "analysis", "mcp"}

#: Populated by the graph builder with the specialist nodes that exist in this
#: deployment, so the supervisor cannot route somewhere unregistered.
_ROUTABLE: set[str] = {"retrieval_agent", "research_agent"}


def set_routable(destinations: set[str]) -> None:
    """Declare which specialist nodes the supervisor may target."""
    _ROUTABLE.clear()
    _ROUTABLE.update(destinations)


async def supervisor(state: AgentState, settings: Settings) -> Command[str]:
    """Decompose the turn and route to a specialist.

    Returns a ``Command`` rather than setting a routing key: the destination is
    then data the model produced, visible in the trace and in the activity panel,
    rather than a lookup hidden in a conditional edge.
    """
    events.node_enter("supervisor")
    budget = state["budget"]

    if budget.exhausted:
        # Not an error. A partial answer the user can account for beats a
        # truncated one they cannot.
        events.budget_update("supervisor", budget)
        events.node_exit("supervisor", reason="budget exhausted")
        return Command(goto="response_agent", update={"route": "response_agent"})

    # A plan already in progress means a specialist has reported back.
    if state.get("plan") and all(step.status is TodoStatus.DONE for step in state["plan"]):
        events.node_exit("supervisor", reason="plan complete")
        return Command(goto="response_agent", update={"route": "response_agent"})

    question = _last_user_question(state)
    model = get_model(ModelRole.AGENT, settings.gemini).with_structured_output(SupervisorPlan)

    try:
        decision = cast(
            SupervisorPlan,
            await model.ainvoke(
                [SystemMessage(content=SUPERVISOR_SYSTEM), HumanMessage(content=question)]
            ),
        )
    except Exception as exc:  # routing failure must not end the turn
        log.warning("supervisor_failed", error_type=type(exc).__name__)
        events.degradation(
            "supervisor",
            component="supervisor model",
            reason=type(exc).__name__,
            fallback="direct retrieval",
        )
        plan = [TodoItem(id="1", description="Search for relevant documents", agent="retrieval")]
        return Command(
            goto="retrieval_agent",
            update={
                "plan": plan,
                "route": "retrieval_agent",
                "errors": [
                    ErrorRecord(
                        stage="supervisor",
                        kind="model_unavailable",
                        detail=type(exc).__name__,
                        degraded_to="retrieval",
                    )
                ],
                "budget": state["budget"].spend(supervisor_steps=1),
            },
        )

    steps = [
        TodoItem(
            id=str(index),
            description=step.description,
            agent=step.agent if step.agent in _VALID_AGENTS else "retrieval",
        )
        for index, step in enumerate(decision.plan[:4], start=1)
    ] or [TodoItem(id="1", description="Search for relevant documents", agent="retrieval")]

    steps[0].status = TodoStatus.IN_PROGRESS
    events.plan_update("supervisor", steps)

    # Routing is model-produced, so it is checked against the nodes that were
    # actually registered. A graph built without the tool guard has no analysis
    # or MCP node, and routing to a missing node would fail the turn.
    destination = f"{steps[0].agent}_agent"
    if destination not in _ROUTABLE:
        destination = "retrieval_agent"
    events.node_exit("supervisor", route=destination, reasoning=decision.reasoning[:160])

    return Command(
        goto=destination,
        update={
            "plan": steps,
            "route": destination,
            "budget": state["budget"].spend(supervisor_steps=1),
        },
    )


# ── retrieval agent ─────────────────────────────────────────────────────


class SearchPlan(BaseModel):
    query: str = Field(description="search text, expanded with likely document terms")
    departments: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)


async def retrieval_agent(
    state: AgentState, settings: Settings, retriever: HybridRetriever
) -> dict[str, Any]:
    """Rewrite the question into a query, search, and quarantine what comes back."""
    events.node_enter("retrieval_agent")
    question = _last_user_question(state)
    principal = state["principal"]

    search_plan = SearchPlan(query=question)
    try:
        planner = get_model(ModelRole.AGENT, settings.gemini).with_structured_output(SearchPlan)
        search_plan = cast(
            SearchPlan,
            await planner.ainvoke(
                [SystemMessage(content=RETRIEVAL_QUERY_SYSTEM), HumanMessage(content=question)]
            ),
        )
        events.retrieval_stage(
            "retrieval_agent", "query_understanding", count=1, query=search_plan.query[:120]
        )
    except Exception as exc:
        # The raw question is a serviceable query. Losing the rewrite costs
        # recall, not correctness.
        log.warning("query_rewrite_failed", error_type=type(exc).__name__)
        events.degradation(
            "retrieval_agent",
            component="query rewriter",
            reason=type(exc).__name__,
            fallback="raw user question",
        )

    result = await retriever.retrieve(
        search_plan.query,
        principal,
        departments=set(search_plan.departments) or None,
        document_types=set(search_plan.document_types) or None,
    )

    passages = quarantine_chunks(result.chunks)
    flagged = [p for p in passages if p.flagged]
    if flagged:
        events.guard_result(
            "retrieval_agent",
            verdict="quarantined",
            score=max(item.decision.score for item in flagged),
            signals=[f"{item.chunk_id}: injected content" for item in flagged[:3]]
            + flagged[0].decision.signals[:2],
        )

    plan = [
        step.model_copy(update={"status": TodoStatus.DONE, "result_ref": "evidence.md"})
        if step.agent == "retrieval"
        else step
        for step in state.get("plan", [])
    ]

    errors: list[ErrorRecord] = []
    if result.degraded:
        errors.append(
            ErrorRecord(
                stage="retrieval",
                kind="dependency_unavailable",
                detail=result.degraded_reason or "primary index unavailable",
                degraded_to="bm25",
            )
        )

    events.node_exit("retrieval_agent", chunks=len(result.chunks))

    return {
        "retrieved": result.chunks,
        "citations": [Citation.of(r.chunk) for r in result.chunks],
        "files": {"evidence.md": build_evidence_block(passages)},
        "plan": plan,
        "degraded_retrieval": result.degraded,
        "errors": errors,
        "budget": state["budget"].spend(tool_calls=1),
    }


# ── response agent ──────────────────────────────────────────────────────


async def response_agent(state: AgentState, settings: Settings) -> dict[str, Any]:
    """Compose the answer from the evidence gathered so far."""
    events.node_enter("response_agent")

    question = _last_user_question(state)
    files = state.get("files", {})
    principal = state["principal"]
    repair_note = (files.get("repair.md") or "").strip()

    # Source material is whatever the specialists produced, which is not always
    # retrieved chunks: the research agent leaves a synthesis, the analysis agent
    # a computed result, the MCP agent structured records. Gating on `retrieved`
    # alone made a successful enterprise lookup report "no evidence" — the same
    # mistake, twice, so the check now names every artifact a specialist writes.
    has_material = bool(state.get("retrieved")) or any(
        (files.get(name) or "").strip()
        for name in ("research.md", "evidence.md", "analysis.md", "enterprise.md")
    )
    if not has_material:
        events.node_exit("response_agent", reason="no source material")
        return {
            "messages": [AIMessage(content=insufficient_evidence(principal.role.value))],
            "citations": [],
        }

    # Source material comes from whichever specialist ran. The research agent
    # writes a synthesis rather than raw passages — that is the point of the
    # RLM, since its 52 passages must not all re-enter context here — so the
    # response agent composes from the synthesis when one exists.
    #
    # Reading only evidence.md meant a completed investigation was silently
    # discarded and the answer became "no evidence was provided".
    research = (files.get("research.md") or "").strip()
    evidence = (files.get("evidence.md") or "").strip()
    computed = (files.get("analysis.md") or "").strip()
    enterprise = (files.get("enterprise.md") or "").strip()

    if research:
        source = (
            "An investigation across many documents produced the synthesis below. "
            "Its [chunk_id] citations refer to passages that were read. Base your "
            "answer on it and preserve those citations exactly.\n\n" + research
        )
        if evidence:
            source += "\n\n" + evidence
    else:
        source = evidence

    # Computed results are stated before the passages: a figure produced by code
    # is more reliable than one the model would infer from prose, so it should
    # anchor the answer rather than compete with it.
    if computed:
        source = (
            "The following result was computed by running code over the "
            "retrieved passages. Prefer it over any figure you would estimate "
            "yourself.\n\n" + computed + "\n\n" + source
        )
    if enterprise:
        source = (
            "Structured enterprise records retrieved for this question:\n\n"
            + enterprise
            + "\n\n"
            + source
        )

    recalled = (files.get("memory.md") or "").strip()
    if recalled:
        # Placed before the evidence and labelled as context: memory explains
        # what the user means, but it is not a source and must not be cited.
        source = recalled + "\n\n" + source

    prompt_parts = [evidence_preamble(), "", source, "", f"Question: {question}"]
    if repair_note:
        prompt_parts += ["", "Your previous draft had these problems:", repair_note]
    if state.get("degraded_retrieval"):
        prompt_parts += [
            "",
            "Note: the primary search index was unavailable and these results "
            "come from a keyword-only fallback. Say so at the end of your answer.",
        ]
    if state["budget"].exhausted:
        prompt_parts += [
            "",
            f"Note: {state['budget'].exhausted_reason}. Answer with what is here "
            "and say the answer may be incomplete.",
        ]

    model = get_model(ModelRole.RESPONSE, settings.gemini)
    try:
        reply = await model.ainvoke(
            [
                SystemMessage(content=RESPONSE_SYSTEM),
                HumanMessage(content="\n".join(prompt_parts)),
            ]
        )
        answer = _text_of(reply)
    except Exception as exc:
        log.warning("response_model_failed", error_type=type(exc).__name__)
        events.degradation(
            "response_agent",
            component="response model",
            reason=type(exc).__name__,
            fallback="fallback model",
        )
        try:
            fallback = get_model(ModelRole.GUARD, settings.gemini)
            answer = _text_of(
                await fallback.ainvoke(
                    [
                        SystemMessage(content=RESPONSE_SYSTEM),
                        HumanMessage(content="\n".join(prompt_parts)),
                    ]
                )
            )
        except Exception:
            log.exception("response_fallback_failed")
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I could not generate an answer because the language "
                            "model is currently unavailable. Please try again, and "
                            f"quote correlation id {state.get('correlation_id', '')}."
                        )
                    )
                ],
                "errors": [
                    ErrorRecord(
                        stage="response",
                        kind="model_unavailable",
                        detail=type(exc).__name__,
                    )
                ],
            }

    events.node_exit("response_agent", characters=len(answer))
    return {
        "messages": [AIMessage(content=answer)],
        # Clear the repair note now it has been applied, so a later pass is not
        # corrected against feedback for a draft that no longer exists.
        "files": {"repair.md": ""},
        "route": "validator",
        "budget": state["budget"].spend(tokens=len(answer) // 4),
    }


# ── validator ───────────────────────────────────────────────────────────


async def validator(state: AgentState, settings: Settings) -> dict[str, Any]:
    """Check grounding and egress before the answer is released."""
    events.node_enter("validator")

    messages = state.get("messages", [])
    answer = _text_of(messages[-1]) if messages else ""
    attempt = state.get("repair_attempts", 0)
    principal = state["principal"]

    evidence = {r.chunk.chunk_id: r.chunk.text for r in state.get("retrieved", [])}

    # A research synthesis is an aggregate over many documents, so it is checked
    # against the findings it was reduced from as well as the passages, and is
    # not required to cite a passage per sentence. Fabricated citations and
    # unsupported figures remain failures in both modes — see the rationale in
    # app.security.grounding.check_grounding.
    files = state.get("files", {})
    # Aggregate mode also covers answers built from computed results or
    # structured records: neither is a restatement of a passage, so neither can
    # carry a per-sentence chunk citation.
    is_synthesis = any(
        (files.get(name) or "").strip() for name in ("research.md", "analysis.md", "enterprise.md")
    )
    if is_synthesis:
        for name, text in files.items():
            if name.startswith("findings/"):
                evidence[name] = text

    grounding = check_grounding(
        answer,
        retrieved_chunks=evidence,
        require_citation_per_claim=not is_synthesis,
    )
    egress = scan_egress(answer, source_passages=evidence)

    passed = grounding.passed and not egress.must_block
    events.validation_result(
        "validator",
        passed=passed,
        grounded_claims=grounding.grounded_claims,
        ungrounded_claims=grounding.ungrounded_claims,
        attempt=attempt,
    )

    if egress.must_block:
        # Egress failures are not repairable by rewriting: the content itself is
        # the problem, so the answer is withheld outright.
        log.warning("egress_blocked", findings=[f.value for f, _ in egress.blocking])
        events.node_exit("validator", outcome="blocked")
        return {
            "route": "done",
            "messages": [
                AIMessage(
                    content=(
                        f"{BLOCKED_ANSWER} Correlation id: "
                        f"{state.get('correlation_id', 'unknown')}."
                    )
                )
            ],
            "errors": [
                ErrorRecord(
                    stage="validator",
                    kind="egress_blocked",
                    detail=egress.summary(),
                )
            ],
        }

    if passed:
        cited = grounding.cited_chunk_ids
        events.node_exit("validator", outcome="passed", citations=len(cited))
        return {
            "route": "done",
            # Narrow the citation list to what the answer actually referenced,
            # so the UI shows sources rather than everything retrieved.
            "citations": [c for c in state.get("citations", []) if c.chunk_id in cited],
            "messages": [AIMessage(content=redact(answer))] if "![" in answer else [],
        }

    if attempt >= settings.graph.max_repair_attempts:
        log.info("validation_exhausted", attempts=attempt)
        events.node_exit("validator", outcome="insufficient_evidence")
        return {
            "route": "done",
            "messages": [AIMessage(content=insufficient_evidence(principal.role.value))],
            "citations": [],
            "errors": [
                ErrorRecord(
                    stage="validator",
                    kind="ungrounded_answer",
                    detail=f"{grounding.ungrounded_claims} unsupported claim(s)",
                    degraded_to="insufficient evidence",
                )
            ],
        }

    events.node_exit("validator", outcome="repair", attempt=attempt + 1)
    return {
        "route": "repair",
        "repair_attempts": attempt + 1,
        "files": {"repair.md": grounding.repair_instructions()},
    }
