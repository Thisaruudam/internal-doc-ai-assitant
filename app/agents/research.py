"""The research agent.

Handles the questions retrieval cannot: those that span many documents and need
aggregation rather than lookup — the brief's own example being "summarise all
outage reports related to payment failures during the last year and identify
recurring root causes".

The sequence:

1. Build a manifest of chunks this caller may read — metadata only, no bodies.
2. Ask the model to write a Python program that partitions that manifest.
3. Run the program in the sandbox. It has no network, so it selects; it does not
   reason.
4. Resolve the selected ids back to chunks, re-checking authorization.
5. Map each batch through a sub-agent, then reduce the findings.

Every step degrades rather than fails. Codegen that misfires falls back to a
date-ordered partition; a batch that errors becomes a stated gap; a failed
reduce returns the raw findings. The alternative — abandoning the investigation
because one step was imperfect — is worse for the user in every case.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.config import Settings
from app.graph import events
from app.graph.state import AgentState, ErrorRecord, TodoStatus
from app.observability.logging import get_logger
from app.retrieval.schema import Chunk, Citation, ScoredChunk
from app.rlm.corpus_api import RlmCorpus, parse_plan_result
from app.rlm.planner import fallback_batches, generate_plan_code, run_plan
from app.rlm.recursion import ResearchOutcome, map_reduce

log = get_logger(__name__)

_NODE = "research_agent"

#: Ceilings applied to whatever the plan returns. The plan is model-authored, so
#: its shape is a suggestion until it has been bounded.
MAX_BATCHES = 6
MAX_IDS_PER_BATCH = 12


async def research_agent(
    state: AgentState, settings: Settings, all_chunks: list[Chunk]
) -> dict[str, Any]:
    """Investigate a question that spans many documents."""
    events.node_enter(_NODE)

    principal = state["principal"]
    budget = state["budget"]
    question = _question_of(state)

    if budget.depth <= 0:
        # Recursion limit reached. Not an error — the turn continues to the
        # response agent with whatever has already been gathered.
        events.node_exit(_NODE, reason="recursion budget exhausted")
        return {
            "errors": [
                ErrorRecord(
                    stage="research",
                    kind="budget_exhausted",
                    detail="recursion depth limit reached",
                    degraded_to="existing evidence",
                )
            ]
        }

    corpus = RlmCorpus.for_principal(principal, all_chunks)
    manifest = corpus.manifest()

    events.retrieval_stage(
        _NODE, "manifest", count=len(manifest), note="metadata only, no document text"
    )

    if not manifest:
        events.node_exit(_NODE, reason="no readable documents")
        return {"retrieved": [], "citations": []}

    outcome = await _investigate(question, corpus, manifest, settings, depth=budget.depth)

    chunks = _chunks_for(corpus, outcome)
    events.node_exit(
        _NODE,
        batches=outcome.batches_attempted,
        failed=outcome.batches_failed,
        chunks_read=outcome.chunks_read,
    )

    plan = [
        step.model_copy(update={"status": TodoStatus.DONE, "result_ref": "research.md"})
        if step.agent == "research"
        else step
        for step in state.get("plan", [])
    ]

    errors: list[ErrorRecord] = []
    if outcome.used_fallback_plan:
        errors.append(
            ErrorRecord(
                stage="research",
                kind="plan_generation_failed",
                detail=outcome.plan_rationale or "the generated plan could not be used",
                degraded_to="date-ordered batching",
            )
        )
    if outcome.partial:
        errors.append(
            ErrorRecord(
                stage="research",
                kind="partial_results",
                detail=f"{outcome.batches_failed} of {outcome.batches_attempted} groups failed",
                degraded_to="partial findings",
            )
        )

    files = dict(outcome.files)
    files["research.md"] = _research_artifact(outcome)

    # state["retrieved"] holds ScoredChunk everywhere, so the validator and the
    # response agent see one shape regardless of which specialist produced it.
    # Research selects by plan rather than by score, so there is no retrieval
    # rank to report — the provenance says how it was found instead.
    scored = [ScoredChunk(chunk=chunk, retrievers=["rlm"]) for chunk in chunks]

    return {
        # The reduced answer is carried as an assistant message so the response
        # agent composes from a synthesis rather than re-reading every passage.
        "messages": [AIMessage(content=outcome.answer)] if outcome.answer else [],
        "retrieved": scored,
        "citations": [Citation.of(chunk) for chunk in chunks],
        "files": files,
        "plan": plan,
        "errors": errors,
        "budget": budget.spend(depth=1, tool_calls=1, tokens=len(outcome.answer) // 4),
    }


async def _investigate(
    question: str,
    corpus: RlmCorpus,
    manifest: list[dict[str, Any]],
    settings: Settings,
    *,
    depth: int,
) -> ResearchOutcome:
    """Plan, then map/reduce over what the plan selected."""
    summary = corpus.summarise()

    batch_ids: list[list[str]] = []
    rationale = ""
    used_fallback = False

    try:
        code = await generate_plan_code(question, summary, settings.gemini)
        events.retrieval_stage(_NODE, "plan_generated", count=len(code.splitlines()))

        result, error = await run_plan(
            code,
            manifest,
            timeout_s=settings.graph.sandbox_timeout_s,
            memory_mb=settings.graph.sandbox_memory_mb,
        )
        if error:
            raise RuntimeError(error)

        plan = parse_plan_result(result, max_batches=MAX_BATCHES, max_per_batch=MAX_IDS_PER_BATCH)
        batch_ids = plan.batches
        rationale = plan.rationale
    except Exception as exc:
        log.warning("rlm_planning_failed", error_type=type(exc).__name__)
        rationale = str(exc)[:200]

    if not batch_ids:
        used_fallback = True
        events.degradation(
            _NODE,
            component="plan generation",
            reason=rationale or "empty plan",
            fallback="date-ordered batching",
        )
        batch_ids = fallback_batches(manifest, batch_size=MAX_IDS_PER_BATCH, max_batches=4)

    events.retrieval_stage(
        _NODE,
        "batches",
        count=len(batch_ids),
        chunk_ids=sum(len(batch) for batch in batch_ids),
        rationale=rationale[:120],
    )

    batches: list[list[Chunk]] = []
    for ids in batch_ids:
        resolved, _refused = corpus.resolve(ids)
        if resolved:
            batches.append(resolved)

    outcome = await map_reduce(batches, question, settings.gemini, settings.graph, depth=depth)
    outcome.used_fallback_plan = used_fallback
    outcome.plan_rationale = rationale
    return outcome


def _chunks_for(corpus: RlmCorpus, outcome: ResearchOutcome) -> list[Chunk]:
    """Every chunk the investigation actually read, de-duplicated in order."""
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for finding in outcome.findings:
        if not finding.ok:
            continue
        for chunk_id in finding.chunk_ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk = corpus.chunks.get(chunk_id)
            if chunk is not None:
                chunks.append(chunk)
    return chunks


def _research_artifact(outcome: ResearchOutcome) -> str:
    """A record of how the investigation was conducted, for the trace and panel."""
    lines = [
        "# Research summary",
        "",
        f"- groups analysed: {outcome.batches_attempted - outcome.batches_failed}"
        f" of {outcome.batches_attempted}",
        f"- passages read: {outcome.chunks_read}",
        f"- planning: {'fallback batching' if outcome.used_fallback_plan else 'generated plan'}",
    ]
    if outcome.plan_rationale:
        lines.append(f"- plan rationale: {outcome.plan_rationale}")
    caveat = outcome.caveat()
    if caveat:
        lines += ["", caveat]
    lines += ["", "## Synthesis", "", outcome.answer or "(no findings)"]
    return "\n".join(lines)


def _question_of(state: AgentState) -> str:
    from langchain_core.messages import HumanMessage

    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            text = getattr(message, "text", None)
            if isinstance(text, str):
                return text
            return str(message.content)
    return ""
