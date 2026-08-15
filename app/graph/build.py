"""Graph assembly.

The whole topology is visible in ``build_graph`` — deliberately, because the
routing is the architecture and burying it across decorator registrations makes
it unreviewable.

Dependencies (settings, the retriever) are bound into the nodes with
``functools.partial`` rather than read from module globals. That is what lets a
test drive the graph with a stub retriever, and it keeps the nodes honest about
what they actually depend on.

The validator loop is the one piece of control flow worth reading twice: a failed
validation routes *back* to the response agent with repair instructions in the
virtual filesystem, up to a bounded number of attempts, after which the turn ends
with an explicit "insufficient evidence" rather than an unsupported answer.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents import nodes
from app.agents.research import research_agent
from app.agents.specialists import analysis_agent, mcp_agent
from app.config import Settings
from app.graph.state import AgentState, RiskAssessment
from app.observability.logging import get_logger
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.schema import Chunk
from app.tools.guard import ToolGuard

log = get_logger(__name__)


def _after_guard(state: AgentState) -> str:
    risk: RiskAssessment = state.get("risk") or RiskAssessment()
    return "refusal" if risk.blocked else "supervisor"


def _after_validator(state: AgentState) -> str:
    """Repair or finish.

    Driven by an explicit ``route`` the validator sets, not by the presence of a
    repair file. ``files`` is a merge-reducer channel, so anything written to it
    persists for the rest of the turn — using it as a loop condition means the
    condition never clears and response↔validator cycles until the recursion
    limit. That is exactly what happened the first time this ran.
    """
    if state.get("route") == "repair":
        return "response_agent"
    return END


def build_graph(
    settings: Settings,
    retriever: HybridRetriever,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    corpus_chunks: list[Chunk] | None = None,
    tool_guard: ToolGuard | None = None,
) -> Any:
    """Compile the agent graph.

    ``corpus_chunks`` backs the research agent's manifest. It is injected rather
    than loaded here so the graph stays constructible in tests without a corpus
    on disk, and so the API loads it once at startup rather than per turn.
    """
    graph = StateGraph(AgentState)

    graph.add_node("ingress_guard", partial(nodes.ingress_guard, settings=settings))
    graph.add_node("refusal", nodes.refusal)
    graph.add_node("supervisor", partial(nodes.supervisor, settings=settings))
    graph.add_node(
        "retrieval_agent",
        partial(nodes.retrieval_agent, settings=settings, retriever=retriever),
    )
    graph.add_node(
        "research_agent",
        partial(research_agent, settings=settings, all_chunks=corpus_chunks or []),
    )
    if tool_guard is not None:
        graph.add_node(
            "analysis_agent", partial(analysis_agent, settings=settings, guard=tool_guard)
        )
        graph.add_node("mcp_agent", partial(mcp_agent, settings=settings, guard=tool_guard))
    graph.add_node("response_agent", partial(nodes.response_agent, settings=settings))
    graph.add_node("validator", partial(nodes.validator, settings=settings))

    routable = {"retrieval_agent", "research_agent"}
    if tool_guard is not None:
        routable |= {"analysis_agent", "mcp_agent"}
    nodes.set_routable(routable)

    graph.add_edge(START, "ingress_guard")
    graph.add_conditional_edges(
        "ingress_guard", _after_guard, {"refusal": "refusal", "supervisor": "supervisor"}
    )
    graph.add_edge("refusal", END)

    # The supervisor returns a Command naming its own destination, so its edges
    # are declared as the set of places it may go rather than as a condition
    # evaluated here.
    graph.add_edge("retrieval_agent", "response_agent")
    graph.add_edge("research_agent", "response_agent")
    if tool_guard is not None:
        graph.add_edge("analysis_agent", "response_agent")
        graph.add_edge("mcp_agent", "response_agent")
    graph.add_edge("response_agent", "validator")
    graph.add_conditional_edges(
        "validator", _after_validator, {"response_agent": "response_agent", END: END}
    )

    compiled = graph.compile(checkpointer=checkpointer, name="atrium")
    log.info("graph_compiled", nodes=len(graph.nodes), checkpointed=checkpointer is not None)
    return compiled


def graph_topology() -> dict[str, list[str]]:
    """The topology, for documentation and the UI.

    Returned as data so the diagram in the docs and the panel in the UI cannot
    drift from the graph that actually runs.
    """
    return {
        "ingress_guard": ["refusal", "supervisor"],
        "refusal": [END],
        "supervisor": [
            "retrieval_agent",
            "research_agent",
            "analysis_agent",
            "mcp_agent",
            "response_agent",
        ],
        "retrieval_agent": ["response_agent"],
        "research_agent": ["response_agent"],
        "analysis_agent": ["response_agent"],
        "mcp_agent": ["response_agent"],
        "response_agent": ["validator"],
        "validator": ["response_agent", END],
    }
