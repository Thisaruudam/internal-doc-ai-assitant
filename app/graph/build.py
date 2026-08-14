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
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents import nodes
from app.config import Settings
from app.graph.state import AgentState, RiskAssessment
from app.observability.logging import get_logger
from app.retrieval.hybrid import HybridRetriever

log = get_logger(__name__)


def _after_guard(state: AgentState) -> Literal["refusal", "supervisor"]:
    risk: RiskAssessment = state.get("risk") or RiskAssessment()
    return "refusal" if risk.blocked else "supervisor"


def _after_validator(state: AgentState) -> Literal["response_agent", "__end__"]:
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
) -> Any:
    """Compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("ingress_guard", partial(nodes.ingress_guard, settings=settings))
    graph.add_node("refusal", nodes.refusal)
    graph.add_node("supervisor", partial(nodes.supervisor, settings=settings))
    graph.add_node(
        "retrieval_agent",
        partial(nodes.retrieval_agent, settings=settings, retriever=retriever),
    )
    graph.add_node("response_agent", partial(nodes.response_agent, settings=settings))
    graph.add_node("validator", partial(nodes.validator, settings=settings))

    graph.add_edge(START, "ingress_guard")
    graph.add_conditional_edges(
        "ingress_guard", _after_guard, {"refusal": "refusal", "supervisor": "supervisor"}
    )
    graph.add_edge("refusal", END)

    # The supervisor returns a Command naming its own destination, so its edges
    # are declared as the set of places it may go rather than as a condition
    # evaluated here.
    graph.add_edge("retrieval_agent", "response_agent")
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
        "supervisor": ["retrieval_agent", "response_agent"],
        "retrieval_agent": ["response_agent"],
        "response_agent": ["validator"],
        "validator": ["response_agent", END],
    }
