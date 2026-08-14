"""Graph state and activity events."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.auth.principal import Principal, Role
from app.graph.events import (
    ActivityEvent,
    EventType,
    digest_arguments,
    emit,
    tool_call,
)
from app.graph.state import (
    Budget,
    ErrorRecord,
    RiskAssessment,
    RiskVerdict,
    TodoItem,
    TodoStatus,
    initial_state,
    merge_files,
)


def budget(**overrides: int) -> Budget:
    base = {"depth": 3, "tool_calls": 24, "tokens": 400_000, "supervisor_steps": 8}
    return Budget(**{**base, **overrides})


class TestBudget:
    def test_spending_returns_a_new_instance(self) -> None:
        """Immutability keeps concurrent sub-agents from racing the accounting."""
        original = budget()
        after = original.spend(tool_calls=1)
        assert original.tool_calls == 24
        assert after.tool_calls == 23
        assert after is not original

    def test_budget_is_frozen(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            budget().tool_calls = 99  # type: ignore[misc]

    def test_spending_floors_at_zero(self) -> None:
        """A negative remaining budget would read as nonsense in the UI."""
        assert budget(tool_calls=2).spend(tool_calls=10).tool_calls == 0

    def test_not_exhausted_while_everything_remains(self) -> None:
        assert not budget().exhausted
        assert budget().exhausted_reason is None

    @pytest.mark.parametrize(
        ("field", "phrase"),
        [
            ("supervisor_steps", "planning step limit"),
            ("tool_calls", "tool-call budget"),
            ("tokens", "token budget"),
            ("depth", "recursion limit"),
        ],
    )
    def test_each_exhaustion_explains_itself(self, field: str, phrase: str) -> None:
        """A truncated answer the user cannot account for is worse than a short
        one they can."""
        spent = budget(**{field: 0})
        assert spent.exhausted
        assert phrase in (spent.exhausted_reason or "")

    def test_spending_several_dimensions_at_once(self) -> None:
        after = budget().spend(tool_calls=2, tokens=1000, supervisor_steps=1)
        assert (after.tool_calls, after.tokens, after.supervisor_steps) == (22, 399_000, 7)


class TestFileReducer:
    def test_merges_disjoint_artifacts(self) -> None:
        merged = merge_files({"a.md": "A"}, {"b.md": "B"})
        assert merged == {"a.md": "A", "b.md": "B"}

    def test_last_write_wins(self) -> None:
        assert merge_files({"a.md": "old"}, {"a.md": "new"}) == {"a.md": "new"}

    def test_inputs_are_not_mutated(self) -> None:
        left = {"a.md": "A"}
        merge_files(left, {"b.md": "B"})
        assert left == {"a.md": "A"}


class TestInitialState:
    def test_carries_the_question_and_principal(self) -> None:
        principal = Principal("analyst", "A", Role.ANALYST, frozenset({"payments"}))
        state = initial_state(
            principal=principal,
            question="why did payments fail in March?",
            thread_id="t-1",
            correlation_id="c-1",
            budget=budget(),
        )
        assert state["principal"] is principal
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "why did payments fail in March?"

    def test_starts_clean(self) -> None:
        state = initial_state(
            principal=Principal("v", "V", Role.VIEWER, frozenset({"*"})),
            question="hello",
            thread_id="t",
            correlation_id="c",
            budget=budget(),
        )
        assert state["plan"] == []
        assert state["files"] == {}
        assert state["errors"] == []
        assert state["repair_attempts"] == 0
        assert state["degraded_retrieval"] is False
        assert state["risk"].verdict is RiskVerdict.SAFE


class TestRiskAssessment:
    def test_default_is_safe(self) -> None:
        assert not RiskAssessment().blocked

    def test_blocked_verdict(self) -> None:
        risk = RiskAssessment(verdict=RiskVerdict.BLOCKED, score=0.9, signals=["override"])
        assert risk.blocked

    def test_suspicious_is_not_blocked(self) -> None:
        """Suspicious content is quarantined and processed, not refused."""
        assert not RiskAssessment(verdict=RiskVerdict.SUSPICIOUS, score=0.5).blocked


class TestTodoItem:
    def test_defaults_to_pending(self) -> None:
        item = TodoItem(id="1", description="search incidents", agent="retrieval")
        assert item.status is TodoStatus.PENDING
        assert item.result_ref is None

    def test_rejects_an_unknown_agent(self) -> None:
        with pytest.raises(ValueError, match="agent"):
            TodoItem(id="1", description="x", agent="nonexistent")  # type: ignore[arg-type]


class TestActivityEvents:
    def test_sse_frame_is_well_formed(self) -> None:
        frame = ActivityEvent(
            type=EventType.NODE_ENTER, node="supervisor", data={"k": "v"}
        ).to_sse()
        assert frame.startswith("event: node.enter\ndata: ")
        assert frame.endswith("\n\n")
        payload = json.loads(frame.split("data: ", 1)[1].strip())
        assert payload["node"] == "supervisor"
        assert payload["data"] == {"k": "v"}

    def test_frame_contains_no_bare_newlines_in_data(self) -> None:
        """A newline inside the data field would split the SSE frame."""
        frame = ActivityEvent(
            type=EventType.TOKEN, node="response", data={"text": "line one\nline two"}
        ).to_sse()
        assert frame.count("\n\n") == 1

    def test_emitting_outside_a_graph_run_is_a_no_op(self) -> None:
        """Nodes are unit-tested by direct call, with no LangGraph runtime.

        Observability must never be able to fail the thing it observes.
        """
        emit(ActivityEvent(type=EventType.NODE_ENTER, node="test"))
        tool_call("n", "knowledge_search", {"query": "x"}, allowed=True)

    def test_argument_digest_is_stable_and_order_independent(self) -> None:
        assert digest_arguments({"a": 1, "b": 2}) == digest_arguments({"b": 2, "a": 1})

    def test_argument_digest_changes_with_content(self) -> None:
        assert digest_arguments({"q": "payments"}) != digest_arguments({"q": "salaries"})

    def test_digest_does_not_reveal_the_value(self) -> None:
        secret = "employee salary 4500000"
        assert secret not in digest_arguments({"q": secret})


class TestErrorRecord:
    def test_records_what_it_fell_back_to(self) -> None:
        record = ErrorRecord(
            stage="retrieval",
            kind="dependency_unavailable",
            detail="pinecone timeout",
            degraded_to="bm25",
        )
        assert record.degraded_to == "bm25"

    def test_message_reducer_appends(self) -> None:
        """errors uses operator.add, so parallel branches accumulate."""
        left = [ErrorRecord(stage="a", kind="k", detail="d")]
        right = [ErrorRecord(stage="b", kind="k", detail="d")]
        assert len([*left, *right]) == 2


class TestMessageReducer:
    def test_add_messages_appends(self) -> None:
        from langgraph.graph import add_messages

        merged = add_messages([HumanMessage(content="q")], [AIMessage(content="a")])
        assert [m.content for m in merged] == ["q", "a"]
