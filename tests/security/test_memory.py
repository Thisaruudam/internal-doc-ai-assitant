"""Memory: recall scoping, role-change invalidation, and compaction.

Memory is the one path that carries content from one turn into another while
bypassing retrieval entirely, so none of the three authorization layers apply to
it. These tests exist because that makes memory a plausible leak with no other
control in front of it.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.auth.principal import Principal, Role
from app.config import GraphSettings
from app.memory.longterm import LongTermMemory, format_recalled
from app.memory.summarizer import (
    SUMMARY_PREFIX,
    estimate_tokens,
    should_compact,
    split_history,
)

pytestmark = pytest.mark.security


def principal(role: Role, user_id: str = "u1") -> Principal:
    return Principal(user_id, "Test", role, frozenset({"*"}))


@pytest.fixture
def memory() -> LongTermMemory:
    return LongTermMemory()


class TestRoleChangeInvalidation:
    """The property with no other control in front of it.

    A fact written while a user was an administrator must not be replayed after
    they become a viewer. The underlying document stays unreachable either way —
    but a summary of it, stored in memory, would not be.
    """

    def test_a_downgraded_user_loses_facts_written_at_a_higher_role(
        self, memory: LongTermMemory
    ) -> None:
        admin = principal(Role.ADMINISTRATOR)
        memory.remember(admin, "Works on the restricted compensation review")

        downgraded = principal(Role.VIEWER)
        assert memory.recall(downgraded, "compensation review") == []

    def test_the_same_user_at_the_same_role_still_recalls(
        self, memory: LongTermMemory
    ) -> None:
        admin = principal(Role.ADMINISTRATOR)
        memory.remember(admin, "Works on the compensation review")
        assert len(memory.recall(admin, "compensation review")) == 1

    def test_an_upgraded_user_recalls_facts_written_lower(
        self, memory: LongTermMemory
    ) -> None:
        """Promotion widens reach; it does not hide what was already visible."""
        viewer = principal(Role.VIEWER)
        memory.remember(viewer, "Works on payments reliability")

        promoted = principal(Role.ANALYST)
        assert len(memory.recall(promoted, "payments reliability")) == 1

    @pytest.mark.parametrize(
        ("writer", "reader", "recalled"),
        [
            (Role.VIEWER, Role.VIEWER, 1),
            (Role.ANALYST, Role.VIEWER, 0),
            (Role.ADMINISTRATOR, Role.ANALYST, 0),
            (Role.ANALYST, Role.ADMINISTRATOR, 1),
        ],
    )
    def test_ceiling_comparison_across_roles(
        self, memory: LongTermMemory, writer: Role, reader: Role, recalled: int
    ) -> None:
        memory.remember(principal(writer), "payments incident review work")
        assert len(memory.recall(principal(reader), "payments incident review")) == recalled


class TestUserScoping:
    def test_facts_never_cross_users(self, memory: LongTermMemory) -> None:
        memory.remember(principal(Role.ANALYST, "alice"), "Works on payments settlement")
        assert memory.recall(principal(Role.ANALYST, "bob"), "payments settlement") == []

    def test_same_role_different_user_is_still_separate(
        self, memory: LongTermMemory
    ) -> None:
        memory.remember(principal(Role.ADMINISTRATOR, "alice"), "owns the ledger migration")
        assert memory.count("alice") == 1
        assert memory.count("bob") == 0

    def test_forget_removes_everything_for_one_user(self, memory: LongTermMemory) -> None:
        alice = principal(Role.ANALYST, "alice")
        memory.remember(alice, "works on payments")
        memory.remember(principal(Role.ANALYST, "bob"), "works on platform")

        assert memory.forget("alice") == 1
        assert memory.count("alice") == 0
        assert memory.count("bob") == 1


class TestRecallRelevance:
    def test_unrelated_facts_are_not_injected(self, memory: LongTermMemory) -> None:
        """Recency alone is not a reason to put a fact in an unrelated question."""
        user = principal(Role.ANALYST)
        memory.remember(user, "Prefers concise answers about the payments domain")
        assert memory.recall(user, "what is the remote working policy") == []

    def test_relevant_facts_are_recalled(self, memory: LongTermMemory) -> None:
        user = principal(Role.ANALYST)
        memory.remember(user, "Responsible for settlement service reliability")
        recalled = memory.recall(user, "settlement service incidents")
        assert len(recalled) == 1

    def test_recall_is_capped(self, memory: LongTermMemory) -> None:
        user = principal(Role.ANALYST)
        for index in range(20):
            memory.remember(user, f"works on payments settlement topic {index}")
        assert len(memory.recall(user, "payments settlement")) <= 4

    def test_stored_facts_are_bounded(self, memory: LongTermMemory) -> None:
        user = principal(Role.ANALYST)
        for index in range(120):
            memory.remember(user, f"fact number {index} about payments")
        assert memory.count("u1") <= 60

    def test_recalled_block_forbids_citation(self, memory: LongTermMemory) -> None:
        """A fact is a compressed memory, not a passage. Citing one would put an
        unverifiable source into an answer that claims to be grounded."""
        user = principal(Role.ANALYST)
        memory.remember(user, "Works on payments")
        block = format_recalled(memory.recall(user, "payments"))
        assert "never cite" in block.lower()

    def test_empty_recall_renders_nothing(self) -> None:
        assert format_recalled([]) == ""


class TestCompaction:
    def settings(self, **overrides: int) -> GraphSettings:
        return GraphSettings(
            memory_verbatim_turns=overrides.get("verbatim", 4),
            memory_compact_threshold=overrides.get("threshold", 100),
        )

    def test_short_conversations_are_not_compacted(self) -> None:
        messages = [HumanMessage(content="hello"), AIMessage(content="hi")]
        assert not should_compact(messages, self.settings())

    def test_long_conversations_are_compacted(self) -> None:
        messages: list = []
        for index in range(12):
            messages.append(HumanMessage(content=f"question {index} " + "x" * 200))
            messages.append(AIMessage(content=f"answer {index} " + "y" * 200))
        assert should_compact(messages, self.settings())

    def test_recent_turns_are_kept_verbatim(self) -> None:
        messages: list = [
            HumanMessage(content=f"turn {index}") for index in range(10)
        ]
        _, retired, kept = split_history(messages, self.settings(verbatim=4))
        assert len(kept) == 4
        assert len(retired) == 6
        assert kept[-1].content == "turn 9"

    def test_an_existing_summary_is_replaced_not_stacked(self) -> None:
        """Otherwise every compaction adds another summary and the history grows
        in exactly the way compaction exists to prevent."""
        messages: list = [
            SystemMessage(content=f"{SUMMARY_PREFIX} earlier context"),
            *[HumanMessage(content=f"turn {index}") for index in range(8)],
        ]
        existing, retired, kept = split_history(messages, self.settings(verbatim=4))
        assert existing == "earlier context"
        assert all(SUMMARY_PREFIX not in str(m.content) for m in retired + kept)

    def test_token_estimate_grows_with_history(self) -> None:
        short = [HumanMessage(content="hi")]
        long = [HumanMessage(content="x" * 4000)]
        assert estimate_tokens(long) > estimate_tokens(short)
