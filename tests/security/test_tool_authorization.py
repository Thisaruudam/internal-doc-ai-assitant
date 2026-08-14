"""Layers 1 and 2 of the authorization stack.

Written adversarially: each test assumes the model has been fully compromised
and is emitting the most useful call it can for an attacker.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from app.auth.principal import Principal, Role
from app.tools.guard import (
    ApprovalRequired,
    ToolArgumentsInvalidError,
    ToolDeniedError,
    ToolGuard,
    ToolTimeoutError,
    ToolUnknownError,
)
from app.tools.registry import RiskLevel, ToolRegistry, ToolSpec, build_default_registry

pytestmark = pytest.mark.security


def principal(role: Role, departments: set[str] | None = None) -> Principal:
    return Principal("t", "Test", role, frozenset(departments or {"*"}))


async def _echo(args: BaseModel, _principal: Principal) -> dict:
    return {"ok": True, "args": args.model_dump()}


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_registry(
        dict.fromkeys(
            [
                "knowledge_search",
                "python_analysis",
                "employee_directory",
                "service_catalog",
                "incident_records",
                "admin_reindex",
                "admin_purge_memory",
            ],
            _echo,
        )
    )


@pytest.fixture
def guard(registry: ToolRegistry) -> ToolGuard:
    return ToolGuard(registry)


class TestLayerOneBinding:
    """A tool the caller may not use is absent from the model's schema."""

    def test_viewer_is_offered_only_search(self, registry: ToolRegistry) -> None:
        assert registry.names_for(principal(Role.VIEWER)) == ["knowledge_search"]

    def test_analyst_gains_analytics_and_mcp(self, registry: ToolRegistry) -> None:
        names = registry.names_for(principal(Role.ANALYST))
        assert "python_analysis" in names
        assert "employee_directory" in names
        assert not any(n.startswith("admin_") for n in names)

    def test_administrator_is_offered_everything(self, registry: ToolRegistry) -> None:
        assert registry.names_for(principal(Role.ADMINISTRATOR)) == [
            spec.name for spec in registry.all()
        ]

    def test_admin_tools_are_invisible_to_lower_roles(self, registry: ToolRegistry) -> None:
        """Not merely refused — absent. There is no name for the model to emit."""
        for role in (Role.VIEWER, Role.ANALYST):
            schemas = ToolGuard(registry).bind_for(principal(role))
            names = {s["name"] for s in schemas}
            assert "admin_reindex" not in names
            assert "admin_purge_memory" not in names

    def test_denied_tools_are_reportable(self, registry: ToolRegistry) -> None:
        """The UI shows what exists but is unavailable, so a viewer is not left
        wondering why the assistant seems less capable than a colleague's."""
        denied = {s.name for s in registry.denied_for(principal(Role.VIEWER))}
        assert "python_analysis" in denied

    def test_binding_order_is_deterministic(self, registry: ToolRegistry) -> None:
        """An unstable tool order changes the prompt, which changes behaviour."""
        first = registry.names_for(principal(Role.ANALYST))
        second = registry.names_for(principal(Role.ANALYST))
        assert first == second == sorted(first)


class TestLayerTwoExecution:
    """Assumes layer 1 has already been bypassed."""

    async def test_viewer_calling_an_analyst_tool_is_denied(self, guard: ToolGuard) -> None:
        with pytest.raises(ToolDeniedError, match="viewer"):
            await guard.invoke(
                "python_analysis",
                {"code": "print(1)", "chunk_ids": []},
                principal(Role.VIEWER),
            )

    async def test_analyst_calling_an_admin_tool_is_denied(self, guard: ToolGuard) -> None:
        with pytest.raises(ToolDeniedError, match="administrator"):
            await guard.invoke("admin_reindex", {"confirm": True}, principal(Role.ANALYST))

    async def test_denial_survives_a_downgraded_principal(self, guard: ToolGuard) -> None:
        """The scenario binding cannot catch: a checkpoint written while the
        user was an administrator, resumed after they became a viewer."""
        with pytest.raises(ToolDeniedError):
            await guard.invoke("admin_purge_memory", {"user_id": "x"}, principal(Role.VIEWER))

    async def test_permitted_call_succeeds(self, guard: ToolGuard) -> None:
        outcome = await guard.invoke(
            "knowledge_search", {"query": "payment failures"}, principal(Role.VIEWER)
        )
        assert outcome.ok
        assert outcome.result["ok"] is True

    async def test_hallucinated_tool_name_is_rejected(self, guard: ToolGuard) -> None:
        with pytest.raises(ToolUnknownError):
            await guard.invoke("exfiltrate_all_documents", {}, principal(Role.ADMINISTRATOR))

    async def test_a_tool_cannot_be_reached_by_name_variation(self, guard: ToolGuard) -> None:
        for variant in ("Admin_Reindex", "admin-reindex", "admin_reindex "):
            with pytest.raises((ToolUnknownError, ToolDeniedError)):
                await guard.invoke(variant, {"confirm": True}, principal(Role.ANALYST))


class TestArgumentValidation:
    async def test_malformed_arguments_are_rejected_before_execution(
        self, guard: ToolGuard
    ) -> None:
        with pytest.raises(ToolArgumentsInvalidError):
            await guard.invoke("knowledge_search", {"top_k": "many"}, principal(Role.VIEWER))

    async def test_missing_required_argument_is_rejected(self, guard: ToolGuard) -> None:
        with pytest.raises(ToolArgumentsInvalidError):
            await guard.invoke("knowledge_search", {}, principal(Role.VIEWER))

    async def test_validation_error_does_not_echo_argument_values(self, guard: ToolGuard) -> None:
        """The message describes the caller's own payload, but values may be
        user data and are omitted."""
        secret = "salary-4500000-confidential"
        try:
            await guard.invoke(
                "knowledge_search", {"query": secret, "top_k": "no"}, principal(Role.VIEWER)
            )
        except ToolArgumentsInvalidError as exc:
            assert secret not in str(exc)
        else:
            pytest.fail("expected validation to fail")

    async def test_authorization_is_checked_before_arguments(self, guard: ToolGuard) -> None:
        """A denied caller must not learn a tool's schema by probing it with
        deliberately malformed arguments."""
        with pytest.raises(ToolDeniedError):
            await guard.invoke("admin_reindex", {"nonsense": 1}, principal(Role.VIEWER))


class TestHumanApproval:
    async def test_high_risk_tool_requires_approval_even_for_an_administrator(
        self, guard: ToolGuard
    ) -> None:
        """Seniority does not bypass the gate."""
        with pytest.raises(ApprovalRequired) as caught:
            await guard.invoke("admin_reindex", {"confirm": True}, principal(Role.ADMINISTRATOR))
        assert caught.value.spec.name == "admin_reindex"

    async def test_approved_call_proceeds(self, guard: ToolGuard) -> None:
        outcome = await guard.invoke(
            "admin_reindex", {"confirm": True}, principal(Role.ADMINISTRATOR), approved=True
        )
        assert outcome.ok

    async def test_approval_does_not_bypass_authorization(self, guard: ToolGuard) -> None:
        """approved=True confirms a specific call; it never grants a role."""
        with pytest.raises(ToolDeniedError):
            await guard.invoke(
                "admin_reindex", {"confirm": True}, principal(Role.ANALYST), approved=True
            )

    async def test_low_risk_tools_need_no_approval(self, guard: ToolGuard) -> None:
        outcome = await guard.invoke("knowledge_search", {"query": "x"}, principal(Role.ANALYST))
        assert outcome.ok


class TestFailureHandling:
    async def test_timeout_is_enforced(self) -> None:
        async def hangs(_args: BaseModel, _principal: Principal) -> None:
            await asyncio.sleep(5)

        registry = ToolRegistry()
        from app.tools.registry import KnowledgeSearchArgs

        registry.register(
            ToolSpec(
                name="slow",
                description="hangs",
                args_schema=KnowledgeSearchArgs,
                handler=hangs,
                required_role=Role.VIEWER,
                timeout_s=0.05,
            )
        )
        with pytest.raises(ToolTimeoutError, match=r"0\.05"):
            await ToolGuard(registry).invoke("slow", {"query": "x"}, principal(Role.VIEWER))

    async def test_a_failing_tool_degrades_rather_than_raising(self) -> None:
        """One broken tool must not kill the turn."""

        async def boom(_args: BaseModel, _principal: Principal) -> None:
            raise RuntimeError("upstream exploded")

        registry = ToolRegistry()
        from app.tools.registry import KnowledgeSearchArgs

        registry.register(
            ToolSpec(
                name="broken",
                description="fails",
                args_schema=KnowledgeSearchArgs,
                handler=boom,
                required_role=Role.VIEWER,
            )
        )
        outcome = await ToolGuard(registry).invoke("broken", {"query": "x"}, principal(Role.VIEWER))
        assert outcome.ok is False
        assert "upstream exploded" in (outcome.error or "")


class TestRegistryIntegrity:
    def test_duplicate_registration_is_rejected(self, registry: ToolRegistry) -> None:
        spec = registry.get("knowledge_search")
        assert spec is not None
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)

    def test_every_tool_declares_a_timeout(self, registry: ToolRegistry) -> None:
        assert all(spec.timeout_s > 0 for spec in registry.all())

    def test_every_mutating_tool_is_high_risk_and_non_idempotent(
        self, registry: ToolRegistry
    ) -> None:
        for spec in registry.all():
            if spec.name.startswith("admin_"):
                assert spec.risk_level is RiskLevel.HIGH
                assert not spec.idempotent

    def test_specs_are_immutable(self, registry: ToolRegistry) -> None:
        spec = registry.get("knowledge_search")
        assert spec is not None
        with pytest.raises((AttributeError, TypeError, ValueError)):
            spec.required_role = Role.VIEWER  # type: ignore[misc]
