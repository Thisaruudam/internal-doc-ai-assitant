"""Tool execution guard — layer 2 of the authorization stack.

Binding (layer 1) means the model is only offered tools its caller may use. This
layer assumes that guarantee has already failed and checks again at the moment of
invocation.

That is not paranoia for its own sake. Binding is a property of how the graph was
wired, and it can be defeated by things that have nothing to do with the model
misbehaving:

* a replayed or resumed conversation whose checkpoint was written under a
  different role, then restored after the user was downgraded;
* a tool call reconstructed from message history rather than freshly bound;
* a future refactor that binds the full catalogue by mistake — the kind of change
  that passes review because nothing looks wrong.

Every invocation therefore re-derives permission from the ``Principal`` in state.
The guard also validates arguments before the handler sees them, enforces the
declared timeout, and emits the activity events that make the call visible.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.auth.principal import Principal
from app.graph import events
from app.observability.logging import get_logger
from app.tools.registry import ToolRegistry, ToolSpec

log = get_logger(__name__)


class ToolDeniedError(Exception):
    """The caller is not permitted to run this tool."""


class ToolUnknownError(Exception):
    """No such tool. Usually a hallucinated tool name."""


class ToolArgumentsInvalidError(Exception):
    """Arguments failed schema validation before execution."""


class ToolTimeoutError(Exception):
    """The handler exceeded its declared timeout."""


class ApprovalRequired(Exception):  # noqa: N818
    """A high-risk tool needs human confirmation.

    Deliberately not named ``...Error``: this is control flow, not a
    failure, in the same sense as ``StopIteration``. The graph catches it
    and routes to the approval node.

    Raised rather than returned so it cannot be ignored by a caller that
    forgot to check a flag.
    """

    def __init__(self, spec: ToolSpec, arguments: dict[str, Any]) -> None:
        super().__init__(f"tool {spec.name!r} requires human approval")
        self.spec = spec
        self.arguments = arguments


@dataclass(slots=True)
class ToolOutcome:
    """The result of one guarded invocation."""

    tool: str
    ok: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    denied: bool = False


class ToolGuard:
    """Invokes tools on behalf of a principal, checking every time."""

    def __init__(self, registry: ToolRegistry, *, node: str = "tool_guard") -> None:
        self._registry = registry
        self._node = node

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        principal: Principal,
        *,
        depth: int = 0,
        approved: bool = False,
    ) -> ToolOutcome:
        """Run a tool, or refuse to.

        ``approved`` is set only by the human-in-the-loop node after a person
        confirmed this specific call. It is never derived from model output.
        """
        started = time.perf_counter()

        spec = self._registry.get(name)
        if spec is None:
            # A hallucinated tool name, or one the caller was never offered.
            # Logged at warning: it is a signal that something is off, whether a
            # confused model or an injection attempt.
            events.tool_call(self._node, name, arguments, allowed=False, depth=depth)
            log.warning("tool_unknown", tool=name, role=principal.role.value)
            raise ToolUnknownError(f"no tool named {name!r}")

        # ── The re-check. Layer 1 already filtered; this assumes it failed. ──
        if not spec.permitted_for(principal):
            events.tool_call(self._node, name, arguments, allowed=False, depth=depth)
            log.warning(
                "tool_denied",
                tool=name,
                role=principal.role.value,
                required_role=spec.required_role.value,
            )
            raise ToolDeniedError(
                f"the {principal.role.value} role may not use {name!r}; "
                f"it requires {spec.required_role.value} or higher"
            )

        # Arguments are model output, so validate before the handler sees them.
        try:
            validated = spec.args_schema.model_validate(arguments)
        except ValidationError as exc:
            events.tool_call(self._node, name, arguments, allowed=False, depth=depth)
            log.warning("tool_arguments_invalid", tool=name, errors=exc.error_count())
            raise ToolArgumentsInvalidError(
                f"arguments for {name!r} failed validation: {_summarise(exc)}"
            ) from exc

        # High-risk tools stop here unless a human already said yes. Checked
        # after validation so the approval prompt shows well-formed arguments.
        if spec.requires_approval and not approved:
            events.tool_call(self._node, name, arguments, allowed=False, depth=depth)
            log.info("tool_awaiting_approval", tool=name, role=principal.role.value)
            raise ApprovalRequired(spec, arguments)

        events.tool_call(self._node, name, arguments, allowed=True, depth=depth)

        try:
            result = await asyncio.wait_for(
                spec.handler(validated, principal), timeout=spec.timeout_s
            )
        except TimeoutError as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            events.tool_result(
                self._node,
                name,
                ok=False,
                duration_ms=duration_ms,
                summary="timed out",
                depth=depth,
            )
            log.warning("tool_timeout", tool=name, timeout_s=spec.timeout_s)
            raise ToolTimeoutError(f"{name!r} exceeded its {spec.timeout_s}s timeout") from exc
        except Exception as exc:  # a failing tool degrades the turn, never kills it
            duration_ms = (time.perf_counter() - started) * 1000
            events.tool_result(
                self._node,
                name,
                ok=False,
                duration_ms=duration_ms,
                summary=type(exc).__name__,
                depth=depth,
            )
            log.warning("tool_failed", tool=name, error_type=type(exc).__name__)
            return ToolOutcome(
                tool=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
            )

        duration_ms = (time.perf_counter() - started) * 1000
        events.tool_result(
            self._node,
            name,
            ok=True,
            duration_ms=duration_ms,
            summary=_summarise_result(result),
            depth=depth,
        )
        return ToolOutcome(tool=name, ok=True, result=result, duration_ms=duration_ms)

    def bind_for(self, principal: Principal) -> list[dict[str, Any]]:
        """Layer 1 — the tool schemas offered to the model for this caller."""
        return [spec.to_model_schema() for spec in self._registry.tools_for(principal)]


def _summarise(exc: ValidationError) -> str:
    """Field-level detail only.

    Safe to return: it describes the caller's own arguments. Values are omitted
    because they may contain user data.
    """
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3])


def _summarise_result(result: Any) -> str:
    """A short shape description for the activity panel — never the payload."""
    if isinstance(result, list):
        return f"{len(result)} item(s)"
    if isinstance(result, dict):
        return f"{len(result)} field(s)"
    if isinstance(result, BaseModel):
        return type(result).__name__
    return type(result).__name__
