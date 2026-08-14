"""FastAPI dependencies.

The single place a ``Principal`` enters the system. Every protected route depends
on ``current_principal``; nothing else in the codebase parses a token or reads an
``Authorization`` header.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.errors import AuthenticationError, AuthorizationError
from app.auth.principal import Principal, Role, role_satisfies
from app.auth.tokens import TokenError, verify_token
from app.config import Settings, get_settings
from app.observability.logging import bind_request_context

# auto_error=False so a missing header produces our problem+json shape rather
# than Starlette's default JSON body.
_bearer = HTTPBearer(auto_error=False)


def settings_dep() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dep)]


async def current_principal(
    request: Request,
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Resolve the caller from the bearer token.

    Also binds the caller into the log context, so every downstream line in this
    request is attributable without any call site passing the user through.
    """
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Provide a bearer token.")

    try:
        principal = verify_token(credentials.credentials, settings.security)
    except TokenError as exc:
        raise AuthenticationError(str(exc)) from exc

    request.state.principal = principal
    bind_request_context(
        correlation_id=request.state.correlation_id,
        user_id=principal.user_id,
        role=principal.role.value,
    )
    return principal


PrincipalDep = Annotated[Principal, Depends(current_principal)]


def require_role(minimum: Role) -> Callable[[Principal], Awaitable[Principal]]:
    """Route guard enforcing a minimum role.

    This is the *API-layer* half of authorization. It is not the only check:
    tool execution re-verifies against the registry, and retrieval filters on the
    principal's access ceiling. See ``docs/architecture.md`` section 6 for why
    the same decision is made in three places.
    """

    async def _guard(principal: PrincipalDep) -> Principal:
        if not role_satisfies(principal.role, minimum):
            raise AuthorizationError(f"This action requires the {minimum.value} role or higher.")
        return principal

    return _guard
