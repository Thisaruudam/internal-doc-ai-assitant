"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.api.dependencies import PrincipalDep, SettingsDep
from app.api.errors import AuthenticationError
from app.auth.principal import Principal
from app.auth.tokens import issue_token
from app.auth.users import UnknownUserError, authenticate, load_users
from app.observability.logging import get_logger

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)


class LoginRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class PrincipalView(BaseModel):
    """What the client is told about itself.

    Includes the derived access ceiling so the UI can show the caller what they
    can see. Derived server-side and re-derived on every request — the client is
    never trusted to report its own permissions.
    """

    user_id: str
    display_name: str
    role: str
    departments: list[str]
    access_ceiling: str

    @classmethod
    def of(cls, principal: Principal) -> PrincipalView:
        return cls(
            user_id=principal.user_id,
            display_name=principal.display_name,
            role=principal.role.value,
            departments=sorted(principal.departments),
            access_ceiling=principal.access_ceiling.name.lower(),
        )


class LoginResponse(BaseModel):
    access_token: str
    # OAuth2 bearer scheme name, not a credential.
    token_type: str = "bearer"  # noqa: S105
    expires_in: int
    principal: PrincipalView


@router.post("/login", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def login(payload: LoginRequest, settings: SettingsDep) -> LoginResponse:
    users = load_users(settings.users_file)
    try:
        principal = authenticate(users, payload.user_id, payload.password)
    except UnknownUserError as exc:
        # Log the attempt but return the same message for both failure modes.
        log.warning("login_failed", attempted_user=payload.user_id)
        raise AuthenticationError("Invalid username or password.") from exc

    token, expires_in = issue_token(principal, settings.security)
    log.info("login_succeeded", role=principal.role.value)
    return LoginResponse(
        access_token=token,
        expires_in=expires_in,
        principal=PrincipalView.of(principal),
    )


@router.get("/me", response_model=PrincipalView)
async def me(principal: PrincipalDep) -> PrincipalView:
    return PrincipalView.of(principal)
