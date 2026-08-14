"""JWT issuance and verification.

Deliberately minimal. The token carries identity — who the caller is and what
role they hold — and nothing else. The access ceiling and the tool allowlist are
*derived* from the role at use time rather than stamped into the token, so
tightening a policy takes effect on the next request instead of whenever
outstanding tokens happen to expire.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt

from app.auth.principal import Principal
from app.config import SecuritySettings


class TokenError(Exception):
    """Raised for any invalid, expired, or malformed token."""


def issue_token(principal: Principal, settings: SecuritySettings) -> tuple[str, int]:
    """Mint a signed token. Returns ``(token, expires_in_seconds)``."""
    now = datetime.now(UTC)
    expires_at = now + timedelta(minutes=settings.jwt_ttl_minutes)

    claims = principal.to_claims() | {
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "iss": "atrium",
    }
    token = jwt.encode(
        claims,
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return token, settings.jwt_ttl_minutes * 60


def verify_token(token: str, settings: SecuritySettings) -> Principal:
    """Verify a token and rebuild the caller's ``Principal``.

    ``algorithms`` is pinned to the single configured algorithm. Accepting the
    algorithm named in the token header is the classic JWT confusion bug, and
    passing a list containing ``none`` or an asymmetric variant would let a
    caller forge tokens.
    """
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            issuer="atrium",
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("token is invalid") from exc

    try:
        return Principal.from_claims(claims)
    except (KeyError, ValueError) as exc:
        raise TokenError("token claims are malformed") from exc
