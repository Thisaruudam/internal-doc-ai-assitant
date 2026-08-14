"""Authentication endpoints and token handling."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.auth.principal import Principal, Role
from app.auth.tokens import TokenError, issue_token, verify_token
from app.config import SecuritySettings


@pytest.fixture
def security() -> SecuritySettings:
    return SecuritySettings(jwt_secret=SecretStr("unit-test-signing-key"))


class TestLogin:
    def test_each_demo_role_can_sign_in(self, client: TestClient) -> None:
        for user_id, expected_role in [
            ("viewer", "viewer"),
            ("analyst", "analyst"),
            ("admin", "administrator"),
        ]:
            response = client.post(
                "/auth/login",
                json={"user_id": user_id, "password": f"{user_id}-demo-2026"},
            )
            assert response.status_code == 200
            assert response.json()["principal"]["role"] == expected_role

    def test_wrong_password_and_unknown_user_are_indistinguishable(
        self, client: TestClient
    ) -> None:
        """No account-enumeration oracle: same status, same body."""
        wrong = client.post("/auth/login", json={"user_id": "viewer", "password": "nope"})
        absent = client.post("/auth/login", json={"user_id": "ghost", "password": "nope"})

        assert wrong.status_code == absent.status_code == 401
        assert wrong.json()["detail"] == absent.json()["detail"]
        assert wrong.json()["title"] == absent.json()["title"]

    def test_unknown_user_costs_the_same_as_a_wrong_password(self, client: TestClient) -> None:
        """Timing must not reveal existence either.

        Generous bound — this asserts the dummy-hash comparison happens at all,
        not a precise constant. Without it the absent-user path returns in
        microseconds while the real path pays for bcrypt.
        """

        def elapsed(user_id: str) -> float:
            start = time.perf_counter()
            client.post("/auth/login", json={"user_id": user_id, "password": "nope"})
            return time.perf_counter() - start

        real = min(elapsed("viewer") for _ in range(3))
        absent = min(elapsed("ghost") for _ in range(3))
        assert absent > real * 0.5

    @pytest.mark.parametrize(
        "payload",
        [
            {"user_id": "", "password": "x"},
            {"user_id": "viewer"},
            {"password": "x"},
            {"user_id": "v" * 65, "password": "x"},
        ],
    )
    def test_malformed_payloads_are_rejected(
        self, client: TestClient, payload: dict[str, str]
    ) -> None:
        response = client.post("/auth/login", json=payload)
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/problem+json")


class TestProtectedRoutes:
    def test_me_requires_a_token(self, client: TestClient) -> None:
        response = client.get("/auth/me")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    def test_me_returns_the_caller(self, client: TestClient, auth_header) -> None:
        response = client.get("/auth/me", headers=auth_header("analyst"))
        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == "analyst"
        assert body["access_ceiling"] == "confidential"
        assert body["departments"] == ["payments", "platform"]

    def test_tampered_signature_is_rejected(
        self, client: TestClient, tokens: dict[str, str]
    ) -> None:
        forged = tokens["viewer"][:-4] + "AAAA"
        response = client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"})
        assert response.status_code == 401


class TestTokenVerification:
    def test_round_trip(self, security: SecuritySettings) -> None:
        principal = Principal("analyst", "A", Role.ANALYST, frozenset({"payments"}))
        token, expires_in = issue_token(principal, security)
        assert expires_in == security.jwt_ttl_minutes * 60
        assert verify_token(token, security) == principal

    def test_expired_token_rejected(self, security: SecuritySettings) -> None:
        past = datetime.now(UTC) - timedelta(hours=2)
        token = jwt.encode(
            {
                "sub": "viewer",
                "role": "viewer",
                "departments": [],
                "iat": int(past.timestamp()),
                "exp": int((past + timedelta(minutes=1)).timestamp()),
                "iss": "atrium",
            },
            security.jwt_secret.get_secret_value(),
            algorithm=security.jwt_algorithm,
        )
        with pytest.raises(TokenError, match="expired"):
            verify_token(token, security)

    def test_alg_none_token_is_rejected(self, security: SecuritySettings) -> None:
        """The classic JWT confusion attack.

        ``verify_token`` pins ``algorithms`` to the configured algorithm, so an
        unsigned token claiming ``alg: none`` cannot be accepted.
        """
        unsigned = jwt.encode(
            {
                "sub": "admin",
                "role": "administrator",
                "departments": ["*"],
                "iat": int(datetime.now(UTC).timestamp()),
                "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
                "iss": "atrium",
            },
            key="",
            algorithm="none",
        )
        with pytest.raises(TokenError):
            verify_token(unsigned, security)

    def test_token_signed_with_another_key_is_rejected(self, security: SecuritySettings) -> None:
        attacker = SecuritySettings(jwt_secret=SecretStr("attacker-key"))
        principal = Principal("admin", "A", Role.ADMINISTRATOR, frozenset({"*"}))
        token, _ = issue_token(principal, attacker)
        with pytest.raises(TokenError):
            verify_token(token, security)

    def test_wrong_issuer_is_rejected(self, security: SecuritySettings) -> None:
        token = jwt.encode(
            {
                "sub": "admin",
                "role": "administrator",
                "departments": ["*"],
                "iat": int(datetime.now(UTC).timestamp()),
                "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
                "iss": "somewhere-else",
            },
            security.jwt_secret.get_secret_value(),
            algorithm=security.jwt_algorithm,
        )
        with pytest.raises(TokenError):
            verify_token(token, security)

    def test_missing_required_claims_rejected(self, security: SecuritySettings) -> None:
        token = jwt.encode(
            {"sub": "x", "role": "viewer"},  # no exp / iat / iss
            security.jwt_secret.get_secret_value(),
            algorithm=security.jwt_algorithm,
        )
        with pytest.raises(TokenError):
            verify_token(token, security)

    def test_unknown_role_claim_rejected(self, security: SecuritySettings) -> None:
        token = jwt.encode(
            {
                "sub": "x",
                "role": "superuser",
                "departments": [],
                "iat": int(datetime.now(UTC).timestamp()),
                "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
                "iss": "atrium",
            },
            security.jwt_secret.get_secret_value(),
            algorithm=security.jwt_algorithm,
        )
        with pytest.raises(TokenError, match="malformed"):
            verify_token(token, security)
