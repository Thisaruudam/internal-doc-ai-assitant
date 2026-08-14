"""Shared test fixtures.

Credentials for external services are stubbed at import time so the unit suite
never needs a live Gemini or Pinecone key. Tests that genuinely require those are
marked ``integration`` and skipped unless the real keys are present.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
os.environ.setdefault("PINECONE_API_KEY", "test-key-not-used")
os.environ.setdefault("OBSERVABILITY_LANGSMITH_TRACING", "false")

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import get_settings

#: Mirrors app/auth/users.yaml. Kept here so a change to the demo directory
#: breaks the tests loudly rather than silently weakening the RBAC suite.
DEMO_CREDENTIALS = {
    "viewer": "viewer-demo-2026",
    "analyst": "analyst-demo-2026",
    "admin": "admin-demo-2026",
}


@pytest.fixture(scope="session")
def client() -> TestClient:
    get_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture(scope="session")
def tokens(client: TestClient) -> dict[str, str]:
    """A valid bearer token per demo role."""
    issued: dict[str, str] = {}
    for user_id, password in DEMO_CREDENTIALS.items():
        response = client.post("/auth/login", json={"user_id": user_id, "password": password})
        assert response.status_code == 200, response.text
        issued[user_id] = response.json()["access_token"]
    return issued


@pytest.fixture
def auth_header(tokens: dict[str, str]):
    def _header(user_id: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {tokens[user_id]}"}

    return _header
