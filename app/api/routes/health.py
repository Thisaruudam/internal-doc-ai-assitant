"""Health and dependency-status routes.

Two endpoints with different jobs:

* ``/health`` — liveness. Answers "is this process up", nothing more. Cheap
  enough for a container probe to hit every second.
* ``/health/deps`` — readiness and degradation state. Reports each dependency's
  circuit-breaker state so the evaluator can see *before* asking a question which
  rung of the degradation ladder the system is currently on.

``/health/deps`` never fails the whole response because one dependency is down —
that is the point of it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.dependencies import SettingsDep
from app.observability.logging import get_logger

router = APIRouter(prefix="/health", tags=["health"])
log = get_logger(__name__)

DependencyState = Literal["ok", "degraded", "unavailable", "not_configured"]


class DependencyStatus(BaseModel):
    name: str
    state: DependencyState
    detail: str
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    environment: str
    organization: str


class DependencyReport(BaseModel):
    #: ``ok`` only when every dependency is ok; ``degraded`` when the system can
    #: still answer via a fallback; ``unavailable`` when it cannot.
    status: Literal["ok", "degraded", "unavailable"]
    dependencies: list[DependencyStatus]


@router.get("", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        organization=settings.organization_name,
    )


async def _probe(name: str, probe: Awaitable[str]) -> DependencyStatus:
    """Run one probe, converting any failure into a status rather than an error."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        detail = await asyncio.wait_for(probe, timeout=5.0)
        return DependencyStatus(
            name=name,
            state="ok",
            detail=detail,
            latency_ms=round((loop.time() - started) * 1000, 2),
        )
    except TimeoutError:
        return DependencyStatus(name=name, state="unavailable", detail="probe timed out")
    except Exception as exc:
        return DependencyStatus(name=name, state="unavailable", detail=f"{type(exc).__name__}")


@router.get("/deps", response_model=DependencyReport)
async def dependencies(settings: SettingsDep) -> DependencyReport:
    """Probe every dependency concurrently.

    Registered probes are filled in as each subsystem lands; the shape is fixed
    now so the UI and the compose healthchecks can be written against it.
    """
    probes: dict[str, Awaitable[str]] = {}

    statuses: list[DependencyStatus] = []
    if probes:
        statuses = list(
            await asyncio.gather(*(_probe(name, probe) for name, probe in probes.items()))
        )

    # Placeholder rows keep the contract stable while subsystems are being built.
    if not statuses:
        statuses = [
            DependencyStatus(name=name, state="not_configured", detail="probe not yet registered")
            for name in ("gemini", "pinecone", "postgres", "redis", "mcp")
        ]

    if any(s.state == "unavailable" for s in statuses):
        overall: Literal["ok", "degraded", "unavailable"] = "degraded"
    elif all(s.state == "ok" for s in statuses):
        overall = "ok"
    else:
        overall = "degraded"

    return DependencyReport(status=overall, dependencies=statuses)
