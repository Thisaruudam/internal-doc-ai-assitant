"""FastAPI application factory.

The whole app is assembled in one readable function so a reviewer can see the
middleware order, which is security-relevant: correlation binding must wrap
everything (so failures are traceable), and authentication must run before rate
limiting (so buckets are per-user rather than per-IP).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware.correlation import CorrelationIdMiddleware
from app.api.routes import auth, health
from app.config import Settings, get_settings
from app.observability.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shut-down.

    Configuration is validated here rather than lazily: a deployment with a
    placeholder signing key should refuse to start, not discover the problem
    when someone logs in.
    """
    settings: Settings = get_settings()
    settings.validate_production()

    log.info(
        "api_starting",
        environment=settings.environment,
        organization=settings.organization_name,
        tracing=settings.observability.langsmith_tracing,
    )
    yield
    log.info("api_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    # Configured at construction rather than in lifespan: anything logged during
    # app assembly — or by a test client that never runs lifespan — should still
    # land in the structured stream.
    configure_logging(settings.observability)

    app = FastAPI(
        title="Atrium",
        description=(
            f"Enterprise AI assistant over {settings.organization_name} organizational knowledge."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # Outermost middleware runs first. Correlation binding wraps everything so
    # that even an authentication rejection is logged against an id.
    app.add_middleware(CorrelationIdMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(auth.router)

    return app


app = create_app()
