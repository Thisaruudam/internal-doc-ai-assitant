"""FastAPI application factory.

The whole app is assembled in one readable function so a reviewer can see the
middleware order, which is security-relevant: correlation binding must wrap
everything (so failures are traceable), and authentication must resolve before
rate limiting (so buckets are per-user rather than per-IP).

Expensive, long-lived objects — the Pinecone connection pool, the BM25 index,
the compiled graph — are built once in the lifespan and held on ``app.state``.
Building them per request would add a TLS handshake and an index load to every
question asked.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver

from app.api.errors import register_exception_handlers
from app.api.middleware.correlation import CorrelationIdMiddleware
from app.api.middleware.ratelimit import TokenBucketLimiter
from app.api.routes import auth, chat, health
from app.config import Settings, get_settings
from app.graph.build import build_graph
from app.observability.langsmith import configure_tracing
from app.observability.logging import configure_logging, get_logger
from app.retrieval.bm25_store import load_or_none
from app.retrieval.chunking import chunk_corpus
from app.retrieval.corpus import CorpusError, load_documents
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pinecone_store import AsyncPineconeSearch
from app.tools.factory import build_mcp_client, build_registry
from app.tools.guard import ToolGuard

log = get_logger(__name__)

_PLACEHOLDER_KEYS = {"", "test-key-not-used", "your-key-here"}


def _pinecone_configured(settings: Settings) -> bool:
    return settings.pinecone.api_key.get_secret_value() not in _PLACEHOLDER_KEYS


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shut-down.

    Configuration is validated here rather than lazily: a deployment with a
    placeholder signing key should refuse to start, not discover the problem
    when someone logs in.
    """
    settings: Settings = get_settings()
    settings.validate_production()
    tracing = configure_tracing(settings.observability)

    async with AsyncExitStack() as stack:
        search = None
        if _pinecone_configured(settings):
            search = await stack.enter_async_context(
                AsyncPineconeSearch(settings.pinecone, settings.gemini)
            )
        else:
            # Not fatal. The BM25 index still answers questions, which is the
            # bottom rung of the degradation ladder rather than an outage.
            log.warning("pinecone_not_configured", detail="serving from the local index only")

        bm25 = load_or_none(settings.bm25_index_dir)
        if bm25 is None:
            log.warning("bm25_index_missing", detail="run `make seed` to build the fallback")

        retriever = HybridRetriever(
            search,
            bm25=bm25,
            rrf_k=settings.pinecone.rrf_k,
            alpha=settings.pinecone.hybrid_alpha,
            top_k_per_retriever=settings.pinecone.top_k_per_retriever,
            top_k_final=settings.pinecone.top_k_final,
        )

        # The corpus backs the research agent's manifest and the analysis
        # tool's chunk lookup. Loaded once here rather than per turn.
        try:
            corpus_chunks = chunk_corpus(load_documents(settings.corpus_dir))
        except CorpusError as exc:
            log.warning("corpus_unavailable", detail=str(exc)[:200])
            corpus_chunks = []

        mcp_client = build_mcp_client(settings.mcp_server_url)
        registry = build_registry(
            retriever=retriever,
            corpus={chunk.chunk_id: chunk for chunk in corpus_chunks},
            mcp_client=mcp_client,
            analysis_timeout_s=settings.graph.sandbox_timeout_s,
        )
        tool_guard = ToolGuard(registry)

        app.state.settings = settings
        app.state.retriever = retriever
        app.state.registry = registry
        app.state.tool_guard = tool_guard
        app.state.limiter = TokenBucketLimiter(settings.ratelimit)
        # In-memory checkpointing keeps session memory working without Postgres.
        # AsyncPostgresSaver is the durable swap and needs only this line changed.
        app.state.graph = build_graph(
            settings,
            retriever,
            checkpointer=InMemorySaver(),
            corpus_chunks=corpus_chunks,
            tool_guard=tool_guard,
        )

        log.info(
            "api_starting",
            environment=settings.environment,
            organization=settings.organization_name,
            tracing=tracing,
            pinecone=search is not None,
            bm25_fallback=bm25 is not None,
            corpus_chunks=len(corpus_chunks),
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
    app.include_router(chat.router)

    return app


app = create_app()
