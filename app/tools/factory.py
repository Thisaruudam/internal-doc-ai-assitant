"""Tool wiring.

Binds the tool registry's declarations to real implementations. Kept separate
from ``registry.py`` so the role and risk declarations stay reviewable on their
own — a reviewer checking "who can call what" should not have to read
connection handling to do it.

Handlers close over their dependencies (the retriever, the corpus, the MCP
endpoint) rather than reaching for globals, which is what lets the registry be
built with stubs in tests.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from app.auth.principal import Principal
from app.observability.logging import get_logger
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.schema import Chunk
from app.tools.mcp_client import McpClient
from app.tools.python_analysis import run_analysis
from app.tools.registry import (
    EmployeeDirectoryArgs,
    IncidentRecordsArgs,
    KnowledgeSearchArgs,
    PythonAnalysisArgs,
    ServiceCatalogArgs,
    ToolHandler,
    ToolRegistry,
    build_default_registry,
)

log = get_logger(__name__)


def _knowledge_search_handler(retriever: HybridRetriever) -> ToolHandler:
    async def handler(args: BaseModel, principal: Principal) -> Any:
        assert isinstance(args, KnowledgeSearchArgs)
        result = await retriever.retrieve(
            args.query,
            principal,
            departments=set(args.departments) if args.departments else None,
            document_types=set(args.document_types) if args.document_types else None,
            top_k=args.top_k,
        )
        return {
            "passages": [
                {
                    "chunk_id": scored.chunk.chunk_id,
                    "title": scored.chunk.metadata.title,
                    "section": scored.chunk.metadata.heading_path,
                    "text": scored.chunk.text,
                }
                for scored in result.chunks
            ],
            "degraded": result.degraded,
        }

    return handler


def _python_analysis_handler(corpus: dict[str, Chunk], timeout_s: float) -> ToolHandler:
    async def handler(args: BaseModel, principal: Principal) -> Any:
        assert isinstance(args, PythonAnalysisArgs)
        return await run_analysis(args.code, args.chunk_ids, corpus, principal, timeout_s=timeout_s)

    return handler


def _mcp_handler(client: McpClient | None, tool: str) -> ToolHandler:
    async def handler(args: BaseModel, principal: Principal) -> Any:
        if client is None:
            # Not an exception: the MCP server being down should cost the user
            # that data source, not the turn.
            return {
                "ok": False,
                "error": "the enterprise data service is not configured or unreachable",
            }
        arguments = {k: v for k, v in args.model_dump().items() if v is not None}
        result = await client.call(tool, arguments, principal)
        payload: dict[str, Any] = {"ok": result.ok, **result.payload}
        if result.error:
            payload["error"] = result.error
        note = result.describe()
        if note:
            payload["note"] = note
        return payload

    return handler


async def _admin_unavailable(args: BaseModel, principal: Principal) -> Any:
    """Placeholder for administrative tools.

    Registered rather than omitted so the role model stays complete and the
    human-approval gate is exercised end to end. Refusing here is honest: the
    operations behind these names are not implemented, and a tool that claims to
    have reindexed without doing so would be worse than one that says it cannot.
    """
    return {
        "ok": False,
        "error": "this administrative operation is not available in this deployment",
    }


def build_mcp_client(mcp_url: str, *, timeout_s: float = 10.0) -> McpClient | None:
    """Construct an MCP client that opens a session per call.

    The server runs stateless, so a session per call costs one round trip and
    avoids holding a connection that may have died between questions. If the
    import or the endpoint is unusable the client is ``None`` and every MCP tool
    degrades to a stated absence.
    """

    async def call_tool(tool: str, arguments: dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with (
            streamable_http_client(mcp_url) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            response = await session.call_tool(tool, arguments)

        if not response.content:
            return {}
        text = getattr(response.content[0], "text", "")
        return json.loads(text) if text else {}

    return McpClient(call_tool, timeout_s=timeout_s)


def build_registry(
    *,
    retriever: HybridRetriever,
    corpus: dict[str, Chunk],
    mcp_client: McpClient | None,
    analysis_timeout_s: float = 20.0,
) -> ToolRegistry:
    """Assemble the registry with live handlers."""
    handlers: dict[str, ToolHandler] = {
        "knowledge_search": _knowledge_search_handler(retriever),
        "python_analysis": _python_analysis_handler(corpus, analysis_timeout_s),
        "employee_directory": _mcp_handler(mcp_client, "employee_directory"),
        "service_catalog": _mcp_handler(mcp_client, "service_catalog"),
        "incident_records": _mcp_handler(mcp_client, "incident_records"),
        "admin_reindex": _admin_unavailable,
        "admin_purge_memory": _admin_unavailable,
    }
    registry = build_default_registry(handlers)
    log.info("tool_registry_built", tools=len(registry.all()), mcp=mcp_client is not None)
    return registry


#: Re-exported so nodes can name argument schemas without importing the registry
#: module directly.
ARG_SCHEMAS = {
    "knowledge_search": KnowledgeSearchArgs,
    "python_analysis": PythonAnalysisArgs,
    "employee_directory": EmployeeDirectoryArgs,
    "service_catalog": ServiceCatalogArgs,
    "incident_records": IncidentRecordsArgs,
}

__all__ = [
    "ARG_SCHEMAS",
    "build_mcp_client",
    "build_registry",
]
