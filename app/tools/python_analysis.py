"""The Python analysis tool.

Lets the agent compute over retrieved evidence rather than reason about it in
prose. Counting incidents by root cause, ranking services by outage minutes, or
averaging a duration are things a language model does unreliably and three lines
of Python does exactly — and the difference matters most on precisely the
questions this system exists to answer.

Two constraints shape the interface:

**The code is untrusted.** It is written by a model that may have been injected,
so it runs through ``app.rlm.sandbox`` rather than in this process.

**The data is authorized before it enters the sandbox.** The model supplies
``chunk_ids``; those are model output, and are therefore filtered against the
caller's ``Principal`` before anything is handed over. Skipping that would make
this tool an authorization bypass with extra steps: a model that cannot
*retrieve* a restricted chunk could otherwise still *analyse* one by naming its
id.
"""

from __future__ import annotations

from typing import Any

from app.auth.principal import Principal
from app.graph import events
from app.observability.logging import get_logger
from app.retrieval.schema import Chunk
from app.rlm.sandbox import SandboxResult, run_sandboxed

log = get_logger(__name__)

#: Fields exposed to the analysis code as ``rows``. Metadata plus the text, so
#: grouping and counting work without the code needing to parse anything.
_ROW_FIELDS = (
    "doc_id",
    "department",
    "document_type",
    "access_level",
    "created_date",
    "owner",
    "heading_path",
    "token_estimate",
)

#: Ceiling on rows handed to the sandbox. An analysis over thousands of chunks
#: is not what this tool is for, and the serialised payload crosses a pipe.
MAX_ROWS = 200


def chunk_to_row(chunk: Chunk) -> dict[str, Any]:
    """Flatten a chunk into a plain dict for the sandbox."""
    metadata = chunk.metadata
    row: dict[str, Any] = {"chunk_id": chunk.chunk_id, "text": chunk.text}
    for name in _ROW_FIELDS:
        row[name] = getattr(metadata, name)
    row["tags"] = list(metadata.tags)
    return row


def authorize_rows(
    chunk_ids: list[str],
    available: dict[str, Chunk],
    principal: Principal,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve model-supplied ids into rows this caller may actually read.

    Returns the permitted rows and the ids that were refused. Unknown ids and
    unauthorized ids are both refused, and neither is distinguished in what the
    model is told — confirming that an id exists but is out of reach is itself
    a disclosure.
    """
    rows: list[dict[str, Any]] = []
    refused: list[str] = []

    for chunk_id in chunk_ids[:MAX_ROWS]:
        chunk = available.get(chunk_id)
        if chunk is None:
            refused.append(chunk_id)
            continue
        if not principal.may_read(chunk.metadata.access_level, chunk.metadata.department):
            refused.append(chunk_id)
            continue
        rows.append(chunk_to_row(chunk))

    if refused:
        log.info(
            "analysis_rows_refused",
            refused=len(refused),
            permitted=len(rows),
            role=principal.role.value,
        )
    return rows, refused


def format_outcome(result: SandboxResult, refused: list[str]) -> dict[str, Any]:
    """Shape the sandbox outcome into what the agent sees."""
    payload: dict[str, Any] = {
        "ok": result.ok,
        "result": result.result,
        "stdout": result.stdout,
        "duration_ms": round(result.duration_ms, 1),
    }
    if result.error:
        payload["error"] = result.error
    if result.violations:
        # Returned so the model can correct itself rather than retry blindly.
        payload["violations"] = result.violations
    if refused:
        payload["rows_unavailable"] = len(refused)
        payload["note"] = (
            f"{len(refused)} requested passage(s) were not available for analysis "
            "and were excluded."
        )
    return payload


async def run_analysis(
    code: str,
    chunk_ids: list[str],
    available: dict[str, Chunk],
    principal: Principal,
    *,
    timeout_s: float = 20.0,
    memory_mb: int = 512,
) -> dict[str, Any]:
    """Authorize the inputs, then run the code against them."""
    rows, refused = authorize_rows(chunk_ids, available, principal)

    events.retrieval_stage(
        "analysis_agent",
        "analysis_input",
        count=len(rows),
        refused=len(refused),
    )

    if not rows:
        return {
            "ok": False,
            "result": None,
            "error": (
                "none of the requested passages were available for analysis; "
                "retrieve evidence first, then analyse it"
            ),
            "rows_unavailable": len(refused),
        }

    result = await run_sandboxed(code, rows, timeout_s=timeout_s, memory_mb=memory_mb)

    if result.rejected:
        log.warning("analysis_code_rejected", violations=result.violations[:3])
    elif not result.ok:
        log.info("analysis_failed", error=result.error)

    return format_outcome(result, refused)


#: Handed to the model alongside the tool schema. Describes the environment
#: precisely, because a model that knows the constraints writes code that runs
#: first time rather than discovering the rules through rejections.
ANALYSIS_PROMPT = """\
Write a short Python snippet to compute over the retrieved passages.

Available:
  rows    a list of dicts, one per passage, with keys: chunk_id, text, doc_id,
          department, document_type, access_level, created_date, owner,
          heading_path, token_estimate, tags
  result  assign your final answer to this name
  print() output is captured and returned

You may import only: collections, datetime, itertools, json, math, re,
statistics, string, textwrap.

Not available: file access, network access, imports outside that list,
getattr/setattr, class definitions, try/except, and any name beginning with an
underscore. Write code that works rather than code that probes what is allowed.

Prefer computing an exact answer over describing one. Assign a small, plain
structure to `result` — a dict, list, or number — not a formatted string.
"""
