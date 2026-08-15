"""Plan generation.

The model writes Python that partitions a manifest into batches. It is given the
corpus's actual vocabulary — real departments, real document types, real tags —
because a plan that filters on a plausible-sounding tag that does not exist
returns nothing, and the failure looks like "no relevant documents" rather than
"the plan was wrong".

The generated code runs in ``app.rlm.sandbox``. It has no network, so it cannot
call a model; it selects and partitions, and the orchestrator does the reasoning
over what it selected.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.models import ModelRole, get_model
from app.config import GeminiSettings
from app.observability.logging import get_logger
from app.rlm.sandbox import run_sandboxed

log = get_logger(__name__)

PLANNER_SYSTEM = """\
You write a short Python program that decides which documents an investigation \
should read, and how to group them.

You are given `rows`: a list of dicts, one per document chunk, with keys:
  chunk_id, doc_id, title, department, document_type, created_date (YYYY-MM-DD),
  owner, heading_path, token_estimate, tags

You do NOT get document text. You are selecting what to read, not reading it.

Assign to `result` a dict of exactly this shape:
  {"batches": [[chunk_id, ...], ...], "rationale": "one sentence"}

Rules for a good plan:
- Filter `rows` down to what the question actually needs, using tags,
  document_type, department, and created_date.
- Group the survivors into batches that can each be understood independently —
  by time window for "what recurred over time", by service or department for
  "compare across X". Aim for 3 to 6 batches.
- Keep each batch small enough to read: at most 12 chunk_ids.
- Prefer chunks whose heading_path suggests they answer the question. For a
  root-cause question that means sections named Root Cause, not Summary.
- If very few rows match, return one batch rather than an empty plan.

You may import: collections, datetime, itertools, json, math, re, statistics,
string, textwrap. No file or network access, no getattr, no classes, no
try/except. Write code that works rather than code that probes what is allowed.
"""


def build_planning_prompt(question: str, summary: dict[str, Any]) -> str:
    """Describe the corpus and the task to the planner."""
    return (
        f"Question:\n{question}\n\n"
        f"Corpus available to this user:\n{json.dumps(summary, indent=2)}\n\n"
        "Write the Python program. Assign the plan to `result`."
    )


#: A fenced code block anywhere in the response, with an optional language tag.
_CODE_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the program out of a model response.

    Models routinely explain themselves before the code — "An elegant way to
    solve this is..." — even when asked not to. Handling only a response that
    *begins* with a fence left that prose as line 1 of the program, which the
    sandbox correctly rejected as a syntax error and the agent then reported as
    a failed plan. So the fenced block is extracted from wherever it appears.
    """
    match = _CODE_FENCE.search(text)
    if match:
        return match.group(1).strip()

    # No fence: strip a leading bare ``` if present and hope the rest is code.
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
    return stripped.removesuffix("```").strip()


async def generate_plan_code(
    question: str, summary: dict[str, Any], settings: GeminiSettings
) -> str:
    """Ask the model for a partitioning program."""
    model = get_model(ModelRole.AGENT, settings)
    reply = await model.ainvoke(
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=build_planning_prompt(question, summary)),
        ]
    )
    text = getattr(reply, "text", None)
    if not isinstance(text, str) and callable(text):
        text = text()
    return extract_code(str(text or ""))


async def run_plan(
    code: str,
    manifest: list[dict[str, Any]],
    *,
    timeout_s: float = 20.0,
    memory_mb: int = 512,
) -> tuple[Any, str | None]:
    """Execute a plan in the sandbox. Returns ``(result, error)``."""
    outcome = await run_sandboxed(code, manifest, timeout_s=timeout_s, memory_mb=memory_mb)

    if outcome.rejected:
        # The code itself is logged at debug: a rejected plan is otherwise
        # undiagnosable, and "syntax error on line 12" without line 12 is not
        # an actionable message.
        log.warning(
            "rlm_plan_rejected",
            violations=outcome.violations[:3],
            code_preview=code[:400],
        )
        return None, "the generated plan was rejected: " + "; ".join(outcome.violations[:3])
    if not outcome.ok:
        log.info("rlm_plan_failed", error=outcome.error)
        return None, outcome.error

    return outcome.result, None


def fallback_batches(
    manifest: list[dict[str, Any]], *, batch_size: int = 8, max_batches: int = 4
) -> list[list[str]]:
    """Partition by date when planning fails.

    Not a silent substitute for a plan — the research agent reports that it fell
    back. But an arbitrary-yet-sensible partition still lets the map/reduce run,
    which beats abandoning the investigation because codegen misfired.
    """
    ordered = sorted(manifest, key=lambda row: str(row.get("created_date", "")))
    ids = [str(row["chunk_id"]) for row in ordered]
    return [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)][:max_batches]
