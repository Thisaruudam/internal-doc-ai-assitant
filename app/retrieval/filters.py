"""Retrieval filter construction.

This module is the load-bearing security control in the whole system. Everything
else about authorization — which tools are bound to the model, what the tool
guard re-checks — defends against a model that is *behaving*. This defends
against one that is not.

The rule it enforces has one sentence: **a filter derived from the verified JWT
is always applied, and anything the model contributes can only narrow it.**

A prompt injection can rewrite the model's intent, forge a plausible tool call,
and ask for restricted salary records in fluent policy language. What it cannot
do is change the `access_level` clause that this module stamps onto every query,
because that clause is built from the `Principal` — which was written once by the
API layer from a verified token and is frozen for the life of the request.

The asymmetry is deliberate and is tested explicitly in
``tests/unit/test_filters.py``: model-supplied constraints pass through
``_intersect``, which computes an intersection and never a union.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from app.auth.principal import Principal
from app.observability.logging import get_logger
from app.retrieval.schema import Department, DocumentType

log = get_logger(__name__)

#: Pinecone applies range operators to numbers only, so dates are filtered on
#: the ``created_ts`` mirror of ``created_date``. Kept in one place so the two
#: representations cannot drift.
_DATE_FIELD = "created_ts"


def date_to_ts(value: date) -> int:
    """Match the encoding used when chunks are indexed."""
    return int(value.toordinal() * 86400)


def _intersect(permitted: set[str], requested: set[str] | None) -> set[str]:
    """Narrow ``permitted`` by ``requested``, never widen it.

    ``requested`` originates from model output — a query-understanding step that
    extracts "the payments incident from March" into structured constraints. It
    is therefore untrusted. An empty intersection is returned as-is and yields a
    query that matches nothing, which is the correct outcome: asking for a
    department you cannot see should return no evidence, not all evidence.
    """
    if not requested:
        return permitted
    return permitted & requested


def permitted_departments(principal: Principal) -> set[str]:
    """Every department this caller may read from."""
    if principal.has_all_departments:
        return {d.value for d in Department}
    return set(principal.departments)


def authorization_filter(principal: Principal) -> dict[str, Any]:
    """The non-negotiable clause applied to every query.

    Never takes an argument other than the principal. If a future caller needs
    to relax this, that is a change to the lattice in ``app.auth.principal``,
    reviewed as a security change — not a parameter someone can pass.
    """
    return {
        "access_level": {"$in": principal.visible_access_levels},
        "department": {"$in": sorted(permitted_departments(principal))},
    }


def build_filter(
    principal: Principal,
    *,
    departments: set[str] | None = None,
    document_types: set[str] | None = None,
    tags: set[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> dict[str, Any]:
    """Compose the full metadata filter for a retrieval call.

    All keyword arguments are treated as untrusted narrowing hints. Note what is
    absent: there is no ``access_level`` parameter. It cannot be passed, so it
    cannot be widened, so no amount of model compromise reaches a document above
    the caller's ceiling.
    """
    allowed_departments = _intersect(permitted_departments(principal), departments)

    metadata_filter: dict[str, Any] = {
        # Derived from the JWT. Always present, never parameterised.
        "access_level": {"$in": principal.visible_access_levels},
        "department": {"$in": sorted(allowed_departments)},
    }

    if document_types:
        valid = {t for t in document_types if t in {d.value for d in DocumentType}}
        if valid:
            metadata_filter["document_type"] = {"$in": sorted(valid)}
        else:
            # Every requested type was unrecognised. Narrow to nothing rather
            # than silently dropping the constraint and returning more than the
            # caller asked for.
            metadata_filter["document_type"] = {"$in": []}

    if tags:
        metadata_filter["tags"] = {"$in": sorted(tags)}

    date_clause: dict[str, int] = {}
    if date_from is not None:
        date_clause["$gte"] = date_to_ts(date_from)
    if date_to is not None:
        date_clause["$lte"] = date_to_ts(date_to)
    if date_clause:
        metadata_filter[_DATE_FIELD] = date_clause

    if departments and allowed_departments != set(departments):
        # Worth a log line: this is the signal that a query asked for something
        # outside the caller's scope, whether by user error or injection.
        log.info(
            "retrieval_filter_narrowed",
            requested=sorted(departments),
            granted=sorted(allowed_departments),
            role=principal.role.value,
        )

    return metadata_filter


def matches_nothing(metadata_filter: dict[str, Any]) -> bool:
    """Whether a filter can possibly match.

    Used to short-circuit before spending a network round trip, and to give the
    activity panel an honest reason for an empty result rather than an
    unexplained blank.
    """
    return any(
        isinstance(clause, dict) and clause.get("$in") == [] for clause in metadata_filter.values()
    )
