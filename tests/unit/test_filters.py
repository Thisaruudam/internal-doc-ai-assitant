"""Retrieval filter construction — the primary authorization control.

Framed adversarially. Each test asks: if the model were fully compromised and
emitted the most hostile structured query it could, what would actually reach
Pinecone?
"""

from __future__ import annotations

import inspect
from datetime import date

import pytest

from app.auth.principal import Principal, Role
from app.retrieval.filters import (
    authorization_filter,
    build_filter,
    date_to_ts,
    matches_nothing,
    normalise_document_type,
    permitted_departments,
)
from app.retrieval.schema import Department


def make(role: Role, departments: set[str] | None = None) -> Principal:
    return Principal(
        user_id="t",
        display_name="Test",
        role=role,
        departments=frozenset(departments or {"*"}),
    )


class TestAuthorizationClause:
    def test_every_filter_carries_an_access_level_clause(self) -> None:
        for role in Role:
            assert "access_level" in build_filter(make(role))

    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            (Role.VIEWER, ["public", "internal"]),
            (Role.ANALYST, ["public", "internal", "confidential"]),
            (Role.ADMINISTRATOR, ["public", "internal", "confidential", "restricted"]),
        ],
    )
    def test_access_levels_match_the_role_ceiling(self, role: Role, expected: list[str]) -> None:
        assert build_filter(make(role))["access_level"]["$in"] == expected

    def test_wildcard_expands_to_every_department(self) -> None:
        assert permitted_departments(make(Role.ANALYST, {"*"})) == {d.value for d in Department}

    def test_scoped_principal_is_confined(self) -> None:
        granted = build_filter(make(Role.ANALYST, {"payments"}))["department"]["$in"]
        assert granted == ["payments"]


class TestCannotBeWidened:
    """The asymmetry that makes this control sound."""

    def test_access_level_is_not_a_parameter(self) -> None:
        """There is no argument to pass, so there is nothing to smuggle.

        If someone later adds one, this fails and forces the conversation.
        """
        parameters = set(inspect.signature(build_filter).parameters)
        assert "access_level" not in parameters
        assert "access_levels" not in parameters

    def test_authorization_filter_takes_only_the_principal(self) -> None:
        assert list(inspect.signature(authorization_filter).parameters) == ["principal"]

    def test_requesting_a_forbidden_department_narrows_to_nothing(self) -> None:
        analyst = make(Role.ANALYST, {"payments", "platform"})
        result = build_filter(analyst, departments={"people"})
        assert result["department"]["$in"] == []
        assert matches_nothing(result)

    def test_requesting_a_mix_keeps_only_the_permitted_part(self) -> None:
        analyst = make(Role.ANALYST, {"payments", "platform"})
        result = build_filter(analyst, departments={"payments", "people", "risk"})
        assert result["department"]["$in"] == ["payments"]

    def test_a_viewer_cannot_reach_confidential_by_asking(self) -> None:
        viewer = make(Role.VIEWER)
        result = build_filter(viewer, document_types={"incident"}, tags={"confidential"})
        # The tag is just a tag; the access clause is untouched by it.
        assert result["access_level"]["$in"] == ["public", "internal"]

    def test_unknown_document_types_are_dropped_not_narrowed_to_nothing(self) -> None:
        """document_type is a relevance hint, not a security constraint.

        Dropping it returns a superset of what was asked for, but every document
        in that superset is still inside the caller's access scope because the
        security clauses are untouched. Treating it as a security constraint made
        a query rewriter saying "incident report" collapse the filter to nothing.
        """
        result = build_filter(make(Role.ADMINISTRATOR), document_types={"not_a_real_type"})
        assert "document_type" not in result
        assert not matches_nothing(result)
        assert result["access_level"]["$in"] == [
            "public",
            "internal",
            "confidential",
            "restricted",
        ]

    @pytest.mark.parametrize(
        ("supplied", "expected"),
        [
            ("incident report", "incident"),
            ("Incident Reports", "incident"),
            ("post-mortem", "incident"),
            ("runbooks", "runbook"),
            ("meeting notes", "meeting_notes"),
            ("product spec", "product_spec"),
            ("policy", "policy"),
            ("nonsense", None),
        ],
    )
    def test_document_type_aliases_are_normalised(
        self, supplied: str, expected: str | None
    ) -> None:
        assert normalise_document_type(supplied) == expected

    def test_a_mix_of_valid_and_invalid_types_keeps_the_valid_ones(self) -> None:
        result = build_filter(make(Role.ANALYST), document_types={"incident report", "gibberish"})
        assert result["document_type"]["$in"] == ["incident"]

    def test_empty_narrowing_arguments_leave_authorization_intact(self) -> None:
        result = build_filter(make(Role.VIEWER), departments=set(), document_types=set())
        assert result["access_level"]["$in"] == ["public", "internal"]
        assert result["department"]["$in"] == sorted(d.value for d in Department)


class TestDateWindows:
    def test_range_uses_the_numeric_mirror_field(self) -> None:
        """Pinecone range operators apply to numbers, not ISO strings."""
        result = build_filter(
            make(Role.ANALYST), date_from=date(2025, 8, 1), date_to=date(2026, 8, 1)
        )
        assert "created_ts" in result
        assert "created_date" not in result

    def test_bounds_are_inclusive_and_ordered(self) -> None:
        result = build_filter(
            make(Role.ANALYST), date_from=date(2025, 8, 1), date_to=date(2026, 8, 1)
        )
        clause = result["created_ts"]
        assert clause["$gte"] == date_to_ts(date(2025, 8, 1))
        assert clause["$lte"] == date_to_ts(date(2026, 8, 1))
        assert clause["$gte"] < clause["$lte"]

    def test_one_sided_windows_are_supported(self) -> None:
        assert set(build_filter(make(Role.ANALYST), date_from=date(2026, 1, 1))["created_ts"]) == {
            "$gte"
        }
        assert set(build_filter(make(Role.ANALYST), date_to=date(2026, 1, 1))["created_ts"]) == {
            "$lte"
        }

    def test_no_date_clause_when_no_window_requested(self) -> None:
        assert "created_ts" not in build_filter(make(Role.ANALYST))

    def test_encoding_matches_the_indexer(self) -> None:
        """date_to_ts must agree with ChunkMetadata.created_ts, or every date
        filter silently matches nothing."""
        from datetime import date as d

        from app.retrieval.chunking import chunk_document
        from tests.unit.test_chunking import INCIDENT_BODY, make_document

        chunk = chunk_document(make_document(INCIDENT_BODY))[0]
        assert chunk.metadata.created_ts == date_to_ts(d(2026, 3, 14))


class TestMatchesNothing:
    def test_detects_an_impossible_clause(self) -> None:
        assert matches_nothing({"department": {"$in": []}})

    def test_ordinary_filters_are_satisfiable(self) -> None:
        assert not matches_nothing(build_filter(make(Role.ADMINISTRATOR)))

    def test_range_only_filters_are_satisfiable(self) -> None:
        assert not matches_nothing(build_filter(make(Role.ANALYST), date_from=date(2026, 1, 1)))
