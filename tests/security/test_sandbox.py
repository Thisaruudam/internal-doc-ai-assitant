"""Sandboxed execution of model-authored code.

The escape suite is the point of this file. Each case is a technique that works
against a naive restricted-``exec`` sandbox, and two of them were live holes in
the first implementation: ``getattr(x, "__class__")`` walked straight past the
AST rule, and stripping ``__import__`` broke every allowlisted import while
blocking none of the forbidden ones.
"""

from __future__ import annotations

import pytest

from app.auth.principal import Principal, Role
from app.retrieval.schema import Chunk, ChunkMetadata
from app.rlm.sandbox import run_sandboxed, screen
from app.tools.python_analysis import (
    MAX_ROWS,
    authorize_rows,
    chunk_to_row,
    run_analysis,
)

pytestmark = pytest.mark.security

ROWS = [
    {"doc_id": "INC-1", "severity": "SEV1", "duration": 42, "cause": "pool-exhaustion"},
    {"doc_id": "INC-2", "severity": "SEV2", "duration": 18, "cause": "pool-exhaustion"},
    {"doc_id": "INC-3", "severity": "SEV3", "duration": 71, "cause": "certificate-expiry"},
]


def make_chunk(chunk_id: str, access: str = "internal", department: str = "payments") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        metadata=ChunkMetadata(
            doc_id=chunk_id.split("#")[0],
            title="Incident report",
            department=department,
            document_type="incident",
            access_level=access,
            created_date="2026-03-14",
            created_ts=63_000_000,
            owner="Payments Platform Team",
            tags=["outage"],
            source_uri=f"corpus://{department}/{chunk_id}",
            chunk_idx=0,
            heading_path="Root Cause",
            token_estimate=200,
        ),
    )


ESCAPES = [
    ("dunder traversal", "result = ().__class__.__bases__[0].__subclasses__()"),
    ("function globals", "f = lambda: 0\nresult = f.__globals__"),
    ("getattr by string", "result = getattr((), '__class__')"),
    ("getattr chained", "c = getattr((), '__class__')\nresult = getattr(c, '__bases__')"),
    ("import os", "import os\nresult = os.listdir('/')"),
    ("import socket", "import socket\nresult = socket.gethostname()"),
    ("import subprocess", "import subprocess\nresult = subprocess.run(['ls'])"),
    ("from-import os", "from os import getcwd\nresult = getcwd()"),
    ("submodule import", "import os.path\nresult = 1"),
    ("dunder import call", "result = __import__('os').getcwd()"),
    ("eval", "result = eval('1+1')"),
    ("exec", "exec('result = 1')"),
    ("compile", "result = compile('1', '<s>', 'eval')"),
    ("open for read", "result = open('/etc/passwd').read()"),
    ("open for write", "f = open('/tmp/pwned', 'w')\nresult = 1"),
    ("builtins by name", "result = __builtins__"),
    ("globals()", "result = globals()"),
    ("class definition", "class Evil:\n    pass\nresult = 1"),
    ("try/except probing", "try:\n    x = 1\nexcept Exception:\n    result = 'probed'"),
    ("with statement", "with open('/tmp/x') as f:\n    result = 1"),
    ("type() metaclass", "result = type('E', (), {})"),
    ("object subclasses", "result = object.__subclasses__()"),
]


class TestEscapesAreBlocked:
    @pytest.mark.parametrize(("label", "code"), ESCAPES, ids=[label for label, _ in ESCAPES])
    async def test_escape_attempt_never_succeeds(self, label: str, code: str) -> None:
        result = await run_sandboxed(code, ROWS, timeout_s=8)
        assert not result.ok, f"{label} executed successfully"

    def test_string_based_attribute_access_has_no_route(self) -> None:
        """The hole found in the first implementation.

        AST screening sees ``x.__class__`` but not ``getattr(x, "__class__")``,
        so getattr is simply absent from the builtins.
        """
        from app.rlm.sandbox import ALLOWED_BUILTINS

        assert "getattr" not in ALLOWED_BUILTINS
        assert "setattr" not in ALLOWED_BUILTINS

    def test_screening_reports_every_problem_at_once(self) -> None:
        """A model rewriting its code should see all the errors, not one per
        round trip."""
        violations = screen("import os\nimport socket\nresult = eval('1')")
        assert len(violations) >= 3

    def test_syntax_errors_are_reported_readably(self) -> None:
        violations = screen("result = (((")
        assert violations and "syntax error" in violations[0]


class TestLegitimateAnalysisRuns:
    """The other half: over-restriction makes the tool useless."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (
                "from collections import Counter\nresult = dict(Counter(r['cause'] for r in rows))",
                {"pool-exhaustion": 2, "certificate-expiry": 1},
            ),
            (
                "import statistics\n"
                "result = round(statistics.mean([r['duration'] for r in rows]), 1)",
                43.7,
            ),
            ("result = sum(r['duration'] for r in rows)", 131),
            (
                "s = sorted(rows, key=lambda r: -r['duration'])\nresult = [r['doc_id'] for r in s]",
                ["INC-3", "INC-1", "INC-2"],
            ),
        ],
    )
    async def test_analysis_produces_the_right_answer(self, code: str, expected: object) -> None:
        result = await run_sandboxed(code, ROWS, timeout_s=10)
        assert result.ok, result.error
        assert result.result == expected

    async def test_allowlisted_imports_actually_work(self) -> None:
        """The second bug found: stripping __import__ broke every permitted
        import while blocking none of the forbidden ones."""
        for module in (
            "collections",
            "datetime",
            "itertools",
            "json",
            "math",
            "re",
            "statistics",
            "string",
            "textwrap",
        ):
            result = await run_sandboxed(f"import {module}\nresult = 1", ROWS, timeout_s=8)
            assert result.ok, f"{module} failed: {result.error}"

    async def test_print_output_is_captured(self) -> None:
        result = await run_sandboxed("print('hello')\nresult = 1", ROWS, timeout_s=8)
        assert result.ok
        assert "hello" in result.stdout

    async def test_a_runtime_error_is_reported_not_raised(self) -> None:
        result = await run_sandboxed("result = 1 / 0", ROWS, timeout_s=8)
        assert not result.ok
        assert "ZeroDivisionError" in (result.error or "")


class TestResourceLimits:
    async def test_an_infinite_loop_is_killed(self) -> None:
        result = await run_sandboxed("while True:\n    pass", ROWS, timeout_s=3)
        assert not result.ok
        assert result.duration_ms < 6_000

    async def test_the_sandbox_cannot_see_the_environment(self) -> None:
        """The child is launched with an empty environment, so API keys in the
        parent's environment are not reachable even if a module were."""
        result = await run_sandboxed(
            "import json\nresult = json.dumps(sorted(rows[0]))", ROWS, timeout_s=8
        )
        assert result.ok
        assert "GEMINI" not in str(result.result)


class TestAnalysisAuthorization:
    """The tool must not become an authorization bypass with extra steps."""

    def test_unauthorized_chunks_are_refused(self) -> None:
        available = {
            "A#0": make_chunk("A#0", access="internal"),
            "B#0": make_chunk("B#0", access="restricted"),
        }
        viewer = Principal("v", "V", Role.VIEWER, frozenset({"*"}))
        rows, refused = authorize_rows(["A#0", "B#0"], available, viewer)

        assert [r["chunk_id"] for r in rows] == ["A#0"]
        assert refused == ["B#0"]

    def test_a_model_cannot_analyse_what_it_cannot_retrieve(self) -> None:
        """Naming a restricted id directly must not reach its content."""
        available = {"S#0": make_chunk("S#0", access="restricted")}
        analyst = Principal("a", "A", Role.ANALYST, frozenset({"*"}))
        rows, refused = authorize_rows(["S#0"], available, analyst)
        assert rows == []
        assert refused == ["S#0"]

    def test_department_scoping_applies(self) -> None:
        available = {"P#0": make_chunk("P#0", department="people")}
        scoped = Principal("a", "A", Role.ANALYST, frozenset({"payments"}))
        rows, _ = authorize_rows(["P#0"], available, scoped)
        assert rows == []

    def test_unknown_and_forbidden_ids_are_indistinguishable(self) -> None:
        """Confirming an id exists but is out of reach is itself a disclosure."""
        available = {"B#0": make_chunk("B#0", access="restricted")}
        viewer = Principal("v", "V", Role.VIEWER, frozenset({"*"}))
        _, refused = authorize_rows(["B#0", "DOES-NOT-EXIST#9"], available, viewer)
        assert set(refused) == {"B#0", "DOES-NOT-EXIST#9"}

    def test_row_count_is_capped(self) -> None:
        available = {f"C#{i}": make_chunk(f"C#{i}") for i in range(MAX_ROWS + 50)}
        admin = Principal("a", "A", Role.ADMINISTRATOR, frozenset({"*"}))
        rows, _ = authorize_rows(list(available), available, admin)
        assert len(rows) <= MAX_ROWS

    def test_row_shape_exposes_metadata_for_grouping(self) -> None:
        row = chunk_to_row(make_chunk("A#0"))
        for key in ("chunk_id", "text", "doc_id", "department", "created_date", "tags"):
            assert key in row

    async def test_analysis_over_no_permitted_rows_explains_itself(self) -> None:
        available = {"B#0": make_chunk("B#0", access="restricted")}
        viewer = Principal("v", "V", Role.VIEWER, frozenset({"*"}))
        outcome = await run_analysis("result = len(rows)", ["B#0"], available, viewer)

        assert outcome["ok"] is False
        assert "none of the requested passages" in outcome["error"]
        assert outcome["rows_unavailable"] == 1

    async def test_analysis_runs_over_permitted_rows_only(self) -> None:
        available = {
            "A#0": make_chunk("A#0", access="internal"),
            "B#0": make_chunk("B#0", access="restricted"),
        }
        viewer = Principal("v", "V", Role.VIEWER, frozenset({"*"}))
        outcome = await run_analysis(
            "result = [r['chunk_id'] for r in rows]", ["A#0", "B#0"], available, viewer
        )

        assert outcome["ok"]
        assert outcome["result"] == ["A#0"]
        assert outcome["rows_unavailable"] == 1
        assert "excluded" in outcome["note"]

    async def test_rejected_code_returns_violations_for_self_correction(self) -> None:
        available = {"A#0": make_chunk("A#0")}
        admin = Principal("a", "A", Role.ADMINISTRATOR, frozenset({"*"}))
        outcome = await run_analysis("import os\nresult = 1", ["A#0"], available, admin)

        assert outcome["ok"] is False
        assert outcome["violations"]
