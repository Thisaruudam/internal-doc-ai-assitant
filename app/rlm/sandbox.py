"""Sandboxed execution of model-authored Python.

The analysis tool and the RLM planner both let a language model write code that
this system then runs. That code must be treated as fully hostile: a successful
prompt injection turns "summarise these incidents" into arbitrary code execution
inside the application process.

**Where the actual boundary is.** Two mechanisms, and it matters which one is
load-bearing:

* **AST screening is not a security boundary.** Restricted-``exec`` sandboxes
  built on allowlisted globals are reliably escapable — ``().__class__.__bases__``
  walks to ``object``, ``__subclasses__()`` reaches every loaded class, and from
  there to ``os`` is a short trip. The screening here exists to *reject obvious
  mistakes early with a readable error*, which is a usability feature, not a
  containment one. Treating it as containment is the common and dangerous error.

* **Process isolation is the boundary.** Code runs in a separate interpreter with
  POSIX resource limits, no inherited file descriptors, a wall-clock kill, and no
  access to this process's memory. An escape from the AST rules lands the
  attacker in a disposable subprocess holding nothing of value.

**Known limit, stated plainly.** Process isolation does not block network access
on its own. The import allowlist has no ``socket``, ``urllib``, or
``subprocess``, and dunder traversal is rejected, so reaching the network
requires defeating the screening first. In a deployment that matters, the
sandbox belongs in a container with no network namespace and a read-only root
filesystem; ``docker-compose.yml`` is where that would be configured. For this
build the honest description is: defence in depth, with the last layer missing.
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from app.observability.logging import get_logger

log = get_logger(__name__)

#: Modules the analysis code may import. Everything here is pure computation:
#: nothing that touches the filesystem, the network, the process table, or the
#: interpreter's own machinery.
ALLOWED_IMPORTS = frozenset(
    {
        "collections",
        "datetime",
        "itertools",
        "json",
        "math",
        "re",
        "statistics",
        "string",
        "textwrap",
    }
)

#: Builtins exposed to the code. An allowlist rather than a blocklist: a
#: blocklist is a promise to have thought of everything.
#:
#: ``getattr``, ``setattr`` and ``hasattr`` are deliberately absent. The AST
#: screening refuses literal dunder access (``x.__class__``), but it cannot see
#: through a string: ``getattr(x, "__class__")`` is an ordinary call with an
#: ordinary argument, and from there ``__bases__`` and ``__subclasses__`` are
#: two more calls away. Analysis over lists of dictionaries uses subscripting,
#: so removing them costs nothing real.
ALLOWED_BUILTINS = [
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
]

#: Names that reach the interpreter's internals or the outside world.
FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
        "memoryview",
        "object",
        "type",
        "super",
        "classmethod",
        "staticmethod",
        "property",
        "__import__",
        "__builtins__",
        "exit",
        "quit",
    }
)


# Note: this module raises nothing. Rejection, timeout, and runtime failure are
# all reported through SandboxResult, so a caller cannot forget to handle them
# and a failing analysis degrades the turn instead of ending it.


@dataclass
class SandboxResult:
    ok: bool
    result: Any = None
    stdout: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    #: Set when the code was rejected by screening rather than failing at runtime.
    rejected: bool = False
    violations: list[str] = field(default_factory=list)


class _Screener(ast.NodeVisitor):
    """Walks the parse tree and collects reasons to refuse.

    Collects rather than raises on the first hit, so a model rewriting its code
    sees every problem at once instead of discovering them one round trip at a
    time.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    # ── imports ─────────────────────────────────────────────────────────
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                self.violations.append(f"import of {alias.name!r} is not permitted")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_IMPORTS:
            self.violations.append(f"import from {node.module!r} is not permitted")
        self.generic_visit(node)

    # ── attribute access ────────────────────────────────────────────────
    def visit_Attribute(self, node: ast.Attribute) -> None:
        # The whole escape family — __class__, __bases__, __subclasses__,
        # __globals__, __builtins__ — is dunder traversal. Refusing every
        # underscore-prefixed attribute closes it without enumerating names.
        if node.attr.startswith("_"):
            self.violations.append(
                f"access to the private attribute {node.attr!r} is not permitted"
            )
        self.generic_visit(node)

    # ── names ───────────────────────────────────────────────────────────
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self.violations.append(f"use of {node.id!r} is not permitted")
        if node.id.startswith("__"):
            self.violations.append(f"use of the dunder name {node.id!r} is not permitted")
        self.generic_visit(node)

    # ── statements with no place in an analysis snippet ─────────────────
    def visit_Global(self, node: ast.Global) -> None:
        self.violations.append("global statements are not permitted")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.violations.append("nonlocal statements are not permitted")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class definitions are the usual vehicle for metaclass tricks, and no
        # legitimate analysis snippet needs one.
        self.violations.append("class definitions are not permitted")

    def visit_With(self, node: ast.With) -> None:
        self.violations.append("with statements are not permitted")

    def visit_Try(self, node: ast.Try) -> None:
        # Exception handling is used to probe the sandbox by catching the errors
        # it raises. Analysis code should fail loudly instead.
        self.violations.append("try/except is not permitted")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.generic_visit(node)


def screen(code: str) -> list[str]:
    """Return every reason the code should be refused. Empty means it passed."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return [f"syntax error on line {exc.lineno}: {exc.msg}"]

    screener = _Screener()
    screener.visit(tree)
    return screener.violations


# The harness that runs inside the subprocess. Kept as source rather than an
# importable module so the child starts from a known, minimal state.
_HARNESS = """
import json, sys, io, resource, builtins
from contextlib import redirect_stdout

payload = json.loads(sys.stdin.read())
code, rows = payload["code"], payload["rows"]
mem_bytes = payload["memory_mb"] * 1024 * 1024
cpu_seconds = payload["cpu_seconds"]

# Resource limits. RLIMIT_AS is not enforced consistently on macOS, so the
# wall-clock kill in the parent is the reliable stop; these are the belt.
for limit, value in (
    (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
    (resource.RLIMIT_FSIZE, (0, 0)),          # no file writes at all
    (resource.RLIMIT_NOFILE, (16, 16)),       # no descriptor exhaustion
    (resource.RLIMIT_NPROC, (0, 0)),          # no forking
):
    try:
        resource.setrlimit(limit, value)
    except (ValueError, OSError):
        pass
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except (ValueError, OSError):
    pass

ALLOWED_BUILTINS = payload["allowed_builtins"]
ALLOWED_IMPORTS = set(payload["allowed_imports"])

safe_builtins = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS
                 if hasattr(builtins, name)}
# Exceptions must remain constructible or ordinary code cannot raise.
for name in ("ValueError", "TypeError", "KeyError", "IndexError",
             "ZeroDivisionError", "StopIteration", "Exception"):
    safe_builtins[name] = getattr(builtins, name)

# The import statement compiles to a call to __builtins__.__import__, so
# removing it outright makes even allowlisted imports fail. A guarded version
# enforces the same allowlist the AST screening does — the screening gates the
# syntax, this gates the actual module load, and neither relies on the other.
_real_import = builtins.__import__

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not permitted in the sandbox")
    return _real_import(name, globals, locals, fromlist, level)

safe_builtins["__import__"] = _guarded_import

sandbox_globals = {"__builtins__": safe_builtins, "rows": rows, "result": None}

buffer = io.StringIO()
envelope = {"ok": False, "result": None, "stdout": "", "error": None}
try:
    with redirect_stdout(buffer):
        exec(compile(code, "<analysis>", "exec"), sandbox_globals)
    envelope["ok"] = True
    envelope["result"] = sandbox_globals.get("result")
except MemoryError:
    envelope["error"] = "analysis exceeded its memory limit"
except Exception as exc:
    envelope["error"] = f"{type(exc).__name__}: {exc}"

envelope["stdout"] = buffer.getvalue()[:8000]

def fallback(obj):
    return str(obj)

sys.stdout.write("\\n__ATRIUM_RESULT__" + json.dumps(envelope, default=fallback))
"""

_RESULT_MARKER = "__ATRIUM_RESULT__"


async def run_sandboxed(
    code: str,
    rows: list[dict[str, Any]],
    *,
    timeout_s: float = 20.0,
    memory_mb: int = 512,
) -> SandboxResult:
    """Screen, then execute in an isolated subprocess.

    Async so a long analysis does not block the event loop while other parts of
    the turn — retrieval, other sub-agents — continue.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()

    violations = screen(code)
    if violations:
        log.warning("sandbox_code_rejected", violation_count=len(violations))
        return SandboxResult(
            ok=False,
            rejected=True,
            violations=violations,
            error="the analysis code was rejected: " + "; ".join(violations[:4]),
        )

    payload = json.dumps(
        {
            "code": code,
            "rows": rows,
            "memory_mb": memory_mb,
            # Passed as data rather than templated into the harness source, so
            # the harness stays a plain constant with no substitution step —
            # and no escaping rules to get wrong.
            "allowed_builtins": ALLOWED_BUILTINS,
            "allowed_imports": sorted(ALLOWED_IMPORTS),
            # CPU limit sits below the wall clock so a busy loop is killed by
            # the kernel rather than waiting out the full timeout.
            "cpu_seconds": max(1, int(timeout_s) - 1),
        }
    )

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",  # isolated: ignore PYTHON* env vars and the user site directory
        "-S",  # skip site customisation
        "-c",
        _HARNESS,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Empty environment: nothing about this deployment, including API keys,
        # is visible to the analysis code.
        env={"PATH": "/usr/bin:/bin"},
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(payload.encode()), timeout=timeout_s
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        log.warning("sandbox_timeout", timeout_s=timeout_s)
        return SandboxResult(
            ok=False,
            error=f"the analysis exceeded its {timeout_s:g}s time limit",
            duration_ms=(loop.time() - started) * 1000,
        )

    duration_ms = (loop.time() - started) * 1000
    stdout = stdout_bytes.decode("utf-8", errors="replace")

    if _RESULT_MARKER not in stdout:
        # The child died before writing its envelope — killed by a resource
        # limit, or crashed outright.
        detail = stderr_bytes.decode("utf-8", errors="replace").strip()[-300:]
        log.warning("sandbox_no_result", returncode=process.returncode, detail=detail)
        return SandboxResult(
            ok=False,
            error=(
                "the analysis process was terminated before it produced a result "
                "(most likely a memory or CPU limit)"
            ),
            duration_ms=duration_ms,
        )

    envelope = json.loads(stdout.split(_RESULT_MARKER, 1)[1])
    return SandboxResult(
        ok=bool(envelope["ok"]),
        result=envelope.get("result"),
        stdout=envelope.get("stdout", ""),
        error=envelope.get("error"),
        duration_ms=duration_ms,
    )
