# Atrium

An enterprise AI assistant over organizational knowledge — multi-agent, retrieval-grounded, and
authorization-enforced. Built as the Lead AI Engineer technical assessment, operating as
**Commercial Bank**.

> **Architecture is the deliverable.** Start with **[docs/architecture.md](docs/architecture.md)** —
> it carries nine diagrams covering the agent graph, hybrid retrieval, RLM recursion, security
> control points, and the degradation ladder.

---

## The one-paragraph version

A LangGraph supervisor decomposes each turn into a plan and dispatches to isolated specialist
sub-agents — retrieval, recursive research, sandboxed analysis, and enterprise MCP tools. Retrieval
is hybrid: a Pinecone dense index and a Pinecone learned-sparse index queried concurrently, fused by
Reciprocal Rank Fusion, then reranked. Authorization is enforced in three independent layers, the
strongest of which lives in the *data path* — the access filter is built from the verified JWT, so a
prompt injection that fully hijacks the model still cannot reach a document the caller was never
entitled to see. Every node streams typed activity events over the same SSE connection as the answer
tokens, so the system's reasoning is observable while it happens rather than reconstructed afterwards.

---

## Quick start

```bash
cp .env.example .env      # then fill in GEMINI_API_KEY, PINECONE_API_KEY, LangSmith key
make up                   # build and start all five services
make seed                 # generate the synthetic corpus and index it
```

| Surface | URL |
|---|---|
| Chat UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| Dependency status | http://localhost:8000/health/deps |

Running without Docker:

```bash
make install && make api
```

### Demo credentials

Local demonstration accounts. Sign in as each to see the authorization boundaries move.

| User | Password | Role | Departments | Access ceiling |
|---|---|---|---|---|
| `viewer` | `viewer-demo-2026` | Viewer | all | internal |
| `analyst` | `analyst-demo-2026` | Analyst | payments, platform | confidential |
| `admin` | `admin-demo-2026` | Administrator | all | restricted |

The interesting comparison is `viewer` vs `analyst` on the same question about a confidential
payments incident: the viewer gets *"insufficient evidence"*, the analyst gets a cited answer, and
the LangSmith trace shows the different Pinecone filters that produced the difference.

---

## What is where

| Path | Contents |
|---|---|
| `app/api/` | FastAPI surface — routes, middleware, SSE, RFC 7807 errors |
| `app/auth/` | JWT, the demo user directory, the access-level lattice |
| `app/graph/` | LangGraph state, node wiring, activity-event emission |
| `app/agents/` | Supervisor and the specialist sub-agents |
| `app/retrieval/` | Chunking, Pinecone dense + sparse, RRF, rerank, BM25 fallback |
| `app/rlm/` | Restricted corpus API, Python sandbox, recursive research |
| `app/memory/` | Checkpointer, working-memory compaction, long-term store |
| `app/tools/` | Tool registry with role and risk metadata, tool guard |
| `app/security/` | Injection detection, quarantine, egress scanning, policies |
| `app/resilience/` | Circuit breakers, retries, the degradation ladder |
| `mcp_server/` | MCP server exposing employee, service, and incident data |
| `ui/` | Streamlit chat window and Agent Activity Panel |
| `docs/` | Architecture, ADRs, security notes, assumptions, demo script |

---

## Development

```bash
make check     # lint + types + tests, the same set CI would run
make test      # unit and security suites, no external services needed
make fmt       # autoformat
```

Python is pinned to **3.12** via `uv`. The async `get_stream_writer()` path requires 3.11 or newer,
and 3.14 does not resolve cleanly across this dependency set.

---

## Build status

| Phase | Status |
|---|---|
| 0 — Foundation: config, auth, RBAC primitives, logging, errors, compose | ✅ Done |
| 1 — Retrieval: corpus, chunking, dual Pinecone index, hybrid + rerank | ⬜ Next |
| 2 — Graph core: agents, SSE streaming, Streamlit UI | ⬜ |
| 3 — RLM: corpus API, sandbox, recursive research | ⬜ |
| 4 — Memory and tools: checkpointer, long-term store, MCP, analysis | ⬜ |
| 5 — Security and resilience | ⬜ |
| 6 — HITL, feedback loop, eval harness, ADRs, demo script | ⬜ |
