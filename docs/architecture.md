# Architecture

Enterprise AI Assistant — a multi-agent, retrieval-grounded conversational system over
organizational knowledge, built for **Commercial Bank** as the operating persona.

> **Design thesis.** The interesting engineering in a system like this is not "LLM + vector DB".
> It is everything around that edge: how work is decomposed, how context is kept small, how
> authorization survives a compromised model, how the system behaves when a dependency dies,
> and whether a human can see what happened afterwards. Every diagram below is organized
> around one of those questions.

**Contents**

1. [Design principles](#1-design-principles)
2. [System layers](#2-system-layers)
3. [The agent graph](#3-the-agent-graph)
4. [Retrieval architecture](#4-retrieval-architecture)
5. [Recursive Language Model](#5-recursive-language-model-rlm)
6. [Security control points](#6-security-control-points)
7. [Request lifecycle](#7-request-lifecycle)
8. [Memory design](#8-memory-design)
9. [Model routing](#9-model-routing)
10. [Failure model](#10-failure-model)
11. [Deployment topology](#11-deployment-topology)

---

## 1. Design principles

| # | Principle | Where it shows up |
|---|---|---|
| P1 | **Authorization lives in the data path, never in the prompt.** | Access filters are injected into every Pinecone query from the verified JWT ([§4](#4-retrieval-architecture), [§6](#6-security-control-points)) |
| P2 | **Context is a budget, not a bucket.** | Sub-agent isolation, VFS scratchpad, RLM manifests, memory compaction ([§3](#3-the-agent-graph), [§5](#5-recursive-language-model-rlm), [§8](#8-memory-design)) |
| P3 | **Every internal state transition is observable.** | Typed activity events on the same stream as answer tokens ([§2](#2-system-layers), [§7](#7-request-lifecycle)) |
| P4 | **Degradation is a designed behaviour, not an exception path.** | Explicit ladder, each rung surfaced to the user ([§10](#10-failure-model)) |
| P5 | **Retrieved text is data, never instruction.** | Quarantine + spotlighting of all retrieved content ([§6](#6-security-control-points)) |
| P6 | **An unsupported claim is a bug.** | Validator enforces per-claim citation grounding with a repair loop ([§3](#3-the-agent-graph)) |

---

## 2. System layers

```mermaid
flowchart TB
    subgraph CLIENT["Streamlit client"]
        LOGIN["Login · role badge<br/>permitted-tool list"]
        CHAT["Chat window<br/>multi-turn · token streaming"]
        PANEL["Agent Activity Panel<br/>live node · tool · retrieval ·<br/>memory · validation events"]
        APPR["HITL approval widget"]
    end

    subgraph API["FastAPI — fully async"]
        MW["Middleware chain<br/>1 correlation-id + structlog bind<br/>2 JWT authentication<br/>3 token-bucket rate limit<br/>4 Pydantic request validation<br/>5 RFC7807 error mapping"]
        EP["POST /auth/login · GET /auth/me<br/>POST /chat/stream — SSE<br/>POST /chat/resume — HITL<br/>POST /feedback<br/>GET /health · GET /health/deps"]
        MW --> EP
    end

    subgraph GRAPH["LangGraph orchestrator"]
        SUP["Supervisor + specialist subgraphs<br/>see section 3"]
    end

    subgraph DATA["Data and tool plane"]
        PC[("Pinecone<br/>dense + sparse indexes<br/>namespaces · metadata filters")]
        BM["Local BM25<br/>degraded-mode fallback"]
        MCPS["MCP server<br/>employees · services<br/>incidents"]
        SBX["Sandboxed Python<br/>RLM plans · analysis tool"]
        PG[("PostgreSQL<br/>checkpointer<br/>+ long-term store")]
        RD[("Redis<br/>rate-limit buckets")]
        PC ~~~ SBX
        BM ~~~ PG
        MCPS ~~~ RD
    end

    subgraph OBS["Cross-cutting"]
        LS["LangSmith<br/>distributed tracing"]
        LOG["structlog<br/>JSON, correlation-scoped"]
        CB["Circuit breakers<br/>degradation ladder"]
    end

    CLIENT -->|"Bearer JWT · SSE"| API
    API --> GRAPH
    GRAPH --> DATA
    GRAPH -.->|"typed activity events"| PANEL
    GRAPH -.-> OBS
    DATA -.-> OBS
    API -.-> OBS
```

The **Agent Activity Panel is a product surface, not a debug view.** Answer tokens and internal
state transitions travel the same SSE connection, so what the evaluator watches is the actual
execution, not a reconstruction after the fact.

---

## 3. The agent graph

```mermaid
flowchart TB
    START(["User turn"]) --> IG["<b>ingress_guard</b><br/>injection scan · PII detect<br/>schema validation · brand policy"]
    IG -->|blocked| REF["<b>refusal</b><br/>policy-cited decline"]
    IG -->|safe| ML["<b>memory_load</b><br/>checkpoint replay +<br/>semantic long-term recall"]
    ML --> SUP["<b>supervisor</b><br/>intent · task decomposition<br/>plan write/update · Command routing"]

    subgraph SPEC["Specialist sub-agents — isolated subgraphs, own message lists"]
        RET["<b>retrieval_agent</b><br/>rewrite → hybrid<br/>→ rerank → compress"]
        RES["<b>research_agent</b><br/>RLM recursion<br/>see section 5"]
        ANA["<b>analysis_agent</b><br/>sandboxed pandas<br/>over hits"]
        MCPA["<b>mcp_agent</b><br/>enterprise MCP<br/>tool calls"]
        RET ~~~ ANA
        RES ~~~ MCPA
    end

    SUP -->|dispatch| RET
    SUP -->|dispatch| RES
    SUP -->|dispatch| ANA
    SUP -->|dispatch| MCPA
    RET -->|findings → VFS| SUP
    RES -->|findings → VFS| SUP
    ANA -->|findings → VFS| SUP
    MCPA -->|findings → VFS| SUP

    SUP -->|"high risk"| HITL["<b>hitl_approval</b><br/>interrupt · await human"]
    HITL -->|approved| SPEC
    HITL -->|denied| SUP

    SUP -->|"plan done"| RSP["<b>response_agent</b><br/>compose + cite · streams tokens"]
    RSP --> VAL["<b>validator</b><br/>citation grounding · hallucination<br/>egress scan · brand check"]
    VAL -->|"fail · max 2 repairs"| RSP
    VAL -->|pass| MW["<b>memory_write</b><br/>extract durable facts"]
    MW --> DONE(["Answer + citations"])
    REF --> DONE

    SUP -.->|"budget out"| RSP
```

### Deep-agent patterns, hand-built

The spec names "Deep agents". Rather than adopting a framework abstraction, each pattern is
implemented directly on `StateGraph` so that every transition is inspectable and streamable:

| Pattern | Implementation |
|---|---|
| **Planning** | Supervisor writes a todo list into `state.plan`; each dispatch marks progress. Rendered live in the activity panel. |
| **Sub-agent isolation** | Each specialist is a compiled subgraph with its *own* `messages` list. Only distilled findings return to the parent — the parent context never sees raw tool output. |
| **Virtual filesystem** | `state.files: dict[str, str]` scratchpad. Sub-agents write artifacts and pass *filenames*, not contents. |
| **`Command` routing** | Supervisor returns `Command(goto=..., update=...)`. Routing becomes model-produced data visible in the trace, not a hidden edge table. |
| **Explicit budgets** | `depth`, `tool_calls`, `tokens` live in state. Exhaustion is a normal terminal transition into `response_agent` with a partial answer — never an exception. |

### State

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    principal: Principal  # user_id, role, allowed_tools, max_access_level
    plan: list[TodoItem]  # deep-agent planning
    files: dict[str, str]  # VFS scratchpad — artifacts by name
    retrieved: list[Chunk]
    citations: list[Citation]
    risk: RiskAssessment
    budget: Budget  # depth / tool_calls / tokens remaining
    errors: list[ErrorRecord]  # degradations, surfaced to the UI
```

`principal` is written **once**, by the API layer, from the verified JWT. No node and no model
output can mutate it — this is what makes [§6](#6-security-control-points) layer 3 sound.

### Streaming

The graph runs under `astream(stream_mode=["messages", "updates", "custom"])`. Nodes emit
activity events through `get_stream_writer()`; the API multiplexes them into a single SSE stream:

| Event | Emitted by | Panel rendering |
|---|---|---|
| `node.enter` / `node.exit` | every node | current-state breadcrumb |
| `plan.update` | supervisor | live todo checklist |
| `tool.call` / `tool.result` | tool guard | tool name, args digest, latency, allow/deny |
| `retrieval.stage` | retrieval agent | dense/sparse counts, fusion, rerank deltas |
| `memory.read` / `memory.write` | memory nodes | recalled facts, written facts |
| `validation.result` | validator | grounding score, repair attempts |
| `degradation` | resilience layer | which rung of the ladder fired, and why |
| `token` | response agent | streamed answer text |

---

## 4. Retrieval architecture

```mermaid
flowchart TB
    Q["User query"] --> QU["<b>Query understanding</b><br/>rewrite · expand ·<br/>extract metadata filters"]
    JWT["<b>JWT principal</b><br/>role → max_access_level<br/>role → permitted departments"]
    QU --> F{"Filter<br/>assembly"}
    JWT -->|"server-side<br/>never model-supplied"| F

    F --> D["Pinecone <b>dense</b><br/>gemini-embedding-001<br/>@ 1536 dims"]
    F --> S["Pinecone <b>sparse</b><br/>pinecone-sparse-english-v0"]
    D -->|asyncio.gather| RRF["<b>RRF fusion</b><br/>k = 60<br/>configurable alpha"]
    S -->|asyncio.gather| RRF
    RRF --> RR["<b>Rerank</b><br/>Pinecone hosted<br/>bge-reranker-v2-m3"]
    RR --> CMP["Dedupe +<br/>contextual compression"]
    CMP --> OUT["Top-k chunks<br/>with full attribution"]

    D -.->|"circuit open"| BM["<b>Local BM25</b><br/>degraded banner"]
    S -.->|"circuit open"| BM
    BM --> CMP
    RR -.->|unavailable| CMP
```

### Why two indexes rather than one dot-product index

Pinecone supports both shapes. A single index holding dense + sparse vectors is simpler, but
separate indexes are the documented path when you want the **integrated
`pinecone-sparse-english-v0` model** and **independent reranking** — and we want both. The
practical win is explainability: fusion becomes an explicit, inspectable step rather than a
server-side score blend the demo cannot show.

### Why RRF rather than weighted score blending

Dense cosine similarity and learned-sparse scores are not on a comparable scale, so a weighted
sum needs a normalization step that is itself a tuning liability. Reciprocal Rank Fusion consumes
only rank order:

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)} \qquad k = 60$$

`alpha` stays configurable so the demo can show the dense↔sparse trade-off live.

### Chunking and metadata

Chunking is heading-aware — roughly 800 tokens with 120 overlap, split on structural boundaries,
retaining a `heading_path` so a citation points at *"Incident 2025-114 → Root Cause Analysis"*
rather than an opaque byte offset.

| Field | Purpose |
|---|---|
| `department` | Namespace selector + filter |
| `document_type` | `policy` · `architecture` · `runbook` · `incident` · `product_spec` · `meeting_notes` |
| `access_level` | `public` < `internal` < `confidential` < `restricted` — the authorization lattice |
| `created_date` | Recency filters and the RLM's time-window batching |
| `doc_id`, `chunk_idx`, `heading_path`, `source_uri` | Attribution and citation rendering |

Namespaces partition by department: cheaper filtering, and a hard blast-radius boundary if a
filter is ever mis-constructed.

---

## 5. Recursive Language Model (RLM)

The point of RLM is that **the corpus is an environment the model queries, not a payload it
swallows.** The research agent never sees whole documents; it sees a manifest, writes code
against a restricted API, and recurses.

```mermaid
flowchart TB
    T["<b>Task</b><br/>'Summarize all outage reports related to payment<br/>failures in the last year and identify recurring root causes'"]
    T --> M["<b>corpus.describe(filters)</b><br/>manifest only — doc_id, title, department,<br/>type, date, token_count.<br/>No document bodies enter context."]
    M --> P["<b>LLM writes a Python search plan</b><br/>against a restricted corpus API"]
    P --> SB["<b>Sandboxed execution</b><br/>AST allowlist · no network · no filesystem<br/>wall-clock + memory caps"]

    SB --> B1["batch 1<br/>Sep–Dec"]
    SB --> B2["batch 2<br/>Jan–Apr"]
    SB --> B3["batch 3<br/>May–Aug"]

    B1 --> R1["sub-agent<br/>depth + 1"]
    B2 --> R2["sub-agent<br/>depth + 1"]
    B3 --> R3["sub-agent<br/>depth + 1"]

    R1 --> VFS["<b>Partial findings → VFS</b><br/>files['findings/batch_n.md']"]
    R2 --> VFS
    R3 --> VFS
    VFS --> RED["<b>Reduce</b><br/>aggregate · cluster root causes<br/>reconcile conflicting accounts"]
    RED --> ANS["Structured summary<br/>with per-claim citations"]

    R2 -.->|"needs deeper detail"| SB
```

The restricted API the model codes against:

| Call | Returns |
|---|---|
| `describe(**filters)` | Manifest rows — metadata only, never bodies |
| `search(query, **filters)` | Ranked `chunk_id`s via the [§4](#4-retrieval-architecture) hybrid path |
| `filter(rows, predicate)` | Narrowed manifest |
| `get_chunk(chunk_id)` | One chunk body, counted against the token budget |
| `batch(rows, by=...)` | Partitions — by date window, department, or size |
| `map_llm(batch, instruction)` | Recursive sub-agent call, depth + 1 |
| `reduce_llm(findings, instruction)` | Aggregation over VFS artifacts |

**Recursion limits:** depth ≤ 3, fan-out ≤ 8, shared token budget. A timeout returns **partial
map results** rather than failing the turn — the answer says which batches completed. Every
recursion level is a nested LangSmith span, so the trace shows the tree literally.

---

## 6. Security control points

```mermaid
flowchart TB
    U["User input"]

    subgraph L1G["Layer 1 — Ingress"]
        L1["Heuristic scan<br/>instruction-override patterns · base64,<br/>homoglyph, unicode-tag decoding ·<br/>exfil markers such as markdown-image data URIs"]
        L1B["LLM classifier<br/>structured risk verdict"]
        L1 --> L1B
    end

    subgraph L2G["Layer 2 — Authorization, three enforcement points"]
        L2["<b>Binding</b><br/>only role-permitted tools are<br/>bound to the LLM at all"]
        L3["<b>Execution</b><br/>ToolGuard re-checks the principal<br/>against the registry at invoke time"]
        L4["<b>Data</b><br/>access_level + department filter injected<br/>into every Pinecone query from the JWT"]
        L2 --> L3
        L3 --> L4
    end

    subgraph L5G["Layer 3 — Untrusted content handling"]
        L5["<b>Quarantine</b><br/>retrieved chunks wrapped in delimited<br/>UNTRUSTED DATA blocks with spotlighting"]
        L6["Sanitization<br/>strip markup · cap length · NFKC normalize"]
        L5 --> L6
    end

    subgraph L7G["Layer 4 — Egress"]
        L7["<b>Citation grounding</b><br/>every claim maps to a retrieved chunk_id"]
        L8["Exfiltration scan<br/>outbound URLs · secret patterns ·<br/>citations to never-retrieved documents"]
        L9["Brand policy<br/>no financial advice · no rate or product<br/>commitments · no competitor disparagement"]
        L7 --> L8
        L8 --> L9
    end

    DENY["Refusal + policy citation<br/>logged with correlation id"]
    OUT["Answer delivered"]
    REPAIR["Repair loop, max 2<br/>then 'insufficient evidence'"]

    U --> L1
    L1B -->|blocked| DENY
    L1B -->|safe| L2
    L4 --> L5
    L6 --> L7
    L9 -->|pass| OUT
    L9 -->|fail| REPAIR
    REPAIR --> L7
```

### Why authorization is enforced three times

Defence in depth against three different failure modes:

| Point | Defeats |
|---|---|
| **Binding** | The model cannot even *emit* a call it is not permitted to make — the tool is absent from its schema |
| **Execution** | State tampering, replayed calls, injected tool invocations that bypassed binding |
| **Data** | Retrieval-level leakage — works even if the model is **fully** compromised, because the filter is built from the JWT, not from anything the model produced |

The third point is the one that matters. A prompt injection can hijack a model's intent; it
cannot change what the vector database was asked to return.

### Roles

| Role | Chat | Knowledge search | MCP tools | Analytics | Admin tools |
|---|---|---|---|---|---|
| **Viewer** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Analyst** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Administrator** | ✅ | ✅ | ✅ | ✅ | ✅ |

Every tool in the registry declares `required_role`, `risk_level`, `timeout`, and `idempotent`.
`risk_level = high` routes through the HITL approval node regardless of role.

---

## 7. Request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant U as Streamlit
    participant A as FastAPI
    participant R as Redis
    participant G as LangGraph
    participant P as Pinecone
    participant L as LangSmith

    U->>A: POST /chat/stream (Bearer JWT)
    A->>A: verify JWT → Principal
    A->>R: token bucket consume(user_id)
    alt bucket empty
        R-->>A: denied
        A-->>U: 429 + Retry-After
    else allowed
        A->>G: astream(state with principal)
        G->>L: trace run start
        G-->>U: SSE node.enter ingress_guard
        G-->>U: SSE node.enter retrieval_agent
        par dense
            G->>P: query dense index, filter from JWT
        and sparse
            G->>P: query sparse index, filter from JWT
        end
        P-->>G: hits
        G-->>U: SSE retrieval.stage (counts, fusion, rerank)
        G-->>U: SSE token … token … token
        G-->>U: SSE validation.result
        G->>L: trace run end (+ token usage)
        A-->>U: SSE done (citations, trace_id)
    end
```

Every I/O boundary is `async`. Dense and sparse retrieval run under a single `asyncio.gather`;
RLM batch sub-agents fan out the same way, bounded by a semaphore set to the fan-out limit.

---

## 8. Memory design

Three tiers, each solving a distinct problem:

```mermaid
flowchart LR
    subgraph T1["Tier 1 — Session state"]
        CP[("LangGraph checkpointer<br/>AsyncPostgresSaver<br/>keyed by thread_id")]
    end
    subgraph T2["Tier 2 — Working memory"]
        SUM["Rolling summarization<br/>last N turns verbatim<br/>+ running summary"]
    end
    subgraph T3["Tier 3 — Long-term"]
        ST[("LangGraph Store<br/>Postgres + pgvector<br/>namespace (user_id, 'facts')")]
    end

    TURN["Turn N"] --> CP
    CP -->|"over token threshold"| SUM
    SUM --> CP
    CP -->|"turn ends"| EXT["Fact extractor"]
    EXT --> ST
    ST -->|"semantic recall on<br/>next query"| TURN
```

| Tier | Solves | Key decision |
|---|---|---|
| **Session state** | "Memory should survive multiple turns during a session" | Checkpointing full *graph state*, not just messages, so HITL resume and crash recovery come free |
| **Working memory** | Multi-turn cost growing quadratically | Compaction over truncation — a summary preserves early-turn constraints that naive windowing silently drops |
| **Long-term** | Cross-session personalization | Scoped to `(user_id, "facts")`. **Never crosses users**, and invalidated on role change so a downgraded user cannot read prior-role content out of memory |

The role-change invalidation is easy to miss and is a real leak: memory written while a user was
an Administrator would otherwise be replayed into context after they became a Viewer.

---

## 9. Model routing

One model does not fit every node. Routing is a single config table so the cost/quality
trade-off is explicit and demonstrable.

| Node | Model | Rationale |
|---|---|---|
| `ingress_guard`, `validator` | `gemini-3.5-flash-lite` | Pure classification, runs on every turn — cheapest and lowest latency |
| `supervisor`, `retrieval_agent`, `mcp_agent` | `gemini-3.5-flash` | Positioned by Google for agentic/tool-use workloads |
| `research_agent` (RLM), `analysis_agent` | `gemini-3.5-flash`, escalating to `gemini-3.1-pro-preview` on hard reduce steps | Long-horizon reasoning where answer quality dominates token cost |
| `response_agent` | `gemini-3.6-flash` | Latest GA balanced model; cheaper output tokens than 3.5 Flash |
| Embeddings | `gemini-embedding-001` @ **1536** dims | GA and stable. `gemini-embedding-2-preview` is preview *and* its embedding space is incompatible — adopting it would force a full re-embed later. MRL truncation 3072 → 1536 halves index cost at negligible quality loss |

---

## 10. Failure model

Every dependency call is wrapped in a circuit breaker, jittered exponential backoff, and an
`asyncio.wait_for` timeout. Degradation is a **designed ladder**, and each rung emits a
`degradation` event so the evaluator watches it happen:

```mermaid
flowchart LR
    F1["Pinecone<br/>unavailable"] --> D1["Local BM25 index<br/>degraded banner"]
    F2["Reranker<br/>unavailable"] --> D2["Keep RRF ordering<br/>note reduced precision"]
    F3["MCP server<br/>unavailable"] --> D3["Skip tool<br/>name the missing data"]
    F4["Primary LLM<br/>error"] --> D4["Fall back to<br/>flash-lite"]
    D4 -->|"also fails"| D5["Apology + trace_id<br/>no fabricated content"]
    F5["Sandbox<br/>timeout"] --> D6["Partial map results<br/>name incomplete batches"]
    F6["Budget<br/>exhausted"] --> D7["Clean termination<br/>partial answer"]
    F7["Rate limit<br/>hit"] --> D8["429 + Retry-After<br/>graceful UI message"]
```

The invariant across all of them: **the system never silently returns a worse answer.** Reduced
capability is always stated in the response and visible in the activity panel.

---

## 11. Deployment topology

```mermaid
flowchart LR
    subgraph COMPOSE["docker compose"]
        UI["ui<br/>Streamlit :8501"]
        AP["api<br/>FastAPI + uvicorn :8000"]
        MC["mcp<br/>MCP server :8900"]
        PG[("postgres :5432<br/>checkpointer + store<br/>pgvector")]
        RD[("redis :6379<br/>rate-limit buckets")]
    end

    subgraph EXT["External"]
        GEM["Gemini API"]
        PIN["Pinecone serverless"]
        LSM["LangSmith"]
    end

    UI --> AP
    AP --> MC
    AP --> PG
    AP --> RD
    AP --> GEM
    AP --> PIN
    AP -.-> LSM
```

Python is pinned to **3.12** via `uv`; the async `get_stream_writer()` path requires ≥ 3.11, and
3.14 does not yet resolve cleanly across this dependency set.

```bash
make up      # bring up all five services
make seed    # generate the mock corpus and index it into Pinecone
make test    # unit + integration + security suites
```

---

## Related documents

| Document | Contents |
|---|---|
| [`security.md`](./security.md) | Threat model, injection test corpus, guardrail specifications |
| [`assumptions-and-tradeoffs.md`](./assumptions-and-tradeoffs.md) | What was assumed, what was deliberately not built, and why |
| [`adr/`](./adr/) | Architecture decision records — LLM selection, hybrid retrieval, memory design, RBAC enforcement, RLM design |
| [`demo-script.md`](./demo-script.md) | Walkthrough order for the demo video, including the failure-injection steps |
