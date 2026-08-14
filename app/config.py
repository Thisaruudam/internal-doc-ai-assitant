"""Typed application settings.

Every knob the system exposes is declared here rather than read from ``os.environ``
at the point of use. That keeps configuration auditable — a reviewer can see the
whole surface in one file — and it means a missing or malformed value fails at
startup instead of halfway through a user's first question.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    """Model routing.

    One model does not fit every node. Cheap classification models run on every
    turn; expensive reasoning models are reserved for the nodes where answer
    quality dominates token cost. See ``docs/adr/0001-llm-selection.md``.
    """

    model_config = SettingsConfigDict(env_prefix="GEMINI_")

    api_key: SecretStr

    #: Guard and validator nodes — pure classification, runs on every turn.
    model_fast: str = "gemini-3.5-flash-lite"
    #: Supervisor, retrieval and MCP agents — agentic tool-use workloads.
    model_agent: str = "gemini-3.5-flash"
    #: Final answer composition — latest GA balanced model.
    model_response: str = "gemini-3.6-flash"
    #: Escalation target for hard RLM reduce steps only.
    model_deep: str = "gemini-3.1-pro-preview"

    #: GA embedding model. Its preview successor uses an incompatible embedding
    #: space, so adopting that would force a full re-index later.
    embedding_model: str = "gemini-embedding-001"
    #: Matryoshka truncation from the native 3072 — halves index cost at
    #: negligible retrieval quality loss.
    embedding_dimensions: int = 1536

    request_timeout_s: float = 60.0
    max_retries: int = 3


class PineconeSettings(BaseSettings):
    """Vector store.

    Two indexes rather than one hybrid index: the sparse index uses Pinecone's
    integrated ``pinecone-sparse-english-v0`` model, and separate indexes let the
    fusion step stay explicit and inspectable. See ``docs/adr/0002-hybrid-retrieval.md``.
    """

    model_config = SettingsConfigDict(env_prefix="PINECONE_")

    api_key: SecretStr

    dense_index: str = "atrium-dense"
    sparse_index: str = "atrium-sparse"
    sparse_model: str = "pinecone-sparse-english-v0"
    rerank_model: str = "bge-reranker-v2-m3"

    cloud: str = "aws"
    region: str = "us-east-1"

    #: Candidates pulled from *each* retriever before fusion.
    top_k_per_retriever: int = 20
    #: Documents surviving rerank and passed to the response agent.
    top_k_final: int = 8
    #: Reciprocal Rank Fusion smoothing constant.
    rrf_k: int = 60
    #: 1.0 = pure dense, 0.0 = pure sparse. Exposed so the trade-off is demonstrable.
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    request_timeout_s: float = 15.0


class SecuritySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    #: HS256 signing key. Generated per-deployment; the committed default is a
    #: placeholder that ``Settings.validate_production`` rejects outside dev.
    jwt_secret: SecretStr = SecretStr("dev-only-insecure-signing-key-change-me")
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 60

    #: Refuse the turn when the injection classifier scores at or above this.
    injection_block_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    #: Warn and quarantine but proceed above this.
    injection_warn_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    #: Hard ceiling on retrieved-content length fed to any model, in characters.
    max_retrieved_chars: int = 60_000


class RateLimitSettings(BaseSettings):
    """Per-user token bucket.

    Two buckets per user: one on *requests* to blunt abusive polling, one on
    *LLM tokens* to bound spend. Capacity and refill are role-scoped so an
    administrator is not throttled like an anonymous viewer.
    """

    model_config = SettingsConfigDict(env_prefix="RATELIMIT_")

    enabled: bool = True

    request_capacity: int = 20
    request_refill_per_minute: float = 10.0

    token_capacity: int = 200_000
    token_refill_per_minute: float = 50_000.0

    #: Multiplier applied to both buckets, keyed by role name.
    role_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"viewer": 1.0, "analyst": 2.0, "administrator": 4.0}
    )


class GraphSettings(BaseSettings):
    """Agent budgets.

    Budgets live in graph state and exhausting one is a *normal* terminal
    transition producing a partial answer — never an exception. See
    ``docs/architecture.md`` section 3.
    """

    model_config = SettingsConfigDict(env_prefix="GRAPH_")

    #: Supervisor dispatch rounds before the graph must compose an answer.
    max_supervisor_steps: int = 8
    #: RLM recursion depth. Depth 3 is enough for corpus → batch → document.
    max_recursion_depth: int = 3
    #: Concurrent sub-agents per RLM map step.
    max_fan_out: int = 8
    #: Total tool invocations per turn, across all agents.
    max_tool_calls: int = 24
    #: Total model tokens per turn, across all nodes.
    max_tokens: int = 400_000
    #: Validator repair attempts before falling back to "insufficient evidence".
    max_repair_attempts: int = 2

    #: Wall-clock cap on a single sandboxed RLM plan.
    sandbox_timeout_s: float = 20.0
    #: Address-space cap for the sandbox subprocess, in megabytes.
    sandbox_memory_mb: int = 512

    #: Turns kept verbatim before rolling summarization compacts the history.
    memory_verbatim_turns: int = 6
    #: Token count that triggers compaction.
    memory_compact_threshold: int = 12_000


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_")

    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "atrium"
    langsmith_tracing: bool = True

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    #: JSON for machines, console for a readable local dev stream.
    log_format: Literal["json", "console"] = "json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["dev", "staging", "production"] = "dev"

    #: The persona the assistant operates as. Drives the brand guardrail policy.
    organization_name: str = "Commercial Bank"

    api_host: str = "0.0.0.0"  # noqa: S104 — container-internal bind
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    postgres_dsn: str = "postgresql://atrium:atrium@localhost:5432/atrium"
    redis_url: str = "redis://localhost:6379/0"
    mcp_server_url: str = "http://localhost:8900/mcp"

    users_file: str = "app/auth/users.yaml"
    corpus_dir: str = "data/corpus"
    bm25_index_dir: str = "data/bm25_index"

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    ratelimit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    def validate_production(self) -> None:
        """Fail fast on configuration that is only acceptable in development.

        Called from the API lifespan so a misconfigured deployment refuses to
        start rather than silently running with a known signing key.
        """
        if self.environment == "dev":
            return

        problems: list[str] = []
        secret = self.security.jwt_secret.get_secret_value()
        if "dev-only" in secret:
            problems.append("SECURITY_JWT_SECRET is still the development placeholder")
        # RFC 7518 §3.2: an HMAC-SHA256 key shorter than the 32-byte hash output
        # weakens the signature. PyJWT warns; outside dev we refuse.
        if len(secret.encode("utf-8")) < 32:
            problems.append("SECURITY_JWT_SECRET must be at least 32 bytes for HS256")
        if self.api_base_url.startswith("http://"):
            problems.append("API_BASE_URL must use TLS outside development")
        if problems:
            raise RuntimeError(f"Refusing to start in {self.environment}: " + "; ".join(problems))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that importing modules never re-parse the environment, and so
    tests can clear the cache to inject an alternate configuration.
    """
    return Settings()
