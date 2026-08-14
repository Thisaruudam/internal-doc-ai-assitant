"""Per-user token-bucket rate limiting.

Two buckets per user, because they bound different things and one does not
substitute for the other:

* **Requests** blunt abusive polling and runaway clients.
* **LLM tokens** bound spend. A single recursive research question can cost more
  than a hundred simple lookups, so a request-count limit alone lets one user
  burn the budget while looking well-behaved.

Capacity is role-scoped: an administrator running an investigation should not be
throttled like an anonymous viewer clicking around.

**Where this runs, and the limit of it.** The buckets are in-process. That is
correct for a single API instance and wrong for several: each replica would hold
its own buckets and grant the full allowance, so the effective limit multiplies
by replica count. A shared Redis-backed implementation is the fix and is not
written yet — ``TokenBucketLimiter`` is the only implementation here.

The limitation is stated rather than glossed because a rate limiter that
silently scales with replicas is worse than none: it looks like protection while
providing proportionally less of it as the system grows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.auth.principal import Principal
from app.config import RateLimitSettings
from app.observability.logging import get_logger

log = get_logger(__name__)


@dataclass
class Bucket:
    """A classic token bucket.

    Tokens refill continuously rather than on a fixed window boundary, so a user
    is never told to wait a full minute for an allowance that is already partly
    restored.
    """

    capacity: float
    refill_per_second: float
    tokens: float = field(init=False)
    updated_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = self.capacity

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated_at = now

    def take(self, amount: float = 1.0) -> bool:
        now = time.monotonic()
        self._refill(now)
        if self.tokens < amount:
            return False
        self.tokens -= amount
        return True

    def retry_after_seconds(self, amount: float = 1.0) -> int:
        """How long until ``amount`` tokens are available. Always at least 1."""
        if self.refill_per_second <= 0:
            return 60
        shortfall = max(0.0, amount - self.tokens)
        return max(1, int(shortfall / self.refill_per_second) + 1)


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0
    reason: str = ""


class TokenBucketLimiter:
    """In-process limiter. One request bucket and one token bucket per user."""

    def __init__(self, settings: RateLimitSettings) -> None:
        self._settings = settings
        self._requests: dict[str, Bucket] = {}
        self._tokens: dict[str, Bucket] = {}

    def _multiplier(self, principal: Principal) -> float:
        return self._settings.role_multipliers.get(principal.role.value, 1.0)

    def _request_bucket(self, principal: Principal) -> Bucket:
        key = principal.user_id
        if key not in self._requests:
            scale = self._multiplier(principal)
            self._requests[key] = Bucket(
                capacity=self._settings.request_capacity * scale,
                refill_per_second=self._settings.request_refill_per_minute * scale / 60.0,
            )
        return self._requests[key]

    def _token_bucket(self, principal: Principal) -> Bucket:
        key = principal.user_id
        if key not in self._tokens:
            scale = self._multiplier(principal)
            self._tokens[key] = Bucket(
                capacity=self._settings.token_capacity * scale,
                refill_per_second=self._settings.token_refill_per_minute * scale / 60.0,
            )
        return self._tokens[key]

    def check_request(self, principal: Principal) -> RateLimitDecision:
        """Consume one request token."""
        if not self._settings.enabled:
            return RateLimitDecision(allowed=True)

        bucket = self._request_bucket(principal)
        if bucket.take(1.0):
            return RateLimitDecision(allowed=True)

        retry_after = bucket.retry_after_seconds(1.0)
        log.warning("rate_limited", kind="requests", role=principal.role.value)
        return RateLimitDecision(
            allowed=False,
            retry_after_seconds=retry_after,
            reason="too many requests in a short period",
        )

    def check_tokens(self, principal: Principal, estimated: int) -> RateLimitDecision:
        """Reserve an estimated LLM token spend before starting a turn."""
        if not self._settings.enabled:
            return RateLimitDecision(allowed=True)

        bucket = self._token_bucket(principal)
        if bucket.take(float(estimated)):
            return RateLimitDecision(allowed=True)

        retry_after = bucket.retry_after_seconds(float(estimated))
        log.warning("rate_limited", kind="tokens", role=principal.role.value)
        return RateLimitDecision(
            allowed=False,
            retry_after_seconds=retry_after,
            reason="the language-model budget for this account is temporarily exhausted",
        )

    def refund_tokens(self, principal: Principal, amount: int) -> None:
        """Return unspent reservation after a turn finishes.

        Reserving up front and refunding the remainder keeps a burst of
        concurrent turns from collectively overspending, which a
        charge-on-completion scheme allows.
        """
        if not self._settings.enabled or amount <= 0:
            return
        bucket = self._token_bucket(principal)
        bucket.tokens = min(bucket.capacity, bucket.tokens + amount)

    def snapshot(self, principal: Principal) -> dict[str, float]:
        """Remaining allowance, for the UI."""
        now = time.monotonic()
        requests = self._request_bucket(principal)
        tokens = self._token_bucket(principal)
        requests._refill(now)
        tokens._refill(now)
        return {
            "requests_remaining": round(requests.tokens, 1),
            "requests_capacity": requests.capacity,
            "tokens_remaining": round(tokens.tokens),
            "tokens_capacity": tokens.capacity,
        }
