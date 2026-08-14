"""The authenticated caller, and the authorization lattice they sit in.

A ``Principal`` is constructed exactly once per request, by the API layer, from a
verified JWT. It is then carried through the whole LangGraph run as immutable
state. Nothing downstream — no node, no tool, and critically no model output —
can widen it.

That immutability is what makes the data-layer authorization control sound: the
Pinecone metadata filter is derived from the ``Principal``, so a prompt injection
that fully hijacks the model's intent still cannot reach a document the caller
was never entitled to see.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

#: Sentinel department granting access to every namespace.
ALL_DEPARTMENTS = "*"


class Role(StrEnum):
    """The three roles required by the specification."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMINISTRATOR = "administrator"


class AccessLevel(IntEnum):
    """Document sensitivity, ordered least to most restricted.

    Modelled as an ``IntEnum`` so the lattice comparison is the integer
    comparison — a caller may read any level at or below their ceiling. Using an
    ordered type rather than a free-form string means an unknown level can never
    accidentally compare as permitted.
    """

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def parse(cls, value: str) -> AccessLevel:
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unknown access level: {value!r}") from exc


#: The single source of truth for how far each role can see.
ROLE_ACCESS_CEILING: dict[Role, AccessLevel] = {
    Role.VIEWER: AccessLevel.INTERNAL,
    Role.ANALYST: AccessLevel.CONFIDENTIAL,
    Role.ADMINISTRATOR: AccessLevel.RESTRICTED,
}

#: Role seniority, used to answer "does this role satisfy that requirement".
ROLE_RANK: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.ANALYST: 1,
    Role.ADMINISTRATOR: 2,
}


def role_satisfies(actual: Role, required: Role) -> bool:
    """Whether ``actual`` is at least as senior as ``required``."""
    return ROLE_RANK[actual] >= ROLE_RANK[required]


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller.

    Frozen and slotted: the graph carries this object through every node, and it
    must be impossible to mutate in flight.
    """

    user_id: str
    display_name: str
    role: Role
    departments: frozenset[str]

    @property
    def access_ceiling(self) -> AccessLevel:
        """The most sensitive document class this caller may read."""
        return ROLE_ACCESS_CEILING[self.role]

    @property
    def visible_access_levels(self) -> list[str]:
        """Access levels for a metadata ``$in`` filter, lowest first.

        Returned as the lowercase strings used in chunk metadata so this drops
        straight into a Pinecone filter without further translation.
        """
        return [level.name.lower() for level in AccessLevel if level <= self.access_ceiling]

    @property
    def has_all_departments(self) -> bool:
        return ALL_DEPARTMENTS in self.departments

    def may_read(self, access_level: str, department: str) -> bool:
        """Post-retrieval assertion that a chunk was legitimately returned.

        The Pinecone filter should already guarantee this. It is re-checked on
        the way out because a filter that is silently mis-constructed — a typo, a
        schema drift, a future refactor — would otherwise leak without a signal.
        """
        try:
            level = AccessLevel.parse(access_level)
        except ValueError:
            return False  # unknown level: refuse rather than guess
        if level > self.access_ceiling:
            return False
        return self.has_all_departments or department in self.departments

    def to_claims(self) -> dict[str, Any]:
        """Serialise into JWT claims."""
        return {
            "sub": self.user_id,
            "name": self.display_name,
            "role": self.role.value,
            "departments": sorted(self.departments),
        }

    @classmethod
    def from_claims(cls, claims: Mapping[str, Any]) -> Principal:
        """Rebuild from verified JWT claims.

        Note what is *not* stored in the token: the access ceiling and the tool
        allowlist. Both are derived from the role at use time, so tightening a
        policy takes effect immediately rather than when outstanding tokens
        happen to expire.

        A malformed ``departments`` claim raises rather than degrading to an
        empty set — silently narrowing is confusing, silently widening is a
        vulnerability, so neither is guessed at.
        """
        raw_departments = claims.get("departments", [])
        if not isinstance(raw_departments, list | tuple | set | frozenset):
            raise ValueError("departments claim must be a sequence")

        return cls(
            user_id=str(claims["sub"]),
            display_name=str(claims.get("name", claims["sub"])),
            role=Role(str(claims["role"])),
            departments=frozenset(str(d) for d in raw_departments),
        )
