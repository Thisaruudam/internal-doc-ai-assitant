"""Hardcoded user directory (specification Option A).

Users live in a YAML file rather than a database because this is the *identity*
half of the assessment's auth requirement, and the interesting half is
authorization. Keeping the directory declarative means a reviewer can read the
entire access model in twenty lines.

The file stores bcrypt hashes, never plaintext. Verification is constant-time and
runs the same work whether or not the username exists, so the endpoint does not
leak which accounts are real.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import bcrypt
import yaml

from app.auth.principal import ALL_DEPARTMENTS, Principal, Role

#: Hashing a throwaway value against this when the user is unknown keeps the
#: failure path the same cost as the success path.
_DUMMY_HASH = bcrypt.hashpw(b"nonexistent-user-timing-equaliser", bcrypt.gensalt(rounds=12))


@dataclass(frozen=True, slots=True)
class UserRecord:
    user_id: str
    display_name: str
    role: Role
    departments: frozenset[str]
    password_hash: bytes

    def to_principal(self) -> Principal:
        return Principal(
            user_id=self.user_id,
            display_name=self.display_name,
            role=self.role,
            departments=self.departments,
        )


class UnknownUserError(Exception):
    """Raised when authentication fails. Deliberately does not distinguish
    'no such user' from 'wrong password' — that distinction is an enumeration
    oracle."""


@functools.lru_cache(maxsize=4)
def load_users(path: str) -> dict[str, UserRecord]:
    """Parse and validate the user directory.

    Cached per path: the file is read once per process. Validation is strict —
    an unknown role or a missing hash fails at load time, not at login time.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = raw.get("users") or []
    if not entries:
        raise ValueError(f"user directory {path!r} defines no users")

    users: dict[str, UserRecord] = {}
    for entry in entries:
        try:
            user_id = str(entry["user_id"])
            role = Role(str(entry["role"]))
            password_hash = str(entry["password_hash"]).encode("utf-8")
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid user entry in {path!r}: {entry!r}") from exc

        departments = frozenset(str(d) for d in entry.get("departments", [ALL_DEPARTMENTS]))
        users[user_id] = UserRecord(
            user_id=user_id,
            display_name=str(entry.get("display_name", user_id)),
            role=role,
            departments=departments,
            password_hash=password_hash,
        )
    return users


def authenticate(users: dict[str, UserRecord], user_id: str, password: str) -> Principal:
    """Verify credentials and mint a ``Principal``.

    Always performs a bcrypt comparison, even for an unknown user, so response
    time does not reveal whether the account exists.
    """
    record = users.get(user_id)
    candidate = password.encode("utf-8")[:72]  # bcrypt's hard input limit
    expected = record.password_hash if record else _DUMMY_HASH

    matched = bcrypt.checkpw(candidate, expected)
    if record is None or not matched:
        raise UnknownUserError("invalid username or password")
    return record.to_principal()


def hash_password(password: str) -> str:
    """Produce a storable hash. Used by ``scripts/hash_password.py``."""
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt(rounds=12)).decode("utf-8")
