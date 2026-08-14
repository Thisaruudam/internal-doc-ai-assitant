"""The authorization lattice.

These tests protect the single most load-bearing claim in the architecture: that
a caller's reach is determined by their role, computed server-side, and cannot be
widened by anything downstream.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.auth.principal import (
    AccessLevel,
    Principal,
    Role,
    role_satisfies,
)


def make(role: Role, departments: set[str] | None = None) -> Principal:
    return Principal(
        user_id="t",
        display_name="Test",
        role=role,
        departments=frozenset(departments or {"*"}),
    )


class TestAccessCeiling:
    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            (Role.VIEWER, AccessLevel.INTERNAL),
            (Role.ANALYST, AccessLevel.CONFIDENTIAL),
            (Role.ADMINISTRATOR, AccessLevel.RESTRICTED),
        ],
    )
    def test_ceiling_per_role(self, role: Role, expected: AccessLevel) -> None:
        assert make(role).access_ceiling is expected

    def test_visible_levels_are_a_prefix_of_the_lattice(self) -> None:
        assert make(Role.VIEWER).visible_access_levels == ["public", "internal"]
        assert make(Role.ANALYST).visible_access_levels == [
            "public",
            "internal",
            "confidential",
        ]
        assert make(Role.ADMINISTRATOR).visible_access_levels == [
            "public",
            "internal",
            "confidential",
            "restricted",
        ]

    def test_viewer_cannot_reach_confidential(self) -> None:
        assert make(Role.VIEWER).may_read("confidential", "payments") is False

    def test_analyst_cannot_reach_restricted(self) -> None:
        assert make(Role.ANALYST).may_read("restricted", "payments") is False


class TestDepartmentScoping:
    def test_scoped_principal_is_confined_to_its_departments(self) -> None:
        analyst = make(Role.ANALYST, {"payments", "platform"})
        assert analyst.may_read("confidential", "payments") is True
        assert analyst.may_read("confidential", "hr") is False

    def test_wildcard_grants_every_department(self) -> None:
        assert make(Role.ANALYST, {"*"}).may_read("confidential", "anything") is True


class TestFailClosed:
    """An unrecognised access level must never compare as permitted.

    This is the reason ``AccessLevel`` is an ordered enum rather than a string:
    a typo or a schema drift in chunk metadata should deny, not allow.
    """

    @pytest.mark.parametrize("bogus", ["", "top-secret", "confidential;--", "0", "internal\x00"])
    def test_unknown_level_denies(self, bogus: str) -> None:
        # Even an administrator, who may read every *known* level, is denied an
        # unrecognised one.
        assert make(Role.ADMINISTRATOR).may_read(bogus, "payments") is False

    def test_parse_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown access level"):
            AccessLevel.parse("top-secret")

    def test_parse_is_case_and_space_insensitive(self) -> None:
        assert AccessLevel.parse("  Confidential ") is AccessLevel.CONFIDENTIAL


class TestImmutability:
    def test_principal_cannot_be_mutated(self) -> None:
        principal = make(Role.VIEWER)
        with pytest.raises(dataclasses.FrozenInstanceError):
            principal.role = Role.ADMINISTRATOR  # type: ignore[misc]

    def test_principal_has_no_dict_to_smuggle_attributes_into(self) -> None:
        # slots=True: a compromised node cannot stash an override on the object.
        # frozen + slots rejects unknown attributes with TypeError rather than
        # AttributeError, so accept either — what matters is that it is refused.
        with pytest.raises((AttributeError, TypeError)):
            make(Role.VIEWER).injected_flag = True  # type: ignore[attr-defined]

    def test_principal_really_has_no_instance_dict(self) -> None:
        assert not hasattr(make(Role.VIEWER), "__dict__")


class TestRoleSeniority:
    def test_ordering(self) -> None:
        assert role_satisfies(Role.ADMINISTRATOR, Role.VIEWER)
        assert role_satisfies(Role.ANALYST, Role.ANALYST)
        assert not role_satisfies(Role.VIEWER, Role.ANALYST)
        assert not role_satisfies(Role.ANALYST, Role.ADMINISTRATOR)


class TestClaimRoundTrip:
    def test_round_trip_preserves_identity(self) -> None:
        original = make(Role.ANALYST, {"payments", "platform"})
        assert Principal.from_claims(original.to_claims()) == original

    def test_claims_do_not_carry_derived_permissions(self) -> None:
        """The ceiling is derived at use time, not stamped into the token.

        A token minted before a policy tightened must not keep the old reach.
        """
        claims = make(Role.ANALYST).to_claims()
        assert "access_ceiling" not in claims
        assert "allowed_tools" not in claims

    def test_forged_ceiling_claim_is_ignored(self) -> None:
        claims = make(Role.VIEWER).to_claims() | {"access_ceiling": "restricted"}
        rebuilt = Principal.from_claims(claims)
        assert rebuilt.access_ceiling is AccessLevel.INTERNAL
