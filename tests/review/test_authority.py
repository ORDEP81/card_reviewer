import pytest

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.knowledge.models import Confidence, EvidenceType
from card_reviewer.review.enums import Authority
from card_reviewer.review.policies.authority_v1 import authority_of, may_establish_reject


class _R:
    def __init__(self, et, c):
        self.evidence_type, self.confidence, self.id = et, c, "TEST_001"


@pytest.mark.parametrize("et,conf,expected", [
    (EvidenceType.OBJECTIVE, Confidence.HIGH, Authority.BINDING),
    (EvidenceType.OBJECTIVE, Confidence.LOW, Authority.BINDING),
    (EvidenceType.EXPERIENCE_BASED, Confidence.HIGH, Authority.BINDING),
    (EvidenceType.EXPERIENCE_BASED, Confidence.MEDIUM, Authority.ADVISORY),
    (EvidenceType.EXPERIENCE_BASED, Confidence.LOW, Authority.ADVISORY),
    (EvidenceType.OPINION, Confidence.HIGH, Authority.ADVISORY),
    (EvidenceType.UNVERIFIED, Confidence.HIGH, Authority.ADVISORY),
    (EvidenceType.CONTRADICTED, Confidence.HIGH, Authority.INERT),
])
def test_the_authority_lattice_matches_the_declared_table(et, conf, expected):
    assert authority_of(_R(et, conf)) is expected


def test_objective_rules_stay_binding_regardless_of_confidence():
    """Confidence demotes within experience-based; it never demotes across
    from objective, which is grounded in PSA's published standards."""
    for conf in Confidence:
        assert authority_of(_R(EvidenceType.OBJECTIVE, conf)) is Authority.BINDING


def test_a_contradicted_rule_is_inert_rather_than_deleted():
    """Rule 11: never delete a historical rule, change its status."""
    r = _R(EvidenceType.CONTRADICTED, Confidence.HIGH)
    assert authority_of(r) is Authority.INERT
    assert may_establish_reject(r) is False


def test_only_binding_authority_may_establish_a_reject():
    assert may_establish_reject(_R(EvidenceType.OBJECTIVE, Confidence.HIGH))
    assert not may_establish_reject(_R(EvidenceType.OPINION, Confidence.HIGH))
    assert not may_establish_reject(_R(EvidenceType.EXPERIENCE_BASED, Confidence.MEDIUM))


def test_every_live_rubric_rule_maps_to_a_defined_authority():
    for rule in load_active_rubric().rules:
        assert authority_of(rule) in set(Authority)


def test_the_live_rubric_yields_no_inert_rules_today():
    """Tripwire: if this fails, a contradicted rule went active and the
    reason authority exists has changed."""
    assert all(authority_of(r) is not Authority.INERT
               for r in load_active_rubric().rules)
