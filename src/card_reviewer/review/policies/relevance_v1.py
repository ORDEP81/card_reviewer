"""Which rubric rules actually apply to a given finding (Decision 4).

Matching is by category, plus the product scoping already applied upstream.
That is the finest the rubric supports: subsystem B scopes rules by
`category` and `card_types`/`sets`, and no rule declares a defect type.
Matching finer would mean inventing defect-type semantics the rubric does
not contain and the owner never sanctioned.

What this does prevent is cross-category leakage — a corner finding never
inherits a centering rule's authority.
"""

from __future__ import annotations

from card_reviewer.knowledge.models import Rule

from ..findings import Finding
from ..versions import RELEVANCE_POLICY_VERSION

__all__ = ["RELEVANCE_POLICY_VERSION", "rule_matches_finding"]


def rule_matches_finding(rule: Rule, finding: Finding) -> bool:
    return rule.category.value == finding.category
