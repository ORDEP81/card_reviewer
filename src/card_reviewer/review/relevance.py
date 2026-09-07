"""Resolve each finding to the rules that govern it, and to an authority.

Two failures this exists to prevent:

1. A high-authority rule in an unrelated category lending its weight to a
   defect it says nothing about.
2. An unmapped finding defaulting to BINDING, which would let any
   unrecognized anomaly reject a card — the false rejection the governing
   asymmetry forbids.
"""

from __future__ import annotations

from pydantic import BaseModel

from .enums import Authority, RuleEvaluability
from .findings import Finding
from .policies.authority_v1 import authority_of
from .policies.relevance_v1 import RELEVANCE_POLICY_VERSION, rule_matches_finding
from .taxonomy import CATEGORIES

__all__ = ["RELEVANCE_POLICY_VERSION", "ResolvedFinding", "resolve_relevance"]


class ResolvedFinding(BaseModel):
    finding: Finding
    rule_ids: list[str]
    authority: Authority
    psa10_relevant: bool
    policy_version: str = RELEVANCE_POLICY_VERSION


def resolve_relevance(
    findings: list[Finding], scoped_rules: list
) -> list[ResolvedFinding]:
    # Only APPLICABLE rules participate: a product-scoped rule we could not
    # evaluate must not lend its authority to anything. Inert rules
    # (contradicted) contribute nothing either.
    usable = [
        s.rule
        for s in scoped_rules
        if s.evaluability is RuleEvaluability.APPLICABLE
        and authority_of(s.rule) is not Authority.INERT
    ]

    out: list[ResolvedFinding] = []
    for finding in findings:
        matched = [r for r in usable if rule_matches_finding(r, finding)]
        # Relevance is decided by OUR grading taxonomy, not by the provider's
        # claim and NOT by whether a rule happened to match. Gating on
        # `matched` would make an unexplained corner defect psa10_relevant
        # False, dropping it from both the verdict and the score: an observed
        # defect would ship as a clean gem candidate.
        relevant = finding.category in CATEGORIES
        rule_ids = [r.id for r in matched]
        out.append(
            ResolvedFinding(
                finding=finding.model_copy(
                    update={"rule_ids": rule_ids, "psa10_relevant": relevant}
                ),
                rule_ids=rule_ids,
                # Unmapped is ADVISORY, never BINDING: it still penalizes and
                # still routes to REVIEW, but it cannot reject.
                authority=(
                    max(authority_of(r) for r in matched)
                    if matched
                    else Authority.ADVISORY
                ),
                psa10_relevant=relevant,
            )
        )
    return out
