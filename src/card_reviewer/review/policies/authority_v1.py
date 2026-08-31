"""How much a rubric rule may influence the outcome (Decision 4).

This is non-negotiable rule 8 made executable: the pipeline does not treat
every claim the video pipeline learned as equally binding.

The discipline that matters: authority answers "if this defect exists, how
much does it matter?" It NEVER answers "does this defect exist?" That
question belongs to the finding's own state and detectability. The two axes
meet only in a penalty-table lookup — there is no multiplication of one by
the other.
"""

from __future__ import annotations

from card_reviewer.knowledge.models import Confidence, EvidenceType

from ..enums import Authority
from ..versions import AUTHORITY_POLICY_VERSION

__all__ = ["AUTHORITY_POLICY_VERSION", "authority_of", "may_establish_reject"]


def authority_of(rule) -> Authority:
    """Map subsystem B's evidence taxonomy onto an authority tier.

    `confidence` may demote within experience_based, but never promotes
    across a tier: a high-confidence opinion is still an opinion.
    """
    match rule.evidence_type:
        case EvidenceType.OBJECTIVE:
            # Grounded in PSA's published standards or official material.
            return Authority.BINDING
        case EvidenceType.EXPERIENCE_BASED:
            return (
                Authority.BINDING
                if rule.confidence is Confidence.HIGH
                else Authority.ADVISORY
            )
        case EvidenceType.OPINION | EvidenceType.UNVERIFIED:
            return Authority.ADVISORY
        case EvidenceType.CONTRADICTED:
            # Inert, not absent: rule 11 says never delete, change status.
            return Authority.INERT
    raise ValueError(f"unhandled evidence_type {rule.evidence_type!r}")


def may_establish_reject(rule) -> bool:
    """Only binding authority can carry a card to REJECT."""
    return authority_of(rule) is Authority.BINDING
