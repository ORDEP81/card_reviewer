"""Findings, and the one invariant that is pure logic over them.

Both the heuristic layer and the vision layer emit findings in this shared
vocabulary (spec §9). Defining it once, upstream of both, is what lets
combine, the coverage policy and the invariants be written once — and what
makes OFF mode well-defined, since there the heuristic is the only producer.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .enums import FindingState
from .provenance import EvidenceRef, NormalizedBox


class FindingProducer(StrEnum):
    HEURISTIC = "heuristic"
    VISION = "vision"


class Severity(StrEnum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class Finding(BaseModel):
    """One observation about the card, from either producer."""

    defect_type: str
    category: str
    state: FindingState
    producer: FindingProducer
    confidence: float = Field(ge=0.0, le=1.0)
    psa10_relevant: bool
    evidence: list[EvidenceRef] = Field(min_length=1)
    severity: Severity | None = None
    location: NormalizedBox | None = None
    rule_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    demotion_reason: str = ""

    model_config = {"frozen": True}


def i3_satisfied(finding: Finding) -> bool:
    """I3 — enhancement alone never confirms.

    An anomaly visible only under enhancement may be a `suspected` candidate
    but can never independently reach `observed`. Agreement across several
    enhancement paths is deliberately NOT a corroboration route: independent
    enhancements of the same pixels are not independent evidence.
    """
    if finding.state is not FindingState.OBSERVED:
        return True
    return any(not ref.is_enhanced for ref in finding.evidence)


def enforce_i3(findings: list[Finding]) -> list[Finding]:
    """Demote — never drop — findings that violate I3.

    An enhancement-only anomaly is still real information about where to
    look; it simply may not establish a confirmed defect. Dropping it would
    hide a limitation, which non-negotiable rule 3 forbids.
    """
    out: list[Finding] = []
    for finding in findings:
        if i3_satisfied(finding):
            out.append(finding)
            continue
        methods = sorted({ref.enhancement or "" for ref in finding.evidence})
        out.append(
            finding.model_copy(
                update={
                    "state": FindingState.SUSPECTED,
                    "demotion_reason": (
                        f"I3: visible only under enhancement ({', '.join(methods)}); "
                        "demoted from observed to suspected"
                    ),
                }
            )
        )
    return out
