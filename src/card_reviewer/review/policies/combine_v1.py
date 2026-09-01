"""Verdict resolution and the three invariants (spec §14, §15).

The four states are mutually exclusive and evaluated in STRICT ORDER — first
match wins. Stating them as independent conditions would leave a card with
both an observed crease and PARTIAL coverage matching two rows at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ..enums import (
    Authority, Coverage, FindingState, Psa10Candidate, Scale, Verdict,
)
from ..findings import Finding, i3_satisfied
from ..versions import COMBINATION_POLICY_VERSION

if TYPE_CHECKING:
    from ..fusion import FusedFinding

__all__ = [
    "COMBINATION_POLICY_VERSION",
    "MIN_DETECTABILITY_FOR_REJECT",
    "REJECT_CONFIDENCE_FLOOR",
    "VerdictResult",
    "decide_verdict",
    "i1_satisfied",
]

MIN_DETECTABILITY_FOR_REJECT = Scale.MODERATE

# Spec §15 declares the floor as HIGH on the shared scale. Findings carry a
# float confidence, so the mapping is stated once here rather than a bare
# 0.8 appearing as an unexplained magic number.
CONFIDENCE_BANDS: dict[Scale, float] = {
    Scale.LOW: 0.0,
    Scale.MODERATE: 0.5,
    Scale.HIGH: 0.8,
}
REJECT_CONFIDENCE_FLOOR = CONFIDENCE_BANDS[Scale.HIGH]


class VerdictResult(BaseModel):
    verdict: Verdict
    psa10_candidate: Psa10Candidate
    reasons: list[str] = Field(default_factory=list)
    policy_version: str = COMBINATION_POLICY_VERSION


_CANDIDATE: dict[Verdict, Psa10Candidate] = {
    Verdict.PASS: Psa10Candidate.YES,
    Verdict.REVIEW: Psa10Candidate.UNCERTAIN,
    Verdict.REJECT: Psa10Candidate.NO,
    Verdict.INSUFFICIENT_IMAGES: Psa10Candidate.UNKNOWN,
}


def i1_satisfied(
    finding: Finding,
    detectability: Scale,
    others: list[tuple[Finding, Scale]],
    *,
    material_contradiction: bool = False,
) -> bool:
    """I1 — ambiguity never rejects.

    The adequacy prong binds the ASSERTING finding rather than hoping for a
    contradicting one: on a badly photographed card no contradicting finding
    could reach MODERATE, so a contradiction-only test would weaken exactly
    where it is needed most.
    """
    if finding.state is not FindingState.OBSERVED:
        return False
    if not i3_satisfied(finding):
        return False
    if detectability < MIN_DETECTABILITY_FOR_REJECT:
        return False
    if finding.confidence < REJECT_CONFIDENCE_FLOOR:
        return False
    # Carried from fusion, which saw the raw sources. Fusion selects the
    # STRONGEST state, so by the time a fused finding reaches here the
    # contradicting NOT_OBSERVED source is no longer in `others` — without
    # this flag a contested defect would reject as though uncontested.
    if material_contradiction:
        return False
    return not _material_contradiction(finding, others)


def _material_contradiction(
    finding: Finding, others: list[tuple[Finding, Scale]]
) -> bool:
    for other, other_detectability in others:
        if other is finding or other.defect_type != finding.defect_type:
            continue
        if finding.location is None or other.location is None:
            continue
        if not finding.location.overlaps(other.location):
            continue
        if (
            other.state is FindingState.NOT_OBSERVED
            and other_detectability >= MIN_DETECTABILITY_FOR_REJECT
        ):
            return True
        if other.state is not finding.state and other.producer is not finding.producer:
            return True
    return False


def decide_verdict(
    findings: list[tuple[Finding, Authority, Scale]],
    coverage: Coverage,
    *,
    ambiguity: bool,
    contradicted: set[int] | None = None,
) -> VerdictResult:
    others = [(f, d) for f, _, d in findings]
    contradicted = contradicted or set()
    reasons: list[str] = []

    # Rule 1 — REJECT. A confidently observed disqualifier is knowledge, not
    # absence of it, so it outranks inadequate coverage: a missing back bars
    # passing, never rejecting.
    for finding, authority, detectability in findings:
        if not finding.psa10_relevant or authority is not Authority.BINDING:
            continue
        if i1_satisfied(
            finding, detectability, others,
            material_contradiction=id(finding) in contradicted,
        ):
            return _result(
                Verdict.REJECT,
                [f"{finding.category}/{finding.defect_type} observed and "
                 "I1-satisfying"],
            )

    # Rule 2 — INSUFFICIENT_IMAGES.
    if coverage is Coverage.INADEQUATE:
        return _result(Verdict.INSUFFICIENT_IMAGES, ["coverage INADEQUATE"])

    # Rule 3 — REVIEW. Includes an observed disqualifier that FAILS I1:
    # something looked like a defect and could not be established. That is an
    # unresolved concern, not an absence of one, and must never reach PASS.
    if coverage is Coverage.PARTIAL:
        reasons.append("coverage PARTIAL")
    for finding, _authority, _d in findings:
        if not finding.psa10_relevant:
            continue
        if finding.state is FindingState.OBSERVED:
            reasons.append(
                f"{finding.category}/{finding.defect_type} observed but not "
                "adequately evidenced to reject"
            )
        elif finding.state is FindingState.SUSPECTED:
            reasons.append(f"{finding.category}/{finding.defect_type} suspected")
    if ambiguity:
        reasons.append("unresolved ambiguity")
    if reasons:
        return _result(Verdict.REVIEW, reasons)

    # Rule 4 — otherwise. Reached only with SUFFICIENT coverage; stating it
    # as `otherwise` is what makes the function total.
    return _result(Verdict.PASS, ["coverage SUFFICIENT, no disqualifier"])


def _result(verdict: Verdict, reasons: list[str]) -> VerdictResult:
    return VerdictResult(
        verdict=verdict, psa10_candidate=_CANDIDATE[verdict], reasons=reasons
    )
