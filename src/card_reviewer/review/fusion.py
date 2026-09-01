"""Correlate findings across producers into one assessment per defect.

Raw findings stay unfused underneath, per producer, because calibration
against real PSA outcomes needs to know what each source said alone.
Scoring and the verdict consume the fused view, so one physical defect
penalizes once however many producers saw it — otherwise a card looked at
*harder* scores *worse*, which is the opposite of what more evidence should
do.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import FindingState
from .findings import Finding, FindingProducer, Severity
from .provenance import EvidenceRef, NormalizedBox
from .versions import FUSION_VERSION

__all__ = ["FUSION_VERSION", "FusedFinding", "fuse"]

#: Strongest first: one producer confirming what another suspected is
#: corroboration, not contradiction.
_STATE_RANK = {
    FindingState.OBSERVED: 3,
    FindingState.SUSPECTED: 2,
    FindingState.NOT_OBSERVED: 1,
    FindingState.NOT_ASSESSABLE: 0,
}
_SEVERITY_RANK = {Severity.MINOR: 1, Severity.MODERATE: 2, Severity.SEVERE: 3}


class FusedFinding(BaseModel):
    category: str
    defect_type: str
    state: FindingState
    confidence: float
    psa10_relevant: bool
    severity: Severity | None = None
    location: NormalizedBox | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    #: The raw per-producer findings, kept for calibration.
    sources: list[Finding] = Field(default_factory=list)
    producers_disagreed: bool = False
    #: Computed from the RAW sources, before the strongest state is selected.
    #: Without it a heuristic OBSERVED fused with a well-evidenced vision
    #: NOT_OBSERVED would present as a clean OBSERVED and could reject the
    #: card on evidence that was actually contested.
    material_contradiction: bool = False
    #: A real field, not an attribute bolted on later: `combine` is a cached
    #: stage, and anything that is not a field is dropped by model_dump,
    #: losing the I3 reason on the first cache write.
    demotion_reason: str = ""
    fusion_version: str = FUSION_VERSION

    def as_finding(self) -> Finding:
        """A Finding view, so I3 and the verdict operate unchanged.

        The producer is the one that supplied the winning state, never a
        hardcoded value: I1's contradiction test compares producers, so
        stamping everything HEURISTIC would make that clause dead code and
        misattribute vision-only findings in the report.
        """
        strongest = max(self.sources, key=lambda f: _STATE_RANK[f.state])
        return Finding(
            defect_type=self.defect_type, category=self.category,
            state=self.state, producer=strongest.producer,
            confidence=self.confidence, psa10_relevant=self.psa10_relevant,
            severity=self.severity, location=self.location,
            evidence=self.evidence,
            rule_ids=sorted({r for f in self.sources for r in f.rule_ids}),
            demotion_reason=self.demotion_reason,
        )


def fuse(findings: list[Finding]) -> list[FusedFinding]:
    groups: list[list[Finding]] = []
    for finding in findings:
        for group in groups:
            if _correlates(group[0], finding):
                group.append(finding)
                break
        else:
            groups.append([finding])
    return [_fuse_group(group) for group in groups]


def _correlates(a: Finding, b: Finding) -> bool:
    """Same defect AND overlapping region.

    Region is required: two findings about different corners are not one
    defect, and merging them would suppress a real flaw. Without locations
    we cannot establish they are the same thing, so we do not merge.
    """
    if (a.category, a.defect_type) != (b.category, b.defect_type):
        return False
    if a.location is None or b.location is None:
        return False
    return a.location.overlaps(b.location)


def _fuse_group(group: list[Finding]) -> FusedFinding:
    strongest = max(group, key=lambda f: _STATE_RANK[f.state])
    severities = [f.severity for f in group if f.severity]

    evidence: list[EvidenceRef] = []
    seen: set[str] = set()
    for finding in group:
        for ref in finding.evidence:
            if ref.artifact_id not in seen:
                seen.add(ref.artifact_id)
                evidence.append(ref)

    producers = {f.producer for f in group}
    states = {f.state for f in group}
    # The spec gives two prongs, and only the first was implemented. Because
    # decide_verdict runs on FUSED findings, combine's own cross-producer
    # check could never fire for a pair that fused — and fusion's merge
    # condition is exactly that check's precondition, so the clause was dead.
    disagreed = len(producers) > 1 and len(states) > 1
    contradicted = (
        # One source says it is there, another looked with adequate evidence
        # and says it is not.
        (any(f.state is FindingState.OBSERVED for f in group)
         and any(f.state is FindingState.NOT_OBSERVED for f in group))
        # ...or the two layers simply report different states for the same
        # defect at an overlapping location.
        or disagreed
    )

    # The confidence of the state we ADOPTED, never the group maximum. Taking
    # the max across mismatched states let a SUSPECTED finding at 1.0 hand its
    # number to an OBSERVED one at 0.5, and the fused finding then cleared a
    # REJECT floor that neither source reached on its own.
    at_state = [f.confidence for f in group if f.state is strongest.state]

    return FusedFinding(
        category=strongest.category, defect_type=strongest.defect_type,
        state=strongest.state, confidence=max(at_state),
        psa10_relevant=any(f.psa10_relevant for f in group),
        severity=(max(severities, key=lambda s: _SEVERITY_RANK[s])
                  if severities else None),
        location=strongest.location, evidence=evidence, sources=list(group),
        producers_disagreed=disagreed,
        material_contradiction=contradicted,
    )
