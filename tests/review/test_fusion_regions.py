"""Two named regions are two places, whatever their boxes do.

Edge findings carry full-length strips, so `top` and `left` necessarily
overlap at the corner they share — geometrically unavoidable, and enough for
`_correlates` to call them one defect. On `listings/clean/card06_back.webp`
a MINOR top-edge finding and a SEVERE left-edge finding fused into a single
severe defect, suppressing a distinct flaw.

`region_of_finding` already returns the value that separates them and was
never consulted. Overlap remains the test when either region is unknown —
a surface view has no region, and refusing to fuse there would double-count
corroboration instead.
"""

from card_reviewer.review.enums import FindingState
from card_reviewer.review.findings import Finding, FindingProducer, Severity
from card_reviewer.review.fusion import fuse
from card_reviewer.review.provenance import (
    EvidenceOrigin, EvidenceRef, NormalizedBox,
)
from card_reviewer.review.roles import ImageRole

TOP = NormalizedBox(x0=0.0, y0=0.0, x1=1.0, y1=0.2)
LEFT = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=1.0)


def _edge(view, box, severity, producer=FindingProducer.HEURISTIC):
    return Finding(
        defect_type="chipping", category="edges", state=FindingState.SUSPECTED,
        producer=producer, confidence=0.6, psa10_relevant=True,
        severity=severity, location=box,
        evidence=[EvidenceRef(artifact_id=f"a-{view}", image_hash="front-hash",
                              origin=EvidenceOrigin.NORMALIZED, view=view,
                              region=box)])


ROLES = {"front-hash": ImageRole.FRONT}


def test_two_named_edges_are_two_defects_even_though_they_overlap():
    assert TOP.overlaps(LEFT), "fixture no longer exercises the overlap"
    fused = fuse([_edge("edge_top", TOP, Severity.MINOR),
                  _edge("edge_left", LEFT, Severity.SEVERE)], ROLES)
    assert len(fused) == 2, (
        "a minor top-edge chip and a severe left-edge chip fused into one "
        "defect through the corner they share")


def test_the_severity_of_one_edge_does_not_travel_to_another():
    """The consequence, stated as itself: the merged finding took the worst
    severity, so the top edge was reported as severely chipped."""
    fused = fuse([_edge("edge_top", TOP, Severity.MINOR),
                  _edge("edge_left", LEFT, Severity.SEVERE)], ROLES)
    by_severity = sorted(f.severity.value for f in fused if f.severity)
    assert by_severity == ["minor", "severe"], by_severity


def test_two_producers_on_the_same_named_region_still_fuse():
    """The behaviour fusion exists for. Same region, same defect, two
    producers: one defect, two sources."""
    fused = fuse([_edge("edge_top", TOP, Severity.MINOR),
                  _edge("edge_top", TOP, Severity.MODERATE,
                        FindingProducer.VISION)], ROLES)
    assert len(fused) == 1
    assert len(fused[0].sources) == 2


def test_an_unregioned_finding_still_fuses_by_overlap():
    """A surface view carries no region. Refusing to fuse where the region
    is unknown would double-penalize corroboration, so overlap remains the
    test there."""
    unregioned = _edge("surface_original", TOP, Severity.MODERATE,
                       FindingProducer.VISION)
    fused = fuse([_edge("edge_top", TOP, Severity.MINOR), unregioned], ROLES)
    assert len(fused) == 1, (
        "a finding with no region was treated as a different place rather "
        "than an unknown one")
