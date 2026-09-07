"""Fusion must not manufacture a REJECT out of a disagreement.

The spec defines a material unresolved contradiction as EITHER another
finding reporting not_observed at an overlapping location with adequate
detectability, OR "the heuristic and vision layers report different states
for the same defect type and overlapping location". An unresolved one blocks
rule 1 and the card falls through to REVIEW.
"""

import pytest
from detectability_helpers import regions_for

from card_reviewer.review.enums import (
    Authority, Coverage, FindingState, Scale, Verdict,
)
from card_reviewer.review.findings import Finding, FindingProducer
from card_reviewer.review.fusion import fuse
from card_reviewer.review.heuristic import HeuristicResult
from card_reviewer.review.policies.combine_v1 import CombinedResult, combine
from card_reviewer.review.policies.coverage_v1 import CoverageResult
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef, NormalizedBox
from card_reviewer.review.roles import ImageRole

BOX = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)


def _finding(state, producer, confidence, category="corners",
             defect_type="rounding", box=BOX):
    return Finding(
        defect_type=defect_type, category=category, state=state,
        producer=producer, confidence=confidence, psa10_relevant=True,
        location=box,
        evidence=[EvidenceRef(artifact_id=f"a{producer}", image_hash="h1",
                              origin=EvidenceOrigin.ORIGINAL,
                              view="corner_top_left", region=box)])


def test_a_suspicion_does_not_lend_its_confidence_to_an_observation():
    """confidence was max() across the whole group regardless of state, so a
    SUSPECTED finding at 1.0 handed its number to an OBSERVED one at 0.5 —
    and the fused finding then cleared the REJECT confidence floor that
    neither source reached."""
    fused = fuse([
        _finding(FindingState.SUSPECTED, FindingProducer.HEURISTIC, 1.0),
        _finding(FindingState.OBSERVED, FindingProducer.VISION, 0.5),
    ])
    assert len(fused) == 1
    assert fused[0].state is FindingState.OBSERVED
    assert fused[0].confidence == 0.5, (
        "the adopted state's own confidence, not the group maximum")


def test_confidence_is_the_strongest_among_findings_at_the_adopted_state():
    fused = fuse([
        _finding(FindingState.OBSERVED, FindingProducer.HEURISTIC, 0.7),
        _finding(FindingState.OBSERVED, FindingProducer.VISION, 0.9),
        _finding(FindingState.SUSPECTED, FindingProducer.HEURISTIC, 1.0),
    ])
    assert fused[0].confidence == 0.9


def test_producers_disagreeing_on_state_is_a_material_contradiction():
    fused = fuse([
        _finding(FindingState.SUSPECTED, FindingProducer.HEURISTIC, 0.9),
        _finding(FindingState.OBSERVED, FindingProducer.VISION, 0.95),
    ])
    assert fused[0].material_contradiction is True


def test_one_producer_alone_is_not_a_contradiction():
    fused = fuse([
        _finding(FindingState.OBSERVED, FindingProducer.VISION, 0.95),
        _finding(FindingState.SUSPECTED, FindingProducer.VISION, 0.9),
    ])
    assert fused[0].material_contradiction is False


def test_agreeing_producers_are_not_a_contradiction():
    fused = fuse([
        _finding(FindingState.OBSERVED, FindingProducer.HEURISTIC, 0.9),
        _finding(FindingState.OBSERVED, FindingProducer.VISION, 0.95),
    ])
    assert fused[0].material_contradiction is False


def _det(scale=Scale.HIGH, category="corners", defect="rounding"):
    return {(ImageRole.FRONT, region, category, defect): scale
            for region in regions_for(category)}


def test_a_suspicion_cannot_flip_a_review_into_a_reject(rubric_scoped):
    """The whole point, end to end. Adding a merely SUSPECTED CV candidate
    beside a vision finding whose own confidence is below the REJECT floor
    used to produce REJECT."""
    def verdict(findings):
        return combine(
            HeuristicResult(findings=[f for f in findings
                                      if f.producer is FindingProducer.HEURISTIC]),
            None,
            CoverageResult(outcome=Coverage.SUFFICIENT, rankable=True),
            card_context_known=True, scoped_rules=rubric_scoped,
            detectability=_det(),
        ).verdict

    weak_vision_only = [_finding(FindingState.OBSERVED,
                                 FindingProducer.HEURISTIC, 0.5)]
    with_a_suspicion = weak_vision_only + [
        _finding(FindingState.SUSPECTED, FindingProducer.HEURISTIC, 1.0)]

    assert verdict(weak_vision_only) is not Verdict.REJECT
    assert verdict(with_a_suspicion) is not Verdict.REJECT, (
        "a suspicion made the card worse than the observation alone")


def test_a_contradiction_is_recorded_not_silently_resolved():
    """Spec: contradictions are recorded, never silently resolved."""
    fused = fuse([
        _finding(FindingState.SUSPECTED, FindingProducer.HEURISTIC, 0.9),
        _finding(FindingState.OBSERVED, FindingProducer.VISION, 0.95),
    ])
    assert len(fused[0].sources) == 2
    assert fused[0].producers_disagreed is True


def test_a_contradiction_must_be_about_the_same_category_too():
    """`whitening` exists in both corners and edges, and a corner box
    overlaps the top-edge strip. Comparing only defect_type let an unrelated
    edges/whitening NOT_OBSERVED suppress a genuine corners/whitening
    OBSERVED — conservative in direction, but it also strips the finding's I1
    status in the scoring path so a real defect under-penalizes. Fusion's own
    _correlates already compares the (category, defect_type) pair.
    """
    from card_reviewer.review.policies.combine_v1 import _material_contradiction

    corner = _finding(FindingState.OBSERVED, FindingProducer.VISION, 0.95,
                      category="corners", defect_type="whitening")
    edge = _finding(FindingState.NOT_OBSERVED, FindingProducer.HEURISTIC, 0.9,
                    category="edges", defect_type="whitening")
    same_category = _finding(FindingState.NOT_OBSERVED,
                             FindingProducer.HEURISTIC, 0.9,
                             category="corners", defect_type="whitening")

    assert _material_contradiction(corner, [(edge, Scale.HIGH)]) is False
    assert _material_contradiction(corner, [(same_category, Scale.HIGH)]) is True


def test_one_producer_contradicting_itself_is_still_material():
    """The two prongs overlap for cross-producer disagreement, so only this
    case separates them: the SAME layer reporting the defect present at one
    view and absent at an overlapping one. `disagreed` cannot see it — there
    is only one producer — and it is exactly the "another finding reports
    not_observed at an overlapping location" case the first prong names.
    """
    fused = fuse([
        _finding(FindingState.OBSERVED, FindingProducer.VISION, 0.95),
        _finding(FindingState.NOT_OBSERVED, FindingProducer.VISION, 0.9),
    ])
    assert fused[0].producers_disagreed is False, "one producer, by construction"
    assert fused[0].material_contradiction is True
