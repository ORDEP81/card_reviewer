import json

from card_reviewer.review.enums import FindingState
from card_reviewer.review.findings import (
    Finding, FindingProducer, Severity, i3_satisfied,
)
from card_reviewer.review.fusion import FusedFinding, fuse
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef, NormalizedBox


def _f(producer, state, box, defect="scratches", severity=None,
       origin=EvidenceOrigin.ORIGINAL, aid="a"):
    return Finding(
        defect_type=defect, category="surface", state=state, producer=producer,
        confidence=0.9, psa10_relevant=True, severity=severity,
        location=NormalizedBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
        evidence=[EvidenceRef(
            artifact_id=aid, image_hash="h", origin=origin,
            enhancement="clahe:clip=2.0" if origin is EvidenceOrigin.ENHANCED
            else None, view="v")])


def test_the_same_defect_seen_by_both_producers_fuses_into_one():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.SUSPECTED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (.1, .1, .4, .4))])
    assert len(out) == 1


def test_the_same_defect_type_in_a_different_region_stays_separate():
    """Two corners really are two flaws."""
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .2, .2)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (.7, .7, .9, .9))])
    assert len(out) == 2


def test_different_defect_types_in_one_region_stay_separate():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3),
                   defect="scratches"),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3),
                   defect="print_lines")])
    assert len(out) == 2


def test_the_fused_state_is_the_strongest_among_sources():
    """One producer confirming what another suspected is corroboration."""
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.SUSPECTED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert out[0].state is FindingState.OBSERVED


def test_the_source_findings_are_retained_for_calibration():
    """CV and Claude assessments must stay independently recoverable."""
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.SUSPECTED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert len(out[0].sources) == 2
    assert {f.producer for f in out[0].sources} == {
        FindingProducer.HEURISTIC, FindingProducer.VISION}


def test_evidence_refs_are_unioned_so_i3_sees_everything():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3),
                   origin=EvidenceOrigin.ENHANCED, aid="a1"),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3),
                   origin=EvidenceOrigin.ORIGINAL, aid="a2")])
    assert {r.origin for r in out[0].evidence} == {
        EvidenceOrigin.ENHANCED, EvidenceOrigin.ORIGINAL}


def test_a_fusion_of_only_enhanced_sources_still_fails_i3():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3),
                   origin=EvidenceOrigin.ENHANCED, aid="a1"),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3),
                   origin=EvidenceOrigin.ENHANCED, aid="a2")])
    assert i3_satisfied(out[0].as_finding()) is False


def test_an_observed_versus_not_observed_pair_is_a_material_contradiction():
    """Selecting OBSERVED as the strongest state must not erase the fact that
    another producer looked and did not see it."""
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.NOT_OBSERVED,
                   (.1, .1, .4, .4))])
    assert out[0].state is FindingState.OBSERVED
    assert out[0].material_contradiction is True


def test_corroboration_is_not_a_material_contradiction():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.SUSPECTED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert out[0].material_contradiction is False


def test_disagreement_between_producers_is_recorded():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.NOT_OBSERVED,
                   (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert out[0].producers_disagreed is True


def test_agreement_is_not_recorded_as_disagreement():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert out[0].producers_disagreed is False


def test_the_worst_severity_among_sources_is_kept():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3),
                   severity=Severity.MINOR),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3),
                   severity=Severity.SEVERE)])
    assert out[0].severity is Severity.SEVERE


def test_as_finding_keeps_the_winning_producer_not_a_hardcoded_one():
    """I1's contradiction test compares producers; stamping everything
    HEURISTIC would make that clause dead code."""
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.SUSPECTED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert out[0].as_finding().producer is FindingProducer.VISION


def test_findings_without_a_location_never_silently_merge():
    a = _f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3))
    b = _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))
    a = a.model_copy(update={"location": None})
    b = b.model_copy(update={"location": None})
    assert len(fuse([a, b])) == 2


def test_a_fused_finding_round_trips_and_canonicalizes():
    from card_reviewer.review.canonical import canonicalize

    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED,
                   (0, 0, .3, .3))])[0]
    assert FusedFinding.model_validate(json.loads(out.model_dump_json())) == out
    assert canonicalize(out.model_dump(mode="json"))


def test_the_demotion_reason_is_a_real_field_so_it_survives_the_cache():
    """combine is a cached stage; an attribute that is not a field is
    dropped by model_dump, losing the I3 reason on the first cache write."""
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED,
                   (0, 0, .3, .3))])[0]
    revived = FusedFinding.model_validate(
        json.loads(out.model_copy(update={"demotion_reason": "I3: x"})
                   .model_dump_json()))
    assert revived.demotion_reason == "I3: x"
