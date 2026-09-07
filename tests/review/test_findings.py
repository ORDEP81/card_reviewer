import pytest
from pydantic import ValidationError

from card_reviewer.review.enums import FindingState
from card_reviewer.review.findings import (
    Finding,
    FindingProducer,
    Severity,
    enforce_i3,
    i3_satisfied,
)
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef


def _ev(origin=EvidenceOrigin.ORIGINAL, enhancement=None, aid="a1"):
    return EvidenceRef(
        artifact_id=aid, image_hash="h1", origin=origin,
        enhancement=enhancement, view="front_face",
    )


def _finding(state, evidence, **kw):
    base = dict(
        defect_type="scratches", category="surface", state=state,
        producer=FindingProducer.HEURISTIC, confidence=0.9,
        psa10_relevant=True, evidence=evidence,
    )
    return Finding(**(base | kw))


def test_observed_finding_backed_only_by_enhanced_views_violates_i3():
    f = _finding(FindingState.OBSERVED,
                 [_ev(EvidenceOrigin.ENHANCED, "clahe:clip=2.0")])
    assert i3_satisfied(f) is False


def test_observed_finding_with_one_unenhanced_ref_satisfies_i3():
    f = _finding(FindingState.OBSERVED, [
        _ev(EvidenceOrigin.ENHANCED, "clahe:clip=2.0", aid="a1"),
        _ev(EvidenceOrigin.ORIGINAL, aid="a2"),
    ])
    assert i3_satisfied(f) is True


def test_a_normalized_crop_counts_as_corroboration():
    f = _finding(FindingState.OBSERVED, [_ev(EvidenceOrigin.NORMALIZED, aid="a3")])
    assert i3_satisfied(f) is True


def test_a_suspected_finding_is_never_an_i3_violation():
    f = _finding(FindingState.SUSPECTED,
                 [_ev(EvidenceOrigin.ENHANCED, "sharpen:amount=1.5")])
    assert i3_satisfied(f) is True


def test_agreement_across_two_enhancements_still_fails_i3():
    """Independent enhancements of the same pixels are not independent
    evidence — that route was deliberately excluded."""
    f = _finding(FindingState.OBSERVED, [
        _ev(EvidenceOrigin.ENHANCED, "clahe:clip=2.0", aid="a1"),
        _ev(EvidenceOrigin.ENHANCED, "sharpen:amount=1.5", aid="a2"),
        _ev(EvidenceOrigin.ENHANCED, "edge:canny", aid="a3"),
    ])
    assert i3_satisfied(f) is False


def test_enforce_i3_demotes_rather_than_drops():
    """An enhancement-only anomaly is still information about where to look;
    dropping it would hide a limitation."""
    findings = [_finding(FindingState.OBSERVED,
                         [_ev(EvidenceOrigin.ENHANCED, "clahe:clip=2.0")])]
    out = enforce_i3(findings)
    assert len(out) == 1
    assert out[0].state is FindingState.SUSPECTED
    assert "I3" in out[0].demotion_reason


def test_enforce_i3_leaves_compliant_findings_untouched():
    f = _finding(FindingState.OBSERVED, [_ev(EvidenceOrigin.ORIGINAL)])
    assert enforce_i3([f])[0] == f


def test_a_finding_requires_at_least_one_evidence_ref():
    with pytest.raises(ValidationError):
        _finding(FindingState.OBSERVED, [])


def test_a_finding_round_trips_through_json():
    """Findings are embedded in the cached heuristic and combine outputs."""
    import json

    f = _finding(FindingState.SUSPECTED, [_ev()], severity=Severity.MODERATE,
                 rule_ids=["CORNERS_COLORED_001"])
    revived = Finding.model_validate(json.loads(f.model_dump_json()))
    assert revived == f
