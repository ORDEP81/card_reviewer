import pytest

from card_reviewer.review.enums import (
    Coverage, FindingState, ReviewConfidence, Scale, Verdict,
)
from card_reviewer.review.findings import Finding, FindingProducer
from card_reviewer.review.heuristic import HeuristicResult
from card_reviewer.review.policies.combine_v1 import CombinedResult, combine
from card_reviewer.review.policies.coverage_v1 import CoverageResult
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef, NormalizedBox
from card_reviewer.review.vision.provider import Assessment, GemView, VisionFinding

_BOX = NormalizedBox(x0=0.0, y0=0.0, x1=0.3, y1=0.3)
_ALL_ASSESSABLE = {"centering": True, "corners": True, "edges": True,
                   "surface": True}


def _f(producer=FindingProducer.HEURISTIC, state=FindingState.OBSERVED,
       category="corners", defect="rounding", enhanced=False, conf=0.95):
    origin = EvidenceOrigin.ENHANCED if enhanced else EvidenceOrigin.ORIGINAL
    return Finding(
        defect_type=defect, category=category, state=state, producer=producer,
        confidence=conf, psa10_relevant=True, location=_BOX,
        evidence=[EvidenceRef(artifact_id="a", image_hash="h", origin=origin,
                              enhancement="clahe:clip=2.0" if enhanced else None,
                              view="corner_top_left")])


def _cov(outcome=Coverage.SUFFICIENT):
    return CoverageResult(outcome=outcome,
                          rankable=outcome is not Coverage.INADEQUATE)


def _det(scale=Scale.HIGH, category="corners", defect="rounding"):
    from card_reviewer.review.roles import ImageRole

    return {(ImageRole.FRONT, category, defect): scale}


def test_off_mode_produces_a_complete_result_without_any_vision(rubric_scoped):
    r = combine(HeuristicResult(), None, _cov(), card_context_known=True,
                scoped_rules=rubric_scoped)
    assert r.verdict is Verdict.PASS
    assert r.psa10_rank_score == 100
    assert r.vision_present is False


def test_a_confidently_observed_defect_rejects(rubric_scoped):
    r = combine(HeuristicResult(findings=[_f()]), None, _cov(),
                card_context_known=True, scoped_rules=rubric_scoped,
                detectability=_det())
    assert r.verdict is Verdict.REJECT


def test_i3_is_enforced_before_the_verdict_is_decided(rubric_scoped):
    """An enhancement-only observed finding is demoted, so it cannot reject."""
    r = combine(HeuristicResult(findings=[_f(enhanced=True)]), None, _cov(),
                card_context_known=True, scoped_rules=rubric_scoped,
                detectability=_det())
    assert r.verdict is not Verdict.REJECT
    assert any("I3" in f.demotion_reason for f in r.fused)


def test_the_same_defect_from_both_producers_penalizes_once(rubric_scoped):
    """Looking harder must not make the score worse."""
    one = combine(HeuristicResult(findings=[_f(state=FindingState.SUSPECTED)]),
                  None, _cov(), card_context_known=True,
                  scoped_rules=rubric_scoped, detectability=_det())
    both = combine(
        HeuristicResult(findings=[
            _f(state=FindingState.SUSPECTED),
            _f(FindingProducer.VISION, FindingState.SUSPECTED)]),
        None, _cov(), card_context_known=True, scoped_rules=rubric_scoped,
        detectability=_det())
    assert one.psa10_rank_score == both.psa10_rank_score
    assert len(both.findings) == 2 and len(both.fused) == 1


def test_raw_findings_are_retained_alongside_the_fused_view(rubric_scoped):
    r = combine(
        HeuristicResult(findings=[
            _f(state=FindingState.SUSPECTED),
            _f(FindingProducer.VISION, FindingState.SUSPECTED)]),
        None, _cov(), card_context_known=True, scoped_rules=rubric_scoped,
        detectability=_det())
    assert {f.producer for f in r.findings} == {
        FindingProducer.HEURISTIC, FindingProducer.VISION}


def test_a_heuristic_observed_against_a_vision_not_observed_reviews(rubric_scoped):
    """Fusion picks OBSERVED as the strongest state, but the contradiction
    must still block the reject."""
    r = combine(
        HeuristicResult(findings=[
            _f(),
            _f(FindingProducer.VISION, FindingState.NOT_OBSERVED)]),
        None, _cov(), card_context_known=True, scoped_rules=rubric_scoped,
        detectability=_det())
    assert r.verdict is Verdict.REVIEW
    assert r.review_confidence is ReviewConfidence.LOW


def test_a_finding_with_no_matching_rules_cannot_reject(rubric_scoped):
    """Advisory authority means it cannot REJECT — not that it disappears."""
    r = combine(HeuristicResult(findings=[_f()]), None, _cov(),
                card_context_known=True, scoped_rules=[], detectability=_det())
    assert r.verdict is Verdict.REVIEW
    assert r.psa10_rank_score < 100


def test_i1_cannot_be_satisfied_when_detectability_is_absent(rubric_scoped):
    """An empty detectability map must block a reject, never license one."""
    r = combine(HeuristicResult(findings=[_f()]), None, _cov(),
                card_context_known=True, scoped_rules=rubric_scoped,
                detectability={})
    assert r.verdict is not Verdict.REJECT


def test_the_score_is_null_when_coverage_is_inadequate(rubric_scoped):
    r = combine(HeuristicResult(), None, _cov(Coverage.INADEQUATE),
                card_context_known=True, scoped_rules=rubric_scoped)
    assert r.psa10_rank_score is None and r.rankable is False
    assert r.verdict is Verdict.INSUFFICIENT_IMAGES


def test_unknown_card_context_lowers_confidence_but_never_rejects(rubric_scoped):
    r = combine(HeuristicResult(), None, _cov(), card_context_known=False,
                scoped_rules=rubric_scoped)
    assert r.review_confidence is ReviewConfidence.MEDIUM
    assert r.verdict is not Verdict.REJECT


def test_a_missing_required_face_is_low_confidence_yet_still_rankable(
        rubric_scoped):
    r = combine(HeuristicResult(), None, _cov(Coverage.PARTIAL),
                card_context_known=True, scoped_rules=rubric_scoped,
                required_face_missing=True)
    assert r.review_confidence is ReviewConfidence.LOW
    assert r.rankable is True


def test_grade_and_score_are_reported_independently(rubric_scoped):
    r = combine(HeuristicResult(), None, _cov(Coverage.PARTIAL),
                card_context_known=True, scoped_rules=rubric_scoped)
    assert r.estimated_psa_grade == "9-10"
    assert r.psa10_rank_score == 90


def test_vision_findings_are_resolved_through_the_manifest_index(rubric_scoped):
    """Provenance must survive the round trip — a rebuilt ref would launder
    an enhancement-only finding into a rejectable one."""
    ref = EvidenceRef(artifact_id="a1", image_hash="realhash",
                      origin=EvidenceOrigin.ENHANCED,
                      enhancement="clahe:clip=2.0", view="surface_clahe")
    vision = Assessment(
        findings=[VisionFinding(
            defect_type="scratches", category="surface", state="observed",
            confidence=0.99, psa10_relevant=True,
            evidence_artifact_ids=["a1"], location=_BOX, explanation="")],
        category_assessability=_ALL_ASSESSABLE,
        gem_view=GemView.VISIBLE_DISQUALIFIER)
    r = combine(HeuristicResult(), vision, _cov(), card_context_known=True,
                scoped_rules=rubric_scoped, manifest_index={"a1": ref},
                detectability=_det(category="surface", defect="scratches"))
    assert r.vision_present is True
    assert r.verdict is not Verdict.REJECT
    assert any("I3" in f.demotion_reason for f in r.fused)


def test_the_result_round_trips_and_canonicalizes(rubric_scoped):
    import json

    from card_reviewer.review.canonical import canonicalize

    r = combine(HeuristicResult(findings=[_f()]), None, _cov(),
                card_context_known=True, scoped_rules=rubric_scoped,
                detectability=_det())
    assert CombinedResult.model_validate(json.loads(r.model_dump_json())) == r
    assert canonicalize(r.model_dump(mode="json"))


def test_an_observed_finding_failing_i1_stays_rankable(rubric_scoped):
    """It routes to REVIEW, so it must sort above a confirmed reject.

    Flooring it to zero would make an unresolved concern indistinguishable
    from an established disqualifier in the triage ordering the score exists
    to provide.
    """
    r = combine(HeuristicResult(findings=[_f()]), None, _cov(),
                card_context_known=True, scoped_rules=rubric_scoped,
                detectability=_det(Scale.LOW))
    assert r.verdict is Verdict.REVIEW
    assert 0 < r.psa10_rank_score < 100


def test_a_confirmed_reject_scores_below_an_unresolved_concern(rubric_scoped):
    """The ordering that matters for triage."""
    confirmed = combine(HeuristicResult(findings=[_f()]), None, _cov(),
                        card_context_known=True, scoped_rules=rubric_scoped,
                        detectability=_det(Scale.HIGH))
    unresolved = combine(HeuristicResult(findings=[_f()]), None, _cov(),
                         card_context_known=True, scoped_rules=rubric_scoped,
                         detectability=_det(Scale.LOW))
    assert confirmed.psa10_rank_score < unresolved.psa10_rank_score
