import pytest

from card_reviewer.review.enums import (
    Authority, Coverage, FindingState, ReviewConfidence,
)
from card_reviewer.review.findings import Finding, FindingProducer, Severity
from card_reviewer.review.policies.scoring_v1 import (
    estimated_grade, rank_score, review_confidence,
)
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef


def _f(state, severity=None, relevant=True, authority=Authority.BINDING, i1=True):
    return (
        Finding(defect_type="rounding", category="corners", state=state,
                producer=FindingProducer.HEURISTIC, confidence=0.9,
                psa10_relevant=relevant, severity=severity,
                evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                                      origin=EvidenceOrigin.ORIGINAL,
                                      view="front")]),
        authority, i1,
    )


# --- rank score ------------------------------------------------------------

def test_a_clean_card_with_full_coverage_scores_100():
    assert rank_score([], Coverage.SUFFICIENT) == 100


def test_the_score_is_null_when_coverage_is_inadequate():
    """Do not manufacture a neutral-looking number for an unrankable card."""
    assert rank_score([], Coverage.INADEQUATE) is None


def test_adding_a_credible_negative_finding_never_raises_the_score():
    assert rank_score([_f(FindingState.SUSPECTED, i1=False)],
                      Coverage.SUFFICIENT) <= rank_score([], Coverage.SUFFICIENT)


def test_promoting_suspected_to_observed_never_raises_the_score():
    s = rank_score([_f(FindingState.SUSPECTED, i1=False)], Coverage.SUFFICIENT)
    o = rank_score([_f(FindingState.OBSERVED, i1=False)], Coverage.SUFFICIENT)
    assert o <= s


def test_improving_coverage_without_adding_a_defect_never_lowers_the_score():
    assert rank_score([], Coverage.SUFFICIENT) >= rank_score([], Coverage.PARTIAL)


def test_only_an_i1_satisfying_binding_disqualifier_floors_the_score():
    assert rank_score([_f(FindingState.OBSERVED, i1=True)], Coverage.SUFFICIENT) == 0


def test_an_observed_finding_failing_i1_stays_meaningfully_rankable():
    """It routes to REVIEW, so it must sort above a confirmed reject rather
    than collapsing to the same 0."""
    score = rank_score([_f(FindingState.OBSERVED, i1=False)], Coverage.SUFFICIENT)
    assert 0 < score < 100


def test_an_unresolved_finding_scores_worse_than_a_merely_suspected_one():
    assert rank_score([_f(FindingState.OBSERVED, i1=False)], Coverage.SUFFICIENT) < (
        rank_score([_f(FindingState.SUSPECTED, i1=False)], Coverage.SUFFICIENT))


def test_advisory_authority_costs_less_than_binding():
    a = rank_score([_f(FindingState.SUSPECTED, authority=Authority.ADVISORY,
                       i1=False)], Coverage.SUFFICIENT)
    b = rank_score([_f(FindingState.SUSPECTED, authority=Authority.BINDING,
                       i1=False)], Coverage.SUFFICIENT)
    assert a > b


def test_not_assessable_findings_cost_nothing_in_the_score():
    """Missing evidence is already paid for in coverage and confidence;
    charging it here too would double-count absence as a defect."""
    assert rank_score([_f(FindingState.NOT_ASSESSABLE, i1=False)],
                      Coverage.SUFFICIENT) == rank_score([], Coverage.SUFFICIENT)


def test_an_inert_rule_contributes_no_penalty(monkeypatch):
    """Guards the INERT skip itself, not the absence of a table entry.

    PENALTIES has no INERT rows today, so this would pass even if the skip
    were deleted. Injecting one proves the authority check is what excludes
    a contradicted rule — otherwise adding a row later would silently let a
    rule the rubric retracted start penalizing cards again.
    """
    from card_reviewer.review.policies import scoring_v1

    monkeypatch.setitem(scoring_v1.PENALTIES,
                        (FindingState.OBSERVED, Authority.INERT, True), 40)
    assert rank_score([_f(FindingState.OBSERVED, authority=Authority.INERT)],
                      Coverage.SUFFICIENT) == 100


def test_an_irrelevant_finding_contributes_no_penalty():
    assert rank_score([_f(FindingState.OBSERVED, relevant=False)],
                      Coverage.SUFFICIENT) == 100


def test_two_separate_defects_cost_more_than_one():
    one = rank_score([_f(FindingState.SUSPECTED, i1=False)], Coverage.SUFFICIENT)
    two = rank_score([_f(FindingState.SUSPECTED, i1=False),
                      _f(FindingState.SUSPECTED, i1=False)], Coverage.SUFFICIENT)
    assert two < one


def test_the_score_is_always_within_bounds():
    assert rank_score([_f(FindingState.OBSERVED) for _ in range(20)],
                      Coverage.PARTIAL) == 0


def test_an_unmapped_finding_defaults_to_advisory_never_binding():
    bare = _f(FindingState.OBSERVED)[0]
    assert rank_score([bare], Coverage.SUFFICIENT) > 0


# --- grade estimate --------------------------------------------------------

def test_a_clean_fully_covered_card_estimates_a_10():
    assert estimated_grade([], Coverage.SUFFICIENT) == "10"


def test_partial_coverage_widens_the_estimate_rather_than_lowering_it():
    assert estimated_grade([], Coverage.PARTIAL) == "9-10"


@pytest.mark.parametrize("severity,expected", [
    (Severity.MINOR, "9"), (Severity.MODERATE, "8-9"), (Severity.SEVERE, "<=8"),
])
def test_the_grade_follows_the_worst_confirmed_defect(severity, expected):
    assert estimated_grade([_f(FindingState.OBSERVED, severity)],
                           Coverage.SUFFICIENT) == expected


def test_two_moderate_defects_estimate_below_8():
    fs = [_f(FindingState.OBSERVED, Severity.MODERATE),
          _f(FindingState.OBSERVED, Severity.MODERATE)]
    assert estimated_grade(fs, Coverage.SUFFICIENT) == "<=8"


def test_the_grade_is_null_when_coverage_is_inadequate():
    assert estimated_grade([], Coverage.INADEQUATE) is None


def test_suspected_findings_do_not_lower_the_grade_estimate():
    assert estimated_grade([_f(FindingState.SUSPECTED, Severity.SEVERE, i1=False)],
                           Coverage.SUFFICIENT) == "10"


def test_an_observed_finding_failing_i1_does_not_lower_the_grade():
    """It is an unresolved concern, not a confirmed defect — it costs score
    and routes to REVIEW, but the grade reports what is established."""
    assert estimated_grade([_f(FindingState.OBSERVED, Severity.SEVERE, i1=False)],
                           Coverage.SUFFICIENT) == "10"


def test_the_grade_is_not_a_conversion_of_the_score():
    """They answer different questions and must be able to disagree."""
    fs = [_f(FindingState.SUSPECTED, i1=False) for _ in range(4)]
    assert rank_score(fs, Coverage.SUFFICIENT) < 90
    assert estimated_grade(fs, Coverage.SUFFICIENT) == "10"


# --- review confidence -----------------------------------------------------

def test_confidence_is_low_when_coverage_is_inadequate():
    assert review_confidence(Coverage.INADEQUATE, [], False, True) is (
        ReviewConfidence.LOW)


def test_a_missing_required_face_is_low_confidence_even_though_partial():
    """A front-only card is PARTIAL and rankable, but we never saw half of
    it — the assessment deserves LOW confidence."""
    assert review_confidence(Coverage.PARTIAL, [], False, True,
                             required_face_missing=True) is ReviewConfidence.LOW


def test_other_partial_coverage_remains_medium():
    assert review_confidence(Coverage.PARTIAL, [], False, True) is (
        ReviewConfidence.MEDIUM)


def test_confidence_is_low_on_an_unresolved_contradiction():
    assert review_confidence(Coverage.SUFFICIENT, ["x"], False, True) is (
        ReviewConfidence.LOW)


def test_confidence_is_medium_when_the_two_producers_disagreed():
    assert review_confidence(Coverage.SUFFICIENT, [], True, True) is (
        ReviewConfidence.MEDIUM)


def test_confidence_is_medium_when_card_context_is_unknown():
    assert review_confidence(Coverage.SUFFICIENT, [], False, False) is (
        ReviewConfidence.MEDIUM)


def test_confidence_is_high_only_when_everything_is_resolved():
    assert review_confidence(Coverage.SUFFICIENT, [], False, True) is (
        ReviewConfidence.HIGH)


def test_a_high_score_can_carry_low_confidence():
    """'Probably clean, but we could barely see it' must be expressible."""
    assert rank_score([], Coverage.PARTIAL) >= 85
    assert review_confidence(Coverage.PARTIAL, [], False, True,
                             required_face_missing=True) is ReviewConfidence.LOW
