"""Guards the mutation review found nothing holding.

Each of these existed in the production code and survived being deleted or
inverted, because the tests that named them could not reach them, or asserted
something the mutant also satisfied. They are grouped here because they share
a cause rather than a subject: a threshold is only guarded by a test that
sits on BOTH sides of it.
"""

import pytest
from detectability_helpers import detectability_map, regions_for, set_every_region

from card_reviewer.review.enums import Authority, Coverage, FindingState, Scale
from card_reviewer.review.findings import Severity
from card_reviewer.review.heuristic import (
    MIN_CONFIDENCE_FOR_OBSERVED, MIN_DETECTABILITY_FOR_OBSERVED, _state_for,
)
from card_reviewer.review.roles import ImageRole
from card_reviewer.review.taxonomy import Promotion, promotion_of

MEASURED = ("centering", "border_ratio")


def test_the_only_measurement_promotion_type_is_the_one_these_tests_use():
    """The reason the old tests could not reach either floor: they used
    corners/rounding, which the taxonomy classifies INTERPRETIVE, so the
    interpretive branch returned SUSPECTED before either floor was
    consulted. Both tests were named "even for measurement types" and
    neither used one."""
    assert promotion_of(*MEASURED) is Promotion.MEASUREMENT
    assert promotion_of("corners", "rounding") is Promotion.INTERPRETIVE


def test_low_detectability_prevents_observed_for_a_measurement_type():
    assert _state_for(*MEASURED, 0.99, Scale.LOW) is FindingState.SUSPECTED


def test_adequate_detectability_allows_observed_for_a_measurement_type():
    """The other side of the same line, without which the floor could be
    lowered to NONE unnoticed."""
    assert _state_for(*MEASURED, 0.99,
                      MIN_DETECTABILITY_FOR_OBSERVED) is FindingState.OBSERVED


def test_low_confidence_prevents_observed_for_a_measurement_type():
    """Absolute values, not `MIN_CONFIDENCE_FOR_OBSERVED - 0.01`: derived
    from the constant, the assertion moves WITH any change to it and the
    floor could be dropped to 0.1 unnoticed. That is the same shape of
    mistake this module exists to catch."""
    assert _state_for(*MEASURED, 0.5, Scale.HIGH) is FindingState.SUSPECTED
    assert _state_for(*MEASURED, 0.2, Scale.HIGH) is FindingState.SUSPECTED


def test_high_confidence_allows_observed():
    assert _state_for(*MEASURED, 0.95, Scale.HIGH) is FindingState.OBSERVED


def test_the_confidence_floor_is_where_it_is_declared():
    """Pins the boundary itself, so moving it is a deliberate act."""
    assert MIN_CONFIDENCE_FOR_OBSERVED == 0.8
    assert _state_for(*MEASURED, 0.79, Scale.HIGH) is FindingState.SUSPECTED
    assert _state_for(*MEASURED, 0.80, Scale.HIGH) is FindingState.OBSERVED


def test_an_interpretive_type_is_suspected_however_good_the_evidence():
    """CV cannot establish an interpretive defect at all."""
    assert _state_for("corners", "rounding", 1.0,
                      Scale.HIGH) is FindingState.SUSPECTED


# --- the PARTIAL / INADEQUATE line ----------------------------------------

def _front_assessing(categories):
    from card_reviewer.review.taxonomy import CATEGORIES, defect_types_for

    detectability = detectability_map((ImageRole.FRONT,), Scale.NONE)
    for category in categories:
        for defect_type in defect_types_for(category):
            set_every_region(detectability, ImageRole.FRONT, category,
                             defect_type, Scale.HIGH)
    return detectability


@pytest.mark.parametrize("count,expected", [
    (1, Coverage.INADEQUATE),
    (2, Coverage.PARTIAL),
    (3, Coverage.PARTIAL),
])
def test_the_partial_line_is_pinned_on_both_sides(count, expected):
    """MIN_FRONT_CATEGORIES_FOR_PARTIAL decides whether a barely-assessable
    front is rankable at all. Moving it to 1 or to 3 both survived; only 4
    was caught."""
    from card_reviewer.review.policies.coverage_v1 import evaluate_coverage
    from card_reviewer.review.taxonomy import CATEGORIES

    detectability = _front_assessing(CATEGORIES[:count])
    result = evaluate_coverage(detectability, {}, {}, (ImageRole.FRONT,))
    assert result.outcome is expected


# --- rule authority --------------------------------------------------------

def test_a_contradicted_rule_cannot_make_a_finding_relevant(rubric):
    """resolve_relevance skips INERT rules, and the skip survived deletion
    because the live rubric contains no CONTRADICTED rules today. The
    scoring tests solved the same problem by constructing one; relevance
    never did."""
    from card_reviewer.review.policies.authority_v1 import authority_of
    from card_reviewer.review.evaluability import ScopedRule
    from card_reviewer.review.enums import RuleEvaluability

    contradicted = None
    for rule in rubric.rules:
        candidate = rule.model_copy(update={"evidence_type": "contradicted"})
        if authority_of(candidate) is Authority.INERT:
            contradicted = candidate
            break
    if contradicted is None:
        pytest.skip("no rule shape in this rubric maps to INERT")

    scoped = [ScopedRule(rule=contradicted,
                         evaluability=RuleEvaluability.APPLICABLE, reason="")]
    from card_reviewer.review.relevance import resolve_relevance
    from card_reviewer.review.findings import Finding, FindingProducer
    from card_reviewer.review.provenance import (
        EvidenceOrigin, EvidenceRef, NormalizedBox,
    )

    box = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)
    finding = Finding(
        defect_type="whitening", category=contradicted.category.value,
        state=FindingState.OBSERVED, producer=FindingProducer.HEURISTIC,
        confidence=0.95, psa10_relevant=True, location=box,
        evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                              origin=EvidenceOrigin.ORIGINAL,
                              view="corner_top_left", region=box)])

    resolved = resolve_relevance([finding], scoped)
    assert resolved[0].authority is not Authority.BINDING, (
        "a contradicted rule reached BINDING authority")
    assert contradicted.id not in resolved[0].finding.rule_ids, (
        "a contradicted rule was cited as support for a finding")


# --- centering severity ----------------------------------------------------

@pytest.mark.parametrize("horizontal,expected", [
    # Either side of CENTERING_SEVERE_PP (15.0), and both clear of
    # CENTERING_TOLERANCE_PP so a finding is produced at all.
    (50.0 + 20.0, Severity.SEVERE),
    (50.0 + 12.0, Severity.MODERATE),
])
def test_centering_severity_is_pinned(horizontal, expected, rubric_scoped):
    """Severity drives estimated_psa_grade — `<=8` against `8-9` — and
    CENTERING_SEVERE_PP could be moved from 15.0 to 60.0 unnoticed."""
    from card_reviewer.review.assembly import Assembled
    from card_reviewer.review.heuristic import evaluate
    from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef

    key = Assembled.key(ImageRole.FRONT, "center", "centering", "border_ratio")
    assembled = Assembled(
        detectability_flat={key: Scale.HIGH.label},
        centering={"measurable": True, "horizontal": horizontal,
                   "vertical": 50.0},
        faces_present=["front"],
        evidence_refs={"centering:border_ratio": [
            EvidenceRef(artifact_id="a", image_hash="h",
                        origin=EvidenceOrigin.NORMALIZED,
                        view="surface_original")]})

    findings = [f for f in evaluate(assembled, rubric_scoped).findings
                if f.category == "centering"]
    assert findings, "no centering finding was produced"
    assert findings[0].severity is expected


# --- metadata gaps are not photograph gaps --------------------------------

def test_a_metadata_gap_never_generates_a_photo_request():
    """The `continue` after `identify = True` survived deletion because
    UNKNOWN_PRODUCT_CONTEXT has no PHOTO_REQUESTS template today. That is
    the same argument the authors used for the STRUCTURAL branch, where they
    monkeypatched a template in to prove the class check is load-bearing —
    written for one branch and not the other.

    CLAUDE.md: do not convert metadata problems into photography problems.
    """
    from card_reviewer.review.enums import UndetectabilityClass
    from card_reviewer.review.policies import coverage_v1
    from card_reviewer.review.policies.coverage_v1 import Limitation, _requests

    limitation = Limitation(
        face="front", category="surface", defect_type="*",
        reason_code="UNKNOWN_PRODUCT_CONTEXT",
        undetectability_class=UndetectabilityClass.METADATA_RESOLVABLE)

    original = dict(coverage_v1.PHOTO_REQUESTS)
    try:
        coverage_v1.PHOTO_REQUESTS["UNKNOWN_PRODUCT_CONTEXT"] = (
            "a better photograph of the {face} {category}")
        photos, identify = _requests([limitation])
    finally:
        coverage_v1.PHOTO_REQUESTS.clear()
        coverage_v1.PHOTO_REQUESTS.update(original)

    assert identify is True, "the card identification request was not raised"
    assert photos == [], (
        "a metadata gap produced a photo request; no photograph resolves it")


# --- authority defaults ----------------------------------------------------

def test_an_unmatched_finding_scores_as_advisory_not_binding():
    """`rank_score(...) > 0` was satisfied by both: ADVISORY scores 75 and
    BINDING 65, so flipping the default survived. The value has to be
    pinned, or compared against an explicit ADVISORY call."""
    from card_reviewer.review.findings import Finding, FindingProducer
    from card_reviewer.review.policies.scoring_v1 import rank_score
    from card_reviewer.review.provenance import (
        EvidenceOrigin, EvidenceRef, NormalizedBox,
    )

    box = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)
    unmatched = Finding(
        defect_type="rounding", category="corners",
        state=FindingState.OBSERVED, producer=FindingProducer.HEURISTIC,
        confidence=0.9, psa10_relevant=True, location=box, rule_ids=[],
        evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                              origin=EvidenceOrigin.ORIGINAL,
                              view="corner_top_left", region=box)])

    # The third element is i1_satisfied, a bool. Passing a Scale here reads
    # as truthy but misses every PENALTIES key, so both calls returned 100
    # and the comparison proved nothing — the same shape of mistake this
    # module exists to catch.
    default = rank_score([(unmatched, Authority.ADVISORY, True)],
                         Coverage.SUFFICIENT)
    binding = rank_score([(unmatched, Authority.BINDING, True)],
                         Coverage.SUFFICIENT)

    assert default != binding, "the two authorities are indistinguishable"
    assert default > binding, (
        "an ADVISORY rule cost the card as much as a BINDING one")
