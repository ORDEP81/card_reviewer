import pytest

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.assembly import Assembled
from card_reviewer.review.context import CardContext
from card_reviewer.review.enums import FindingState, Provenance, Scale
from card_reviewer.review.evaluability import scope_rules
from card_reviewer.review.heuristic import best_detectability, evaluate
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef, NormalizedBox
from card_reviewer.review.roles import ImageRole


@pytest.fixture
def rules_known():
    ctx = CardContext(canonical_card_types=["chrome"],
                      provenance=Provenance.SUPPLIED, confidence=1.0)
    return scope_rules(load_active_rubric().for_card(["chrome"], None), ctx)


@pytest.fixture
def rules_unknown():
    return scope_rules(load_active_rubric().for_card(None, None), CardContext())


def _ev(view="corner_bottom_left"):
    return EvidenceRef(artifact_id="a1", image_hash="h1",
                       origin=EvidenceOrigin.NORMALIZED, view=view,
                       region=NormalizedBox(x0=0.0, y0=0.8, x1=0.2, y1=1.0))


def _assembled(**kw):
    flat = {Assembled.key(ImageRole.FRONT, c, d): Scale.HIGH.label
            for c, d in (("corners", "rounding"), ("corners", "whitening"),
                         ("surface", "scratches"), ("surface", "print_lines"),
                         ("centering", "border_ratio"))}
    base = dict(
        centering={"horizontal": 52.0, "vertical": 51.0, "measurable": True},
        detectability_flat=flat, anomalies=[], faces_present=["front"],
        evidence_refs={"corners:rounding": [_ev()],
                       "surface:print_lines": [_ev("surface_original")],
                       "centering:border_ratio": [_ev("surface_original")]})
    return Assembled(**(base | kw))


def test_best_detectability_takes_the_max_across_faces():
    d = {(ImageRole.FRONT, "corners", "rounding"): Scale.LOW,
         (ImageRole.BACK, "corners", "rounding"): Scale.HIGH}
    assert best_detectability(d, "corners", "rounding") is Scale.HIGH


def test_best_detectability_returns_none_when_nothing_is_registered():
    """Absent evidence must never look like adequate evidence."""
    assert best_detectability({}, "corners", "rounding") is Scale.NONE


def test_a_measurement_type_may_reach_observed(rules_known):
    """Corner rounding is geometric — CV can establish it outright."""
    a = _assembled(anomalies=[{"defect_type": "rounding", "category": "corners",
                               "region": "bottom_left", "confidence": 0.95,
                               "severity": "moderate"}])
    f = next(f for f in evaluate(a, rules_known).findings
             if f.defect_type == "rounding")
    assert f.state is FindingState.OBSERVED


def test_an_interpretive_type_can_never_exceed_suspected_from_cv_alone(rules_known):
    """This is what stops OFF mode manufacturing confident defects out of
    high-contrast pixels."""
    a = _assembled(anomalies=[{"defect_type": "print_lines", "category": "surface",
                               "confidence": 0.99, "severity": "severe"}])
    f = next(f for f in evaluate(a, rules_known).findings
             if f.defect_type == "print_lines")
    assert f.state is FindingState.SUSPECTED


def test_low_detectability_prevents_observed_even_for_measurement_types(rules_known):
    a = _assembled(
        detectability_flat={Assembled.key(ImageRole.FRONT, "corners", "rounding"):
                            Scale.LOW.label},
        anomalies=[{"defect_type": "rounding", "category": "corners",
                    "region": "bottom_left", "confidence": 0.99,
                    "severity": "severe"}])
    f = next(f for f in evaluate(a, rules_known).findings
             if f.defect_type == "rounding")
    assert f.state is not FindingState.OBSERVED


def test_low_confidence_prevents_observed_even_for_measurement_types(rules_known):
    a = _assembled(anomalies=[{"defect_type": "rounding", "category": "corners",
                               "region": "bottom_left", "confidence": 0.2}])
    f = next(f for f in evaluate(a, rules_known).findings
             if f.defect_type == "rounding")
    assert f.state is FindingState.SUSPECTED


def test_every_finding_carries_a_location_so_fusion_can_correlate(rules_known):
    """A finding without a location can never fuse, so the same defect seen
    by both producers would be penalized twice."""
    a = _assembled(anomalies=[{"defect_type": "rounding", "category": "corners",
                               "region": "bottom_left", "confidence": 0.95}])
    assert all(f.location is not None for f in evaluate(a, rules_known).findings)


def test_unevaluable_rules_never_attach_to_a_finding(rules_unknown):
    a = _assembled(anomalies=[{"defect_type": "scratches", "category": "surface",
                               "confidence": 0.9}],
                   evidence_refs={"surface:scratches": [_ev("surface_original")],
                                  "centering:border_ratio": [_ev("surface_original")]})
    result = evaluate(a, rules_unknown)
    assert result.findings
    assert all("SURFACE_SHINY_001" not in f.rule_ids for f in result.findings)
    assert "UNKNOWN_PRODUCT_CONTEXT" in result.unevaluable_reasons


def test_centering_within_psa_tolerance_produces_no_disqualifier(rules_known):
    """CENTERING_PSA10_STANDARD_002: approximately 55/45, explicitly not a
    hard arithmetic cutoff — 52/48 is comfortably inside."""
    assert not [f for f in evaluate(_assembled(), rules_known).findings
                if f.category == "centering"]


def test_a_grossly_miscut_card_does_produce_a_centering_finding(rules_known):
    """Centering is a measurement, not an anomaly candidate, so it needs its
    own evaluation or a 75/25 card produces no finding at all."""
    a = _assembled(centering={"horizontal": 75.0, "vertical": 50.0,
                              "measurable": True})
    assert [f for f in evaluate(a, rules_known).findings
            if f.category == "centering" and f.state is FindingState.OBSERVED]


def test_an_unmeasurable_centering_produces_no_finding(rules_known):
    a = _assembled(centering={"measurable": False,
                              "reason": "BORDERLESS_OR_NO_RELIABLE_REFERENCE"})
    assert not [f for f in evaluate(a, rules_known).findings
                if f.category == "centering"]


def test_an_anomaly_without_evidence_refs_is_dropped(rules_known):
    """A finding with no evidence cannot support anything downstream."""
    a = _assembled(anomalies=[{"defect_type": "dimples", "category": "surface",
                               "confidence": 0.9}])
    assert not [f for f in evaluate(a, rules_known).findings
                if f.defect_type == "dimples"]


def test_the_result_round_trips_through_json(rules_known):
    import json

    from card_reviewer.review.heuristic import HeuristicResult

    a = _assembled(anomalies=[{"defect_type": "rounding", "category": "corners",
                               "region": "bottom_left", "confidence": 0.95}])
    r = evaluate(a, rules_known)
    assert HeuristicResult.model_validate(json.loads(r.model_dump_json())) == r
