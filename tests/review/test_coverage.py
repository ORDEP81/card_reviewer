from card_reviewer.review.enums import Coverage, Scale, UndetectabilityClass
from card_reviewer.review.policies.coverage_v1 import (
    MIN_ASSESSED,
    REQUIRED_FACES,
    UnevaluableRule,
    evaluate_coverage,
)
from card_reviewer.review.roles import ImageRole
from card_reviewer.review.taxonomy import CATEGORIES, defect_types_for


from detectability_helpers import detectability_map, regions_for, set_every_region


def _good(faces=(ImageRole.FRONT, ImageRole.BACK)):
    return detectability_map(faces)


def test_full_evidence_on_both_faces_is_sufficient():
    r = evaluate_coverage(_good(), {}, {}, REQUIRED_FACES)
    assert r.outcome is Coverage.SUFFICIENT
    assert r.rankable is True


def test_a_white_bordered_card_can_still_reach_sufficient():
    """Demanding evidence no photograph could supply would make PASS
    unreachable for most modern base cards — a false-rejection machine."""
    det, reasons = _good(), {}
    for face in REQUIRED_FACES:
        for cat in ("corners", "edges"):
            set_every_region(det, face, cat, "whitening", Scale.LOW)
            for _r in regions_for(cat):
                reasons[(face, _r, cat, "whitening")] = "WHITE_BORDER"
    r = evaluate_coverage(det, reasons, {}, REQUIRED_FACES)
    assert r.outcome is Coverage.SUFFICIENT
    assert any(l.undetectability_class is UndetectabilityClass.STRUCTURAL
               for l in r.limitations)


def test_glare_on_the_same_corner_is_circumstantial_and_blocks_sufficient():
    det = _good()
    set_every_region(det, ImageRole.FRONT, "corners", "whitening", Scale.LOW)
    r = evaluate_coverage(det, {(ImageRole.FRONT, "corners", "whitening"): "GLARE"},
                          {}, REQUIRED_FACES)
    assert r.outcome is Coverage.PARTIAL


def test_a_usable_front_only_card_is_partial_and_rankable():
    r = evaluate_coverage(_good((ImageRole.FRONT,)), {}, {}, (ImageRole.FRONT,))
    assert r.outcome is Coverage.PARTIAL
    assert r.rankable is True


def test_an_unassessable_front_is_inadequate_and_unrankable():
    det = detectability_map((ImageRole.FRONT,), Scale.NONE)
    r = evaluate_coverage(det, {}, {}, (ImageRole.FRONT,))
    assert r.outcome is Coverage.INADEQUATE
    assert r.rankable is False


def test_vision_saying_not_assessable_overrides_good_cv_suitability():
    """CV measures whether the pixels COULD carry evidence; vision reports
    whether anything could actually be concluded."""
    r = evaluate_coverage(_good(), {}, {"surface": False}, REQUIRED_FACES)
    assert r.outcome is Coverage.PARTIAL


def test_an_unapplied_product_rule_is_a_metadata_resolvable_limitation():
    """It arrives as itself, not simulated by lowering pixel detectability."""
    r = evaluate_coverage(_good(), {}, {}, REQUIRED_FACES,
                          unevaluable_rules=[UnevaluableRule(
                              rule_id="SURFACE_SHINY_001", category="surface",
                              reason_code="UNKNOWN_PRODUCT_CONTEXT")])
    assert any(l.undetectability_class is UndetectabilityClass.METADATA_RESOLVABLE
               for l in r.limitations)


def test_perfect_photos_with_unknown_product_are_partial_and_rankable():
    """Nothing is wrong with the photographs, so no photo request is
    warranted — we simply do not know what card this is."""
    r = evaluate_coverage(_good(), {}, {}, REQUIRED_FACES,
                          unevaluable_rules=[UnevaluableRule(
                              rule_id="SURFACE_SHINY_001", category="surface",
                              reason_code="UNKNOWN_PRODUCT_CONTEXT")])
    assert r.outcome is Coverage.PARTIAL
    assert r.rankable is True
    assert r.card_identification_request is True
    assert r.recommended_additional_photos == []


def test_photo_requests_derive_from_circumstantial_limitations_only():
    det = _good()
    set_every_region(det, ImageRole.FRONT, "corners", "whitening", Scale.LOW)
    set_every_region(det, ImageRole.FRONT, "surface", "scratches", Scale.LOW)
    reasons = {}
    for _r in regions_for("corners"):
        reasons[(ImageRole.FRONT, _r, "corners", "whitening")] = "WHITE_BORDER"
    for _r in regions_for("surface"):
        reasons[(ImageRole.FRONT, _r, "surface", "scratches")] = "GLARE"
    r = evaluate_coverage(det, reasons, {}, REQUIRED_FACES)
    assert any("diffuse" in p.lower() for p in r.recommended_additional_photos)
    assert not any("white" in p.lower() for p in r.recommended_additional_photos)


def test_a_missing_face_is_a_circumstantial_limitation():
    r = evaluate_coverage(_good((ImageRole.FRONT,)), {}, {}, (ImageRole.FRONT,))
    assert any(l.reason_code == "MISSING_FACE" for l in r.limitations)
    assert any("back" in p.lower() for p in r.recommended_additional_photos)


def test_min_assessed_is_the_declared_moderate_threshold():
    assert MIN_ASSESSED is Scale.MODERATE


def test_the_result_round_trips_through_json():
    import json

    from card_reviewer.review.policies.coverage_v1 import CoverageResult

    r = evaluate_coverage(_good(), {}, {}, REQUIRED_FACES)
    assert CoverageResult.model_validate(json.loads(r.model_dump_json())) == r


def test_a_structural_limitation_yields_no_photo_request_even_if_templated(
        monkeypatch):
    """Guards the structural check itself, not the absence of a template.

    WHITE_BORDER has no entry in PHOTO_REQUESTS today, so the previous test
    would still pass if the structural skip were deleted. Giving it one
    proves the class check is what excludes it — otherwise adding a template
    later would silently start asking for a better photograph of something
    no photograph can show.
    """
    from card_reviewer.review.policies import coverage_v1

    monkeypatch.setitem(coverage_v1.PHOTO_REQUESTS, "WHITE_BORDER",
                        "a photograph of the {face} {category} on dark backing")
    det, reasons = _good(), {}
    for face in REQUIRED_FACES:
        set_every_region(det, face, "corners", "whitening", Scale.LOW)
        for _r in regions_for("corners"):
            reasons[(face, _r, "corners", "whitening")] = "WHITE_BORDER"
    r = evaluate_coverage(det, reasons, {}, REQUIRED_FACES)
    assert any(l.reason_code == "WHITE_BORDER" for l in r.limitations)
    assert r.recommended_additional_photos == []
