"""A region we could not see is not a photograph we cannot use.

Requiring EVERY region of a defect type to be assessable is right for
SUFFICIENT — that is what stops a clean corner vouching for a glared one.
Carried into the PARTIAL/INADEQUATE decision it went too far the other way:
corners, edges and surface all share the corner regions, so ONE glared corner
blocked three of the four categories, left a single category assessed, and
the card was dropped as INSUFFICIENT_IMAGES.

Coverage and condition are separate concepts, and PARTIAL exists precisely
for a card worth a human look that cannot yet PASS. A card with three good
corners and one glared one belongs there, with a limitation and a photo
request naming the corner.
"""

import pytest
from detectability_helpers import detectability_map, regions_for, set_one_region

from card_reviewer.review.enums import Coverage, Scale
from card_reviewer.review.policies.coverage_v1 import REQUIRED_FACES, evaluate_coverage
from card_reviewer.review.roles import ImageRole
from card_reviewer.review.taxonomy import CATEGORIES, defect_types_for


def _one_glared_corner(region="top_left"):
    detectability = detectability_map(REQUIRED_FACES)
    reasons = {}
    for category in CATEGORIES:
        if region not in regions_for(category):
            continue
        for defect_type in defect_types_for(category):
            key = (ImageRole.FRONT, region, category, defect_type)
            detectability[key] = Scale.LOW
            reasons[key] = "GLARE"
    return detectability, reasons


def test_one_glared_corner_does_not_make_the_card_unusable():
    detectability, reasons = _one_glared_corner()
    result = evaluate_coverage(detectability, reasons, {}, REQUIRED_FACES)
    assert result.outcome is Coverage.PARTIAL
    assert result.rankable is True


def test_one_glared_corner_still_prevents_pass():
    """The half that must not be lost: a corner nobody could see is missing
    evidence, and missing evidence cannot pass."""
    detectability, reasons = _one_glared_corner()
    assert evaluate_coverage(detectability, reasons, {},
                             REQUIRED_FACES).outcome is not Coverage.SUFFICIENT


def test_the_glared_corner_is_named_in_a_photo_request():
    detectability, reasons = _one_glared_corner()
    result = evaluate_coverage(detectability, reasons, {}, REQUIRED_FACES)
    assert any("diffuse" in photo.lower()
               for photo in result.recommended_additional_photos)
    assert any(limitation.region == "top_left"
               for limitation in result.limitations)


def test_a_front_where_nothing_is_assessable_is_still_inadequate():
    """The other end of the scale must keep working."""
    nothing = detectability_map((ImageRole.FRONT,), Scale.NONE)
    assert evaluate_coverage(nothing, {}, {},
                             (ImageRole.FRONT,)).outcome is Coverage.INADEQUATE


def test_every_corner_glared_is_inadequate_not_partial():
    """Four glared corners is a different photograph from one."""
    detectability = detectability_map(REQUIRED_FACES)
    reasons = {}
    for category in CATEGORIES:
        for region in regions_for(category):
            for defect_type in defect_types_for(category):
                key = (ImageRole.FRONT, region, category, defect_type)
                detectability[key] = Scale.LOW
                reasons[key] = "GLARE"
    result = evaluate_coverage(detectability, reasons, {}, REQUIRED_FACES)
    assert result.outcome is Coverage.INADEQUATE
    assert result.rankable is False


def test_a_rubric_blocked_category_is_not_partly_assessed():
    """A product-scoped rule we cannot apply is a METADATA gap: the pixels
    may be perfect and the category still cannot be judged. Counting it as
    partly assessed would let an unidentified card look better covered than
    it is, which is the opposite of what UNKNOWN_PRODUCT_CONTEXT means."""
    from card_reviewer.review.policies.coverage_v1 import UnevaluableRule

    detectability = detectability_map((ImageRole.FRONT,), Scale.NONE)
    for defect_type in defect_types_for("centering"):
        set_one_region(detectability, ImageRole.FRONT, "center", "centering",
                       defect_type, Scale.HIGH)
    for defect_type in defect_types_for("surface"):
        for region in regions_for("surface"):
            set_one_region(detectability, ImageRole.FRONT, region, "surface",
                           defect_type, Scale.HIGH)

    blocked = [UnevaluableRule(rule_id="SURFACE_SHINY_001", category="surface",
                               reason_code="UNKNOWN_PRODUCT_CONTEXT")]
    result = evaluate_coverage(detectability, {}, {}, (ImageRole.FRONT,),
                               unevaluable_rules=blocked)

    assert "surface" not in result.assessed.get("front", [])
    assert result.outcome is Coverage.INADEQUATE, (
        "a category blocked by an unapplied rule was counted as coverage")


def test_a_card_with_no_front_is_inadequate_however_good_its_back():
    """Detectability entries can exist for a face no photograph shows —
    nothing prunes them — so the presence check has to be explicit.

    The front is the face the PARTIAL decision reads, so this is where it
    bites: a back-only listing whose map still carries front keys would
    otherwise count those as coverage and be forwarded as rankable, on the
    strength of a photograph that does not exist.
    """
    both_faces = detectability_map(REQUIRED_FACES)
    back_only = evaluate_coverage(both_faces, {}, {}, (ImageRole.BACK,))

    assert back_only.outcome is Coverage.INADEQUATE
    assert back_only.rankable is False
    assert back_only.assessed.get("front", []) == []
    assert any(limitation.reason_code == "MISSING_FACE"
               for limitation in back_only.limitations)


def test_a_front_only_card_is_still_partial_and_rankable():
    """The front-only policy: usable front, no back, no disqualifier stays
    rankable. The presence check must not cost that."""
    both_faces = detectability_map(REQUIRED_FACES)
    front_only = evaluate_coverage(both_faces, {}, {}, (ImageRole.FRONT,))

    assert front_only.outcome is Coverage.PARTIAL
    assert front_only.rankable is True
