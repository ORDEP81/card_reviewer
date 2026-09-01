"""Detectability belongs to a region, and must survive assembly as one.

Observability measures per (region, category, defect_type). Assembly used to
re-key that to (role, category, defect_type) keeping the MAXIMUM across
regions, so a clean top-left corner spoke for a glared bottom-right one and
the GLARE limitation was deleted outright. The card then reached SUFFICIENT
and PASS with a corner nobody could see, and the report said "Limitations
(0)".

The spec ties I1's adequacy prong to detectability "at the finding's location
on the image that established it", which is not expressible once the region
is gone.
"""

import pytest

from card_reviewer.review.assembly import Assembled, ImageEvidence, assemble
from card_reviewer.review.enums import Coverage, Scale
from card_reviewer.review.enums import Provenance
from card_reviewer.review.roles import ImageRole, ResolvedRole
from card_reviewer.review.taxonomy import CATEGORIES, defect_types_for

CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _full(value=Scale.HIGH):
    from card_reviewer.review.imaging.observability import REGIONS_FOR_CATEGORY

    return {(region, category, defect_type): value
            for category in CATEGORIES
            for region in REGIONS_FOR_CATEGORY[category]
            for defect_type in defect_types_for(category)}


def _evidence(image_hash, detectability, reason_codes=None):
    return ImageEvidence(
        image_hash=image_hash, detectability=detectability,
        reason_codes=reason_codes or {}, sharpness=100.0,
        centering={"measurable": True, "horizontal": 50.0, "vertical": 50.0})


def _roles(*hashes):
    return {h: ResolvedRole(image_hash=h, role=ImageRole.FRONT,
                            provenance=Provenance.SUPPLIED, confidence=1.0)
            for h in hashes}


def test_a_glared_corner_is_not_covered_by_a_clean_one():
    detectability = _full()
    detectability[("bottom_right", "corners", "rounding")] = Scale.LOW
    reasons = {("bottom_right", "corners", "rounding"): "GLARE"}

    assembled = assemble([_evidence("h1", detectability, reasons)],
                         _roles("h1"))

    key = Assembled.key(ImageRole.FRONT, "bottom_right", "corners", "rounding")
    assert assembled.detectability_flat[key] == Scale.LOW.label, (
        "the glared corner's own value must survive assembly")
    assert assembled.reason_codes_flat[key] == "GLARE", (
        "and so must the reason, or it can never become a photo request")


def test_a_clean_corner_keeps_its_own_high_value():
    detectability = _full()
    detectability[("bottom_right", "corners", "rounding")] = Scale.LOW
    assembled = assemble([_evidence("h1", detectability)], _roles("h1"))

    key = Assembled.key(ImageRole.FRONT, "top_left", "corners", "rounding")
    assert assembled.detectability_flat[key] == Scale.HIGH.label


def test_a_second_photograph_of_the_same_region_still_wins_best_of():
    """Best-of is across IMAGES of a region, which is the real intent:
    a defect visible in ANY photograph of that region is observable."""
    poor = _full()
    poor[("bottom_right", "corners", "rounding")] = Scale.LOW
    good = _full()

    assembled = assemble(
        [_evidence("h1", poor, {("bottom_right", "corners", "rounding"): "GLARE"}),
         _evidence("h2", good)],
        _roles("h1", "h2"))

    key = Assembled.key(ImageRole.FRONT, "bottom_right", "corners", "rounding")
    assert assembled.detectability_flat[key] == Scale.HIGH.label
    assert key not in assembled.reason_codes_flat, (
        "a better photograph resolved it; it is no longer a limitation")


def test_the_key_round_trips_through_its_flat_form():
    key = Assembled.key(ImageRole.BACK, "top_left", "corners", "whitening")
    assert Assembled._unkey(key) == (
        ImageRole.BACK, "top_left", "corners", "whitening")


def test_a_glared_corner_stops_coverage_reaching_sufficient():
    """The end of the chain: an unassessable region is missing evidence, and
    missing evidence must not pass."""
    from card_reviewer.review.policies.coverage_v1 import (
        REQUIRED_FACES, evaluate_coverage,
    )

    def coverage(reason):
        detectability = {}
        reasons = {}
        for face in REQUIRED_FACES:
            for (region, category, defect_type), value in _full().items():
                detectability[(face, region, category, defect_type)] = value
            key = (face, "bottom_right", "corners", "rounding")
            detectability[key] = Scale.LOW
            reasons[key] = reason
        return evaluate_coverage(detectability, reasons, {}, REQUIRED_FACES)

    assert coverage("GLARE").outcome is not Coverage.SUFFICIENT
    assert any("GLARE" == lim.reason_code for lim in coverage("GLARE").limitations)


def test_a_structurally_invisible_region_still_waives_rather_than_blocks():
    """A white border hides whitening on every corner and no photograph
    changes that. PASS has to stay reachable for most modern base cards."""
    from card_reviewer.review.policies.coverage_v1 import (
        REQUIRED_FACES, evaluate_coverage,
    )

    detectability = {}
    reasons = {}
    for face in REQUIRED_FACES:
        for (region, category, defect_type), value in _full().items():
            detectability[(face, region, category, defect_type)] = value
            if defect_type == "whitening" and category in ("corners", "edges"):
                detectability[(face, region, category, defect_type)] = Scale.LOW
                reasons[(face, region, category, defect_type)] = "WHITE_BORDER"

    assert evaluate_coverage(detectability, reasons, {},
                             REQUIRED_FACES).outcome is Coverage.SUFFICIENT
