import json

import pytest

from card_reviewer.review.enums import Scale, UndetectabilityClass
from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.observability import ObservabilityResult, analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


def _obs(spec, store):
    data = render_png(spec)
    return analyze(geom(data, store, "h1"), store, "h1")


def test_detectability_is_reported_per_region_and_per_defect_type(store):
    r = _obs(CardSpec(), store)
    assert ("bottom_left", "corners", "whitening") in r.detectability
    assert ("bottom_left", "corners", "rounding") in r.detectability


def test_a_white_corner_cannot_show_whitening_and_says_so_structurally(store):
    """CORNERS_COLORED_001 as physics: the code is WHITE_BORDER and its class
    is structural, so coverage waives it rather than demanding a photograph
    that could never help."""
    r = _obs(CardSpec(border_color=(255, 255, 255)), store)
    key = ("bottom_left", "corners", "whitening")
    assert r.detectability[key] < Scale.MODERATE
    assert r.reason_codes[key] == "WHITE_BORDER"
    assert r.reason_class(key) is UndetectabilityClass.STRUCTURAL


def test_the_same_white_corner_is_still_assessable_for_rounding(store):
    """This is what keeps PASS reachable for white-bordered cards."""
    r = _obs(CardSpec(border_color=(255, 255, 255)), store)
    assert r.detectability[("bottom_left", "corners", "rounding")] >= Scale.MODERATE


def test_a_dark_border_gives_high_whitening_detectability(store):
    r = _obs(CardSpec(border_color=(20, 20, 20)), store)
    assert r.detectability[("bottom_left", "corners", "whitening")] >= Scale.MODERATE


def test_glare_is_circumstantial_not_structural(store):
    r = _obs(CardSpec(border_color=(20, 20, 20), glare_regions=["top_left"],
                      seed=2), store)
    key = ("top_left", "surface", "scratches")
    assert r.reason_codes.get(key) == "GLARE"
    assert r.reason_class(key) is UndetectabilityClass.CIRCUMSTANTIAL


def test_a_photo_can_be_good_for_centering_and_useless_for_surface(store):
    """A glare spot must not condemn a whole image."""
    r = _obs(CardSpec(border_color=(20, 20, 20), glare_regions=["top_left"],
                      seed=2), store)
    assert Scale(r.suitability["centering"]) >= Scale.MODERATE
    assert Scale(r.suitability["surface"]) < Scale(r.suitability["centering"])


def test_corner_detectability_is_not_diluted_by_the_card_centre(store):
    """If the centre contributed to `corners`, a white-bordered card would
    look HIGH for corner whitening after assembly takes the max across
    regions, and the structural exemption would never fire end to end."""
    r = _obs(CardSpec(border_color=(255, 255, 255)), store)
    assert ("center", "corners", "whitening") not in r.detectability
    assert all(v < Scale.MODERATE
               for (_region, c, d), v in r.detectability.items()
               if c == "corners" and d == "whitening")


def test_every_shortfall_below_moderate_carries_a_declared_reason_code(store):
    r = _obs(CardSpec(border_color=(255, 255, 255)), store)
    for key, value in r.detectability.items():
        if value < Scale.MODERATE:
            assert key in r.reason_codes
            assert r.reason_class(key) is not None


def test_unusable_geometry_yields_no_detectability_anywhere(store):
    """Absent evidence must never read as adequate evidence."""
    from card_reviewer.review.imaging.geometry import GeometryResult

    r = analyze(GeometryResult(boundary_confidence=0.1), store, "h1")
    assert all(v is Scale.NONE for v in r.detectability.values())


def test_the_output_revives_from_json_with_identical_detectability(store):
    """Cache round trip: tuple keys must survive JSON or the image tier is
    not cacheable at all."""
    fresh = _obs(CardSpec(), store)
    revived = ObservabilityResult.model_validate(json.loads(fresh.model_dump_json()))
    assert revived.detectability == fresh.detectability
    assert revived.reason_codes == fresh.reason_codes


def test_glare_and_occlusion_masks_are_stored_as_artifacts(store):
    """Masks are pixel data, so they are referenced rather than embedded."""
    r = _obs(CardSpec(glare_regions=["top_left"], seed=2), store)
    assert r.glare_mask_artifact_id and store.read(r.glare_mask_artifact_id)
    assert r.occlusion_mask_artifact_id


def test_the_output_canonicalizes_for_a_fingerprint(store):
    from card_reviewer.review.canonical import canonicalize

    assert canonicalize(_obs(CardSpec(), store).model_dump(mode="json"))


def test_the_serialized_output_stays_small_because_masks_are_referenced(store):
    r = _obs(CardSpec(), store)
    assert len(r.model_dump_json()) < 20000


def test_a_white_border_is_structural_while_glare_on_a_dark_card_is_not(store):
    """The distinction the whole coverage policy rests on.

    Both are simply bright pixels. What separates them is whether the
    brightness belongs to the card (white border, everywhere by design) or
    to the photograph (a highlight in one spot). Getting it backwards would
    ask the owner for a better photograph of something no photograph can
    change.
    """
    white = _obs(CardSpec(border_color=(255, 255, 255)), store)
    glared = _obs(CardSpec(border_color=(20, 20, 20), glare_regions=["top_left"],
                           seed=2), store)
    assert white.reason_class(("bottom_left", "corners", "whitening")) is (
        UndetectabilityClass.STRUCTURAL)
    assert glared.reason_class(("top_left", "surface", "scratches")) is (
        UndetectabilityClass.CIRCUMSTANTIAL)


def test_a_uniformly_white_card_does_not_become_a_photo_request(store):
    """A white-bordered card is bright everywhere; treating that as glare
    would generate a photo request no photograph could satisfy.

    The fixture used to include glare_regions=["top_left"], which is not
    what this rationale describes — a localized highlight is exactly what a
    better photograph fixes. Asserting no GLARE there pinned the defect that
    a flashed corner on a white card was reported as fully detectable.
    """
    r = _obs(CardSpec(border_color=(255, 255, 255), seed=2), store)
    assert all(code != "GLARE" for code in r.reason_codes.values())


def test_a_flashed_corner_on_a_white_card_does_become_a_photo_request(store):
    """The other half: this one a better photograph can fix."""
    r = _obs(CardSpec(border_color=(255, 255, 255), glare_regions=["top_left"],
                      seed=2), store)
    assert r.reason_codes[("top_left", "corners", "rounding")] == "GLARE"


def test_a_borderless_card_reports_centering_as_structurally_undetectable(store):
    """No border reference means no ratio to measure — a property of the
    card's design, not of the photograph, so coverage waives it rather than
    asking for a better picture."""
    r = _obs(CardSpec(borderless=True), store)
    key = ("center", "centering", "border_ratio")
    assert r.detectability[key] < Scale.MODERATE
    assert r.reason_codes[key] == "BORDERLESS_DESIGN"
    assert r.reason_class(key) is UndetectabilityClass.STRUCTURAL


def test_a_bordered_card_reports_centering_as_detectable(store):
    r = _obs(CardSpec(border_color=(255, 255, 255)), store)
    assert r.detectability[("center", "centering", "border_ratio")] >= Scale.MODERATE
