"""Glare hides more than whitening, and it hides it on white cards too.

A white border makes WHITENING structurally invisible — no photograph of that
card can show it. It does not make a rounded or frayed corner visible through
a blown-out highlight. Reporting HIGH detectability there is the system
asserting "highly detectable here, and none observed" about pixels that are
pure white, which is precisely the claim CLAUDE.md says must never be
manufactured.
"""

import pytest

from card_reviewer.review.enums import Scale
from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
from card_reviewer.review.imaging.observability import analyze as observability_analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.taxonomy import UndetectabilityClass, class_of


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _observe(spec, store):
    data = render_png(spec)
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    return observability_analyze(geometry, store, image_hash)


GLARED_CORNER = ["top_left"]


def test_glare_hides_corner_damage_on_a_white_bordered_card(store):
    result = _observe(CardSpec(glare_regions=GLARED_CORNER), store)
    key = ("top_left", "corners", "rounding")
    assert result.detectability[key] < Scale.MODERATE, (
        "a blown-out corner cannot show rounding, whatever colour the border")
    assert result.reason_codes[key] == "GLARE"


def test_glare_on_a_white_border_is_circumstantial_not_structural(store):
    """A better photograph fixes glare; nothing fixes a white border."""
    result = _observe(CardSpec(glare_regions=GLARED_CORNER), store)
    code = result.reason_codes[("top_left", "corners", "rounding")]
    assert class_of(code) is UndetectabilityClass.CIRCUMSTANTIAL


def test_a_white_border_still_hides_whitening_structurally(store):
    """The exemption this guard exists for must survive the fix."""
    result = _observe(CardSpec(), store)
    key = ("top_left", "corners", "whitening")
    assert result.detectability[key] < Scale.MODERATE
    assert result.reason_codes[key] == "WHITE_BORDER"
    assert class_of(result.reason_codes[key]) is UndetectabilityClass.STRUCTURAL


def test_an_unglared_corner_on_the_same_card_stays_detectable(store):
    """The shortfall belongs to the glared region, not the whole card."""
    result = _observe(CardSpec(glare_regions=GLARED_CORNER), store)
    assert result.detectability[("bottom_right", "corners", "rounding")] >= (
        Scale.MODERATE)


def test_glare_hides_corner_damage_on_a_dark_bordered_card_too(store):
    """This half already worked; it must keep working."""
    result = _observe(
        CardSpec(border_color=(20, 20, 20), glare_regions=GLARED_CORNER), store)
    key = ("top_left", "corners", "rounding")
    assert result.detectability[key] < Scale.MODERATE
    assert result.reason_codes[key] == "GLARE"


def test_a_card_blown_out_at_every_corner_is_still_glared(store):
    """The relative test is blind here — when every region is equally blown
    out, nothing stands out from the median. The absolute arm is what stops
    a uniformly flashed dark card reading as fully assessable."""
    result = _observe(
        CardSpec(border_color=(20, 20, 20),
                 glare_regions=["top_left", "top_right",
                                "bottom_left", "bottom_right"]),
        store)
    for region in ("top_left", "bottom_right"):
        key = (region, "corners", "rounding")
        assert result.detectability[key] < Scale.MODERATE, (
            f"{region} reads assessable on a card blown out everywhere")
        assert result.reason_codes[key] == "GLARE"
