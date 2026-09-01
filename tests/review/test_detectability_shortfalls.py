"""Detectability must reflect every reason a region could not be seen.

The producer could only ever emit WHITE_BORDER, GLARE and BORDERLESS_DESIGN;
everything else fell through to `Scale.HIGH`. The occlusion mask was computed
and then discarded, and OCCLUSION, BLUR, LOW_RESOLUTION and SEVERE_PERSPECTIVE
were declared in the taxonomy and mapped in PHOTO_REQUESTS but unreachable
from the pipeline.

So a card shot in a top-loader with a thumb over one corner, or at the edge
of focus, reported HIGH detectability everywhere: SUFFICIENT coverage, PASS
available, and I1's adequacy prong satisfied for any finding placed there.
Spec §7.3 asks for per-region occlusion masks, perspective severity and
per-region effective resolution — the mask was already being built.
"""

import numpy as np
import pytest

from card_reviewer.review.enums import Scale
from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
from card_reviewer.review.imaging.observability import analyze as observability_analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.taxonomy import REASON_CODES, class_of


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _observe(data, store):
    image_hash = store.put_image(data)
    return observability_analyze(
        geometry_analyze(data, store, image_hash), store, image_hash)


def _with_occlusion(spec, corner="top_left", size=260):
    """A thumb over one corner OF THE CARD.

    Applied to the card before it is placed on its background: painting the
    frame's corner instead lands on the backdrop, which the rectified card
    never contains.
    """
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card, _place_on_background

    rng = np.random.default_rng(spec.seed)
    card = _draw_card(spec, rng)
    height, width = card.shape[:2]
    ys = slice(0, size) if "top" in corner else slice(height - size, height)
    xs = slice(0, size) if "left" in corner else slice(width - size, width)
    card[ys, xs] = 4
    return cv2.imencode(".png", _place_on_background(card, spec, cv2))[1].tobytes()


def _soft_corner(spec, corner="top_left", size=260, ksize=31):
    """One corner out of focus on an otherwise sharp card.

    Whole-image softness is preflight's job and it already rejects it
    (a 31-pixel blur takes global sharpness to 0.7 against a floor of 25).
    What preflight cannot see is a card that is sharp overall and soft in one
    place, which is what a shallow depth of field on an angled card gives.
    """
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card, _place_on_background

    card = _draw_card(spec, np.random.default_rng(spec.seed))
    height, width = card.shape[:2]
    ys = slice(0, size) if "top" in corner else slice(height - size, height)
    xs = slice(0, size) if "left" in corner else slice(width - size, width)
    card[ys, xs] = cv2.GaussianBlur(card[ys, xs], (ksize, ksize), 0)
    return cv2.imencode(".png", _place_on_background(card, spec, cv2))[1].tobytes()


def test_an_occluded_corner_is_not_reported_as_assessable(store):
    result = _observe(_with_occlusion(CardSpec()), store)
    key = ("top_left", "corners", "rounding")
    assert result.detectability[key] < Scale.MODERATE, (
        "a corner under an obstruction read as fully assessable")
    assert result.reason_codes[key] == "OCCLUSION"


def test_an_unobstructed_corner_on_the_same_card_stays_assessable(store):
    result = _observe(_with_occlusion(CardSpec()), store)
    assert result.detectability[
        ("bottom_right", "corners", "rounding")] >= Scale.MODERATE


def test_a_soft_corner_is_deliberately_not_claimed_to_be_detectable(store):
    """There is no region-level blur test, and this records why rather than
    leaving its absence to look like an oversight.

    Two measures were calibrated over 8 seeds x 3 border colours and both
    fail the same way — a region can be smooth BY DESIGN, and neither
    separates that from out of focus:

        Laplacian variance vs siblings
            clean minimum ratio 0.137, genuinely soft corner 0.489
        Laplacian normalised by the region's own contrast
            clean minimum 0.0475, genuinely soft corner up to 0.0937

    In the first the clean card looks SOFTER than the blurred one. Any
    threshold either accuses clean cards or never fires, so none is shipped:
    a detector that cannot detect is worse than an acknowledged gap, because
    it reads as coverage.
    """
    result = _observe(_soft_corner(CardSpec()), store)
    assert "BLUR" not in result.reason_codes.values()

    # And say what the consequence IS, rather than only what it is not: the
    # region reports HIGH with no reason code. That is the "highly
    # detectable here, and none observed" claim CLAUDE.md forbids, and it is
    # the actual cost of having no working measure — recorded so the gap is
    # legible rather than implied by an absence.
    key = ("top_left", "corners", "rounding")
    assert result.detectability[key] is Scale.HIGH
    assert key not in result.reason_codes


def test_the_sharp_corners_of_that_card_stay_assessable(store):
    result = _observe(_soft_corner(CardSpec()), store)
    assert result.detectability[
        ("bottom_right", "corners", "rounding")] >= Scale.MODERATE


def test_whole_image_softness_is_rejected_at_preflight_not_here(store):
    """Documents the division of labour the region test relies on."""
    import cv2

    from card_reviewer.review.imaging.preflight import analyze as preflight

    img = cv2.imdecode(np.frombuffer(render_png(CardSpec()), np.uint8),
                       cv2.IMREAD_COLOR)
    data = cv2.imencode(".png", cv2.GaussianBlur(img, (31, 31), 0))[1].tobytes()
    result = preflight(data)
    assert result.usable is False
    assert result.reason_code == "BLUR"


def test_every_shortfall_reason_is_declared_and_classifiable(store):
    for data in (_with_occlusion(CardSpec()), _soft_corner(CardSpec()),
                 render_png(CardSpec())):
        for code in _observe(data, store).reason_codes.values():
            assert code in REASON_CODES, f"{code} is not declared"
            class_of(code)


def test_a_clean_sharp_card_reports_no_new_shortfalls(store):
    """The guards must not fire on the cards they are not about."""
    result = _observe(render_png(CardSpec(border_color=(20, 20, 20))), store)
    assert not any(
        code in {"OCCLUSION", "BLUR", "LOW_RESOLUTION", "SEVERE_PERSPECTIVE"}
        for code in result.reason_codes.values())


def _small_in_frame(spec, scale=0.30):
    """A card photographed small: real, common, and previously invisible.

    The rectified card is always the same size, so a region's normalized
    dimensions say nothing about the detail actually captured. Only the
    original pixels behind it do.
    """
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card

    card = _draw_card(spec, np.random.default_rng(spec.seed))
    height, width = card.shape[:2]
    small = cv2.resize(card, (int(width * scale), int(height * scale)),
                       interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 10, np.uint8)
    y0 = (height - small.shape[0]) // 2
    x0 = (width - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return cv2.imencode(".png", canvas)[1].tobytes()


def test_a_card_photographed_too_small_reports_low_resolution(store):
    result = _observe(_small_in_frame(CardSpec(), scale=0.12), store)
    if not result.detectability:
        pytest.skip("geometry declined this framing; nothing to assess")
    assert any(code == "LOW_RESOLUTION"
               for code in result.reason_codes.values()), (
        "a card captured in a handful of pixels claimed full detectability")


def test_a_card_filling_the_frame_is_not_flagged_low_resolution(store):
    result = _observe(_small_in_frame(CardSpec(), scale=1.0), store)
    assert not any(code == "LOW_RESOLUTION"
                   for code in result.reason_codes.values())


def test_effective_resolution_falls_as_the_card_shrinks_in_frame(store):
    """The measure itself, independent of where the threshold sits."""
    from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
    from card_reviewer.review.imaging.observability import _capture_scale

    scales = []
    for fraction in (1.0, 0.5, 0.25):
        data = _small_in_frame(CardSpec(), scale=fraction)
        image_hash = store.put_image(data)
        geometry = geometry_analyze(data, store, image_hash)
        if not geometry.usable:
            continue
        scales.append(_capture_scale(geometry, (840, 600)))

    assert len(scales) >= 2
    assert scales == sorted(scales, reverse=True), (
        f"effective resolution did not fall with card size: {scales}")


def test_a_partly_obstructed_corner_is_flagged_too(store):
    """A thumb rarely covers a whole corner crop.

    The main fixture covers the crop entirely, so any threshold below 1.0
    fires on it and the threshold's value goes untested. This one leaves most
    of the corner visible — measured at 0.12 of the crop against a 0.10
    threshold — so it pins where the line actually sits.
    """
    result = _observe(_with_occlusion(CardSpec(), size=140), store)
    key = ("top_left", "corners", "rounding")
    assert result.detectability[key] < Scale.MODERATE
    assert result.reason_codes[key] == "OCCLUSION"


def test_a_speck_is_not_an_obstruction(store):
    """The other side of that line: dark print, a dust mote or a small dark
    design element must not cost the region its assessability."""
    result = _observe(_with_occlusion(CardSpec(), size=60), store)
    assert result.detectability[
        ("top_left", "corners", "rounding")] >= Scale.MODERATE
