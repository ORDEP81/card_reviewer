"""Detection must not return a confident quad for the wrong rectangle.

Independent review found that on a tightly cropped listing photograph — the
most common marketplace framing — the flood fill seeds land ON the card's own
border, the border is claimed as background, and the surviving contour is the
PRINTED ART PANEL. Confidence came back 1.000 and usable True, so every
downstream measurement described the wrong object:

    a true 90/10 miscut, with a background margin -> centering 85.71
    the same card cropped to its own edges        -> centering 50.00

A silently wrong quad is worse than no quad at all: 50.00 reads as a
perfectly centred card.
"""

import numpy as np
import pytest

from card_reviewer.review.imaging.geometry import analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _framed(spec, margin_fraction, backdrop=(10, 10, 10)):
    """The card with exactly `margin_fraction` of backdrop around it.

    `render` already places the card on a backdrop of its own, so building a
    genuinely tight crop means cropping in to the card's own bounds first and
    then adding back only the margin under test. Skipping that step tests the
    renderer's framing rather than the seller's.
    """
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card

    rng = np.random.default_rng(spec.seed)
    card = _draw_card(spec, rng)
    height, width = card.shape[:2]
    mx, my = int(width * margin_fraction), int(height * margin_fraction)
    if mx == 0 and my == 0:
        return cv2.imencode(".png", card)[1].tobytes()
    canvas = np.full((height + 2 * my, width + 2 * mx, 3), backdrop, np.uint8)
    canvas[my : my + height, mx : mx + width] = card
    return cv2.imencode(".png", canvas)[1].tobytes()


def _detect(data, store):
    return analyze(data, store, store.put_image(data))


def _covers_the_card(result, data, tolerance=0.06):
    """Does the detected quad describe the whole card, not a panel of it?"""
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    quad = np.asarray(result.quad, dtype=float)
    detected = ((quad[:, 0].max() - quad[:, 0].min())
                * (quad[:, 1].max() - quad[:, 1].min()))
    return detected >= (1.0 - tolerance) ** 2 * height * width


@pytest.mark.parametrize("margin", [0.0, 0.005, 0.01, 0.02])
def test_a_tightly_cropped_card_never_yields_a_border_it_cannot_trust(
        margin, store):
    """Detect the whole card, or withhold the border reference. What must
    never happen is a confident sub-panel WITH a border to measure against,
    because centering would then describe the artwork."""
    data = _framed(CardSpec(), margin)
    result = _detect(data, store)

    if result.usable and result.has_reliable_border:
        assert _covers_the_card(result, data), (
            f"margin {margin}: a sub-panel came with a reliable border at "
            f"confidence {result.boundary_confidence:.3f}")


def test_a_card_cropped_to_its_edges_produces_no_centering_number(store):
    """The consequence that matters. Centering is the only measurement that
    can reject a card on its own, so it must not describe a rectangle we
    cannot confirm is the card."""
    from card_reviewer.review.imaging.measure.centering import measure_centering

    result = _detect(_framed(CardSpec(), 0.0), store)
    if not _covers_the_card(result, _framed(CardSpec(), 0.0)):
        assert measure_centering(result, store).measurable is False


def test_a_cropped_miscut_is_never_reported_as_well_centred(store):
    """An automatic PSA-10 disqualifier must not be turned into its
    opposite by the framing: 50.00 on a 90/10 card is the worst outcome
    available, worse than declining."""
    from card_reviewer.review.imaging.measure.centering import measure_centering

    data = _framed(CardSpec(h_centering=90.0), 0.0)
    result = _detect(data, store)
    measurement = measure_centering(result, store)

    if measurement.measurable:
        assert measurement.horizontal > 60.0, (
            f"a 90/10 miscut measured {measurement.horizontal} once cropped")


def test_a_card_with_a_normal_margin_is_still_detected_precisely(store):
    """The fix must not cost the case that already worked."""
    data = _framed(CardSpec(), 0.03)
    result = _detect(data, store)
    assert result.usable
    assert result.boundary_confidence > 0.5
    assert result.has_reliable_border
