"""The rules that decide when a detected boundary cannot be trusted.

Exercised directly, because the synthetic corpus cannot produce every
combination end to end — and a guard the corpus never reaches is a guard
nothing is holding.
"""

import numpy as np
import pytest

from card_reviewer.review.imaging import geometry


@pytest.fixture
def cv2():
    import cv2 as module

    return module


def _card_like(width, height, border_value, inner_value, noise=0.0, rng=None):
    """A frame whose outer band is `border_value` and centre `inner_value`."""
    img = np.full((height, width, 3), float(inner_value))
    band = max(4, int(min(width, height) * 0.10))
    img[:band, :] = border_value
    img[-band:, :] = border_value
    img[:, :band] = border_value
    img[:, -band:] = border_value
    if noise:
        rng = rng or np.random.default_rng(0)
        img += rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _quad(x0, y0, x1, y1):
    return geometry._order(
        np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))


def test_a_quad_that_is_already_the_frame_is_not_second_guessed(cv2):
    """With nothing discarded there is no rival reading, and re-deciding
    would only risk withholding a border that is plainly there."""
    img = _card_like(600, 840, border_value=255, inner_value=60)
    whole = _quad(0, 0, 599, 839)
    assert geometry._boundary_may_be_the_artwork(img, whole, cv2) is False


def test_a_frame_with_no_border_of_its_own_is_no_rival_reading(cv2):
    """A frame band that is itself ragged is not a border, however much
    worse the inner reading looks. Preferring it would trade one unusable
    reference for another while discarding the region we actually detected.
    """
    rng = np.random.default_rng(1)
    # A frame band built from two values, so its robust spread is set
    # deliberately: MAD is d, and the reported spread is 1.4826 * d. d = 22
    # puts it just past RELIABLE_BORDER_STD.
    img = np.clip(rng.normal(128, 70.0, (840, 600, 3)), 0, 255).astype(np.uint8)
    # ...and an inner region far more varied still, at a scale the warp's
    # resampling preserves. Pixel noise will not do: downsampling averages it
    # away, so the inner band caps out below the frame's.
    grid = np.indices((680, 480))
    checker = ((grid[0] // 60) + (grid[1] // 60)) % 2
    img[80:760, 60:540] = np.where(checker[:, :, None] == 1, 255, 0)
    inner = _quad(60, 80, 539, 759)

    dst = np.float32([[0, 0], [geometry.NORM_W, 0],
                      [geometry.NORM_W, geometry.NORM_H], [0, geometry.NORM_H]])
    spreads = []
    for quad in (inner, _quad(0, 0, 599, 839)):
        view = cv2.warpPerspective(
            img, cv2.getPerspectiveTransform(quad.astype(np.float32), dst),
            (geometry.NORM_W, geometry.NORM_H))
        mask, _ = geometry._segment_border(view)
        spreads.append(geometry._band_spread(view.mean(axis=2), mask))
    inner_spread, whole_spread = spreads

    assert whole_spread >= geometry.RELIABLE_BORDER_STD, (
        "fixture no longer gives the frame an unusable band")
    assert whole_spread * geometry.BORDER_UNIFORMITY_MARGIN < inner_spread, (
        "fixture no longer makes the ratio favour the frame")
    assert geometry._boundary_may_be_the_artwork(img, inner, cv2) is False


def test_a_quad_already_filling_the_frame_is_not_flagged(cv2):
    """A card that legitimately fills its photograph keeps its border.

    The early return for this case is a performance short-circuit rather
    than the guard: with the quad and the frame sampling almost the same
    pixels the comparison reaches the same answer unaided, which mutation
    testing confirms. This pins the OUTCOME, which is what matters, and the
    short-circuit is commented as such at the source.
    """
    rng = np.random.default_rng(4)
    img = np.full((840, 600, 3), 250.0)
    # Artwork right up to a hair inside the frame: the inner reading's band
    # is artwork and the frame's is clean, so the comparison WOULD fire.
    img[8:832, 6:594] = np.clip(rng.normal(120, 50.0, (824, 588, 3)), 0, 255)
    inner = _quad(4, 4, 595, 835)

    assert geometry._quad_area(inner) / (600 * 840) >= (
        geometry.FILLS_FRAME_AREA_RATIO), "fixture no longer fills the frame"
    assert geometry._boundary_may_be_the_artwork(img, inner, cv2) is False


def test_a_markedly_cleaner_frame_makes_the_inner_reading_ambiguous(cv2):
    """The cropped-card case: the frame's band is a uniform border while the
    region we detected is bounded by artwork."""
    rng = np.random.default_rng(2)
    img = np.full((840, 600, 3), 255.0)
    img[80:760, 60:540] = np.clip(
        rng.normal(110, 45.0, (680, 480, 3)), 0, 255)
    inner = _quad(60, 80, 539, 759)
    assert geometry._boundary_may_be_the_artwork(
        img, inner, cv2) is True


def test_a_frame_only_slightly_cleaner_is_not_treated_as_ambiguous(cv2):
    """The margin is a conservatism buffer: withholding a border costs the
    only measurement that can reject a card, so a near-tie must not do it."""
    rng = np.random.default_rng(3)
    img = _card_like(600, 840, border_value=200, inner_value=200,
                     noise=6.0, rng=rng)
    inner = _quad(40, 40, 559, 799)

    spreads = []
    dst = np.float32([[0, 0], [geometry.NORM_W, 0],
                      [geometry.NORM_W, geometry.NORM_H], [0, geometry.NORM_H]])
    for quad in (inner, _quad(0, 0, 599, 839)):
        view = cv2.warpPerspective(
            img, cv2.getPerspectiveTransform(quad.astype(np.float32), dst),
            (geometry.NORM_W, geometry.NORM_H))
        mask, _ = geometry._segment_border(view)
        spreads.append(geometry._band_spread(view.mean(axis=2), mask))

    inner_spread, whole_spread = spreads
    assert whole_spread < inner_spread * geometry.BORDER_UNIFORMITY_MARGIN, (
        "fixture no longer represents a near-tie")
    assert geometry._boundary_may_be_the_artwork(img, inner, cv2) is False


def test_confidence_measures_certainty_not_size(cv2):
    """`min(1, area_ratio * 1.6) * rectangularity` mixed two different
    questions, and the asymmetry ran exactly the wrong way: the detector was
    CONFIDENT where it was wrong (a cropped photo's artwork panel scored
    1.000) and UNCONFIDENT where it was right, discarding a perfectly
    detected card below about 31% of the frame.

    How big the card is in frame is a resolution fact, and it belongs to
    detectability, where it becomes a LOW_RESOLUTION limitation and a photo
    request. It is not evidence about whether we found the right rectangle.
    """
    from card_reviewer.review.imaging.synthetic import CardSpec, _draw_card

    rng = np.random.default_rng(0)
    card = _draw_card(CardSpec(), rng)

    confidences = []
    for fraction in (0.9, 0.5, 0.25):
        height, width = card.shape[:2]
        small = cv2.resize(card, (int(width * fraction), int(height * fraction)),
                           interpolation=cv2.INTER_AREA)
        canvas = np.full((height, width, 3), 10, np.uint8)
        y0 = (height - small.shape[0]) // 2
        x0 = (width - small.shape[1]) // 2
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        quad, confidence = geometry._detect_quad(canvas, cv2)
        assert quad is not None, f"a clean card at {fraction:.0%} was not found"
        confidences.append(confidence)

    assert all(c > 0.5 for c in confidences), (
        f"a crisply detected card was declined for being small: {confidences}")
