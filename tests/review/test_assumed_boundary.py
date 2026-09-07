"""A boundary we assumed is not a boundary we saw.

When the flood finds no background, geometry falls back to reading the frame
as the card. That is often nearly right — the card really does fill the
photograph — but the last few pixels at each edge are a strip of backdrop or
holder that the flood could not separate.

Corners and edges measure a DEPARTURE FROM THE CARD'S BORDER. Handed a quad
whose outer pixels are not card, they measure that strip and report it as
damage. Verified on the real corpus: 15 of 88 accepted photographs are
whole-frame accepts, and four of them are cards a human labelled CLEAN that
emitted severe corner and edge anomalies — measured off the photograph's own
edges.

Centering already declines here, which is why this never produced a false
REJECT. The border-relative producers had no equivalent guard.
"""

import numpy as np
import pytest

from card_reviewer.review.imaging.geometry import analyze
from card_reviewer.review.imaging.measure import measure_all
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _card_with_a_sliver(strip=8):
    """A card filling its frame, with a thin dark strip the flood cannot
    separate — exactly what the failing corpus photographs look like."""
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card

    card = _draw_card(CardSpec(), np.random.default_rng(0))
    height, width = card.shape[:2]
    canvas = np.full((height + 2 * strip, width + 2 * strip, 3), 40, np.uint8)
    canvas[strip:strip + height, strip:strip + width] = card
    return cv2.imencode(".png", canvas)[1].tobytes()


def test_a_real_boundary_is_reported_as_observed(store):
    data = render_png(CardSpec())
    result = analyze(data, store, store.put_image(data))
    assert result.usable
    assert result.boundary_observed is True


def test_a_frame_fallback_is_reported_as_assumed(store):
    data = _card_with_a_sliver()
    result = analyze(data, store, store.put_image(data))
    if not result.usable:
        pytest.skip("fixture no longer reaches the frame fallback")
    assert result.boundary_observed is False, (
        "the frame was taken as the card, but that is an assumption rather "
        "than an observation and must say so")


def test_an_assumed_boundary_emits_no_border_relative_findings(store):
    """The fix. Corners and edges measure departure from the card's border,
    so with a boundary we only assumed, the outermost pixels may not be card
    at all — and that is precisely where they measure."""
    data = _card_with_a_sliver()
    result = analyze(data, store, store.put_image(data))
    if not result.usable or result.boundary_observed:
        pytest.skip("fixture no longer reaches the frame fallback")

    measured = measure_all(result, store, store.put_image(data))
    border_relative = [a for a in measured.anomalies
                       if a["category"] in ("corners", "edges")]
    assert border_relative == [], (
        f"{len(border_relative)} border-relative findings from a boundary "
        "that was never observed: "
        f"{[(a['category'], a.get('region')) for a in border_relative]}")


def test_an_assumed_boundary_still_yields_crops_for_the_vision_layer(store):
    """Declining to CONCLUDE is not declining to look. The crops are what the
    vision layer inspects, and they stay."""
    data = _card_with_a_sliver()
    result = analyze(data, store, store.put_image(data))
    if not result.usable or result.boundary_observed:
        pytest.skip("fixture no longer reaches the frame fallback")

    measured = measure_all(result, store, store.put_image(data))
    assert measured.corners.crops, "no corner crops for the vision layer"
    assert measured.edges.crops, "no edge crops for the vision layer"


def test_an_observed_boundary_still_finds_real_damage(store):
    """The guard must not buy safety by silencing the working case."""
    data = render_png(CardSpec(border_color=(20, 20, 20),
                               corner_damage={"bottom_left": 0.9}))
    result = analyze(data, store, store.put_image(data))
    assert result.boundary_observed is True

    measured = measure_all(result, store, store.put_image(data))
    assert any(a["category"] == "corners" and a.get("region") == "bottom_left"
               for a in measured.anomalies)


def test_the_confidence_floor_is_where_it_is_declared():
    """0.55 could be dropped to 0.30 with nothing failing. Measured on the
    real corpus, genuine cards run 0.562-1.000, so a floor much below that
    stops refusing anything at all."""
    from card_reviewer.review.imaging.geometry import MIN_BOUNDARY_CONFIDENCE

    assert 0.5 <= MIN_BOUNDARY_CONFIDENCE <= 0.65


def test_a_shapeless_detection_is_still_refused(store):
    """What the floor is for, now that aspect carries card-likeness: a
    region whose bounding box is card-shaped but which is not a rectangle.
    Rectangularity is the only thing standing between this and acceptance.
    """
    import cv2

    # A T-shape whose BOUNDING BOX is 5:7 — passes aspect, is not a card.
    canvas = np.full((900, 640, 3), 15, np.uint8)
    cv2.rectangle(canvas, (70, 120), (570, 300), (225, 225, 225), -1)
    cv2.rectangle(canvas, (250, 120), (390, 820), (225, 225, 225), -1)

    data = cv2.imencode(".png", canvas)[1].tobytes()
    result = analyze(data, store, store.put_image(data))
    assert not result.usable, (
        f"a T-shape was accepted as a card at confidence "
        f"{result.boundary_confidence:.3f}")


def test_the_artwork_frame_fill_also_reports_an_assumed_boundary(store):
    """There are TWO paths that read the frame as the card: the flood
    finding no background, and the ambiguity check deciding the detected
    region was the artwork inside a frame-filling card. Both are assumptions
    about a boundary that is not in the picture, and both must say so — the
    second was reported as observed."""
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card

    # A bare card: the flood claims its white border as background and the
    # artwork panel survives, which is the ambiguity path rather than the
    # no-background path.
    card = _draw_card(CardSpec(), np.random.default_rng(0))
    data = cv2.imencode(".png", card)[1].tobytes()
    result = analyze(data, store, store.put_image(data))
    if not result.usable:
        pytest.skip("fixture no longer reaches the ambiguity path")

    quad = np.asarray(result.quad, dtype=float)
    height, width = card.shape[:2]
    covered = ((quad[:, 0].max() - quad[:, 0].min())
               * (quad[:, 1].max() - quad[:, 1].min())) / (width * height)
    if covered < 0.9:
        pytest.skip("fixture did not take the frame")

    assert result.boundary_observed is False
    measured = measure_all(result, store, store.put_image(data))
    assert not [a for a in measured.anomalies
                if a["category"] in ("corners", "edges")]
