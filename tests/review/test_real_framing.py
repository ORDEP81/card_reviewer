"""Detection has to work on how real listing photographs are actually framed.

Every threshold in the imaging layer was calibrated on synthetic fixtures
where the card occupies 59% of the frame on a uniform backdrop. The first 28
real listing images put the card at 94-99.8% of the frame, and the flood fill
found almost no background at all — so 24 of 26 raw cards were refused by
geometry before anything was measured.

Two independent causes, and both are fixed here:

  BACKGROUND_TOLERANCE was a fixed +/-6 levels. A synthetic backdrop is
  uniform, so 6 is generous; a real wall, granite worktop or shadow varies by
  far more, so the flood stopped almost immediately and called the whole
  frame card. Raising it globally is not the answer — at 12 it swallows the
  synthetic dark-bordered card, whose backdrop (10) and border (20) are only
  ten levels apart. The tolerance has to come from the image.

  MAX_AREA_RATIO then rejected the result. That guard is right when the flood
  has failed, but a card really can fill its frame — a seller cropping to the
  card is ordinary — and then the frame IS the card.
"""

import numpy as np
import pytest

from card_reviewer.review.imaging import geometry
from card_reviewer.review.imaging.geometry import analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _decode(data):
    import cv2

    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _textured_backdrop(width, height, base, spread, seed=0):
    """A backdrop that varies the way a real surface does."""
    rng = np.random.default_rng(seed)
    field = rng.normal(base, spread, (height, width, 3))
    blur = np.linspace(-spread, spread, width)[None, :, None]
    return np.clip(field + blur, 0, 255).astype(np.uint8)


def _card_on(backdrop, margin):
    """The synthetic card inset into a backdrop by `margin` of the card."""
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card

    card = _draw_card(CardSpec(), np.random.default_rng(0))
    ch, cw = card.shape[:2]
    mx, my = int(cw * margin), int(ch * margin)
    canvas = cv2.resize(backdrop, (cw + 2 * mx, ch + 2 * my))
    canvas[my:my + ch, mx:mx + cw] = card
    return cv2.imencode(".png", canvas)[1].tobytes()


# --- the tolerance must come from the image -------------------------------

def test_a_uniform_backdrop_keeps_the_tight_tolerance():
    """The synthetic case, and the reason a global raise is wrong: with
    backdrop 10 and border 20 only ten levels apart, a loose tolerance floods
    straight across the card's edge and swallows it."""
    image = _decode(render_png(CardSpec(border_color=(20, 20, 20))))
    assert geometry._background_tolerance(image) <= 10


def test_a_varied_backdrop_gets_a_tolerance_that_can_cross_it():
    """A real wall, worktop or shadow. Measured on the 28 real photographs:
    a tolerance near 6 finds essentially no background at all."""
    backdrop = _textured_backdrop(400, 560, base=120, spread=18)
    image = _decode(_card_on(backdrop, margin=0.08))
    assert geometry._background_tolerance(image) > 20


def test_the_tolerance_is_bounded_at_both_ends():
    """Unbounded, a chaotic background would flood the card away."""
    chaos = _textured_backdrop(400, 560, base=128, spread=90)
    image = _decode(_card_on(chaos, margin=0.08))
    tolerance = geometry._background_tolerance(image)
    assert geometry.MIN_BACKGROUND_TOLERANCE <= tolerance
    assert tolerance <= geometry.MAX_BACKGROUND_TOLERANCE


@pytest.mark.parametrize("spread", [8, 15, 25])
def test_a_card_on_a_varied_backdrop_is_detected(spread, store):
    """The case that failed on every real photograph."""
    data = _card_on(_textured_backdrop(400, 560, 120, spread), margin=0.08)
    result = analyze(data, store, store.put_image(data))
    assert result.usable, (
        f"backdrop spread {spread}: declined at "
        f"confidence {result.boundary_confidence:.3f}")


# --- a card really can fill its frame -------------------------------------

def test_a_card_filling_its_frame_is_detected_as_the_frame(store):
    """A seller cropping to the card is ordinary. The boundary is not in the
    picture, so the frame IS the card — refusing to measure at all loses the
    listing."""
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card

    card = _draw_card(CardSpec(), np.random.default_rng(0))
    data = cv2.imencode(".png", card)[1].tobytes()
    result = analyze(data, store, store.put_image(data))

    assert result.usable
    quad = np.asarray(result.quad, dtype=float)
    height, width = card.shape[:2]
    covered = ((quad[:, 0].max() - quad[:, 0].min())
               * (quad[:, 1].max() - quad[:, 1].min())) / (width * height)
    assert covered > 0.9, "the frame-filling card was read as a sub-panel"


def test_a_frame_filling_image_that_is_not_card_shaped_is_declined(store):
    """The guard that keeps the above honest: a photograph of a wall is not
    a card just because nothing else is in it."""
    import cv2

    wall = _textured_backdrop(600, 600, base=140, spread=10)   # square
    data = cv2.imencode(".png", wall)[1].tobytes()
    assert not analyze(data, store, store.put_image(data)).usable


# --- nothing that already worked may stop working -------------------------

@pytest.mark.parametrize("spec", [
    CardSpec(),
    CardSpec(border_color=(20, 20, 20)),
    CardSpec(border_color=(150, 150, 150)),
    CardSpec(borderless=True),
    CardSpec(rotation_deg=10.0),
    CardSpec(h_centering=72.0, v_centering=58.0),
])
def test_the_synthetic_corpus_is_unaffected(spec, store):
    data = render_png(spec)
    result = analyze(data, store, store.put_image(data))
    assert result.usable
    assert result.boundary_confidence > 0.75


# --- what actually separates a card from a not-card -----------------------

def test_aspect_is_what_rejects_a_non_card_not_rectangularity():
    """Measured on the real corpus, rectangularity CANNOT tell them apart.

    Real cards run 0.611 to 1.000, and a cross-shaped blob sits at 0.63 —
    inside that range. A floor high enough to exclude the cross excludes
    genuine cards, and one of those had a bounding aspect of 0.714, which is
    a textbook card. Aspect ratio is the signal that separates them, so it
    carries the card-likeness gate and rectangularity is left to catch only
    detections that are degenerate rather than merely ragged.
    """
    import cv2

    from card_reviewer.review.imaging.geometry import (
        ASPECT_TOLERANCE, CARD_ASPECT, _aspect, _foreground_mask,
    )

    def bounding_aspect(img):
        mask = _foreground_mask(img, cv2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        box = cv2.minAreaRect(max(contours, key=cv2.contourArea))[1]
        return _aspect(box[0], box[1])

    cross = np.full((900, 700, 3), 10, np.uint8)
    side, arm = 700, 700 // 3
    top, mid = (900 - side) // 2, (side - arm) // 2
    cv2.rectangle(cross, (0, top + mid), (side - 1, top + mid + arm),
                  (240, 240, 240), -1)
    cv2.rectangle(cross, (mid, top), (mid + arm, top + side - 1),
                  (240, 240, 240), -1)

    square = np.full((840, 600, 3), 10, np.uint8)
    square[250:550, 150:450] = 160

    for name, img in (("cross", cross), ("square", square)):
        assert abs(bounding_aspect(img) - CARD_ASPECT) > ASPECT_TOLERANCE, (
            f"{name} is not rejected by aspect, so nothing rejects it")


def test_a_card_shaped_region_with_a_ragged_mask_is_accepted(store):
    """Real photographs give ragged flood masks — shadows, sleeve edges,
    a gradient backdrop. Raggedness is not evidence that the thing is not a
    card, and treating it as such rejected a region whose aspect was 0.714.
    """
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card

    rng = np.random.default_rng(3)
    card = _draw_card(CardSpec(), rng)
    ch, cw = card.shape[:2]
    canvas = _textured_backdrop(cw + 120, ch + 120, base=110, spread=14)
    canvas[60:60 + ch, 60:60 + cw] = card
    # Nibble the edges the way a soft shadow does.
    for _ in range(40):
        y = int(rng.integers(60, 60 + ch)); x0 = 60 + int(rng.integers(-6, 6))
        canvas[y, 60:x0 + 6] = canvas[y, 0]

    data = cv2.imencode(".png", canvas)[1].tobytes()
    assert analyze(data, store, store.put_image(data)).usable


def test_a_card_shaped_region_in_a_non_card_frame_stays_on_the_region(store):
    """The aspect check inside the frame-filling branch, which needs a case
    the earlier gate does not already catch.

    A CARD-SHAPED region is detected, so _detect_quad's own aspect gate is
    satisfied and we reach the branch. The region's band is ragged and the
    frame's is clean, so the ambiguity check fires. The only thing left
    saying "do not treat the frame as the card" is that the FRAME is not
    card-shaped — the premise of the whole reading is "a card fills this
    photograph", and it does not.
    """
    import cv2

    width, height = 600, 706                 # frame aspect 0.850
    inner_w, inner_h = 480, 672              # region aspect 0.714, 76% of it
    panel = np.full((height, width, 3), 238, np.uint8)
    x0, y0 = (width - inner_w) // 2, (height - inner_h) // 2
    panel[y0:y0 + inner_h, x0:x0 + inner_w] = _textured_backdrop(
        inner_w, inner_h, base=115, spread=45, seed=7)

    data = cv2.imencode(".png", panel)[1].tobytes()
    result = analyze(data, store, store.put_image(data))
    assert result.usable, "fixture no longer reaches the branch under test"

    quad = np.asarray(result.quad, dtype=float)
    covered = ((quad[:, 0].max() - quad[:, 0].min())
               * (quad[:, 1].max() - quad[:, 1].min())) / (width * height)
    assert covered < 0.9, (
        "the whole frame was read as the card, though the frame is not "
        "card-shaped")
