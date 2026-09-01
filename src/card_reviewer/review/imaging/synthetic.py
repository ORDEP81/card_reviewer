"""Synthetic trading cards with known ground truth.

Real photographs have no ground truth — nobody can say a listing photo is
"exactly 54/46". This generator is the oracle: it renders a card whose
centering, border colour, damage and distortion are known by construction,
so every measurement test asserts against a value rather than a guess.

It is a piece of software in its own right, with its own tests, and not a
subtask of the CV work.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "CardSpec",
    "achieved_centering",
    "card_bounds",
    "card_region",
    "render",
    "render_png",
]

#: Every render sits on a margin of background so the card has a findable
#: boundary. A render that fills the frame gives geometry nothing to detect,
#: and the CV tests would be measuring the printed art instead of the card.
BACKGROUND_MARGIN = 0.15

_CORNERS = {
    "top_left": (0, 0), "top_right": (0, 1),
    "bottom_left": (1, 0), "bottom_right": (1, 1),
}


class CardSpec(BaseModel):
    card_w: int = 600
    card_h: int = 840
    border_px: int = 40
    #: The leading border's share of total border width, as a percentage.
    h_centering: float = Field(default=50.0, ge=1.0, le=99.0)
    v_centering: float = Field(default=50.0, ge=1.0, le=99.0)
    border_color: tuple[int, int, int] = (255, 255, 255)
    art_color: tuple[int, int, int] = (40, 90, 160)
    borderless: bool = False
    #: Renders a dense grid of small dark blocks instead of one large art
    #: panel, so a card BACK can be simulated with known ground truth.
    text_heavy: bool = False
    corner_damage: dict[str, float] = Field(default_factory=dict)
    rotation_deg: float = 0.0
    perspective: float = 0.0
    glare_regions: list[str] = Field(default_factory=list)
    background: tuple[int, int, int] = (10, 10, 10)
    seed: int = 0


def achieved_centering(spec: CardSpec) -> tuple[float, float]:
    """The centering actually rendered, as (horizontal, vertical).

    Borders land on integer pixels, so a requested 62.0 may render as 62.5.
    The rendered value is the ground truth a measurement should be compared
    against — comparing against the request would charge the measurement for
    the generator's rounding.
    """
    art_w = spec.card_w - 2 * spec.border_px
    art_h = spec.card_h - 2 * spec.border_px
    slack_w, slack_h = spec.card_w - art_w, spec.card_h - art_h
    left = int(round(slack_w * spec.h_centering / 100.0))
    top = int(round(slack_h * spec.v_centering / 100.0))
    return (
        100.0 * left / slack_w if slack_w else 50.0,
        100.0 * top / slack_h if slack_h else 50.0,
    )


def card_bounds(spec: CardSpec) -> tuple[int, int, int, int]:
    """Where the card sits inside the rendered frame: (top, left, h, w).

    Part of the ground truth this generator promises. Without it a caller
    cannot tell card pixels from background, and every assertion about the
    card would silently be an assertion about the backdrop.

    Only meaningful for an undistorted render; rotation and perspective move
    the card off the axis.
    """
    pad_y = int(spec.card_h * BACKGROUND_MARGIN)
    pad_x = int(spec.card_w * BACKGROUND_MARGIN)
    return pad_y, pad_x, spec.card_h, spec.card_w


def card_region(spec: CardSpec, image: np.ndarray) -> np.ndarray:
    top, left, h, w = card_bounds(spec)
    return image[top:top + h, left:left + w]


def render(spec: CardSpec) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(spec.seed)
    card = _draw_card(spec, rng)
    return _place_on_background(card, spec, cv2)


def _draw_card(spec: CardSpec, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((spec.card_h, spec.card_w, 3), np.uint8)
    img[:] = spec.border_color

    if spec.borderless:
        # Edge-to-edge artwork, not a uniform panel: a real borderless card
        # has picture content running to the trim, which is precisely why no
        # reliable border reference exists. Rendering it as flat colour plus
        # fine noise would produce a band uniform enough to measure against,
        # making the generator disagree with the thing it is modelling.
        block = 60
        for y in range(0, spec.card_h, block):
            for x in range(0, spec.card_w, block):
                img[y:y + block, x:x + block] = rng.integers(
                    0, 255, 3, dtype=np.uint8
                )
    else:
        art_w = spec.card_w - 2 * spec.border_px
        art_h = spec.card_h - 2 * spec.border_px
        left = int(round((spec.card_w - art_w) * spec.h_centering / 100.0))
        top = int(round((spec.card_h - art_h) * spec.v_centering / 100.0))
        img[top:top + art_h, left:left + art_w] = spec.art_color

    if spec.text_heavy:
        step = 24
        for y in range(step, spec.card_h - step, step):
            for x in range(step, spec.card_w - step, step):
                img[y:y + 8, x:x + 16] = (30, 30, 30)

    for name, severity in spec.corner_damage.items():
        row, col = _CORNERS[name]
        size = max(1, int(40 * severity))
        ys = slice(0, size) if row == 0 else slice(spec.card_h - size, spec.card_h)
        xs = slice(0, size) if col == 0 else slice(spec.card_w - size, spec.card_w)
        img[ys, xs] = rng.integers(180, 255, (size, size, 3), dtype=np.uint8)

    for region in spec.glare_regions:
        row, col = _CORNERS[region]
        ys = slice(0, 120) if row == 0 else slice(spec.card_h - 120, spec.card_h)
        xs = slice(0, 120) if col == 0 else slice(spec.card_w - 120, spec.card_w)
        # Real glare is a specular highlight: it saturates rather than
        # merely brightening. A +90 offset leaves a dark border at 110,
        # which no clipping test would ever call glare.
        img[ys, xs] = np.clip(
            img[ys, xs].astype(np.int16) * 0.05 + 245, 0, 255
        ).astype(np.uint8)

    return img


def _place_on_background(card: np.ndarray, spec: CardSpec, cv2) -> np.ndarray:
    """Always leave a background margin, then apply any distortion."""
    h, w = card.shape[:2]
    pad_y, pad_x = int(h * BACKGROUND_MARGIN), int(w * BACKGROUND_MARGIN)
    canvas = np.zeros((h + 2 * pad_y, w + 2 * pad_x, 3), np.uint8)
    canvas[:] = spec.background
    canvas[pad_y:pad_y + h, pad_x:pad_x + w] = card

    if not spec.rotation_deg and not spec.perspective:
        return canvas

    src = np.float32([[pad_x, pad_y], [pad_x + w, pad_y],
                      [pad_x + w, pad_y + h], [pad_x, pad_y + h]])
    shift = spec.perspective * w
    dst = np.float32([[pad_x + shift, pad_y], [pad_x + w, pad_y + shift],
                      [pad_x + w - shift, pad_y + h], [pad_x, pad_y + h - shift]])
    out = cv2.warpPerspective(
        canvas, cv2.getPerspectiveTransform(src, dst),
        (canvas.shape[1], canvas.shape[0]), borderValue=spec.background,
    )
    if spec.rotation_deg:
        centre = (canvas.shape[1] / 2, canvas.shape[0] / 2)
        matrix = cv2.getRotationMatrix2D(centre, spec.rotation_deg, 1.0)
        out = cv2.warpAffine(out, matrix, (canvas.shape[1], canvas.shape[0]),
                             borderValue=spec.background)
    return out


def render_png(spec: CardSpec) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", render(spec))
    if not ok:
        raise RuntimeError("failed to encode synthetic card as PNG")
    return buf.tobytes()
