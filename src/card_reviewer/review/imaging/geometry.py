"""Boundary detection, perspective correction, normalization, border
segmentation (spec §7.2).

Establishes the ONE normalized card coordinate system every later stage,
defect location and future model output refers to.

Border segmentation belongs here rather than downstream: both
`observability` (is this corner white, and therefore unable to show
whitening?) and `cv_measurements` (is there a reliable border reference to
measure centering against?) need it, and neither should own a result the
other depends on.

Cache safety shapes the output model. This stage is stored as JSON in
SQLite, so it cannot carry live NumPy arrays — pixel products go to the
artifact store and the output carries their ids.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from pydantic import BaseModel

from ..storage.artifacts import ArtifactStore
from ..versions import GEOMETRY_VERSION

__all__ = [
    "GEOMETRY_VERSION",
    "GeometryArtifacts",
    "GeometryResult",
    "analyze",
    "load_geometry",
]

NORM_W, NORM_H = 600, 840
#: Confidence is now rectangularity alone, so the floor is a statement about
#: SHAPE. Measured: a real card comes back at 0.999 whatever fraction of the
#: frame it occupies, while a cross-shaped blob reaches only 0.63. 0.75 sits
#: in that gap with room on both sides.
MIN_BOUNDARY_CONFIDENCE = 0.75
#: A photographed card always sits against something. A contour covering
#: essentially the whole frame is the frame, not a card — random noise
#: produces exactly that, and without this guard it scored full confidence.
MAX_AREA_RATIO = 0.92

#: How close in intensity a pixel must be to the backdrop to count as more
#: backdrop. Small, because a card only slightly darker than its surroundings
#: is still a distinct surface.
BACKGROUND_TOLERANCE = 6
BORDER_BAND_PX = 24
#: A border band this uniform can serve as a centering reference. Measured
#: robustly (see _segment_border) so one glared corner does not disqualify
#: the whole reference.
RELIABLE_BORDER_STD = 30.0

#: Past this share of the frame the detected quad already IS the frame, so
#: there is no second reading to consider.
FILLS_FRAME_AREA_RATIO = 0.92

#: How much more uniform the full frame's band must be than the inner
#: reading's before we conclude the inner reading was the artwork. A factor
#: rather than a difference, since the two readings are being ranked against
#: each other rather than against an absolute idea of "uniform".
BORDER_UNIFORMITY_MARGIN = 3.0

#: A trading card is 2.5 x 3.5 inches.
CARD_ASPECT = 2.5 / 3.5

#: How far a detected region's aspect ratio may sit from a card's before it
#: stops being credible as a card. This is the card-likeness signal that
#: replaced area_ratio in the confidence formula — dropping area removed the
#: only check on WHAT was detected, and nothing took its place, so a plain
#: grey rectangle scored 1.000 and PASSed at grade 10 with no findings.
#: Measured: real cards span 0.714-0.791 including tilt and perspective,
#: while a 200x120 rectangle reads 0.602 and a square 1.000.
ASPECT_TOLERANCE = 0.10


class GeometryResult(BaseModel):
    """Cache-safe: scalars and artifact ids only."""

    boundary_confidence: float
    quad: list[list[float]] | None = None
    transform: list[list[float]] | None = None
    normalized_artifact_id: str | None = None
    border_mask_artifact_id: str | None = None
    has_reliable_border: bool = False
    version: str = GEOMETRY_VERSION

    @property
    def usable(self) -> bool:
        return self.normalized_artifact_id is not None


class GeometryArtifacts(NamedTuple):
    """The resolved pixel view, never persisted."""

    result: GeometryResult
    normalized: np.ndarray | None
    border_mask: np.ndarray | None


def load_geometry(result: GeometryResult, store: ArtifactStore) -> GeometryArtifacts:
    """Resolve a cached geometry result back to pixels.

    Downstream CV stages call this rather than receiving arrays directly, so
    a cache hit and a fresh computation are indistinguishable to them.
    """
    import cv2

    def _read(artifact_id: str | None, flags: int) -> np.ndarray | None:
        if artifact_id is None:
            return None
        return cv2.imdecode(np.frombuffer(store.read(artifact_id), np.uint8), flags)

    return GeometryArtifacts(
        result=result,
        normalized=_read(result.normalized_artifact_id, cv2.IMREAD_COLOR),
        border_mask=_read(result.border_mask_artifact_id, cv2.IMREAD_GRAYSCALE),
    )


def analyze(
    image_bytes: bytes, store: ArtifactStore, image_hash: str
) -> GeometryResult:
    import cv2

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return GeometryResult(boundary_confidence=0.0)

    quad, confidence = _detect_quad(img, cv2)
    if quad is None or confidence < MIN_BOUNDARY_CONFIDENCE:
        # Decline geometry-dependent work rather than producing plausible
        # numbers from a bad quad.
        return GeometryResult(boundary_confidence=confidence)

    dst = np.float32([[0, 0], [NORM_W, 0], [NORM_W, NORM_H], [0, NORM_H]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    normalized = cv2.warpPerspective(img, matrix, (NORM_W, NORM_H))
    mask, reliable = _segment_border(normalized)
    if reliable and _boundary_may_be_the_artwork(img, quad, cv2):
        # The frame has a markedly cleaner border than the region we
        # detected, so the region may be the artwork inside a cropped card.
        # Centering must not describe a rectangle we are unsure of.
        reliable = False

    # `face/` is geometry's own directory; measurement crops live under
    # corners/, edges/ and surface/ and are invalidated by their own stage.
    return GeometryResult(
        boundary_confidence=confidence,
        quad=quad.tolist(),
        transform=matrix.tolist(),
        normalized_artifact_id=store.put_derived(
            image_hash, "face", "normalized.png",
            cv2.imencode(".png", normalized)[1].tobytes(),
        ),
        border_mask_artifact_id=store.put_derived(
            image_hash, "face", "border_mask.png",
            cv2.imencode(".png", mask)[1].tobytes(),
        ),
        has_reliable_border=reliable,
    )


def _detect_quad(img: np.ndarray, cv2) -> tuple[np.ndarray | None, float]:
    """Find the card as a foreground region against its background.

    Canny alone fails on borderless designs: edge-to-edge artwork produces
    internal edges everywhere, the largest external contour comes out
    irregular, and the card is not detected at all — so a borderless card
    would receive no measurements rather than merely no centering
    reference. Separating foreground from background by intensity is the
    property a photographed card actually has, whatever is printed on it.

    Confidence is how RECTANGULAR the region is, so a ragged blob scores low
    however large it is. It deliberately says nothing about how much of the
    frame the card fills: that is a resolution question, answered by
    observability as effective resolution, and conflating the two made the
    detector confident where it was wrong and unconfident where it was right.
    """
    mask = _foreground_mask(img, cv2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area <= 0:
        return None, 0.0

    area_ratio = area / (img.shape[0] * img.shape[1])
    if area_ratio > MAX_AREA_RATIO:
        return None, 0.1

    rect = cv2.minAreaRect(largest)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area <= 0:
        return None, 0.0
    rectangularity = float(min(1.0, area / rect_area))
    # Certainty about the SHAPE, and about the shape being a CARD's — never
    # about size. Multiplying in area_ratio mixed size with certainty and got
    # the asymmetry backwards: a cropped photo's artwork panel scored 1.000
    # while a crisply detected card below ~31% of the frame was discarded.
    # How large the card sits in frame is a RESOLUTION fact and belongs to
    # detectability. But removing area also removed the only check on WHAT
    # was detected, so aspect ratio carries that: a trading card is 2.5x3.5
    # whatever else varies, and a shape that is not card-shaped is not a
    # card however cleanly it was found.
    box = cv2.minAreaRect(largest)[1]
    if abs(_aspect(box[0], box[1]) - CARD_ASPECT) > ASPECT_TOLERANCE:
        return None, 0.0
    confidence = rectangularity

    # Prefer the contour's own four corners. A photographed card is usually a
    # trapezoid, not a rectangle, and minAreaRect wraps a trapezoid in a
    # bounding box that pulls background wedges into the rectified image —
    # enough to contaminate the border band and cost the card its centering
    # reference on any angled shot.
    corners = _quad_from_contour(largest, cv2)
    if corners is None:
        # Fall back to the bounding rectangle: better an approximate card
        # than no card, which is what a borderless design would otherwise get.
        corners = np.asarray(cv2.boxPoints(rect), dtype=np.float32)
    try:
        return _order(corners), confidence
    except ValueError:
        # Degenerate at this orientation. Declining is the point: the
        # alternative was a finite garbage homography and measurements of a
        # rectangle that was never on the card.
        return None, 0.0


def _boundary_may_be_the_artwork(
    img: np.ndarray, corners: np.ndarray, cv2
) -> bool:
    """Might the region we detected be the artwork rather than the card?

    On a tightly cropped listing photograph the flood-fill seeds land ON the
    card's border, that border is claimed as background, and the surviving
    contour is the PRINTED ART PANEL — returned at confidence 1.000, so every
    later measurement described the wrong rectangle and a 90/10 miscut read
    as perfectly centred.

    On a tightly cropped listing photograph the flood-fill seeds land ON the
    card's border, that border is claimed as background, and the surviving
    contour is the PRINTED ART PANEL — previously returned at confidence
    1.000, so every later measurement described the wrong rectangle and a
    90/10 miscut read as perfectly centred.

    We do not try to CHOOSE between the two readings. Every signal that
    separates them here — which band is more uniform, the frame's aspect
    ratio — also fires on a borderless card sitting on a plain backdrop, and
    picking the frame there hands a borderless design the centering reference
    it definitionally does not have. Manufacturing a border is a worse
    failure than declining to measure one.

    So this reports only that the two readings are not distinguishable, and
    the caller withholds the border reference. Centering — the one
    measurement that can reject a card on its own — then declines rather than
    describing whichever rectangle happened to win.

    The readings are compared on the same measure the rectified card is
    already judged by: how uniform is this reading's outer band?

      - card on a backdrop: the inner reading IS the card, so its band is the
        card's border and is very uniform, while the full frame's band mixes
        backdrop with border and is not.
      - card cropped to its edges: the inner reading is the artwork panel, so
        its band is artwork, while the full frame's band is the card's own
        border.

    The comparison is relative rather than against a fixed threshold, because
    an absolute one cannot separate a smoothly varying artwork edge from a
    border — measured at MAD 14.3 for artwork against a 30.0 threshold, well
    inside "reliable" — whereas a real border sits near zero. Comparing the
    discarded ring against the full frame's outer band instead would be
    circular: they are largely the same pixels.
    """
    height, width = img.shape[:2]
    inner = _quad_area(corners) / float(height * width)
    if inner >= FILLS_FRAME_AREA_RATIO:
        # A short-circuit, NOT a safety gate. When the quad already fills the
        # frame the two readings sample almost the same pixels, so the
        # comparison below returns False on its own — mutation-testing the
        # branch away changes no outcome. It is kept because it saves two
        # warps per image, and labelled so nobody later reads it as the thing
        # protecting this case.
        return False

    dst = np.float32([[0, 0], [NORM_W, 0], [NORM_W, NORM_H], [0, NORM_H]])

    def spread(quad):
        view = cv2.warpPerspective(
            img, cv2.getPerspectiveTransform(quad.astype(np.float32), dst),
            (NORM_W, NORM_H))
        gray = view.mean(axis=2)
        mask, _ = _segment_border(view)
        return _band_spread(gray, mask)

    frame = _order(np.float32([[0, 0], [width - 1, 0],
                               [width - 1, height - 1], [0, height - 1]]))
    whole_spread = spread(frame)
    if whole_spread >= RELIABLE_BORDER_STD:
        # The frame has no usable border either, so it is no rival reading.
        return False
    return whole_spread * BORDER_UNIFORMITY_MARGIN < spread(corners)


def _aspect(width: float, height: float) -> float:
    """Short side over long side, so orientation does not matter."""
    if width <= 0 or height <= 0:
        return 0.0
    return min(width, height) / max(width, height)


def _quad_area(corners: np.ndarray) -> float:
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _ring_between(img: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    """The pixels inside the frame but outside `corners`."""
    import cv2

    height, width = img.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    outside = img.mean(axis=2)[mask == 0]
    return outside if outside.size else None


def _outer_band(normalized: np.ndarray) -> np.ndarray:
    band = max(2, int(min(normalized.shape[:2]) * 0.04))
    gray = normalized.mean(axis=2)
    return np.concatenate([
        gray[:band, :].ravel(), gray[-band:, :].ravel(),
        gray[:, :band].ravel(), gray[:, -band:].ravel(),
    ])


def _quad_from_contour(contour: np.ndarray, cv2) -> np.ndarray | None:
    """The contour's own four corners, if it approximates cleanly to them."""
    perimeter = cv2.arcLength(contour, True)
    for epsilon in (0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _foreground_mask(img: np.ndarray, cv2) -> np.ndarray:
    """Separate card from background by growing the background inward.

    Thresholding by intensity finds whichever boundary happens to be
    strongest, which on a dark card against a dark backdrop is the PRINTED
    ART — the card's own border blends into the background while the artwork
    stands out sharply. Detecting the art and calling it the card silently
    makes every downstream measurement describe the wrong rectangle.

    The property that actually distinguishes them is connectivity: the
    background touches the frame edge and the card does not. Flooding inward
    from the corners therefore claims the backdrop whatever its brightness,
    and whatever is left is the card.
    """
    height, width = img.shape[:2]
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    filled = np.zeros((height + 2, width + 2), np.uint8)

    # Tight tolerance: a backdrop only a little darker than the card is still
    # a different surface, and leaking across that step would swallow the
    # card entirely.
    tolerance = (BACKGROUND_TOLERANCE,) * 3
    for seed in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        # FIXED_RANGE compares each candidate against the SEED, not against
        # its neighbour. With a floating range the fill walks up the blur
        # gradient at the card's edge one step at a time and swallows the
        # card whole.
        cv2.floodFill(
            blurred.copy(), filled, seed, 255, tolerance, tolerance,
            cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8),
        )

    foreground = np.where(filled[1:-1, 1:-1] > 0, 0, 255).astype(np.uint8)
    return cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def _order(pts: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left.

    Raises when the four selections are not four distinct points. Near 45
    degrees `argmin(sum)` and `argmin(diff)` pick the SAME corner, so one is
    emitted twice and another dropped — and `getPerspectiveTransform` accepts
    that silently, returning a finite garbage homography. Every measurement
    downstream then describes a rectangle that was never on the card, which
    is worse than having no quad at all.
    """
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.array(
        [pts[np.argmin(total)], pts[np.argmin(diff)],
         pts[np.argmax(total)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )
    if len({tuple(point) for point in ordered}) != 4:
        raise ValueError(
            "corner ordering did not yield four distinct points; the "
            f"quad {pts.tolist()} is degenerate at this orientation"
        )
    return ordered


def _segment_border(normalized: np.ndarray) -> tuple[np.ndarray, bool]:
    """Mark the outer band, and say whether it can serve as a reference.

    'Reliable' means uniform enough to measure centering against — a
    borderless design's outer band is printed art and varies, so it cannot.
    """
    gray = normalized.mean(axis=2)
    mask = np.zeros(gray.shape, np.uint8)
    mask[:BORDER_BAND_PX, :] = 255
    mask[-BORDER_BAND_PX:, :] = 255
    mask[:, :BORDER_BAND_PX] = 255
    mask[:, -BORDER_BAND_PX:] = 255

    # Robust spread, not the plain standard deviation. A specular highlight
    # on one corner is a small minority of the band but inflates the plain
    # std enough to fail the threshold — and losing the centering reference
    # to a single glare patch is exactly the over-conservatism that costs
    # recall. Median absolute deviation ignores that minority; a genuinely
    # varied borderless band still fails.
    robust_std = _band_spread(gray, mask)
    return mask, bool(robust_std < RELIABLE_BORDER_STD)


def _band_spread(gray: np.ndarray, mask: np.ndarray) -> float:
    values = gray[mask > 0]
    median = np.median(values)
    return 1.4826 * float(np.median(np.abs(values - median)))
