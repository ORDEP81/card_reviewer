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
MIN_BOUNDARY_CONFIDENCE = 0.5
#: A photographed card always sits against something. A contour covering
#: essentially the whole frame is the frame, not a card — random noise
#: produces exactly that, and without this guard it scored full confidence.
MAX_AREA_RATIO = 0.92
BORDER_BAND_PX = 24
#: A border band this uniform can serve as a centering reference.
RELIABLE_BORDER_STD = 30.0


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

    Confidence combines how much of the frame the card occupies with how
    rectangular the region is, so a ragged blob scores low even when large.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

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
    confidence = float(min(1.0, max(0.0, area_ratio * 1.6)) * rectangularity)

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
    return _order(corners), confidence


def _quad_from_contour(contour: np.ndarray, cv2) -> np.ndarray | None:
    """The contour's own four corners, if it approximates cleanly to them."""
    perimeter = cv2.arcLength(contour, True)
    for epsilon in (0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _order(pts: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left."""
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(total)], pts[np.argmin(diff)],
         pts[np.argmax(total)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


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
    return mask, bool(gray[mask > 0].std() < RELIABLE_BORDER_STD)
