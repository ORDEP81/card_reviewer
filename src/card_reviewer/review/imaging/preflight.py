"""Raw-image properties requiring no geometry (spec §7.1).

Marking an image unusable never contributes toward a REJECT verdict. It
reduces coverage, which routes toward REVIEW or INSUFFICIENT_IMAGES —
missing evidence removes evidence, it never creates any.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..versions import PREFLIGHT_VERSION

__all__ = ["PREFLIGHT_VERSION", "PreflightResult", "analyze"]

MIN_WIDTH = 400
MIN_HEIGHT = 400
MIN_SHARPNESS = 25.0
MAX_CLIPPED_FRACTION = 0.6


class PreflightResult(BaseModel):
    usable: bool
    width: int = 0
    height: int = 0
    global_sharpness: float = 0.0
    clipped_fraction: float = 0.0
    #: A declared taxonomy reason code, so coverage can classify it.
    reason_code: str | None = None
    version: str = PREFLIGHT_VERSION


def analyze(image_bytes: bytes) -> PreflightResult:
    import cv2

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # Reported, never raised: one corrupt file must not fail the card.
        return PreflightResult(usable=False, reason_code="DECODE_FAILED")

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Sign-aware. Counting both ends together reported an UNDEREXPOSED
    # photograph as GLARE, whose photo request reads "avoid direct flash" —
    # advice that makes a too-dark picture worse. They are opposite problems
    # with opposite remedies.
    blown = float((gray >= 250).mean())
    crushed = float((gray <= 5).mean())

    # Clipping is tested BEFORE sharpness. A blown-out image has no
    # Laplacian variance *because* it is blown out, so checking sharpness
    # first would report BLUR and ask the owner for a sharper photograph
    # when what they need is diffuse lighting. The reason code becomes a
    # photo request downstream, so naming the cause rather than the symptom
    # is what makes that request useful.
    reason: str | None = None
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        reason = "LOW_RESOLUTION"
    elif blown > MAX_CLIPPED_FRACTION:
        reason = "GLARE"
    elif crushed > MAX_CLIPPED_FRACTION:
        reason = "UNDEREXPOSED"
    elif sharpness < MIN_SHARPNESS:
        reason = "BLUR"

    return PreflightResult(
        usable=reason is None, width=width, height=height,
        global_sharpness=sharpness, clipped_fraction=blown + crushed, reason_code=reason,
    )
