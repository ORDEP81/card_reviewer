"""Photo advice must point the right way.

Clipping was counted with a sign-blind test — `(gray >= 250) | (gray <= 5)` —
so an UNDEREXPOSED photograph came back as GLARE, whose photo request reads
"a diffuse-lit photograph (avoid direct flash)". The owner of a too-dark
picture needs MORE light, and following that advice makes the photo worse.
"""

import numpy as np
import pytest

from card_reviewer.review.imaging.preflight import analyze
from card_reviewer.review.taxonomy import REASON_CODES, class_of


def _png(fill, textured=False):
    """A uniform fill has no Laplacian variance and legitimately reads as
    BLUR, so anything testing the exposure path needs texture on top."""
    import cv2

    img = np.full((900, 700, 3), fill, np.uint8)
    if textured:
        grid = np.indices((900, 700))
        checker = ((grid[0] // 20) + (grid[1] // 20)) % 2
        img = np.where(checker[:, :, None] == 1,
                       np.clip(img.astype(int) + 60, 0, 255), img).astype(np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def test_a_blown_out_photograph_is_reported_as_glare():
    assert analyze(_png(255)).reason_code == "GLARE"


def test_an_underexposed_photograph_is_not_reported_as_glare():
    result = analyze(_png(0))
    assert result.usable is False
    # Named, not merely "not GLARE": falling through to BLUR would also
    # satisfy that, and would ask for a sharper photograph of something the
    # owner cannot sharpen.
    assert result.reason_code == "UNDEREXPOSED"


def test_the_underexposure_code_is_declared_and_circumstantial():
    code = analyze(_png(0)).reason_code
    assert code in REASON_CODES
    class_of(code)


def test_the_advice_for_an_underexposed_photograph_asks_for_more_light():
    from card_reviewer.review.policies.coverage_v1 import PHOTO_REQUESTS

    code = analyze(_png(0)).reason_code
    template = PHOTO_REQUESTS.get(code)
    assert template, f"{code} generates no photo request at all"
    text = template.lower() if isinstance(template, str) else str(template).lower()
    assert "flash" not in text or "avoid" not in text


def test_a_normally_exposed_photograph_is_usable():
    assert analyze(_png(128, textured=True)).usable is True
