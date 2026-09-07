import cv2
import numpy as np

from card_reviewer.review.imaging.preflight import PreflightResult, analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render, render_png


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_a_normal_card_photo_is_usable():
    assert analyze(render_png(CardSpec())).usable is True


def test_a_thumbnail_is_unusable_and_says_why():
    r = analyze(_png(np.zeros((150, 200, 3), np.uint8)))
    assert r.usable is False and r.reason_code == "LOW_RESOLUTION"


def test_marking_an_image_unusable_is_never_a_reject_verdict():
    """An unusable image reduces coverage; it never condemns the card."""
    assert not hasattr(analyze(_png(np.zeros((150, 200, 3), np.uint8))), "verdict")


def test_a_blurred_image_reports_lower_sharpness_than_a_sharp_one():
    blurred = cv2.GaussianBlur(render(CardSpec()), (31, 31), 0)
    assert analyze(_png(blurred)).global_sharpness < (
        analyze(render_png(CardSpec())).global_sharpness)


def test_a_heavily_blurred_image_is_unusable_with_blur():
    blurred = cv2.GaussianBlur(render(CardSpec()), (99, 99), 0)
    r = analyze(_png(blurred))
    assert r.usable is False and r.reason_code == "BLUR"


def test_a_blown_out_image_reports_clipping():
    assert analyze(_png(np.full((840, 600, 3), 254, np.uint8))).clipped_fraction > 0.9


def test_a_blown_out_image_is_unusable_with_glare():
    assert analyze(_png(np.full((840, 600, 3), 254, np.uint8))).reason_code == "GLARE"


def test_corrupt_bytes_are_reported_not_raised():
    r = analyze(b"not an image")
    assert r.usable is False and r.reason_code == "DECODE_FAILED"


def test_every_reason_code_it_emits_is_declared_in_the_taxonomy():
    """A code coverage cannot classify would crash the policy downstream."""
    from card_reviewer.review.taxonomy import REASON_CODES

    emitted = {"LOW_RESOLUTION", "BLUR", "GLARE"}
    assert emitted <= set(REASON_CODES)


def test_dimensions_are_reported_for_a_usable_image():
    r = analyze(render_png(CardSpec()))
    assert r.width > 0 and r.height > 0


def test_the_result_round_trips_through_json():
    import json

    r = analyze(render_png(CardSpec()))
    assert PreflightResult.model_validate(json.loads(r.model_dump_json())) == r
