import numpy as np
import pytest

from card_reviewer.review.imaging.synthetic import (
    CardSpec,
    card_region,
    render,
    render_png,
)


def _border_widths(spec, img):
    """Locate the printed-art rectangle by column variance, within the card.

    Both edges are measured against the CARD's width — measuring the right
    border against the framed image would silently include the background
    margin.
    """
    region = card_region(spec, img)
    col_var = region.std(axis=(0, 2))
    inked = np.where(col_var > col_var.max() * 0.25)[0]
    return int(inked[0]), int(region.shape[1] - inked[-1] - 1)


def test_a_render_produces_a_three_channel_image():
    assert render(CardSpec()).shape[2] == 3


def test_the_card_sits_inside_a_background_margin():
    """Geometry detects the card against its surroundings, so the render
    must never fill the frame — otherwise the only findable boundary is the
    printed art and every measurement is of the wrong rectangle."""
    spec = CardSpec()
    img = render(spec)
    assert img.shape[0] > spec.card_h and img.shape[1] > spec.card_w
    assert img[2, 2].mean() < 30  # background, not card


@pytest.mark.parametrize("ratio", [50.0, 55.0, 60.0, 70.0])
def test_requested_centering_is_reproduced_within_a_pixel(ratio):
    """Ground truth by construction: the generator's geometry is the oracle
    every centering test measures against."""
    spec = CardSpec(h_centering=ratio, card_w=600, card_h=840, border_px=40)
    left, right = _border_widths(spec, render(spec))
    assert abs(100.0 * left / (left + right) - ratio) <= 1.0


def test_white_and_dark_borders_are_both_producible():
    white = CardSpec(border_color=(255, 255, 255))
    dark = CardSpec(border_color=(20, 20, 20))
    assert card_region(white, render(white))[5, 5].mean() > 200
    assert card_region(dark, render(dark))[5, 5].mean() < 60


def test_a_borderless_design_has_no_uniform_border_band():
    spec = CardSpec(borderless=True)
    top = card_region(spec, render(spec))[2:6, :, :].reshape(-1, 3)
    assert top.std(axis=0).mean() > 5.0


def test_a_text_heavy_layout_differs_from_a_plain_front():
    """Backs are simulated this way so role features have ground truth."""
    assert not np.array_equal(render(CardSpec()), render(CardSpec(text_heavy=True)))


def test_corner_damage_appears_only_where_requested():
    spec = CardSpec()
    clean = card_region(spec, render(spec))
    damaged_spec = CardSpec(corner_damage={"bottom_left": 0.8})
    damaged = card_region(damaged_spec, render(damaged_spec))
    assert not np.array_equal(clean, damaged)
    assert np.array_equal(clean[:60, -60:], damaged[:60, -60:])


def test_rotation_and_perspective_are_reproducible_for_a_seed():
    a = render(CardSpec(rotation_deg=7.0, perspective=0.15, seed=42))
    b = render(CardSpec(rotation_deg=7.0, perspective=0.15, seed=42))
    assert np.array_equal(a, b)


def test_glare_brightens_only_the_requested_region():
    spec = CardSpec(border_color=(20, 20, 20), glare_regions=["top_left"], seed=1)
    img = card_region(spec, render(spec))
    assert img[:80, :80].mean() > img[-80:, -80:].mean()


def test_render_png_produces_decodable_bytes():
    import cv2

    data = render_png(CardSpec())
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape[2] == 3


def test_a_distorted_card_sits_on_a_background_so_its_boundary_is_findable():
    """Geometry detects the card against its surroundings; a render that
    fills the frame has no boundary to find."""
    img = render(CardSpec(rotation_deg=6.0, perspective=0.1, seed=3))
    assert img.shape[0] > 840 and img.shape[1] > 600
