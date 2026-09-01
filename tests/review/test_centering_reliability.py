"""A centering number the border cannot support must not be reported.

`centering:border_ratio` is the only MEASUREMENT-promotion defect type, so it
is the one path on which OpenCV alone reaches OBSERVED and can drive a REJECT.
Every fabricated number here is a candidate lost for a defect it does not
have, which is precisely what I1 exists to prevent.
"""

import numpy as np
import pytest

from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
from card_reviewer.review.imaging.measure.centering import measure_centering
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _measure(spec, store):
    data = render_png(spec)
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    return geometry, measure_centering(geometry, store)


def _honest(measurement, truth, tolerance=6.0):
    """Either measure it about right, or decline. Never a confident lie."""
    if not measurement.measurable:
        return True
    return abs(measurement.horizontal - truth) <= tolerance


@pytest.mark.parametrize("tilt", [0.0, 5.0, 10.0, 20.0])
def test_a_tilted_photo_of_a_centred_card_never_reports_a_miscut(tilt, store):
    """A 10-degree tilt is an ordinary phone photograph. Warp resampling
    raises the border's own column variance; the measurement must notice
    rather than pin the border width at the trim."""
    _, measurement = _measure(CardSpec(rotation_deg=tilt), store)
    assert _honest(measurement, 50.0), (
        f"tilt {tilt}deg reported {measurement.horizontal} on a card that is "
        f"50/50 by construction")


def test_a_grey_bordered_centred_card_never_reports_a_miscut(store):
    """A silver or grey border is not a quiet band relative to a bright
    interior, so the fixed 25%-of-peak threshold misplaces the art edge."""
    _, measurement = _measure(
        CardSpec(border_color=(150, 150, 150)), store)
    assert _honest(measurement, 50.0)


def test_a_noisy_border_declines_rather_than_pinning_at_the_trim(store):
    """The failure mode: every interior column clears the threshold, the ink
    band starts at the trim, and the leading border measures as the trim
    width instead of its true width."""
    rng = np.random.default_rng(0)
    _, measurement = _measure(
        CardSpec(border_color=(120, 120, 120), seed=int(rng.integers(1 << 30))),
        store)
    if measurement.measurable:
        assert 44.0 <= measurement.horizontal <= 56.0


def test_a_genuine_miscut_is_still_measured(store):
    """The guard must not buy safety by declining to measure anything."""
    _, measurement = _measure(CardSpec(h_centering=75.0), store)
    assert measurement.measurable, "a clean 75/25 miscut must still measure"
    assert measurement.horizontal > 60.0


def test_a_centred_card_measures_about_fifty(store):
    _, measurement = _measure(CardSpec(), store)
    assert measurement.measurable
    assert 44.0 <= measurement.horizontal <= 56.0


def test_declining_says_why(store):
    """A declined measurement carries a reason code, so coverage can classify
    it and the report can explain it."""
    _, measurement = _measure(CardSpec(borderless=True), store)
    if not measurement.measurable:
        assert measurement.reason


def test_a_border_thinner_than_the_trim_declines_on_either_edge():
    """The trim is a tolerance for an approximate boundary, not a border.

    Both edges need their own check: a card can run its art into the leading
    trim while leaving a wide trailing border, and vice versa. Reporting the
    trim's width as the card's border is how an extreme miscut came out as a
    plausible-looking ratio.
    """
    from card_reviewer.review.imaging.measure.centering import _ratio

    size = 200
    trim = max(1, int(size * 0.02))

    leading_pinned = np.zeros(size)
    leading_pinned[trim:150] = 40.0        # art runs into the leading trim
    assert _ratio(leading_pinned) == (None, "BORDER_NOT_SEPARABLE_FROM_ART")

    trailing_pinned = np.zeros(size)
    trailing_pinned[50 : size - trim] = 40.0   # and into the trailing one
    assert _ratio(trailing_pinned) == (None, "BORDER_NOT_SEPARABLE_FROM_ART")

    both_borders_real = np.zeros(size)
    both_borders_real[50:150] = 40.0
    value, reason = _ratio(both_borders_real)
    assert reason is None and value == pytest.approx(50.0, abs=1.0)
