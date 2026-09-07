import pytest

from card_reviewer.review.imaging.geometry import GeometryResult
from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.measure.centering import (
    PRECISION_PP, measure_centering,
)
from card_reviewer.review.imaging.synthetic import (
    CardSpec,
    achieved_centering,
    render_png,
)
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


@pytest.mark.parametrize("requested", [50.0, 55.0, 60.0, 65.0])
def test_measured_centering_lands_within_the_declared_tolerance(requested, store):
    """Compared against what was RENDERED, not what was requested: borders
    land on integer pixels, so charging the measurement for the generator's
    rounding would test the wrong thing."""
    spec = CardSpec(h_centering=requested)
    truth, _ = achieved_centering(spec)
    m = measure_centering(geom(render_png(spec), store, "h"), store)
    assert m.measurable is True
    assert abs(m.horizontal - truth) <= m.precision_pp


def test_the_measurement_declares_its_own_precision(store):
    m = measure_centering(geom(render_png(CardSpec()), store, "h"), store)
    assert m.precision_pp == PRECISION_PP


def test_vertical_centering_is_measured_too(store):
    spec = CardSpec(v_centering=62.0)
    _, truth = achieved_centering(spec)
    m = measure_centering(geom(render_png(spec), store, "h"), store)
    assert abs(m.vertical - truth) <= m.precision_pp


def test_a_borderless_design_reports_not_measurable_with_a_reason(store):
    m = measure_centering(geom(render_png(CardSpec(borderless=True)), store, "h"),
                          store)
    assert m.measurable is False
    assert m.reason == "BORDERLESS_DESIGN"


def test_no_ratio_is_forced_onto_a_borderless_card(store):
    """CENTERING_BORDERLESS_001 binds directly here."""
    m = measure_centering(geom(render_png(CardSpec(borderless=True)), store, "h"),
                          store)
    assert m.horizontal is None and m.vertical is None


def test_the_measurement_carries_no_acceptability_judgment(store):
    """Product leniency is the heuristic layer's decision, never the
    measurement layer's — CENTERING_PRODUCT_LENIENCY_001 does not apply here."""
    m = measure_centering(geom(render_png(CardSpec(h_centering=62.0)), store, "h"),
                          store)
    assert not hasattr(m, "passes")
    assert not hasattr(m, "acceptable")
    assert not hasattr(m, "grade")


def test_an_undetected_boundary_yields_not_measurable(store):
    m = measure_centering(GeometryResult(boundary_confidence=0.1), store)
    assert m.measurable is False


def test_the_method_is_recorded_as_provenance(store):
    m = measure_centering(geom(render_png(CardSpec()), store, "h"), store)
    assert m.method == "border_geometry"


def test_the_result_round_trips_and_canonicalizes(store):
    import json

    from card_reviewer.review.canonical import canonicalize
    from card_reviewer.review.imaging.measure.centering import CenteringMeasurement

    m = measure_centering(geom(render_png(CardSpec()), store, "h"), store)
    assert CenteringMeasurement.model_validate(json.loads(m.model_dump_json())) == m
    assert canonicalize({"centering": m.model_dump(mode="json")})


def test_an_unreliable_border_blocks_measurement_even_when_pixels_would_yield_one(
        store):
    """Isolates the has_reliable_border check.

    A borderless render happens to produce no ratio anyway, so dropping the
    check changes nothing there. Here the pixels are a normal bordered card —
    a ratio is plainly extractable — and only the reliability flag stops it.
    Measuring against a reference geometry judged unreliable would report a
    number the method cannot support.
    """
    good = geom(render_png(CardSpec()), store, "h")
    assert measure_centering(good, store).measurable is True

    unreliable = good.model_copy(update={"has_reliable_border": False})
    assert measure_centering(unreliable, store).measurable is False


def test_the_ink_threshold_ignores_faint_border_noise():
    """Isolates INK_VARIANCE_FRACTION.

    A synthetic border has exactly zero variance, so any threshold above
    zero behaves the same. Real borders carry sensor noise, and a threshold
    of zero would read the first noisy column as printed art and report the
    border as having no width at all.
    """
    import numpy as np

    from card_reviewer.review.imaging.measure.centering import _ratio

    variance = np.zeros(100)
    variance[:10] = 0.5      # faint noise across the leading border
    variance[90:] = 0.5      # and the trailing one
    variance[30:70] = 40.0   # the printed art
    value, reason = _ratio(variance)
    assert reason is None
    assert value == pytest.approx(50.0, abs=1.0)
