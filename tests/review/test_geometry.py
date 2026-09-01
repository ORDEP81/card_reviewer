import json

import cv2
import numpy as np
import pytest

from card_reviewer.review.imaging.geometry import (
    GeometryResult, analyze, load_geometry,
)
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


def test_a_clean_card_boundary_is_detected_with_high_confidence(store):
    r = analyze(render_png(CardSpec()), store, "h1")
    assert r.boundary_confidence > 0.5 and r.quad is not None


def test_the_detected_quad_is_the_card_not_the_printed_art(store):
    """A card that filled the frame would leave only the art rectangle
    findable, and every later measurement would be of the wrong shape."""
    spec = CardSpec()
    r = analyze(render_png(spec), store, "h1")
    quad = np.array(r.quad)
    width = quad[:, 0].max() - quad[:, 0].min()
    assert width == pytest.approx(spec.card_w, abs=spec.card_w * 0.1)


def test_a_rotated_card_is_rectified_to_the_card_aspect_ratio(store):
    r = analyze(render_png(CardSpec(rotation_deg=6.0, perspective=0.08, seed=3)),
                store, "h1")
    normalized = load_geometry(r, store).normalized
    assert normalized is not None
    assert abs(normalized.shape[1] / normalized.shape[0] - 600 / 840) < 0.05


def test_unreliable_detection_declines_geometry_dependent_work(store):
    """Never produce plausible numbers from a bad quad."""
    noise = np.random.default_rng(0).integers(0, 255, (800, 600, 3), dtype=np.uint8)
    r = analyze(cv2.imencode(".png", noise)[1].tobytes(), store, "h1")
    assert r.usable is False
    assert load_geometry(r, store).normalized is None


def test_undecodable_bytes_yield_zero_confidence(store):
    assert analyze(b"not an image", store, "h1").boundary_confidence == 0.0


def test_a_white_border_is_segmented_as_border(store):
    r = analyze(render_png(CardSpec(border_color=(255, 255, 255))), store, "h1")
    mask = load_geometry(r, store).border_mask
    assert mask is not None and mask[:8, :8].mean() > 128


def test_a_bordered_design_reports_a_reliable_border(store):
    assert analyze(render_png(CardSpec(border_color=(255, 255, 255))),
                   store, "h1").has_reliable_border is True


def test_a_borderless_design_yields_no_reliable_border_band(store):
    assert analyze(render_png(CardSpec(borderless=True)),
                   store, "h1").has_reliable_border is False


def test_the_perspective_transform_is_emitted_as_provenance(store):
    r = analyze(render_png(CardSpec(perspective=0.1, seed=5)), store, "h1")
    assert r.transform is not None and len(r.transform) == 3


# --- cache safety ----------------------------------------------------------

def test_the_output_holds_artifact_ids_not_pixel_arrays(store):
    """This stage is cached as JSON in SQLite. A live NumPy array in the
    output model would make the whole image tier uncacheable."""
    r = analyze(render_png(CardSpec()), store, "h1")
    assert json.loads(r.model_dump_json())["normalized_artifact_id"]


def test_a_cached_result_round_trips_to_identical_downstream_pixels(store):
    """A cache hit and a fresh computation must be indistinguishable to the
    CV stages that consume this."""
    fresh = analyze(render_png(CardSpec(seed=7)), store, "h1")
    revived = GeometryResult.model_validate(json.loads(fresh.model_dump_json()))
    assert np.array_equal(load_geometry(fresh, store).normalized,
                          load_geometry(revived, store).normalized)


def test_geometry_crops_live_under_their_own_directory(store):
    """Owned by this stage, so invalidated by this stage's cache and never
    by the measurement stages'."""
    r = analyze(render_png(CardSpec()), store, "h1")
    assert "/face/" in str(store.path_of(r.normalized_artifact_id))


def test_the_output_canonicalizes_for_a_fingerprint(store):
    from card_reviewer.review.canonical import canonicalize

    r = analyze(render_png(CardSpec()), store, "h1")
    assert canonicalize(r.model_dump(mode="json"))


def test_a_borderless_card_is_still_detected_and_measurable(store):
    """It loses its centering REFERENCE, not its detection. Failing to find
    the card at all would leave a borderless design with no measurements of
    any kind, which is over-conservative to the point of uselessness."""
    r = analyze(render_png(CardSpec(borderless=True)), store, "h1")
    assert r.usable is True
    assert r.has_reliable_border is False


def test_an_angled_card_keeps_its_border_reference(store):
    """A photographed card is a trapezoid. Wrapping it in a bounding
    rectangle pulls background wedges into the rectified image, contaminating
    the border band and costing centering on every angled shot."""
    r = analyze(render_png(CardSpec(rotation_deg=6.0, perspective=0.08, seed=3)),
                store, "h1")
    assert r.usable is True
    assert r.has_reliable_border is True


def test_a_card_occupying_little_of_the_frame_is_low_confidence(store):
    """Confidence scales with how much of the frame the card fills, so a
    distant shot declines geometry-dependent work rather than measuring a
    handful of pixels."""
    import cv2

    card = cv2.imdecode(np.frombuffer(render_png(CardSpec()), np.uint8),
                        cv2.IMREAD_COLOR)
    canvas = np.full((card.shape[0] * 3, card.shape[1] * 3, 3), 10, np.uint8)
    canvas[:card.shape[0], :card.shape[1]] = card
    r = analyze(cv2.imencode(".png", canvas)[1].tobytes(), store, "h1")
    assert r.boundary_confidence < 0.5
    assert r.usable is False


def test_a_non_rectangular_region_is_rejected_even_when_card_sized(store):
    """Rectangularity is part of confidence, and must be able to change the
    decision rather than merely lower a number.

    This cross occupies MORE of the frame than the threshold needs — on
    extent alone it would be accepted as a card. It is rejected only because
    it is not rectangular, so a shape nothing like a card cannot pass for one.
    """
    import cv2

    frame = np.full((900, 700, 3), 10, np.uint8)
    side, arm = 700, 700 // 3
    top, left = (900 - side) // 2, 0
    mid = left + (side - arm) // 2
    cv2.rectangle(frame, (left, top + (side - arm) // 2),
                  (left + side - 1, top + (side + arm) // 2), (240, 240, 240), -1)
    cv2.rectangle(frame, (mid, top), (mid + arm, top + side - 1),
                  (240, 240, 240), -1)
    ragged = analyze(cv2.imencode(".png", frame)[1].tobytes(), store, "h1")
    assert ragged.usable is False
    assert analyze(render_png(CardSpec()), store, "h2").usable is True


def test_the_serialized_output_stays_small_because_pixels_are_referenced(store):
    """A normalized card embedded in the row rather than referenced would be
    hundreds of kilobytes of JSON per image."""
    r = analyze(render_png(CardSpec()), store, "h1")
    assert len(r.model_dump_json()) < 2000
