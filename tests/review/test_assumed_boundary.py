"""A boundary we assumed is not a boundary we saw.

When the flood finds no background, geometry falls back to reading the frame
as the card. That is often nearly right — the card really does fill the
photograph — but the last few pixels at each edge are a strip of backdrop or
holder that the flood could not separate.

Corners and edges measure a DEPARTURE FROM THE CARD'S BORDER. Handed a quad
whose outer pixels are not card, they measure that strip and report it as
damage. Verified on the real corpus: 15 of 88 accepted photographs are
whole-frame accepts, and four of them are cards a human labelled CLEAN that
emitted severe corner and edge anomalies — measured off the photograph's own
edges.

Centering already declines here, which is why this never produced a false
REJECT. The border-relative producers had no equivalent guard.
"""

import numpy as np
import pytest

from card_reviewer.review.imaging.geometry import analyze
from card_reviewer.review.imaging.measure import measure_all
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _card_with_a_sliver(strip=8):
    """A card filling its frame, with a thin dark strip the flood cannot
    separate — exactly what the failing corpus photographs look like."""
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card

    card = _draw_card(CardSpec(), np.random.default_rng(0))
    height, width = card.shape[:2]
    canvas = np.full((height + 2 * strip, width + 2 * strip, 3), 40, np.uint8)
    canvas[strip:strip + height, strip:strip + width] = card
    return cv2.imencode(".png", canvas)[1].tobytes()


def test_a_real_boundary_is_reported_as_observed(store):
    data = render_png(CardSpec())
    result = analyze(data, store, store.put_image(data))
    assert result.usable
    assert result.boundary_observed is True


def test_a_frame_fallback_is_reported_as_assumed(store):
    data = _card_with_a_sliver()
    result = analyze(data, store, store.put_image(data))
    if not result.usable:
        pytest.skip("fixture no longer reaches the frame fallback")
    assert result.boundary_observed is False, (
        "the frame was taken as the card, but that is an assumption rather "
        "than an observation and must say so")


def test_an_assumed_boundary_emits_no_border_relative_findings(store):
    """The fix. Corners and edges measure departure from the card's border,
    so with a boundary we only assumed, the outermost pixels may not be card
    at all — and that is precisely where they measure."""
    data = _card_with_a_sliver()
    result = analyze(data, store, store.put_image(data))
    if not result.usable or result.boundary_observed:
        pytest.skip("fixture no longer reaches the frame fallback")

    measured = measure_all(result, store, store.put_image(data))
    border_relative = [a for a in measured.anomalies
                       if a["category"] in ("corners", "edges")]
    assert border_relative == [], (
        f"{len(border_relative)} border-relative findings from a boundary "
        "that was never observed: "
        f"{[(a['category'], a.get('region')) for a in border_relative]}")


def test_an_assumed_boundary_still_yields_crops_for_the_vision_layer(store):
    """Declining to CONCLUDE is not declining to look. The crops are what the
    vision layer inspects, and they stay."""
    data = _card_with_a_sliver()
    result = analyze(data, store, store.put_image(data))
    if not result.usable or result.boundary_observed:
        pytest.skip("fixture no longer reaches the frame fallback")

    measured = measure_all(result, store, store.put_image(data))
    assert measured.corners.crops, "no corner crops for the vision layer"
    assert measured.edges.crops, "no edge crops for the vision layer"


def test_an_observed_boundary_still_finds_real_damage(store):
    """The guard must not buy safety by silencing the working case."""
    data = render_png(CardSpec(border_color=(20, 20, 20),
                               corner_damage={"bottom_left": 0.9}))
    result = analyze(data, store, store.put_image(data))
    assert result.boundary_observed is True

    measured = measure_all(result, store, store.put_image(data))
    assert any(a["category"] == "corners" and a.get("region") == "bottom_left"
               for a in measured.anomalies)


def test_the_confidence_floor_is_where_it_is_declared():
    """0.55 could be dropped to 0.30 with nothing failing. Measured on the
    real corpus, genuine cards run 0.562-1.000, so a floor much below that
    stops refusing anything at all."""
    from card_reviewer.review.imaging.geometry import MIN_BOUNDARY_CONFIDENCE

    assert 0.5 <= MIN_BOUNDARY_CONFIDENCE <= 0.65


def test_a_shapeless_detection_is_still_refused(store):
    """What the floor is for, now that aspect carries card-likeness: a
    region whose bounding box is card-shaped but which is not a rectangle.
    Rectangularity is the only thing standing between this and acceptance.
    """
    import cv2

    # A T-shape whose BOUNDING BOX is 5:7 — passes aspect, is not a card.
    canvas = np.full((900, 640, 3), 15, np.uint8)
    cv2.rectangle(canvas, (70, 120), (570, 300), (225, 225, 225), -1)
    cv2.rectangle(canvas, (250, 120), (390, 820), (225, 225, 225), -1)

    data = cv2.imencode(".png", canvas)[1].tobytes()
    result = analyze(data, store, store.put_image(data))
    assert not result.usable, (
        f"a T-shape was accepted as a card at confidence "
        f"{result.boundary_confidence:.3f}")


def test_the_artwork_frame_fill_also_reports_an_assumed_boundary(store):
    """There are TWO paths that read the frame as the card: the flood
    finding no background, and the ambiguity check deciding the detected
    region was the artwork inside a frame-filling card. Both are assumptions
    about a boundary that is not in the picture, and both must say so — the
    second was reported as observed."""
    import cv2

    from card_reviewer.review.imaging.synthetic import _draw_card

    # A bare card: the flood claims its white border as background and the
    # artwork panel survives, which is the ambiguity path rather than the
    # no-background path.
    card = _draw_card(CardSpec(), np.random.default_rng(0))
    data = cv2.imencode(".png", card)[1].tobytes()
    result = analyze(data, store, store.put_image(data))
    if not result.usable:
        pytest.skip("fixture no longer reaches the ambiguity path")

    quad = np.asarray(result.quad, dtype=float)
    height, width = card.shape[:2]
    covered = ((quad[:, 0].max() - quad[:, 0].min())
               * (quad[:, 1].max() - quad[:, 1].min())) / (width * height)
    if covered < 0.9:
        pytest.skip("fixture did not take the frame")

    assert result.boundary_observed is False
    measured = measure_all(result, store, store.put_image(data))
    assert not [a for a in measured.anomalies
                if a["category"] in ("corners", "edges")]


def test_an_assumed_boundary_is_recorded_as_unassessed_not_as_clean(store):
    """Silencing the producers was not enough, and made things worse.

    Withholding a measurement without recording that it is MISSING reads as
    cleanliness. Verified end to end: a card labelled `corners:fraying` came
    back score 90, grade 9-10, zero findings — identical to a clean card,
    and BETTER than the 45 it scored before the silencing. Missing evidence
    must remove evidence, never improve the outcome.
    """
    from card_reviewer.review.enums import Scale
    from card_reviewer.review.imaging.observability import analyze as observe

    data = _card_with_a_sliver()
    result = analyze(data, store, store.put_image(data))
    if not result.usable or result.boundary_observed:
        pytest.skip("fixture no longer reaches the frame fallback")

    observed = observe(result, store, store.put_image(data))
    border_relative = {
        (region, category, defect): value
        for (region, category, defect), value in observed.detectability.items()
        if category in ("corners", "edges")
    }
    assert border_relative, "no corner or edge detectability at all"
    assert all(v < Scale.MODERATE for v in border_relative.values()), (
        "an assumed boundary still reports its corners and edges as "
        f"assessable: {border_relative}")

    reasons = {observed.reason_codes.get(k) for k in border_relative}
    assert None not in reasons, "a lowered region with no reason at all"
    assert "BOUNDARY_NOT_OBSERVED" in reasons, (
        f"the gap is not explained: {reasons}")
    # WHITE_BORDER may also appear — a white border hides whitening whatever
    # the boundary — and that one is STRUCTURAL, so it waives rather than
    # blocks. The circumstantial reason has to reach the types it does not
    # cover, or the category would still count as assessed.
    assert "BOUNDARY_NOT_OBSERVED" in {
        observed.reason_codes.get(k) for k in border_relative
        if k[2] != "whitening"
    }


def test_an_assumed_boundary_stops_the_card_reaching_sufficient(store):
    """The consequence that matters: the recorded gap must actually block
    PASS, not merely appear in a list."""
    from card_reviewer.review.enums import Coverage
    from card_reviewer.review.imaging.observability import analyze as observe
    from card_reviewer.review.policies.coverage_v1 import (
        REQUIRED_FACES, evaluate_coverage,
    )
    from card_reviewer.review.roles import ImageRole

    data = _card_with_a_sliver()
    result = analyze(data, store, store.put_image(data))
    if not result.usable or result.boundary_observed:
        pytest.skip("fixture no longer reaches the frame fallback")

    observed = observe(result, store, store.put_image(data))
    detectability, reasons = {}, {}
    for face in REQUIRED_FACES:
        for (region, category, defect), value in observed.detectability.items():
            detectability[(face, region, category, defect)] = value
            code = observed.reason_codes.get((region, category, defect))
            if code:
                reasons[(face, region, category, defect)] = code

    coverage = evaluate_coverage(detectability, reasons, {}, REQUIRED_FACES)
    assert coverage.outcome is not Coverage.SUFFICIENT
    assert any(limitation.reason_code == "BOUNDARY_NOT_OBSERVED"
               for limitation in coverage.limitations)
    assert any("margin" in photo
               for photo in coverage.recommended_additional_photos)


def test_that_gap_is_circumstantial_so_it_asks_for_a_better_photograph(store):
    """A boundary we could not see is a property of THIS photograph — one
    framed with more margin would show it."""
    from card_reviewer.review.taxonomy import REASON_CODES, UndetectabilityClass, class_of

    assert "BOUNDARY_NOT_OBSERVED" in REASON_CODES
    assert class_of("BOUNDARY_NOT_OBSERVED") is UndetectabilityClass.CIRCUMSTANTIAL


def test_the_ambiguity_arm_cannot_yet_separate_cropped_from_borderless(store):
    """A KNOWN GAP, recorded rather than papered over.

    When the ambiguity check fires but the frame-fill test is refused,
    geometry has concluded it cannot tell the card from its artwork. A review
    found that 9 of 88 corpus photographs take this arm and still emit corner
    and edge findings measured against what may be an artwork edge — up to 5
    severe, including on cards labelled clean.

    Marking those unobserved was tried and reverted. The arm has a known
    false positive: a BORDERLESS card on a backdrop trips it every time,
    because its outer band is artwork while the backdrop's is uniform. Its
    boundary is perfectly visible and its corners and edges are measurable,
    so standing the producers down there penalises a design property — the
    structural-versus-circumstantial confusion again, and it made a
    borderless card strictly worse.

    Separating "cropped, so the boundary was never seen" from "borderless, so
    there is no band to measure" needs a signal this branch does not have.
    Until then the border reference is withheld (centering declines) and the
    border-relative producers keep running.
    """
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png

    data = render_png(CardSpec(borderless=True))
    result = analyze(data, store, store.put_image(data))
    assert result.usable
    assert result.has_reliable_border is False, "borderless has no band"
    assert result.boundary_observed is True, (
        "this arm no longer marks a borderless card's boundary unobserved — "
        "if the two cases can now be separated, replace this test with the "
        "positive assertion and record how")


def test_the_geometry_version_moves_when_its_output_changes():
    """A cached row from before this change deserializes boundary_observed to
    its default. Unless the stage version moves, the fix never reaches a card
    already in the cache — and the default is the UNSAFE value."""
    from card_reviewer.review.versions import CV_VERSION, GEOMETRY_VERSION

    assert GEOMETRY_VERSION != "1.0.0", (
        "geometry's output gained a field that changes downstream behaviour; "
        "the version must move or cached rows keep the old answer")
    assert CV_VERSION != "1.0.0", (
        "cv_measurements now consults boundary_observed")
