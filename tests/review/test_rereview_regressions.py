"""Regressions the independent re-review found in the fixes themselves.

Every one of these was introduced by a fix for an earlier finding, which is
the argument for re-reviewing a fix diff rather than trusting it.
"""

import numpy as np
import pytest

from card_reviewer.review.enums import (
    Authority, Coverage, FindingState, Scale,
)
from card_reviewer.review.findings import (
    Finding, FindingProducer, Severity,
)
from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
from card_reviewer.review.imaging.observability import analyze as observe
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.policies.scoring_v1 import estimated_grade
from card_reviewer.review.provenance import (
    EvidenceOrigin, EvidenceRef, NormalizedBox,
)
from card_reviewer.review.storage.artifacts import ArtifactStore

ALL_CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _observe(data, store):
    image_hash = store.put_image(data)
    return observe(geometry_analyze(data, store, image_hash), store, image_hash)


# --- C1: a majority-glared card ------------------------------------------

@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_a_dark_bordered_card_reports_every_glared_corner(count, store):
    """On a border that is not itself near-white the absolute arm carries
    this at any count."""
    result = _observe(
        render_png(CardSpec(border_color=(20, 20, 20),
                            glare_regions=ALL_CORNERS[:count])), store)
    for region in ALL_CORNERS[:count]:
        assert result.reason_codes.get((region, "corners", "rounding")) == "GLARE"


@pytest.mark.parametrize("count", [1, 2])
def test_a_white_bordered_card_reports_a_minority_of_glared_corners(
        count, store):
    result = _observe(
        render_png(CardSpec(border_color=(255, 255, 255),
                            glare_regions=ALL_CORNERS[:count])), store)
    for region in ALL_CORNERS[:count]:
        assert result.reason_codes.get((region, "corners", "rounding")) == "GLARE"


@pytest.mark.parametrize("count", [1, 2, 3])
def test_a_majority_glared_white_card_is_now_detected(count, store):
    """This WAS a recorded I2 hole. It is closed for a majority, and the
    positive assertion replaces the one that pinned the gap.

    Glare was found as excess over the MEDIAN of the sibling regions, so
    once half or more were glared the median moved with them and nothing
    stood out. A white card blown out on three corners reported HIGH
    detectability and reached PASS — and reached it with a BETTER score
    than the same card with one glared corner.

    Closed by comparing against the cleanest comparable sibling instead of
    the median, which cannot be dragged up by the regions under test, and
    by comparing only within a region's own kind. No threshold was fitted:
    the four discriminators this docstring used to list all failed to
    separate on measurement, and a fifth would only have moved the false
    positives onto miscut cards. Over the 40 labelled clean and glare
    photographs the minimum and the median flag identical corners, so the
    robustness costs nothing.

    Regions that are OBSTRUCTED are excluded from the reference — a thumb
    over one corner reads as zero clipping, and without that exclusion the
    card's three ordinary corners stood out from it as glare.

    The all-four case remains open and is a question for the owner rather
    than a bug: see
    `test_glare_monotonicity.py::test_all_four_corners_blown_is_an_open_design_question`.
    """
    result = _observe(
        render_png(CardSpec(border_color=(255, 255, 255),
                            glare_regions=ALL_CORNERS[:count])), store)
    reported = sum(
        1 for region in ALL_CORNERS
        if result.reason_codes.get((region, "corners", "rounding")) == "GLARE")
    assert reported == count, (
        f"{count} glared corners, {reported} reported — not seeing a defect "
        f"must not read as evidence of a clean one")


def test_a_clean_card_is_still_not_glared(store):
    for border in ((255, 255, 255), (20, 20, 20), (150, 150, 150)):
        result = _observe(render_png(CardSpec(border_color=border)), store)
        assert "GLARE" not in result.reason_codes.values(), (
            f"a clean {border} card was reported as glared")


# --- C2: an observed defect that fails I1 --------------------------------

def _finding(state, severity=Severity.SEVERE):
    box = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)
    return Finding(
        defect_type="rounding", category="corners", state=state,
        producer=FindingProducer.HEURISTIC, confidence=0.95,
        psa10_relevant=True, severity=severity, location=box,
        evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                              origin=EvidenceOrigin.ORIGINAL,
                              view="corner_top_left", region=box)])


def test_an_observed_defect_failing_i1_also_stops_the_estimate_claiming_ten():
    """The `unresolved` guard looked only for SUSPECTED, so an OBSERVED
    finding that fails I1 — stronger evidence — fell through to "10" while
    a merely SUSPECTED one gave "9-10". The estimate was non-monotone in
    evidence strength: weaker evidence produced the worse grade.
    """
    observed = estimated_grade(
        [(_finding(FindingState.OBSERVED), Authority.BINDING, False)],
        Coverage.SUFFICIENT)
    suspected = estimated_grade(
        [(_finding(FindingState.SUSPECTED), Authority.BINDING, False)],
        Coverage.SUFFICIENT)

    assert observed != "10", "a confidently observed severe defect graded 10"
    assert observed == suspected == "9-10"


def test_a_confirmed_defect_still_grades_worse_than_an_unresolved_one():
    """The distinction that must survive: I1-satisfying evidence IS a
    confirmed defect and grades accordingly."""
    confirmed = estimated_grade(
        [(_finding(FindingState.OBSERVED), Authority.BINDING, True)],
        Coverage.SUFFICIENT)
    assert confirmed == "<=8"


def test_a_clean_card_still_grades_ten():
    assert estimated_grade([], Coverage.SUFFICIENT) == "10"


# --- C3: a dark card is not an obstructed one ----------------------------

def test_a_dark_card_design_is_not_reported_as_an_obstruction(store):
    """Occlusion used an ABSOLUTE darkness fraction while glare beside it
    used a relative one. A card with dark artwork therefore reported
    OCCLUSION in every region — measured at 0.99 of the centre against a
    0.10 threshold — and was dropped as INSUFFICIENT_IMAGES with a photo
    request to remove an obstruction that does not exist. That converts a
    structural property of the card into a circumstantial one.
    """
    result = _observe(
        render_png(CardSpec(border_color=(30, 30, 30), art_color=(10, 12, 14))),
        store)
    assert "OCCLUSION" not in result.reason_codes.values(), (
        "a dark card was reported as obstructed")


def test_a_real_obstruction_is_still_reported(store):
    """The half that must not be lost."""
    import cv2

    from card_reviewer.review.imaging.synthetic import (
        _draw_card, _place_on_background,
    )

    spec = CardSpec(border_color=(20, 20, 20))
    card = _draw_card(spec, np.random.default_rng(spec.seed))
    card[0:300, 0:300] = 4
    data = cv2.imencode(".png", _place_on_background(card, spec, cv2))[1].tobytes()

    result = _observe(data, store)
    assert result.reason_codes.get(("top_left", "corners", "rounding")) == (
        "OCCLUSION")


# --- C4: a blank rectangle is not a card ---------------------------------

def _blank(width, height):
    import cv2

    canvas = np.full((840, 600, 3), 10, np.uint8)
    canvas[300:300 + height, 200:200 + width] = 160
    return cv2.imencode(".png", canvas)[1].tobytes()


@pytest.mark.parametrize("size", [(200, 120), (300, 300)])
def test_a_rectangle_that_is_not_card_shaped_is_declined(size, store):
    """Dropping `area_ratio` from confidence removed the only card-likeness
    signal, and nothing replaced it: a plain grey rectangle scored 1.000,
    reported a reliable border, and PASSed at grade 10 with no findings.
    CARD_ASPECT and _aspect were written for exactly this and never wired
    up.
    """
    data = _blank(*size)
    result = geometry_analyze(data, store, store.put_image(data))
    assert not result.usable, (
        f"a {size[0]}x{size[1]} blank rectangle was accepted as a card at "
        f"confidence {result.boundary_confidence:.3f}")


@pytest.mark.parametrize("spec", [
    CardSpec(),
    CardSpec(rotation_deg=10.0),
    CardSpec(rotation_deg=20.0),
    CardSpec(perspective=0.25),
    CardSpec(borderless=True),
])
def test_a_real_card_is_still_accepted_at_any_size(spec, store):
    """The reason area was dropped in the first place: a crisply detected
    card must not be declined for being small in frame."""
    import cv2

    card = cv2.imdecode(np.frombuffer(render_png(spec), np.uint8),
                        cv2.IMREAD_COLOR)
    for fraction in (1.0, 0.5, 0.3):
        height, width = card.shape[:2]
        small = cv2.resize(card, (int(width * fraction), int(height * fraction)),
                           interpolation=cv2.INTER_AREA)
        canvas = np.full((height, width, 3), 10, np.uint8)
        y0, x0 = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        data = cv2.imencode(".png", canvas)[1].tobytes()

        result = geometry_analyze(data, store, store.put_image(data))
        assert result.usable, (
            f"a real card at {fraction:.0%} of frame was declined "
            f"(confidence {result.boundary_confidence:.3f})")
