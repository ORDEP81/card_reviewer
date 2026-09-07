"""A pristine card must not accuse itself.

All three anomaly producers scored a region by the standard deviation of its
pixels, which measures whether the region CONTAINS CONTENT, not whether it is
damaged. Printed artwork is content, so a clean card produced ten candidates,
several at confidence 1.0. Each SUSPECTED finding matching a BINDING rule
costs 15 points, so the rank score floored at 0 for every card and its stated
purpose — ordering candidates for a human — was a constant.
"""

import pytest

from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
from card_reviewer.review.imaging.measure import measure_all
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore

CLEAN_BORDERS = [(255, 255, 255), (20, 20, 20), (150, 150, 150)]


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _measure(spec, store):
    data = render_png(spec)
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    return measure_all(geometry, store, image_hash)


def _anomalies(result, category):
    return [a for a in result.anomalies if a["category"] == category]


@pytest.mark.parametrize("border", CLEAN_BORDERS)
@pytest.mark.parametrize("category", ["corners", "edges", "surface"])
def test_a_pristine_card_raises_no_candidates(border, category, store):
    result = _measure(CardSpec(border_color=border), store)
    found = _anomalies(result, category)
    assert found == [], (
        f"{len(found)} spurious {category} candidates on a clean card: "
        f"{[(a.get('region'), round(a.get('contrast', 0), 1)) for a in found]}")


@pytest.mark.parametrize("border", CLEAN_BORDERS)
def test_a_busy_card_design_is_not_damage(border, store):
    """Heavy printing is exactly what the std metric mistook for defects."""
    result = _measure(CardSpec(border_color=border, text_heavy=True), store)
    assert result.anomalies == [], (
        f"{len(result.anomalies)} candidates raised by artwork alone")


def test_a_damaged_corner_is_still_found(store):
    result = _measure(
        CardSpec(border_color=(20, 20, 20),
                 corner_damage={"bottom_left": 0.9}), store)
    assert any(a["region"] == "bottom_left"
               for a in _anomalies(result, "corners"))


def test_the_score_separates_a_clean_card_from_a_damaged_one(store):
    """The end of the chain, and the property the rank score exists for."""
    from card_reviewer.review.enums import Scale
    from card_reviewer.review.policies.scoring_v1 import rank_score

    assert _measure(CardSpec(border_color=(20, 20, 20)), store).anomalies == []
    damaged = _measure(
        CardSpec(border_color=(20, 20, 20),
                 corner_damage={"top_left": 0.9, "bottom_right": 0.9}), store)
    assert len(_anomalies(damaged, "corners")) == 2


def test_the_surface_baseline_is_robust_to_the_damage_it_is_measuring():
    """A mean baseline is dragged upward by the very tiles it is supposed to
    make stand out, so heavy damage hides itself. The median is not."""
    import numpy as np

    from card_reviewer.review.imaging.measure.surface import _local_outlier

    rng = np.random.default_rng(0)
    quiet = rng.normal(128, 2.0, (400, 400))

    scratched = quiet.copy()
    for i in range(6):                      # several loud tiles, not one
        scratched[40 * i : 40 * i + 32, 40 * i : 40 * i + 32] += rng.normal(
            0, 60.0, (32, 32))

    assert _local_outlier(quiet) < _local_outlier(scratched)
    # ...and the damaged card must still clear the threshold. With a mean
    # baseline the loud tiles lift the baseline toward themselves.
    from card_reviewer.review.imaging.measure.surface import ANOMALY_CONTRAST

    assert _local_outlier(scratched) > ANOMALY_CONTRAST


def test_the_backdrop_sliver_at_the_rectified_edge_is_not_wear():
    """A detected boundary is approximate, so the rectified card carries a
    few columns of backdrop, and geometry's border mask includes them. Read
    as border, that backdrop is an enormous departure and fabricates a defect
    on a clean card."""
    import numpy as np

    from card_reviewer.review.imaging.measure.corners import departure_from

    height = width = 400
    inset = int(width * 0.012)                 # the tolerance we allow for
    img = np.full((height, width, 3), 255.0)   # a uniformly white border
    img[:, : inset - 1] = 10.0                 # ...with a backdrop sliver
    img[:, -(inset - 1) :] = 10.0

    region = (slice(0, height), slice(0, 40))
    assert departure_from(img, region, reference=255.0) < 10.0, (
        "the backdrop was read as border wear")

    # The tolerance is a few pixels, not a licence to ignore the edge: a
    # boundary wrong by far more than that must still show up rather than
    # being quietly trimmed away.
    badly_wrong = np.full((height, width, 3), 255.0)
    badly_wrong[:, : inset * 6] = 10.0
    assert departure_from(badly_wrong, region, reference=255.0) > 100.0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("border", CLEAN_BORDERS)
def test_no_clean_card_in_the_corpus_accuses_itself(seed, border, store):
    """One seed is not a calibration. The single-seed numbers this
    producer's thresholds were first set from looked separable by luck; over
    a spread of seeds the clean and scratched populations overlap on every
    view, which is why the thresholds sit above the clean maximum instead.
    """
    result = _measure(CardSpec(border_color=border, seed=seed), store)
    assert result.anomalies == [], (
        f"seed {seed}, border {border}: "
        f"{[(a['category'], round(a.get('contrast', 0), 1)) for a in result.anomalies]}")


def test_the_surface_producer_still_emits_every_view_it_is_there_for():
    """Its real output is the views the vision layer inspects, plus the
    provenance I3 depends on — not the candidates. Going quiet on candidates
    must not go quiet on those."""
    import tempfile
    from pathlib import Path

    from card_reviewer.review.imaging.measure.surface import measure_surface

    store = ArtifactStore(Path(tempfile.mkdtemp()))
    data = render_png(CardSpec())
    image_hash = store.put_image(data)
    result = measure_surface(
        geometry_analyze(data, store, image_hash), store, image_hash)

    assert set(result.crops) >= {"original", "clahe", "sharpen", "edge"}
    origins = {r.origin for r in result.evidence_refs}
    assert len(origins) > 1, "enhanced and unenhanced refs must stay distinct"


def test_visible_in_original_reports_the_unenhanced_view_honestly(store):
    """I3 turns on this field: a defect visible only in an enhanced view can
    never independently establish a confirmed disqualifier. Provenance must
    be measured, never assumed — the threshold that decides it is the
    unenhanced view's own, not whichever view raised the candidate.
    """
    from card_reviewer.review.imaging.measure.surface import (
        VIEW_THRESHOLDS, _local_outlier, measure_surface,
    )
    from card_reviewer.review.imaging.geometry import load_geometry

    data = render_png(CardSpec())
    image_hash = store.put_image(data)
    result = geometry_analyze(data, store, image_hash)
    gray = load_geometry(result, store).normalized.mean(axis=2)

    measured = measure_surface(result, store, image_hash)

    # Absolute, not recomputed from the same constant: a clean card's
    # unenhanced reading sits well under the threshold across the corpus
    # (max 44.5 over 8 seeds x 3 borders), so nothing here may claim to be
    # visible unenhanced. Deriving `expected` from VIEW_THRESHOLDS would move
    # with any change to it and assert nothing.
    assert _local_outlier(gray) < VIEW_THRESHOLDS["original"]
    for anomaly in measured.anomalies:
        assert anomaly["visible_in_original"] is False
        assert anomaly["surfaced_by"] != "original"


def test_the_surface_margin_keeps_the_design_boundary_out_of_the_reading(store):
    """The border/artwork boundary is the strongest texture edge on a plain
    card and it is design. Reading it as surface texture is what made every
    card look scratched."""
    from card_reviewer.review.imaging.geometry import load_geometry
    from card_reviewer.review.imaging.measure.surface import _local_outlier
    import card_reviewer.review.imaging.measure.surface as surface_module

    data = render_png(CardSpec())
    image_hash = store.put_image(data)
    gray = load_geometry(
        geometry_analyze(data, store, image_hash), store).normalized.mean(axis=2)

    with_margin = _local_outlier(gray)
    original = surface_module.SURFACE_MARGIN_FRACTION
    try:
        surface_module.SURFACE_MARGIN_FRACTION = 0.0
        without_margin = _local_outlier(gray)
    finally:
        surface_module.SURFACE_MARGIN_FRACTION = original

    assert with_margin < without_margin, (
        "excluding the design boundary must lower the reading, not raise it")


@pytest.mark.parametrize("h_centering", [50.0, 60.0, 70.0])
def test_an_ordinary_miscut_raises_no_false_candidates(h_centering, store):
    """Up to about 70/30 the border band stays inside the real border."""
    result = _measure(CardSpec(h_centering=h_centering), store)
    assert result.anomalies == [], (
        f"{h_centering}/{100 - h_centering} raised "
        f"{[(a['category'], a.get('region')) for a in result.anomalies]}")


@pytest.mark.parametrize("h_centering", [78.0, 85.0])
def test_a_severe_miscut_raises_known_false_candidates(h_centering, store):
    """A KNOWN GAP, recorded rather than papered over.

    `_segment_border` marks a fixed 24px ring of the rectified card. A
    miscut card's borders are asymmetric by definition, so past roughly
    75/25 the NARROW side's real border is thinner than the ring and the
    band reaches into the artwork. The corner and edge producers then
    measure the design and report it as SEVERE damage at confidence 1.0 —
    on a card with no damage at all, and miscuts are the population this
    tool screens.

    Deriving each side's width from its own variance profile was tried and
    made it worse: a CENTRED card then fired on all four corners, because a
    narrower band leaves too few pixels for a stable reference. Reverted
    rather than shipped.

    The cost is bounded: these are INTERPRETIVE candidates, so they reach
    `suspected` and cost score, never `observed`, and cannot themselves
    REJECT. A severely miscut card is also already being rejected on
    centering, which is measured correctly (77.3 against a rendered 75.0).
    So the practical effect is a worse rank score on a card that is failing
    anyway — not a lost gem.

    The real fix is to derive the band from the measured art boundary that
    centering ALREADY computes, which is plumbing this branch did not take
    on. This test fails loudly if someone does it.
    """
    result = _measure(CardSpec(h_centering=h_centering), store)
    assert result.anomalies, (
        "this gap appears to have been closed — good. Replace this test with "
        "the positive assertion and record how it was done.")


def test_corner_damage_sensitivity_depends_on_border_luminance(store):
    """A second KNOWN GAP in the same producer, recorded for the same reason.

    The departure is |luma - border_median|, so how loudly damage reads is a
    function of the border's brightness rather than of the damage. The same
    0.9-severity corner reads:

        white border   52      grey-200   MISSED ENTIRELY
        grey-150       81      dark       210

    Real corner wear is fibre exposure — a texture change, not only a
    luminance one — so a luminance-difference metric is the wrong instrument
    for a light-bordered card. It is also partly a fixture artefact: the
    generator renders damage as BRIGHT noise, which is close to invisible
    against a bright border by construction. Both need real photographs to
    resolve, and neither should be read as working.
    """
    found = {}
    for border in ((255, 255, 255), (200, 200, 200), (150, 150, 150),
                   (20, 20, 20)):
        result = _measure(
            CardSpec(border_color=border,
                     corner_damage={"bottom_left": 0.9}), store)
        found[border] = bool(_anomalies(result, "corners"))

    assert not all(found.values()), (
        "damage is now found on every border — the gap may be closed. Verify "
        "against real photographs and replace this test.")
