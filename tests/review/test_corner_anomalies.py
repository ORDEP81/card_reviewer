"""A corner anomaly candidate must track corner damage, not border colour.

The metric was the luminance std of the whole corner crop, which on a
bordered card is dominated by the border/artwork edge running through it. It
therefore fired on every white-bordered card whether or not the corner was
damaged, and the destroyed card scored LOWER than the pristine one:

    pristine white    4 candidates, contrast 73.6 71.8 75.1 72.0
    destroyed white   4 candidates, contrast 70.9 69.2 72.5 69.4
    pristine dark     0 candidates
    destroyed dark    4 candidates, contrast 73.8 73.7 73.4 72.9

Eight spurious candidates per card at confidence 1.0 saturated the rank score
to 0 for EVERY card and forced REVIEW on all of them.
"""

import pytest

from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
from card_reviewer.review.imaging.measure.corners import measure_corners
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore

ALL_FOUR = {"top_left": 0.9, "top_right": 0.9,
            "bottom_left": 0.9, "bottom_right": 0.9}


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _corners(spec, store):
    data = render_png(spec)
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    return measure_corners(geometry, store, image_hash)


@pytest.mark.parametrize("border", [(255, 255, 255), (20, 20, 20),
                                    (150, 150, 150)])
def test_a_pristine_card_raises_no_corner_candidates(border, store):
    result = _corners(CardSpec(border_color=border), store)
    assert result.anomalies == [], (
        f"{len(result.anomalies)} spurious candidates on a clean "
        f"{border} border")


@pytest.mark.parametrize("border", [(255, 255, 255), (20, 20, 20)])
def test_a_damaged_corner_is_still_found(border, store):
    result = _corners(
        CardSpec(border_color=border, corner_damage={"bottom_left": 0.9}),
        store)
    regions = {a["region"] for a in result.anomalies}
    assert "bottom_left" in regions, "the damaged corner was missed"


def test_only_the_damaged_corner_is_flagged(store):
    result = _corners(
        CardSpec(border_color=(20, 20, 20),
                 corner_damage={"bottom_left": 0.9}), store)
    assert {a["region"] for a in result.anomalies} == {"bottom_left"}


def test_a_card_damaged_at_every_corner_flags_every_corner(store):
    """The relative reading must not need an undamaged corner to compare
    against — a card can be worn all round."""
    result = _corners(
        CardSpec(border_color=(20, 20, 20), corner_damage=ALL_FOUR), store)
    assert {a["region"] for a in result.anomalies} == set(ALL_FOUR)


def test_every_corner_still_gets_a_crop_and_an_evidence_ref(store):
    """Candidates are for damage; the crops are evidence the vision layer
    needs regardless, and each must keep its own region."""
    result = _corners(CardSpec(), store)
    assert set(result.crops) == set(ALL_FOUR)
    boxes = [(r.region.x0, r.region.y0, r.region.x1, r.region.y1)
             for r in result.evidence_refs]
    assert len(set(boxes)) == 4
