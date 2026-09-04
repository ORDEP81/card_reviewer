"""A producer's region names must be the ones the detectability map declares.

`edges.py` names its regions top / bottom / left / right — an edge runs along
a side, not around a corner. `REGIONS_FOR_CATEGORY["edges"]` declared the four
CORNER names instead, so an edge anomaly at "bottom" looked up detectability
that only existed under "bottom_left" and "bottom_right" and got NONE.

Nothing failed loudly. Every edge finding was forced to `suspected`, because
NONE is below the promotion floor, and I1's adequacy prong could never be
satisfied for the entire edges category. Safe in direction and silently
wrong: the detectability the observability stage carefully computed for edges
described regions no edge finding could ever be at.

Found by someone labelling real photographs and asking why there was no way
to say "top edge".
"""

import pytest

from card_reviewer.review.imaging.observability import REGIONS_FOR_CATEGORY


def _producer_regions(category, spec, store):
    from card_reviewer.review.imaging.geometry import analyze
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.synthetic import render_png

    data = render_png(spec)
    image_hash = store.put_image(data)
    measured = measure_all(analyze(data, store, image_hash), store, image_hash)
    return {a.get("region") for a in measured.anomalies
            if a["category"] == category}


@pytest.fixture
def store(tmp_path):
    from card_reviewer.review.storage.artifacts import ArtifactStore

    return ArtifactStore(tmp_path / "store")


@pytest.mark.parametrize("category", ["corners", "edges"])
def test_every_region_a_producer_emits_is_declared(category, store):
    """The join that was broken. A region the producer reports but the map
    does not declare resolves to NONE detectability, which reads as absent
    evidence — so the finding can never be promoted and the invariant
    quietly stops binding."""
    from card_reviewer.review.imaging.synthetic import CardSpec

    spec = CardSpec(border_color=(20, 20, 20),
                    corner_damage={"bottom_left": 0.9, "top_right": 0.9})
    emitted = _producer_regions(category, spec, store)
    declared = set(REGIONS_FOR_CATEGORY[category])

    assert emitted, f"the {category} producer emitted nothing to check"
    assert emitted <= declared, (
        f"{category} producer emits {sorted(emitted - declared)}, which the "
        f"detectability map does not declare: {sorted(declared)}")


def test_edges_are_named_for_sides_not_corners():
    """An edge runs along a side. Naming edge regions after corners is what
    made the two halves disagree."""
    assert set(REGIONS_FOR_CATEGORY["edges"]) == {
        "top", "bottom", "left", "right"}


def test_corners_are_still_named_for_corners():
    assert set(REGIONS_FOR_CATEGORY["corners"]) == {
        "top_left", "top_right", "bottom_left", "bottom_right"}


def test_an_edge_finding_can_reach_adequate_detectability(store):
    """The consequence, end to end: with the vocabularies agreeing, an edge
    finding resolves to the detectability actually measured for that edge
    instead of to NONE."""
    from card_reviewer.review.enums import Scale
    from card_reviewer.review.heuristic import detectability_for
    from card_reviewer.review.imaging.geometry import analyze
    from card_reviewer.review.imaging.observability import analyze as observe
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png

    data = render_png(CardSpec(border_color=(20, 20, 20)))
    image_hash = store.put_image(data)
    observed = observe(analyze(data, store, image_hash), store, image_hash)
    keyed = {(None, region, category, defect_type): value
             for (region, category, defect_type), value
             in observed.detectability.items()}

    for region in REGIONS_FOR_CATEGORY["edges"]:
        assert detectability_for(keyed, "edges", "chipping", region) > Scale.NONE, (
            f"edges/{region} has no detectability at all")
