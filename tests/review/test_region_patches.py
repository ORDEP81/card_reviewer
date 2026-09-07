"""Every region must map to its own part of the card.

`_patch` had explicit arms for the four corner names and a `case _:` that
returned the CENTRE. When edges were renamed to side names, all four of them
started falling through to that default — so `top`, `bottom`, `left` and
`right` measured the same centre patch of artwork.

Consequences, all of them silent:
  * every edge reported the same detectability, so `stands_out` could never
    fire and glare on an edge was invisible;
  * the centre is artwork, so `bright` was never true and the WHITE_BORDER
    structural exemption never applied to edges;
  * a card whose front is blown out or obstructed reported its EDGES as
    fully assessed, which is I2 exactly;
  * and because edge detectability came back HIGH instead of the old NONE,
    I1's adequacy prong could now be satisfied on evidence nobody had —
    turning a too-conservative bug into a FALSE REJECT.
"""

import numpy as np
import pytest

from card_reviewer.review.imaging.observability import (
    REGIONS_FOR_CATEGORY, _patch,
)


@pytest.fixture
def gray():
    """Distinct values everywhere, so any two regions that overlap show it."""
    return np.arange(200 * 140, dtype=float).reshape(200, 140)


def _declared_regions():
    seen = []
    for regions in REGIONS_FOR_CATEGORY.values():
        for region in regions:
            if region not in seen:
                seen.append(region)
    return seen


@pytest.mark.parametrize("region", _declared_regions())
def test_every_declared_region_has_its_own_patch(region, gray):
    """No region may quietly share another's pixels."""
    mine = _patch(gray, region)
    for other in _declared_regions():
        if other == region:
            continue
        theirs = _patch(gray, other)
        if mine.shape != theirs.shape:
            continue
        assert not np.array_equal(mine, theirs), (
            f"{region!r} and {other!r} return the same pixels")


@pytest.mark.parametrize("region,axis,end", [
    ("top", 0, "start"), ("bottom", 0, "end"),
    ("left", 1, "start"), ("right", 1, "end"),
])
def test_an_edge_band_runs_along_its_own_side(region, axis, end, gray):
    """An edge is a band along a side, and it must be the RIGHT side."""
    band = _patch(gray, region)
    full = gray.shape[1 - axis]
    assert band.shape[1 - axis] == full, (
        f"{region} does not span the card; it is not an edge band")
    assert band.shape[axis] < gray.shape[axis] // 2, (
        f"{region} covers more than half the card")

    expected = (gray[:band.shape[0], :] if (axis, end) == (0, "start") else
                gray[-band.shape[0]:, :] if (axis, end) == (0, "end") else
                gray[:, :band.shape[1]] if (axis, end) == (1, "start") else
                gray[:, -band.shape[1]:])
    assert np.array_equal(band, expected)


def test_an_unknown_region_is_refused_rather_than_given_the_centre(gray):
    """The silent `case _:` is what let the rename break edges without one
    test failing. A region nobody declared is a bug, not a request for the
    middle of the card."""
    with pytest.raises((KeyError, ValueError)):
        _patch(gray, "somewhere_else")


def test_a_blown_out_card_does_not_report_its_edges_as_assessable(tmp_path):
    """The end of the chain: I2. Verified previously to report every edge as
    HIGH with no reason code on a card blown out by glare."""
    from card_reviewer.review.enums import Scale
    from card_reviewer.review.imaging.geometry import analyze
    from card_reviewer.review.imaging.observability import analyze as observe
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    data = render_png(CardSpec(border_color=(20, 20, 20),
                               glare_regions=["top_left", "top_right",
                                              "bottom_left", "bottom_right"]))
    image_hash = store.put_image(data)
    result = observe(analyze(data, store, image_hash), store, image_hash)

    edges = {r: v for (r, c, d), v in result.detectability.items()
             if c == "edges" and d == "chipping"}
    assert edges, "no edge detectability at all"
    assert any(v < Scale.MODERATE for v in edges.values()), (
        f"every edge of a blown-out card reports assessable: {edges}")
