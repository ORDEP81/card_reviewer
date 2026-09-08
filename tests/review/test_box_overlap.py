"""`overlaps` decides whether two findings are one defect.

It is I1's and fusion's correlation test (`fusion.py`, `combine_v1.py`) and
it had no test of its own: mutating `<=` to `<` — which makes boxes that
merely touch count as overlapping — survived the entire suite.

Touching is not overlapping. Adjacent corner crops share an edge exactly,
so treating contact as overlap would fuse two distinct corners into one
defect and suppress a real flaw.
"""

from card_reviewer.review.provenance import NormalizedBox


def box(x0, y0, x1, y1):
    return NormalizedBox(x0=x0, y0=y0, x1=x1, y1=y1)


def test_boxes_that_only_touch_do_not_overlap():
    left = box(0.0, 0.0, 0.5, 1.0)
    right = box(0.5, 0.0, 1.0, 1.0)
    assert not left.overlaps(right)
    assert not right.overlaps(left)


def test_boxes_that_touch_along_a_horizontal_edge_do_not_overlap():
    """The vertical axis needs its own case: this project has already
    shipped a bug where one axis worked and the other did not."""
    top = box(0.0, 0.0, 1.0, 0.5)
    bottom = box(0.0, 0.5, 1.0, 1.0)
    assert not top.overlaps(bottom)
    assert not bottom.overlaps(top)


def test_boxes_sharing_only_a_corner_point_do_not_overlap():
    assert not box(0.0, 0.0, 0.5, 0.5).overlaps(box(0.5, 0.5, 1.0, 1.0))


def test_a_hair_of_genuine_overlap_counts():
    assert box(0.0, 0.0, 0.5, 1.0).overlaps(box(0.49, 0.0, 1.0, 1.0))


def test_containment_counts_in_both_directions():
    outer, inner = box(0.0, 0.0, 1.0, 1.0), box(0.4, 0.4, 0.6, 0.6)
    assert outer.overlaps(inner) and inner.overlaps(outer)


def test_a_box_overlaps_itself():
    corner = box(0.0, 0.0, 0.2, 0.2)
    assert corner.overlaps(corner)


def test_separated_boxes_do_not_overlap():
    assert not box(0.0, 0.0, 0.2, 0.2).overlaps(box(0.8, 0.8, 1.0, 1.0))
