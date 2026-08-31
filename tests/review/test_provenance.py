import pytest
from pydantic import ValidationError

from card_reviewer.review.provenance import (
    EvidenceOrigin,
    EvidenceRef,
    NormalizedBox,
)


def _ref(**kw):
    base = dict(
        artifact_id="a1",
        image_hash="h1",
        origin=EvidenceOrigin.ORIGINAL,
        view="front_face",
    )
    return EvidenceRef(**(base | kw))


def test_enhanced_evidence_must_declare_its_enhancement():
    with pytest.raises(ValidationError, match="enhancement"):
        _ref(origin=EvidenceOrigin.ENHANCED, enhancement=None)


def test_unenhanced_evidence_must_not_declare_an_enhancement():
    with pytest.raises(ValidationError, match="enhancement"):
        _ref(origin=EvidenceOrigin.NORMALIZED, enhancement="clahe:clip=2.0")


def test_enhanced_evidence_with_a_method_is_valid():
    ref = _ref(origin=EvidenceOrigin.ENHANCED, enhancement="clahe:clip=2.0,grid=8")
    assert ref.is_enhanced is True


def test_original_and_normalized_both_count_as_unenhanced():
    assert _ref(origin=EvidenceOrigin.ORIGINAL).is_enhanced is False
    assert _ref(origin=EvidenceOrigin.NORMALIZED).is_enhanced is False


def test_normalized_box_rejects_coordinates_outside_the_unit_square():
    with pytest.raises(ValidationError):
        NormalizedBox(x0=0.1, y0=0.1, x1=1.4, y1=0.5)


def test_normalized_box_rejects_inverted_corners():
    with pytest.raises(ValidationError, match="x1 must exceed x0"):
        NormalizedBox(x0=0.6, y0=0.1, x1=0.2, y1=0.5)


def test_boxes_detect_overlap_for_the_i1_contradiction_test():
    a = NormalizedBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
    b = NormalizedBox(x0=0.4, y0=0.4, x1=0.9, y1=0.9)
    c = NormalizedBox(x0=0.6, y0=0.6, x1=0.9, y1=0.9)
    assert a.overlaps(b) is True
    assert a.overlaps(c) is False


def test_overlap_is_symmetric():
    a = NormalizedBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
    b = NormalizedBox(x0=0.4, y0=0.4, x1=0.9, y1=0.9)
    assert a.overlaps(b) == b.overlaps(a)


def test_an_evidence_ref_round_trips_through_json():
    """Refs are embedded in cached stage outputs, so they must serialize."""
    import json

    ref = _ref(origin=EvidenceOrigin.ENHANCED, enhancement="sharpen:amount=1.5",
               region=NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2))
    revived = EvidenceRef.model_validate(json.loads(ref.model_dump_json()))
    assert revived == ref
