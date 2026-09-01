import json

import pytest

from card_reviewer.review.imaging.geometry import GeometryResult
from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.measure.corners import measure_corners
from card_reviewer.review.imaging.measure.edges import measure_edges
from card_reviewer.review.imaging.measure.surface import measure_surface
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.provenance import EvidenceOrigin
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


def _geom(spec, store, h="h1"):
    return geom(render_png(spec), store, h)


# --- corners ---------------------------------------------------------------

def test_four_corner_crops_are_produced_per_image(store):
    assert len(measure_corners(_geom(CardSpec(), store), store, "h1").crops) == 4


def test_corner_crops_land_in_the_measurement_owned_directory(store):
    r = measure_corners(_geom(CardSpec(), store), store, "h1")
    assert "/corners/" in str(store.path_of(next(iter(r.crops.values()))))


def test_corner_damage_produces_an_anomaly_candidate_not_a_defect(store):
    spec = CardSpec(border_color=(20, 20, 20), corner_damage={"bottom_left": 0.9})
    r = measure_corners(_geom(spec, store), store, "h1")
    assert any(a["region"] == "bottom_left" for a in r.anomalies)
    for a in r.anomalies:
        assert a["kind"] == "candidate"
        assert "defect" not in a


def test_a_clean_dark_card_yields_no_corner_anomalies(store):
    r = measure_corners(_geom(CardSpec(border_color=(20, 20, 20)), store),
                        store, "h1")
    assert r.anomalies == []


def test_every_anomaly_carries_a_confidence_and_severity(store):
    """The heuristic compares confidence against a threshold; an absent key
    defaults to 0.0, which would stop any CV finding ever reaching
    `observed` in the real pipeline."""
    spec = CardSpec(border_color=(20, 20, 20), corner_damage={"bottom_left": 0.9})
    for a in measure_corners(_geom(spec, store), store, "h1").anomalies:
        assert 0.0 <= a["confidence"] <= 1.0
        assert a["severity"] in {"minor", "moderate", "severe"}


def test_corner_evidence_refs_record_normalized_origin_not_enhanced(store):
    r = measure_corners(_geom(CardSpec(), store), store, "h1")
    assert all(ref.origin is EvidenceOrigin.NORMALIZED for ref in r.evidence_refs)


def test_each_corner_ref_carries_its_own_distinct_region(store):
    """Fusion correlates on overlapping regions, so two corners must not
    share a location — otherwise damage at opposite corners merges into one
    defect and the card is charged once for two flaws.

    Asserted on the regions alone: including the view name in the key would
    make the set distinct whatever the boxes were.
    """
    r = measure_corners(_geom(CardSpec(), store), store, "h1")
    boxes = {(ref.region.x0, ref.region.y0, ref.region.x1, ref.region.y1)
             for ref in r.evidence_refs}
    assert len(boxes) == 4
    top_left = next(ref for ref in r.evidence_refs if ref.view == "corner_top_left")
    bottom_right = next(ref for ref in r.evidence_refs
                        if ref.view == "corner_bottom_right")
    assert not top_left.region.overlaps(bottom_right.region)


def test_unusable_geometry_produces_no_corner_crops(store):
    r = measure_corners(GeometryResult(boundary_confidence=0.1), store, "h1")
    assert r.crops == {} and r.anomalies == []


# --- edges -----------------------------------------------------------------

def test_four_edge_strips_are_produced_per_image(store):
    assert len(measure_edges(_geom(CardSpec(), store), store, "h1").crops) == 4


def test_edge_crops_land_in_their_own_directory(store):
    r = measure_edges(_geom(CardSpec(), store), store, "h1")
    assert "/edges/" in str(store.path_of(next(iter(r.crops.values()))))


# --- surface ---------------------------------------------------------------

def test_the_unenhanced_original_is_always_preserved_alongside(store):
    """Non-negotiable rule 7, and the corroboration route I3 depends on:
    without a stored unenhanced view, nothing can ever confirm an
    enhancement-surfaced anomaly."""
    r = measure_surface(_geom(CardSpec(seed=4), store), store, "h1")
    assert any(ref.origin is EvidenceOrigin.NORMALIZED for ref in r.evidence_refs)
    assert "original" in r.crops
    assert store.read(r.crops["original"])


def test_every_enhanced_view_records_its_method(store):
    r = measure_surface(_geom(CardSpec(seed=4), store), store, "h1")
    enhanced = [ref for ref in r.evidence_refs
                if ref.origin is EvidenceOrigin.ENHANCED]
    assert enhanced
    assert all(ref.enhancement for ref in enhanced)


def test_enhancement_parameters_are_reproducible(tmp_path):
    a_store = ArtifactStore(tmp_path / "a")
    b_store = ArtifactStore(tmp_path / "b")
    a = measure_surface(_geom(CardSpec(seed=4), a_store), a_store, "h1")
    b = measure_surface(_geom(CardSpec(seed=4), b_store), b_store, "h1")
    assert {ref.enhancement for ref in a.evidence_refs} == {
        ref.enhancement for ref in b.evidence_refs}


def test_anomalies_record_the_enhancement_level_that_surfaced_them(store):
    r = measure_surface(_geom(CardSpec(seed=4), store), store, "h1")
    for a in r.anomalies:
        assert "surfaced_by" in a
        assert "visible_in_original" in a


# --- the aggregate ---------------------------------------------------------

def test_measure_all_produces_one_json_serializable_document(store):
    """cv_measurements is cached as JSON, so its aggregate must serialize."""
    from card_reviewer.review.canonical import canonicalize
    from card_reviewer.review.imaging.measure import CvMeasurements, measure_all

    m = measure_all(_geom(CardSpec(), store), store, "h1")
    revived = CvMeasurements.model_validate(json.loads(m.model_dump_json()))
    assert revived == m
    assert canonicalize(m.model_dump(mode="json"))


def test_the_aggregate_collects_anomalies_from_every_region(store):
    from card_reviewer.review.imaging.measure import measure_all

    spec = CardSpec(border_color=(20, 20, 20), corner_damage={"bottom_left": 0.9})
    m = measure_all(_geom(spec, store), store, "h1")
    assert any(a["category"] == "corners" for a in m.anomalies)


def test_the_aggregate_carries_the_centering_measurement(store):
    from card_reviewer.review.imaging.measure import measure_all

    m = measure_all(_geom(CardSpec(), store), store, "h1")
    assert m.centering["measurable"] is True
