import json

import pytest

from card_reviewer.review.assembly import (
    Assembled, ImageEvidence, ImageStageOutputs, assemble, to_image_evidence,
)
from card_reviewer.review.enums import Provenance, Scale
from card_reviewer.review.roles import ImageRole, ResolvedRole


def _role(h, role):
    return ResolvedRole(image_hash=h, role=role, provenance=Provenance.SUPPLIED,
                        confidence=1.0)


def _img(h, det=None, reasons=None, sharpness=100.0, centering=None):
    return ImageEvidence(
        image_hash=h, detectability=det or {}, reason_codes=reasons or {},
        sharpness=sharpness,
        centering=centering or {"measurable": True, "horizontal": 52.0},
        anomalies=[], evidence_refs={})


def test_a_corner_glared_in_one_photo_and_clear_in_another_is_observable():
    """Merging detectability across images is the point of this stage."""
    a = _img("h1", {("bottom_left", "corners", "rounding"): Scale.LOW})
    b = _img("h2", {("bottom_left", "corners", "rounding"): Scale.HIGH})
    out = assemble([a, b], {"h1": _role("h1", ImageRole.FRONT),
                            "h2": _role("h2", ImageRole.FRONT)})
    assert out.detectability[(ImageRole.FRONT, "bottom_left", "corners", "rounding")] is Scale.HIGH


def test_assembly_records_which_image_established_a_value():
    a = _img("h1", {("bottom_left", "corners", "rounding"): Scale.LOW})
    b = _img("h2", {("bottom_left", "corners", "rounding"): Scale.HIGH})
    out = assemble([a, b], {"h1": _role("h1", ImageRole.FRONT),
                            "h2": _role("h2", ImageRole.FRONT)})
    assert out.provenance[(ImageRole.FRONT, "bottom_left", "corners", "rounding")] == "h2"


def test_reason_codes_survive_assembly_so_coverage_can_classify_them():
    """Without this, WHITE_BORDER never reaches the coverage policy and every
    structural limitation is misread as circumstantial."""
    a = _img("h1", {("bottom_left", "corners", "whitening"): Scale.LOW},
             {("bottom_left", "corners", "whitening"): "WHITE_BORDER"})
    out = assemble([a], {"h1": _role("h1", ImageRole.FRONT)})
    assert out.reason_codes[(ImageRole.FRONT, "bottom_left", "corners", "whitening")] == (
        "WHITE_BORDER")


def test_a_reason_code_is_dropped_once_another_photo_resolves_the_defect():
    """If one photo shows the corner clearly, the other's glare is no longer
    a limitation on this card."""
    glared = _img("h1", {("bottom_left", "corners", "rounding"): Scale.LOW},
                  {("bottom_left", "corners", "rounding"): "GLARE"})
    clear = _img("h2", {("bottom_left", "corners", "rounding"): Scale.HIGH})
    out = assemble([glared, clear], {"h1": _role("h1", ImageRole.FRONT),
                                     "h2": _role("h2", ImageRole.FRONT)})
    assert (ImageRole.FRONT, "corners", "rounding") not in out.reason_codes


def test_the_sharpest_front_is_selected_for_surface_work():
    dull = _img("h1", sharpness=20.0)
    sharp = _img("h2", sharpness=300.0)
    out = assemble([dull, sharp], {"h1": _role("h1", ImageRole.FRONT),
                                   "h2": _role("h2", ImageRole.FRONT)})
    assert out.best_for["surface"] == "h2"


def test_conflicting_measurements_are_preserved_not_averaged():
    a = _img("h1", centering={"measurable": True, "horizontal": 52.0})
    b = _img("h2", centering={"measurable": True, "horizontal": 61.0})
    out = assemble([a, b], {"h1": _role("h1", ImageRole.FRONT),
                            "h2": _role("h2", ImageRole.FRONT)})
    assert len(out.conflicts) == 1
    assert 52.0 in out.conflicts[0]["values"] and 61.0 in out.conflicts[0]["values"]


def test_close_measurements_are_not_reported_as_a_conflict():
    a = _img("h1", centering={"measurable": True, "horizontal": 52.0})
    b = _img("h2", centering={"measurable": True, "horizontal": 53.0})
    out = assemble([a, b], {"h1": _role("h1", ImageRole.FRONT),
                            "h2": _role("h2", ImageRole.FRONT)})
    assert out.conflicts == []


def test_unknown_role_images_contribute_only_face_independent_work():
    """An image whose face could not be established must not add detectability
    under ANY face — including `unknown` itself, which coverage would then
    have to interpret. Its anomalies still travel, because a defect is a
    defect wherever it was photographed.
    """
    unknown = _img("h1", {("center", "surface", "scratches"): Scale.HIGH})
    unknown.anomalies = [{"category": "surface", "defect_type": "scratches",
                          "confidence": 0.9}]
    out = assemble([unknown], {"h1": _role("h1", ImageRole.UNKNOWN)})
    assert out.detectability == {}
    assert out.faces_present == []
    assert len(out.anomalies) == 1


def test_faces_present_reports_only_confidently_resolved_faces():
    """Stored as label strings so the output is JSON-cacheable."""
    out = assemble([_img("h1")], {"h1": _role("h1", ImageRole.FRONT)})
    assert out.faces_present == ["front"]
    assert out.faces == (ImageRole.FRONT,)


def test_the_output_round_trips_and_canonicalizes():
    from card_reviewer.review.canonical import canonicalize

    a = _img("h1", {("bottom_left", "corners", "whitening"): Scale.LOW},
             {("bottom_left", "corners", "whitening"): "WHITE_BORDER"})
    out = assemble([a], {"h1": _role("h1", ImageRole.FRONT)})
    assert Assembled.model_validate(json.loads(out.model_dump_json())) == out
    assert canonicalize(out.model_dump(mode="json"))


# --- the bridge from measurements to the heuristic ------------------------

@pytest.fixture
def measured(tmp_path):
    from card_reviewer.review.imaging.geometry import analyze as geom
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import analyze as obs
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path)
    spec = CardSpec(border_color=(20, 20, 20), corner_damage={"bottom_left": 0.9})
    g = geom(render_png(spec), store, "h1")
    return ImageStageOutputs(
        image_hash="h1", preflight={"global_sharpness": 120.0},
        geometry=g.model_dump(),
        observability=obs(g, store, "h1").model_dump(),
        cv_measurements=measure_all(g, store, "h1").model_dump(),
    )


def test_the_bridge_produces_refs_under_the_keys_the_heuristic_looks_up(measured):
    """If these keys do not line up, no CV finding is ever emitted and every
    card silently passes — with no error anywhere."""
    evidence = to_image_evidence([measured])[0]
    assert "corners:rounding" in evidence.evidence_refs
    assert "centering:border_ratio" in evidence.evidence_refs


def test_the_bridge_carries_detectability_and_reason_codes_through(measured):
    evidence = to_image_evidence([measured])[0]
    assert evidence.detectability
    assert any(c == "corners" for (_r, c, _d) in evidence.detectability)


def test_the_bridge_carries_anomalies_with_their_confidence(measured):
    evidence = to_image_evidence([measured])[0]
    assert evidence.anomalies
    assert all("confidence" in a for a in evidence.anomalies)


def test_the_bridge_skips_an_image_whose_geometry_failed(measured):
    """One bad photograph out of six must not fail the card."""
    broken = ImageStageOutputs(image_hash="h2", preflight={"usable": False})
    assert [e.image_hash for e in to_image_evidence([measured, broken])] == ["h1"]


def test_region_specific_refs_keep_corners_apart(measured):
    """The heuristic prefers a region-scoped key so an anomaly's location is
    its own corner, not the union of all four."""
    evidence = to_image_evidence([measured])[0]
    assert "corners:rounding:bottom_left" in evidence.evidence_refs
    bl = evidence.evidence_refs["corners:rounding:bottom_left"]
    tr = evidence.evidence_refs["corners:rounding:top_right"]
    assert not bl[0].region.overlaps(tr[0].region)


def test_a_clean_corner_never_speaks_for_a_glared_one():
    """Best-of is across images OF THE SAME REGION. Across regions it is not
    best-of at all: this used to keep the maximum over every corner, so one
    clear corner made a glared one read as fully assessable and the GLARE
    reason was dropped."""
    glared = {("bottom_left", "corners", "rounding"): Scale.LOW,
              ("top_right", "corners", "rounding"): Scale.HIGH}
    out = assemble([_img("h1", glared,
                         {("bottom_left", "corners", "rounding"): "GLARE"})],
                   {"h1": _role("h1", ImageRole.FRONT)})

    assert out.detectability[
        (ImageRole.FRONT, "bottom_left", "corners", "rounding")] is Scale.LOW
    assert out.detectability[
        (ImageRole.FRONT, "top_right", "corners", "rounding")] is Scale.HIGH
    assert out.reason_codes[
        (ImageRole.FRONT, "bottom_left", "corners", "rounding")] == "GLARE"
