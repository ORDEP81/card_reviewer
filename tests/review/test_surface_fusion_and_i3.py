"""Surface defects must be able to fuse, and I3 must not be launderable.

Two findings that compound each other:

  * Surface EvidenceRefs carried no region, so `_location_of` returned None,
    `_correlates` refused to merge, and one physical scratch reported by both
    producers was penalised TWICE — 69 against 94 on the reviewer's numbers.
    Surface is the largest category, 7 of the 14 defect types.

  * `to_image_evidence` handed every surface defect type ALL of the surface
    evidence refs, `surface_original` included, whether or not anything was
    visible there. `combine` fuses before enforcing I3 so the invariant sees
    the union of a defect's evidence — and that union always contained an
    unenhanced ref, so `i3_satisfied` returned True regardless.

The second was latent only because the first stopped surface findings fusing
at all. Fixing the double-penalty turns it on, so both are handled together.
"""

import pytest

from card_reviewer.review.enums import Coverage, FindingState
from card_reviewer.review.findings import Finding, FindingProducer, enforce_i3
from card_reviewer.review.fusion import fuse
from card_reviewer.review.provenance import (
    EvidenceOrigin, EvidenceRef, NormalizedBox,
)

CENTRE = NormalizedBox(x0=0.3, y0=0.3, x1=0.6, y1=0.6)
ELSEWHERE = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)


def _surface(producer, refs, location=CENTRE, state=FindingState.OBSERVED):
    return Finding(
        defect_type="scratches", category="surface", state=state,
        producer=producer, confidence=0.9, psa10_relevant=True,
        location=location, evidence=refs)


def _ref(view, origin=EvidenceOrigin.NORMALIZED, enhancement=None,
         region=CENTRE):
    return EvidenceRef(artifact_id=f"a-{view}", image_hash="h1", origin=origin,
                       enhancement=enhancement, view=view, region=region)


def test_one_scratch_seen_by_both_producers_is_one_defect():
    fused = fuse([
        _surface(FindingProducer.HEURISTIC, [_ref("surface_original")]),
        _surface(FindingProducer.VISION, [_ref("surface_original")]),
    ])
    assert len(fused) == 1, "one physical scratch counted twice"
    assert len(fused[0].sources) == 2


def test_scratches_in_different_places_stay_separate():
    """Fusing on category alone would suppress a real second flaw."""
    fused = fuse([
        _surface(FindingProducer.HEURISTIC, [_ref("surface_original")], CENTRE),
        _surface(FindingProducer.VISION,
                 [_ref("surface_original", region=ELSEWHERE)], ELSEWHERE),
    ])
    assert len(fused) == 2


def test_enhancement_only_evidence_cannot_be_laundered_by_a_neighbour():
    """I3: a feature visible only under enhancement must not reach `observed`
    on the strength of a DIFFERENT finding's unenhanced view."""
    enhancement_only = _surface(
        FindingProducer.VISION,
        [_ref("surface_clahe", EvidenceOrigin.ENHANCED, "clahe")])
    checked = enforce_i3([enhancement_only])
    assert checked[0].state is not FindingState.OBSERVED
    assert checked[0].demotion_reason


def test_a_finding_with_its_own_unenhanced_view_survives_i3():
    plainly_visible = _surface(
        FindingProducer.VISION,
        [_ref("surface_clahe", EvidenceOrigin.ENHANCED, "clahe"),
         _ref("surface_original")])
    assert enforce_i3([plainly_visible])[0].state is FindingState.OBSERVED


def test_a_surface_defect_type_is_not_handed_every_surface_view(tmp_path):
    """The producer wiring behind the laundering: an unenhanced ref was
    attached to every surface defect type regardless of what it showed."""
    from card_reviewer.review.assembly import ImageStageOutputs, to_image_evidence
    from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import (
        analyze as observability_analyze,
    )
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    data = render_png(CardSpec(scratches=[1.0]))
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    outputs = ImageStageOutputs(
        image_hash=image_hash, preflight={"global_sharpness": 120.0},
        geometry=geometry.model_dump(),
        observability=observability_analyze(
            geometry, store, image_hash).model_dump(),
        cv_measurements=measure_all(geometry, store, image_hash).model_dump(),
    )
    evidence = to_image_evidence([outputs])[0]

    for key, refs in evidence.evidence_refs.items():
        if not key.startswith("surface:"):
            continue
        unenhanced = [r for r in refs if r.origin is not EvidenceOrigin.ENHANCED]
        assert len(unenhanced) <= 1, (
            f"{key} carries {len(unenhanced)} unenhanced refs; any one of "
            "them satisfies I3 for the whole group")


def _measured(tmp_path, spec):
    from card_reviewer.review.assembly import ImageStageOutputs, to_image_evidence
    from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import (
        analyze as observability_analyze,
    )
    from card_reviewer.review.imaging.synthetic import render_png
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    data = render_png(spec)
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    return store, ImageStageOutputs(
        image_hash=image_hash, preflight={"global_sharpness": 120.0},
        geometry=geometry.model_dump(),
        observability=observability_analyze(
            geometry, store, image_hash).model_dump(),
        cv_measurements=measure_all(geometry, store, image_hash).model_dump(),
    )


def test_a_real_surface_ref_carries_a_region(tmp_path):
    """The gap the hand-built tests above cannot see. Surface EvidenceRefs
    had no region, so every surface finding the pipeline produced had
    location=None — and `_correlates` refuses to merge without one, so the
    fusion that stops double-penalising could never fire for the largest
    category."""
    from card_reviewer.review.imaging.synthetic import CardSpec

    _, outputs = _measured(tmp_path, CardSpec(scratches=[1.0]))
    from card_reviewer.review.imaging.measure import CvMeasurements

    cv = CvMeasurements.model_validate(outputs.cv_measurements)
    assert cv.surface.evidence_refs, "no surface evidence at all"
    assert any(ref.region is not None for ref in cv.surface.evidence_refs), (
        "no surface ref carries a region, so no surface finding can fuse")
