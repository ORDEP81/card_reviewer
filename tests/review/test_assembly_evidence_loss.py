"""Assembly must not lose evidence it was handed.

Two ways it did. Both were found by independent review, and both make a card
look better than the photographs support — the direction I2 exists to forbid.
"""

import pytest

from card_reviewer.review.assembly import (
    ImageEvidence, ImageStageOutputs, assemble, to_image_evidence,
)
from card_reviewer.review.enums import Provenance, Scale
from card_reviewer.review.roles import ImageRole, ResolvedRole


def _img(image_hash, centering, detectability=None):
    return ImageEvidence(
        image_hash=image_hash, detectability=detectability or {},
        reason_codes={}, sharpness=100.0, centering=centering,
        anomalies=[], evidence_refs={})


def _fronts(*hashes):
    return {h: ResolvedRole(image_hash=h, role=ImageRole.FRONT,
                            provenance=Provenance.SUPPLIED, confidence=1.0)
            for h in hashes}


UNMEASURABLE = {"measurable": False, "reason": "BORDER_NOT_SEPARABLE_FROM_ART"}
MISCUT = {"measurable": True, "horizontal": 78.0, "vertical": 50.0}


def test_a_measured_front_wins_over_an_unmeasurable_one():
    """`fronts[0]` took whichever photograph happened to be listed first, so
    a 78/22 miscut DISAPPEARED when the unmeasurable photo came first — and
    coverage still counted centering as assessed, with no limitation."""
    unmeasurable_first = assemble(
        [_img("a", UNMEASURABLE), _img("b", MISCUT)], _fronts("a", "b"))
    measurable_first = assemble(
        [_img("b", MISCUT), _img("a", UNMEASURABLE)], _fronts("a", "b"))

    assert measurable_first.centering["measurable"] is True
    assert unmeasurable_first.centering["measurable"] is True, (
        "the miscut vanished because of the order the photos were listed in")
    assert (unmeasurable_first.centering["horizontal"]
            == measurable_first.centering["horizontal"] == 78.0)


def test_the_result_does_not_depend_on_the_order_photographs_arrive_in():
    a, b = _img("a", MISCUT), _img("b", {"measurable": True,
                                         "horizontal": 52.0, "vertical": 50.0})
    forward = assemble([a, b], _fronts("a", "b")).centering
    backward = assemble([b, a], _fronts("a", "b")).centering
    assert forward == backward


def test_the_worst_measured_centering_is_the_one_reported():
    """Two photographs of one physical card disagreeing is recorded as a
    conflict, but the value carried forward must not be the flattering one:
    a card is as miscut as it is, and picking the kinder reading is the same
    mistake as picking the first."""
    kind = {"measurable": True, "horizontal": 52.0, "vertical": 50.0}
    harsh = {"measurable": True, "horizontal": 78.0, "vertical": 50.0}
    out = assemble([_img("a", kind), _img("b", harsh)], _fronts("a", "b"))
    assert out.centering["horizontal"] == 78.0


def test_no_front_leaves_centering_empty():
    from card_reviewer.review.roles import ImageRole as Role

    backs = {"a": ResolvedRole(image_hash="a", role=Role.BACK,
                               provenance=Provenance.SUPPLIED, confidence=1.0)}
    assert assemble([_img("a", MISCUT)], backs).centering == {}


def test_image_limitations_reach_the_assembled_evidence(tmp_path):
    """`to_image_evidence` never populated ImageEvidence.limitations, so
    `assembled.limitations` was STRUCTURALLY always empty — and the manifest
    sends it to the provider, which CLAUDE.md requires to carry the image
    limitations. The provider was reading "[]" on every card."""
    from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import (
        analyze as observability_analyze,
    )
    from card_reviewer.review.imaging.preflight import analyze as preflight_analyze
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    data = render_png(CardSpec(glare_regions=["top_left"]))
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    outputs = ImageStageOutputs(
        image_hash=image_hash,
        preflight=preflight_analyze(data).model_dump(),
        geometry=geometry.model_dump(),
        observability=observability_analyze(
            geometry, store, image_hash).model_dump(),
        cv_measurements=measure_all(geometry, store, image_hash).model_dump(),
    )
    evidence = to_image_evidence([outputs])[0]
    assert evidence.limitations, "a glared card reported no limitations at all"
    assert any("GLARE" in limitation for limitation in evidence.limitations)


def test_an_unmeasurable_reading_cannot_win_on_a_stale_number():
    """A declined measurement may still carry the field it failed to fill.
    Letting it into the comparison would let a value nothing stands behind
    outrank one that was actually measured."""
    stale = {"measurable": False, "reason": "BORDER_NOT_SEPARABLE_FROM_ART",
             "horizontal": 95.0, "vertical": 50.0}
    out = assemble([_img("a", stale), _img("b", MISCUT)], _fronts("a", "b"))

    assert out.centering["measurable"] is True
    assert out.centering["horizontal"] == 78.0, (
        "an unmeasurable reading's leftover number was carried forward")


def test_a_declined_centering_measurement_reaches_the_provider(tmp_path):
    """`limitations` was assembled BEFORE the centering downgrade that the
    same change added, so the list could never mention a failed centering
    measurement — and that list is what the manifest sends to the provider
    as image_limitations. The vision layer was never told the border could
    not be measured.
    """
    from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import (
        analyze as observability_analyze,
    )
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    # A tilt steep enough that the border is not separable from the art.
    data = render_png(CardSpec(rotation_deg=20.0))
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    cv = measure_all(geometry, store, image_hash)
    if cv.centering["measurable"]:
        pytest.skip("this fixture no longer declines")

    outputs = ImageStageOutputs(
        image_hash=image_hash, preflight={"global_sharpness": 120.0},
        geometry=geometry.model_dump(),
        observability=observability_analyze(
            geometry, store, image_hash).model_dump(),
        cv_measurements=cv.model_dump())
    evidence = to_image_evidence([outputs])[0]

    assert any("centering" in limitation
               for limitation in evidence.limitations), (
        f"no centering limitation among {evidence.limitations}")
