"""The smaller findings from the independent reviews, with their guards.

Each is minor on its own; each also fails silently, which is what earns it a
test rather than a note.
"""

import numpy as np
import pytest

from card_reviewer.review.enums import Coverage, FindingState, Mode, Scale
from card_reviewer.review.imaging.geometry import _order


# --- a quad with duplicate corners is not a quad ---------------------------

def test_corner_ordering_rejects_a_degenerate_quad():
    """Near 45 degrees `argmin(sum)` and `argmin(diff)` select the SAME
    point, so one corner is emitted twice and another dropped. Nothing
    checked, and getPerspectiveTransform returns a finite garbage homography
    rather than raising — so every measurement downstream describes a
    rectangle that was never on the card. Observed live at 40 and 44 degrees.
    """
    degenerate = np.array([[50, 0], [100, 50], [50, 100], [0, 50]],
                          dtype=np.float32)
    with pytest.raises(ValueError, match="distinct"):
        _order(degenerate)


def test_corner_ordering_accepts_an_ordinary_quad():
    ordered = _order(np.array([[10, 10], [90, 12], [92, 190], [8, 188]],
                              dtype=np.float32))
    assert ordered.shape == (4, 2)
    assert len({tuple(p) for p in ordered}) == 4


def test_a_steeply_tilted_card_declines_rather_than_measuring_a_ghost(tmp_path):
    """The end of that chain: at 44 degrees the rectified 'card' was 100%
    backdrop, yet reported usable with a reliable border."""
    from card_reviewer.review.imaging.geometry import analyze
    from card_reviewer.review.imaging.measure.centering import measure_centering
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    data = render_png(CardSpec(rotation_deg=44.0))
    result = analyze(data, store, store.put_image(data))
    if result.usable:
        assert measure_centering(result, store).measurable is False, (
            "a 44-degree tilt produced a centering number")


# --- routing's cheap path ---------------------------------------------------

def test_a_clean_card_still_triggers_a_confirming_vision_call():
    """The 'strong gem candidate' branch gated on `not findings`, but a real
    card produces many NOT_OBSERVED findings, so a single one suppressed it
    and the branch was effectively dead. What it means is that nothing is
    WRONG with the card — that is `not reasons`."""
    from card_reviewer.review.findings import Finding, FindingProducer
    from card_reviewer.review.policies.routing_v1 import decide_routing
    from card_reviewer.review.provenance import (
        EvidenceOrigin, EvidenceRef, NormalizedBox,
    )

    box = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)
    nothing_wrong = [
        Finding(defect_type="rounding", category="corners",
                state=FindingState.NOT_OBSERVED,
                producer=FindingProducer.HEURISTIC, confidence=0.9,
                psa10_relevant=True, location=box,
                evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                                      origin=EvidenceOrigin.ORIGINAL,
                                      view="corner_top_left", region=box)])
    ]
    decision = decide_routing(Mode.SMART, nothing_wrong, Coverage.SUFFICIENT, {})
    assert decision.call_vision is True
    assert any("gem" in reason for reason in decision.trigger_reasons)


def test_a_card_with_a_real_concern_is_not_called_a_gem_candidate():
    from card_reviewer.review.findings import Finding, FindingProducer
    from card_reviewer.review.policies.routing_v1 import decide_routing
    from card_reviewer.review.provenance import (
        EvidenceOrigin, EvidenceRef, NormalizedBox,
    )
    from card_reviewer.review.roles import ImageRole

    box = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)
    suspected = [
        Finding(defect_type="rounding", category="corners",
                state=FindingState.SUSPECTED,
                producer=FindingProducer.HEURISTIC, confidence=0.6,
                psa10_relevant=True, location=box,
                evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                                      origin=EvidenceOrigin.ORIGINAL,
                                      view="corner_top_left", region=box)])
    ]
    detectability = {(ImageRole.FRONT, "top_left", "corners", "rounding"):
                     Scale.HIGH}
    decision = decide_routing(Mode.SMART, suspected, Coverage.SUFFICIENT,
                              detectability)
    assert not any("gem" in reason for reason in decision.trigger_reasons)


# --- coverage must not invent a reason code --------------------------------

def test_coverage_does_not_guess_a_reason_it_was_never_given():
    """An unrecorded shortfall defaulted to LOW_RESOLUTION, generating a
    "higher-resolution close-up" request that may be wrong.
    `taxonomy.class_of` raises on unknown codes precisely so reason codes are
    not guessed; this bypassed that intent."""
    from card_reviewer.review.policies.coverage_v1 import (
        REQUIRED_FACES, evaluate_coverage,
    )
    from detectability_helpers import detectability_map, set_every_region

    detectability = detectability_map(REQUIRED_FACES)
    set_every_region(detectability, REQUIRED_FACES[0], "corners", "whitening",
                     Scale.LOW)

    result = evaluate_coverage(detectability, {}, {}, REQUIRED_FACES)
    codes = {lim.reason_code for lim in result.limitations}
    assert "LOW_RESOLUTION" not in codes, (
        "a shortfall with no recorded reason was reported as low resolution")
    assert codes, "the shortfall was dropped entirely"


# --- calibration needs the versions that produced the numbers --------------

def test_every_supporting_version_is_stamped_on_a_review():
    """Taxonomy, authority, relevance, scoring, fusion and canonicalization
    all change the numbers, and none was published on the review — leaving
    calibration to reconstruct them by joining stage rows."""
    from card_reviewer.review.versions import SUPPORTING_VERSIONS, effective_versions

    stamped = effective_versions()
    for component in SUPPORTING_VERSIONS:
        assert component in stamped, f"{component} is not stamped on a review"


def test_the_vision_dependent_categories_match_the_taxonomy():
    """The list said corners were CV-establishable; the taxonomy says the
    opposite and deliberately so. A category every one of whose defect types
    is INTERPRETIVE cannot be concluded on without vision, and saying
    otherwise would let a missing vision layer read as an assessed category.
    """
    from card_reviewer.review.pipeline import VISION_DEPENDENT_CATEGORIES
    from card_reviewer.review.taxonomy import (
        CATEGORIES, Promotion, defect_types_for, promotion_of,
    )

    for category in CATEGORIES:
        interpretive = all(
            promotion_of(category, defect_type) is Promotion.INTERPRETIVE
            for defect_type in defect_types_for(category)
        )
        assert interpretive is (category in VISION_DEPENDENT_CATEGORIES), (
            f"{category}: every defect type interpretive={interpretive}, but "
            f"vision-dependent={category in VISION_DEPENDENT_CATEGORIES}")


def test_a_non_finite_value_names_the_field_it_broke_on():
    """The guard already held — quantize's math.floor raises on NaN — but it
    raised "cannot convert float NaN to integer", naming neither the field
    nor the reason. Cache identity is correctness-critical, so its failures
    should say what broke."""
    from card_reviewer.review.canonical import canonicalize

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            canonicalize({"centering": {"horizontal": bad}})


def test_the_repository_protocol_describes_what_the_pipeline_calls():
    """Six methods the pipeline calls on a Repository-typed field were
    missing from the Protocol, so a type checker could not have caught a
    mismatched signature — the one thing it exists for."""
    from card_reviewer.review.storage.repository import Repository, SqliteRepository

    required = {name for name in vars(Repository) if not name.startswith("_")}
    for name in ("save_candidate", "save_image", "link_image",
                 "save_routing_decision", "save_review", "reviews_for"):
        assert name in required, f"{name} is not declared on the Protocol"
        assert hasattr(SqliteRepository, name), (
            f"{name} is declared but the real repository does not have it")
