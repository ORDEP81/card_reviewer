"""Producer -> consumer contract tests for Phase 1.

The failure mode these exist to catch: a producer changes shape and its
consumer silently still expects the old one. Independently constructed
fixtures cannot catch that, so every test here feeds a REAL producer's
output into a REAL consumer, through serialization where the boundary is
a persisted one.
"""

import json

from pydantic import BaseModel

from card_reviewer.review.canonical import canonicalize
from card_reviewer.review.enums import FindingState, Scale
from card_reviewer.review.findings import Finding, FindingProducer, enforce_i3
from card_reviewer.review.fingerprint import fingerprint
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef, NormalizedBox
from card_reviewer.review.taxonomy import DEFECT_TYPES, defect_types_for


def _enhanced_finding() -> Finding:
    return Finding(
        defect_type="scratches", category="surface",
        state=FindingState.OBSERVED, producer=FindingProducer.HEURISTIC,
        confidence=0.95, psa10_relevant=True,
        location=NormalizedBox(x0=0.1, y0=0.1, x1=0.4, y1=0.4),
        evidence=[EvidenceRef(artifact_id="a1", image_hash="h1",
                              origin=EvidenceOrigin.ENHANCED,
                              enhancement="clahe:clip=2.0", view="surface_clahe")],
    )


def test_i3_verdict_survives_a_json_round_trip():
    """combine reads findings back out of a cached stage result. If the
    origin did not survive serialization, an enhancement-only finding would
    revive as confirmable and I3 would be silently defeated."""
    original = _enhanced_finding()
    revived = Finding.model_validate(json.loads(original.model_dump_json()))
    assert revived.evidence[0].origin is EvidenceOrigin.ENHANCED
    assert enforce_i3([revived])[0].state is FindingState.SUSPECTED


def test_a_demoted_finding_keeps_its_reason_through_serialization():
    demoted = enforce_i3([_enhanced_finding()])[0]
    revived = Finding.model_validate(json.loads(demoted.model_dump_json()))
    assert "I3" in revived.demotion_reason


class _StageOutput(BaseModel):
    """Stands in for any cached stage output carrying a Scale."""

    detectability: Scale
    suitability: dict[str, Scale] = {}


def _as_stage_runner_would(model: BaseModel) -> dict:
    """Exactly the serialization path StageRunner uses: model_dump(mode=json)
    then json.dumps into the stage_result row."""
    return json.loads(json.dumps(model.model_dump(mode="json")))


def test_a_scale_field_persists_as_its_label_not_an_integer():
    """Scale is an IntEnum, so the default dump would be an int. Cached
    outputs must carry one readable representation, not two."""
    stored = _as_stage_runner_would(
        _StageOutput(detectability=Scale.MODERATE,
                     suitability={"surface": Scale.LOW})
    )
    assert stored["detectability"] == "moderate"
    assert stored["suitability"]["surface"] == "low"


def test_a_scale_field_revives_from_its_persisted_label_with_order_intact():
    original = _StageOutput(detectability=Scale.MODERATE,
                            suitability={"surface": Scale.LOW})
    revived = _StageOutput.model_validate(_as_stage_runner_would(original))
    assert revived == original
    assert revived.detectability is Scale.MODERATE
    assert revived.detectability >= Scale.MODERATE
    assert revived.suitability["surface"] < revived.detectability


def test_a_persisted_scale_canonicalizes_without_an_unregistered_float():
    """The cache boundary and the fingerprint boundary must agree."""
    assert canonicalize(_as_stage_runner_would(_StageOutput(detectability=Scale.HIGH)))


def test_a_finding_fingerprints_identically_before_and_after_a_round_trip():
    """A cache hit and a fresh computation must be indistinguishable, or every
    re-review recomputes."""
    original = _enhanced_finding()
    revived = Finding.model_validate(json.loads(original.model_dump_json()))
    assert fingerprint(original.model_dump(mode="json")) == fingerprint(
        revived.model_dump(mode="json")
    )


def test_every_finding_the_taxonomy_declares_is_constructible():
    """The taxonomy names defect types; Finding must accept every one, or a
    measurement stage could emit something the vocabulary cannot carry."""
    for spec in DEFECT_TYPES.values():
        f = Finding(
            defect_type=spec.name, category=spec.category,
            state=FindingState.SUSPECTED, producer=FindingProducer.HEURISTIC,
            confidence=0.5, psa10_relevant=True,
            evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                                  origin=EvidenceOrigin.NORMALIZED, view="v")],
        )
        assert f.defect_type in defect_types_for(spec.category)


def test_a_pydantic_dump_of_a_finding_canonicalizes_without_error():
    """canonicalize rejects non-string keys. Feeding it a real model dump is
    the only way to know the model has none."""
    assert canonicalize(_enhanced_finding().model_dump(mode="json"))


def test_two_findings_differing_only_in_confidence_below_precision_agree():
    """Confidence quantizes to 0.01, so 0.9500 and 0.9503 are the same
    evidence and must not produce different cache keys."""
    a = _enhanced_finding().model_copy(update={"confidence": 0.9500})
    b = _enhanced_finding().model_copy(update={"confidence": 0.9503})
    assert fingerprint(a.model_dump(mode="json")) == fingerprint(
        b.model_dump(mode="json")
    )
