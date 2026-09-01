import pytest

from card_reviewer.review.enums import FindingState
from card_reviewer.review.findings import enforce_i3
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef
from card_reviewer.review.vision.provider import (
    Assessment, FakeProvider, GemView, ProviderContractError,
    parse_assessment, resolve_vision_findings,
)


def _payload(**kw):
    base = {
        "findings": [{
            "defect_type": "print_lines", "category": "surface",
            "state": "suspected", "confidence": 0.6, "psa10_relevant": True,
            "evidence_artifact_ids": ["a1"], "explanation": "faint line",
        }],
        "category_assessability": {"centering": True, "corners": True,
                                   "edges": True, "surface": True},
        "gem_view": "possible_psa10_disqualifier",
    }
    return base | kw


def _ref(origin=EvidenceOrigin.ENHANCED, enhancement="clahe:clip=2.0"):
    return EvidenceRef(artifact_id="a1", image_hash="realhash", origin=origin,
                       enhancement=enhancement, view="surface_clahe")


# --- contract --------------------------------------------------------------

def test_a_well_formed_response_parses():
    assert parse_assessment(_payload(), {"a1"}).findings[0].defect_type == (
        "print_lines")


def test_citing_an_artifact_not_in_the_manifest_is_a_contract_violation():
    """A provider that cites an id it was never sent is not to be trusted."""
    with pytest.raises(ProviderContractError, match="not in the manifest"):
        parse_assessment(_payload(), {"other"})


def test_every_category_must_report_assessability():
    with pytest.raises(ProviderContractError, match="assessability"):
        parse_assessment(_payload(category_assessability={"centering": True}),
                         {"a1"})


def test_a_malformed_state_is_rejected_rather_than_coerced():
    bad = _payload(findings=[{
        "defect_type": "x", "category": "surface", "state": "definitely_bad",
        "confidence": 0.9, "psa10_relevant": True,
        "evidence_artifact_ids": ["a1"], "explanation": ""}])
    with pytest.raises(ProviderContractError):
        parse_assessment(bad, {"a1"})


def test_a_missing_gem_view_is_rejected():
    bad = _payload()
    del bad["gem_view"]
    with pytest.raises(ProviderContractError, match="gem_view"):
        parse_assessment(bad, {"a1"})


def test_insufficient_evidence_is_a_first_class_answer():
    a = parse_assessment(
        _payload(gem_view="insufficient_evidence",
                 category_assessability={"centering": True, "corners": True,
                                         "edges": True, "surface": False}),
        {"a1"})
    assert a.category_assessability["surface"] is False


def test_severity_and_location_survive_parsing():
    payload = _payload(findings=[{
        "defect_type": "scratches", "category": "surface", "state": "observed",
        "confidence": 0.9, "psa10_relevant": True,
        "evidence_artifact_ids": ["a1"], "severity": "moderate",
        "location": {"x0": 0.1, "y0": 0.1, "x1": 0.3, "y1": 0.3},
        "explanation": ""}])
    a = parse_assessment(payload, {"a1"})
    assert a.findings[0].severity.value == "moderate"
    assert a.findings[0].location.x1 == 0.3


# --- provenance across the round trip --------------------------------------

def test_a_cited_enhanced_artifact_keeps_its_enhanced_origin():
    """I3 must survive the round trip. Rebuilding this ref as ORIGINAL would
    silently launder an enhancement-only finding into a rejectable one."""
    a = parse_assessment(_payload(), {"a1"})
    got = resolve_vision_findings(a, {"a1": _ref()})[0].evidence[0]
    assert got.origin is EvidenceOrigin.ENHANCED
    assert got.enhancement == "clahe:clip=2.0"
    assert got.image_hash == "realhash"


def test_an_observed_finding_citing_only_enhanced_evidence_is_demoted():
    payload = _payload(findings=[{
        "defect_type": "scratches", "category": "surface", "state": "observed",
        "confidence": 0.99, "psa10_relevant": True,
        "evidence_artifact_ids": ["a1"], "explanation": ""}])
    a = parse_assessment(payload, {"a1"})
    resolved = enforce_i3(resolve_vision_findings(a, {"a1": _ref()}))
    assert resolved[0].state is FindingState.SUSPECTED
    assert "I3" in resolved[0].demotion_reason


def test_an_unresolvable_cited_id_raises_rather_than_defaulting():
    a = parse_assessment(_payload(), {"a1"})
    with pytest.raises(ProviderContractError, match="not in the manifest"):
        resolve_vision_findings(a, {})


def test_a_resolved_finding_is_marked_as_produced_by_vision():
    """CV and Claude findings must remain independently recoverable."""
    from card_reviewer.review.findings import FindingProducer

    a = parse_assessment(_payload(), {"a1"})
    assert resolve_vision_findings(a, {"a1": _ref()})[0].producer is (
        FindingProducer.VISION)


def test_a_location_is_derived_from_cited_regions_when_the_provider_omits_it():
    """Fusion and the contradiction test both need a location."""
    from card_reviewer.review.provenance import NormalizedBox

    ref = EvidenceRef(artifact_id="a1", image_hash="h",
                      origin=EvidenceOrigin.NORMALIZED, view="corner_top_left",
                      region=NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2))
    a = parse_assessment(_payload(), {"a1"})
    assert resolve_vision_findings(a, {"a1": ref})[0].location is not None


# --- provider signature ----------------------------------------------------

def test_the_fake_provider_exposes_a_signature_with_the_declared_keys():
    sig = FakeProvider(Assessment(
        category_assessability={"centering": True, "corners": True,
                                "edges": True, "surface": True},
        gem_view=GemView.NO_DISQUALIFIER)).signature()
    assert set(sig) == {"provider", "model", "prompt_version", "inference_params"}


def test_a_signature_change_is_visible_to_the_cache():
    from card_reviewer.review.fingerprint import signature_for

    base = Assessment(category_assessability={"centering": True, "corners": True,
                                              "edges": True, "surface": True},
                      gem_view=GemView.NO_DISQUALIFIER)
    a = signature_for("vision", FakeProvider(base, model="m1").signature())
    b = signature_for("vision", FakeProvider(base, model="m2").signature())
    assert a != b
