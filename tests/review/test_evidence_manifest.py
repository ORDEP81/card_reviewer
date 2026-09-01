import json

import pytest

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.assembly import Assembled
from card_reviewer.review.enums import Mode, Scale
from card_reviewer.review.manifest import BUDGETS, BuiltManifest, build_manifest
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef
from card_reviewer.review.roles import ImageRole


def _refs(n, origin=EvidenceOrigin.NORMALIZED):
    return [EvidenceRef(artifact_id=f"a{i}", image_hash="h", origin=origin,
                        enhancement="clahe:clip=2.0"
                        if origin is EvidenceOrigin.ENHANCED else None,
                        view=f"corner_{i}")
            for i in range(n)]


def _assembled(refs, **kw):
    base = dict(evidence_refs={"corners:rounding": refs},
                detectability_flat={
                    Assembled.key(ImageRole.FRONT, "corners", "rounding"):
                    Scale.HIGH.label},
                reason_codes_flat={},
                centering={"measurable": True, "horizontal": 52.0},
                anomalies=[], conflicts=[], limitations=[])
    return Assembled(**(base | kw))


def test_smart_and_deep_have_different_declared_budgets():
    assert BUDGETS[Mode.SMART] < BUDGETS[Mode.DEEP]


def test_selection_respects_the_mode_budget():
    m = build_manifest(_assembled(_refs(40)), Mode.SMART, []).payload
    assert len(m["artifacts"]) <= BUDGETS[Mode.SMART]


def test_deep_selects_more_than_smart_but_not_everything():
    """DEEP means maximum USEFUL evidence, not mechanically every artifact."""
    deep = build_manifest(_assembled(_refs(40)), Mode.DEEP, []).payload
    assert BUDGETS[Mode.SMART] < len(deep["artifacts"]) <= BUDGETS[Mode.DEEP]
    assert len(deep["artifacts"]) < 40


def test_duplicate_artifact_ids_are_eliminated():
    m = build_manifest(_assembled(_refs(3) + _refs(3)), Mode.DEEP, []).payload
    ids = [a["artifact_id"] for a in m["artifacts"]]
    assert len(ids) == len(set(ids))


def test_selection_is_deterministic_for_the_same_inputs():
    assert build_manifest(_assembled(_refs(30)), Mode.SMART, []).payload == (
        build_manifest(_assembled(_refs(30)), Mode.SMART, []).payload)


def test_selection_follows_the_declared_view_priority():
    """Determinism alone is not enough — the ORDER has to be the declared
    one, or which evidence survives the budget becomes an accident of
    dictionary iteration. The refs below deliberately vary by image hash so
    a hash-ordered sort would produce a different, still-deterministic,
    selection.
    """
    refs = [
        EvidenceRef(artifact_id="c1", image_hash="zzz",
                    origin=EvidenceOrigin.NORMALIZED, view="corner_top_left"),
        EvidenceRef(artifact_id="s1", image_hash="aaa",
                    origin=EvidenceOrigin.NORMALIZED, view="surface_original"),
        EvidenceRef(artifact_id="e1", image_hash="mmm",
                    origin=EvidenceOrigin.NORMALIZED, view="edge_top"),
    ]
    built = build_manifest(_assembled(refs), Mode.SMART, [])
    assert [a["view"] for a in built.payload["artifacts"]] == [
        "surface_original", "corner_top_left", "edge_top"]


def test_a_tight_budget_keeps_the_highest_priority_evidence():
    """The point of the ordering: when the budget bites, what survives is
    the most useful evidence rather than whatever sorted first."""
    refs = [
        EvidenceRef(artifact_id=f"e{i}", image_hash="aaa",
                    origin=EvidenceOrigin.NORMALIZED, view=f"edge_{i}")
        for i in range(BUDGETS[Mode.SMART])
    ] + [
        EvidenceRef(artifact_id="s1", image_hash="zzz",
                    origin=EvidenceOrigin.NORMALIZED, view="surface_original")
    ]
    built = build_manifest(_assembled(refs), Mode.SMART, [])
    assert "surface_original" in [a["view"] for a in built.payload["artifacts"]]


def test_the_index_resolves_every_sent_artifact_back_to_its_ref():
    """Without this the provider's citations cannot be resolved and
    provenance is lost at the round trip."""
    built = build_manifest(_assembled(_refs(5)), Mode.SMART, [])
    for artifact in built.payload["artifacts"]:
        ref = built.index[artifact["artifact_id"]]
        assert ref.origin.value == artifact["origin"]
        assert ref.image_hash


def test_the_index_contains_exactly_what_was_sent():
    built = build_manifest(_assembled(_refs(40)), Mode.SMART, [])
    assert set(built.index) == {a["artifact_id"]
                                for a in built.payload["artifacts"]}


def test_enhanced_artifacts_declare_their_enhancement_to_the_provider():
    """The provider must be able to tell an enhanced view from an original,
    or it cannot honour the conservative evidence standard."""
    built = build_manifest(_assembled(_refs(3, EvidenceOrigin.ENHANCED)),
                           Mode.DEEP, [])
    assert all(a["enhancement"] for a in built.payload["artifacts"])


def test_the_builder_version_is_not_in_the_provider_payload():
    """It would otherwise enter the vision fingerprint and re-bill every card
    on a builder bump the provider cannot see."""
    built = build_manifest(_assembled(_refs(3)), Mode.SMART, [])
    assert "builder_version" not in built.payload
    assert built.builder_meta["builder_version"]


def test_a_builder_bump_leaves_the_provider_payload_untouched(monkeypatch):
    import card_reviewer.review.manifest as mod

    before = build_manifest(_assembled(_refs(3)), Mode.SMART, []).payload
    monkeypatch.setattr(mod, "MANIFEST_BUILDER_VERSION", "9.9.9")
    assert build_manifest(_assembled(_refs(3)), Mode.SMART, []).payload == before


def test_the_manifest_carries_every_field_the_design_promised(rubric_rules):
    """A silently thinned payload makes the provider's answers worse while
    still looking like a working integration."""
    a = _assembled(
        _refs(3),
        reason_codes_flat={
            Assembled.key(ImageRole.FRONT, "corners", "whitening"):
            "WHITE_BORDER"},
        conflicts=[{"field": "centering.horizontal", "values": [52.0, 61.0]}],
        limitations=["front is glared"],
        anomalies=[{"category": "surface", "defect_type": "scratches"}])
    m = build_manifest(a, Mode.DEEP, rubric_rules).payload
    for field in ("artifacts", "measurements", "detectability",
                  "detectability_reasons", "image_limitations", "conflicts",
                  "anomaly_candidates", "rubric_rules"):
        assert field in m, f"manifest omits {field}"
        # Present AND populated: an empty section is a silently thinned
        # payload, which degrades the provider's answers while still looking
        # like a working integration.
        assert m[field], f"manifest section {field} is empty"


def test_anomaly_candidates_carry_enhancement_provenance():
    a = _assembled(_refs(2), anomalies=[
        {"category": "surface", "defect_type": "scratches",
         "surfaced_by": "clahe", "visible_in_original": False,
         "artifact_id": "x"}])
    m = build_manifest(a, Mode.DEEP, []).payload
    assert m["anomaly_candidates"][0]["visible_in_original"] is False
    assert m["anomaly_candidates"][0]["surfaced_by"] == "clahe"


def test_the_manifest_carries_rubric_rule_content_not_a_version_string(
        rubric_rules):
    m = build_manifest(_assembled(_refs(3)), Mode.SMART, rubric_rules[:2]).payload
    assert isinstance(m["rubric_rules"], list)
    assert "statement" in m["rubric_rules"][0]


def test_no_pricing_information_reaches_the_manifest(rubric_rules):
    import re

    m = build_manifest(_assembled(_refs(3)), Mode.DEEP, rubric_rules).payload
    blob = repr(m).lower()
    for word in ("price", "cost", "profit", "purchase", "resale", "ev", "roi"):
        assert not re.search(rf"\b{word}\b", blob), f"pricing term {word!r} leaked"


def test_the_built_manifest_serializes_for_the_cache():
    """`manifest` is a cached stage, so its output must round-trip as JSON
    with the index intact — that index resolves provider citations."""
    built = build_manifest(_assembled(_refs(4)), Mode.SMART, [])
    revived = BuiltManifest.model_validate(json.loads(built.model_dump_json()))
    assert revived.payload == built.payload
    assert set(revived.index) == set(built.index)
