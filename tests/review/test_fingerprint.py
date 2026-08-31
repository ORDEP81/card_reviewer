import pytest

from card_reviewer.review.fingerprint import (
    STAGE_FINGERPRINT_INPUTS,
    STAGE_SIGNATURE_INPUTS,
    fingerprint,
    signature_for,
)


def test_fingerprint_is_stable_across_key_order():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_fingerprint_changes_when_a_semantic_value_changes():
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_mode_is_a_routing_fingerprint_input_not_a_signature_input():
    """Mode is data the stage consumes, not part of its implementation
    identity — so it belongs in the fingerprint. That is what stops an OFF
    run satisfying a later DEEP lookup."""
    assert "mode" in STAGE_FINGERPRINT_INPUTS["routing"]
    assert "mode" not in STAGE_SIGNATURE_INPUTS["routing"]


def test_mode_is_absent_from_combine_entirely():
    assert "mode" not in STAGE_SIGNATURE_INPUTS["combine"]
    assert "mode" not in STAGE_FINGERPRINT_INPUTS["combine"]


def test_an_off_and_a_deep_run_produce_different_routing_fingerprints():
    off = fingerprint({"mode": "off", "heuristic_output": {}})
    deep = fingerprint({"mode": "deep", "heuristic_output": {}})
    assert off != deep


def test_the_vision_fingerprint_is_the_provider_payload_not_the_builder():
    """A manifest-builder bump producing identical provider-visible content
    must not re-bill a call."""
    assert STAGE_FINGERPRINT_INPUTS["vision"] == ("provider_evidence_payload",)
    assert "manifest_builder_version" not in STAGE_SIGNATURE_INPUTS["vision"]


def test_every_stage_declares_both_a_fingerprint_and_a_signature():
    """A stage in one table but not the other is a stage the pipeline claims
    to cache but cannot."""
    assert set(STAGE_FINGERPRINT_INPUTS) == set(STAGE_SIGNATURE_INPUTS)


def test_versions_covers_every_declared_stage():
    """A stage missing there means a review is stamped with versions that do
    not describe what actually ran."""
    from card_reviewer.review.versions import VERSIONS

    assert set(VERSIONS) == set(STAGE_SIGNATURE_INPUTS)


def test_taxonomy_version_is_in_image_tier_signatures_but_rubric_is_not():
    for stage in ("observability", "cv_measurements"):
        assert "taxonomy_version" in STAGE_SIGNATURE_INPUTS[stage]
        assert "rubric_version" not in STAGE_SIGNATURE_INPUTS[stage]


def test_preflight_and_geometry_do_not_consume_the_taxonomy():
    for stage in ("preflight", "geometry"):
        assert "taxonomy_version" not in STAGE_SIGNATURE_INPUTS[stage]


def test_a_signature_changes_when_any_declared_version_changes():
    a = signature_for("observability", {"observability_version": "1.0.0",
                                        "taxonomy_version": "1.0.0", "config": {}})
    b = signature_for("observability", {"observability_version": "1.0.1",
                                        "taxonomy_version": "1.0.0", "config": {}})
    assert a != b


def test_a_missing_version_key_raises_rather_than_hashing_fewer_inputs():
    with pytest.raises(KeyError, match="taxonomy_version"):
        signature_for("observability", {"observability_version": "1.0.0",
                                        "config": {}})


def test_an_unknown_stage_raises():
    with pytest.raises(KeyError):
        signature_for("not_a_stage", {})
