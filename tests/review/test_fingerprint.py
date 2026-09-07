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


# --- what each stage actually consumes -------------------------------------


def test_heuristic_consumes_the_taxonomy_not_the_authority_policy():
    """The heuristic asks `promotion_of` whether a defect type may reach
    OBSERVED — that is taxonomy. Authority belongs to relevance and scoring,
    which run later; declaring it here would invalidate heuristic results on
    a policy bump it never reads."""
    sig = STAGE_SIGNATURE_INPUTS["heuristic"]
    assert "taxonomy_version" in sig
    assert "authority_policy_version" not in sig


def test_combine_fingerprints_every_value_it_consumes():
    fp = STAGE_FINGERPRINT_INPUTS["combine"]
    for consumed in (
        "heuristic_output",
        "vision_output",
        "coverage_output",
        "applicable_rubric_rule_content",
        "detectability",
        "card_context_known",
        "required_face_missing",
        "manifest_index",
    ):
        assert consumed in fp, f"combine consumes {consumed} but does not hash it"


def test_combine_signature_covers_every_policy_that_changes_its_output():
    sig = STAGE_SIGNATURE_INPUTS["combine"]
    for policy in (
        "combination_policy_version",
        "scoring_policy_version",
        "relevance_policy_version",
        "authority_policy_version",
        "fusion_version",
        "taxonomy_version",
    ):
        assert policy in sig, f"combine output depends on {policy}"


# --- cache identity discrimination -----------------------------------------


def _combine_inputs(**overrides):
    base = {
        "heuristic_output": {"findings": []},
        "vision_output": None,
        "coverage_output": {"outcome": "SUFFICIENT"},
        "applicable_rubric_rule_content": [{"id": "CORNERS_COLORED_001"}],
        "detectability": {"front|corners|whitening": "high"},
        "card_context_known": True,
        "required_face_missing": False,
        "manifest_index": {},
    }
    return base | overrides


def _combine_versions(**overrides):
    base = {
        "combination_policy_version": "1.0.0",
        "scoring_policy_version": "1.0.0",
        "relevance_policy_version": "1.0.0",
        "authority_policy_version": "1.0.0",
        "fusion_version": "1.0.0",
        "taxonomy_version": "1.0.0",
    }
    return base | overrides


@pytest.mark.parametrize(
    "field,changed",
    [
        ("card_context_known", False),
        ("required_face_missing", True),
        ("detectability", {"front|corners|whitening": "low"}),
        ("applicable_rubric_rule_content", [{"id": "SURFACE_SHINY_001"}]),
        ("manifest_index", {"a1": {"origin": "enhanced"}}),
        ("coverage_output", {"outcome": "PARTIAL"}),
    ],
)
def test_changing_any_material_combine_input_changes_its_fingerprint(field, changed):
    before = fingerprint(_combine_inputs())
    after = fingerprint(_combine_inputs(**{field: changed}))
    assert before != after, f"{field} does not affect combine's cache identity"


@pytest.mark.parametrize(
    "version",
    [
        "combination_policy_version",
        "scoring_policy_version",
        "relevance_policy_version",
        "authority_policy_version",
        "fusion_version",
        "taxonomy_version",
    ],
)
def test_changing_any_combine_policy_version_changes_its_signature(version):
    before = signature_for("combine", _combine_versions())
    after = signature_for("combine", _combine_versions(**{version: "2.0.0"}))
    assert before != after


def test_an_unrelated_upstream_producer_bump_leaves_combine_untouched():
    """Values, not signatures: bumping the CV analyzer must not invalidate a
    combine whose inputs are unchanged."""
    before = fingerprint(_combine_inputs())
    after = fingerprint(_combine_inputs())
    assert before == after
    assert "cv_version" not in STAGE_SIGNATURE_INPUTS["combine"]
    assert "preflight_version" not in STAGE_SIGNATURE_INPUTS["combine"]


def test_a_geometry_bump_does_not_appear_in_the_heuristic_signature():
    assert "geometry_version" not in STAGE_SIGNATURE_INPUTS["heuristic"]
