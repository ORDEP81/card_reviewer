from card_reviewer.review.versions import SUPPORTING_VERSIONS, VERSIONS


def test_versions_is_keyed_by_stage_not_by_component():
    """It must be comparable to STAGE_SIGNATURE_INPUTS directly; a
    component-keyed map could not be, and the drift would be invisible."""
    assert "cv_measurements" in VERSIONS
    assert "evidence_assembly" in VERSIONS
    assert "cv" not in VERSIONS


def test_every_declared_stage_has_a_version():
    assert len(VERSIONS) == 14


def test_supporting_versions_are_separate_from_stage_versions():
    assert not (set(VERSIONS) & set(SUPPORTING_VERSIONS))
    assert "taxonomy" in SUPPORTING_VERSIONS


def test_every_supporting_component_a_review_needs_is_stamped():
    """Membership, asserted by name.

    `versions.py` is EXEMPT from the AST bump guard — correctly, it IS the
    constant table — so a key added to or dropped from these maps is
    invisible to it. The tests that walk the map iterate whatever is there
    and so are tautological about membership. `artifact_scheme` was added
    in the same commit that repaired an asserted-but-absent guard, and
    deleting it survived the entire suite.

    A stored review that cannot name the id scheme its artifact references
    were built under cannot be compared across the migration that scheme
    change would be.
    """
    required = {"taxonomy", "vocabulary", "authority", "relevance",
                "scoring", "fusion", "canonicalization", "artifact_scheme"}
    missing = required - set(SUPPORTING_VERSIONS)
    assert not missing, (
        f"a review would be stamped without {sorted(missing)}, so its "
        f"prediction cannot be compared against a later PSA outcome")


# --- the effective run-version map ------------------------------------------


def test_a_skipped_vision_run_is_recorded_as_not_run_not_as_a_placeholder():
    """Writing VERSIONS verbatim would stamp every OFF review with the string
    'provider-supplied', which describes nothing that ran."""
    from card_reviewer.review.versions import effective_versions

    stamped = effective_versions(vision_signature=None)
    assert stamped["vision"] == "not_run"


def test_a_real_vision_run_preserves_provider_model_prompt_and_params():
    """Calibration has to be able to ask which model produced a judgment."""
    from card_reviewer.review.versions import effective_versions

    stamped = effective_versions(
        vision_signature={
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "prompt_version": "1.0.0",
            "adapter_version": "1.0.0",
            "inference_params": {"max_tokens": 4096},
        }
    )
    assert "anthropic" in stamped["vision"]
    assert "claude-sonnet-5" in stamped["vision"]
    assert "1.0.0" in stamped["vision"]
    assert "max_tokens=4096" in stamped["vision"]


def test_the_placeholder_never_reaches_a_stamped_review():
    from card_reviewer.review.versions import effective_versions

    for signature in (None, {"provider": "fake", "model": "m",
                             "prompt_version": "1", "adapter_version": "1.0.0",
                             "inference_params": {}}):
        assert "provider-supplied" not in effective_versions(
            vision_signature=signature
        ).values()


def test_every_other_stage_version_is_carried_through_unchanged():
    from card_reviewer.review.versions import VERSIONS, effective_versions

    from card_reviewer.review.versions import SUPPORTING_VERSIONS

    stamped = effective_versions(vision_signature=None)
    # The supporting versions join the stage versions rather than replacing
    # them: taxonomy, authority, relevance, scoring, fusion and
    # canonicalization all change the numbers, and calibration should not
    # have to reconstruct them by joining stage rows.
    # `rubric` joins them, read at run time from Subsystem B rather than
    # declared in either map — see
    # test_a_review_records_the_rubric_that_graded_it.
    assert set(stamped) == set(VERSIONS) | set(SUPPORTING_VERSIONS) | {"rubric"}
    for stage, version in VERSIONS.items():
        if stage != "vision":
            assert stamped[stage] == version
    for component, version in SUPPORTING_VERSIONS.items():
        assert stamped[component] == version


def test_an_incomplete_vision_signature_is_rejected():
    """A signature missing its model is not a signature — stamping it would
    lose the identity the cache key depends on."""
    import pytest

    from card_reviewer.review.versions import effective_versions

    with pytest.raises(KeyError, match="model"):
        effective_versions(vision_signature={"provider": "anthropic"})


def test_nested_inference_parameters_render_deterministically():
    """Equivalent parameter dicts must produce one run-version string, or the
    same run reads as two different ones in the calibration record."""
    from card_reviewer.review.versions import format_vision_version

    base = {"provider": "anthropic", "model": "m", "prompt_version": "1.0.0",
            "adapter_version": "1.0.0"}
    a = format_vision_version(
        base | {"inference_params": {"thinking": {"budget": 2, "type": "on"},
                                     "max_tokens": 4096}}
    )
    b = format_vision_version(
        base | {"inference_params": {"max_tokens": 4096,
                                     "thinking": {"type": "on", "budget": 2}}}
    )
    assert a == b


def test_different_nested_parameters_still_render_differently():
    from card_reviewer.review.versions import format_vision_version

    base = {"provider": "anthropic", "model": "m", "prompt_version": "1.0.0",
            "adapter_version": "1.0.0"}
    a = format_vision_version(base | {"inference_params": {"thinking": {"budget": 2}}})
    b = format_vision_version(base | {"inference_params": {"thinking": {"budget": 3}}})
    assert a != b


def test_a_review_records_the_rubric_that_graded_it():
    """Non-negotiable rule 7 names the rubric version explicitly, alongside
    the model and analyzer versions.

    Every OTHER version a verdict depends on is stamped, and the one that
    supplied the grading RULES was not — so a stored review could say which
    scorer and which prompt produced it and not which rubric, and Subsystem
    B versions its rubric precisely because those rules change. A prediction
    that cannot name its rubric cannot be compared against the PSA outcome
    it predicted, which is the whole purpose of keeping it.

    Read at run time, not declared as a constant: the active rubric is
    whatever Subsystem B currently publishes, and a hardcoded copy would go
    stale silently — the failure mode this branch has hit repeatedly.
    """
    from card_reviewer.knowledge import load_active_rubric
    from card_reviewer.review.versions import effective_versions

    stamped = effective_versions()
    assert "rubric" in stamped, (
        "a stored review cannot say which rubric produced its verdict")
    assert stamped["rubric"] == load_active_rubric().version
