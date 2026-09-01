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
                             "prompt_version": "1", "inference_params": {}}):
        assert "provider-supplied" not in effective_versions(
            vision_signature=signature
        ).values()


def test_every_other_stage_version_is_carried_through_unchanged():
    from card_reviewer.review.versions import VERSIONS, effective_versions

    stamped = effective_versions(vision_signature=None)
    assert set(stamped) == set(VERSIONS)
    for stage, version in VERSIONS.items():
        if stage != "vision":
            assert stamped[stage] == version


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

    base = {"provider": "anthropic", "model": "m", "prompt_version": "1.0.0"}
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

    base = {"provider": "anthropic", "model": "m", "prompt_version": "1.0.0"}
    a = format_vision_version(base | {"inference_params": {"thinking": {"budget": 2}}})
    b = format_vision_version(base | {"inference_params": {"thinking": {"budget": 3}}})
    assert a != b
