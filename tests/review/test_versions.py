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
