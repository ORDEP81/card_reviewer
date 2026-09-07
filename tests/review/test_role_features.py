import json

import pytest

from card_reviewer.review.imaging.geometry import GeometryResult
from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.role_features import (
    ROLE_FEATURES_VERSION, RoleFeatures, extract_role_features,
)
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


def _features(spec, store):
    return extract_role_features(geom(render_png(spec), store, "h1"), store)


def test_a_card_front_has_a_large_central_image_region(store):
    assert _features(CardSpec(), store).has_central_image_region is True


def test_a_text_heavy_layout_reports_higher_text_density_than_a_front(store):
    """A back is dominated by small high-frequency detail — stat lines, the
    card number, the copyright block — rather than one large image."""
    assert _features(CardSpec(text_heavy=True, seed=3), store).text_density > (
        _features(CardSpec(seed=3), store).text_density)


def test_a_text_heavy_layout_has_no_large_central_image_region(store):
    assert _features(CardSpec(text_heavy=True, seed=3),
                     store).has_central_image_region is False


def test_features_are_bounded(store):
    f = _features(CardSpec(), store)
    assert 0.0 <= f.text_density <= 1.0
    assert 0.0 <= f.layout_confidence <= 1.0


def test_unusable_geometry_yields_zero_confidence_not_a_guess(store):
    """Unknown is a first-class state; never guess a face."""
    f = extract_role_features(GeometryResult(boundary_confidence=0.1), store)
    assert f.layout_confidence == 0.0
    assert f.has_central_image_region is False


def test_the_version_is_declared(store):
    assert _features(CardSpec(), store).version == ROLE_FEATURES_VERSION


def test_the_output_round_trips_through_json(store):
    f = _features(CardSpec(), store)
    assert RoleFeatures.model_validate(json.loads(f.model_dump_json())) == f


def test_the_output_canonicalizes_for_a_fingerprint(store):
    from card_reviewer.review.canonical import canonicalize

    assert canonicalize(_features(CardSpec(), store).model_dump(mode="json"))
