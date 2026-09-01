import json

import pytest

from card_reviewer.review.assembly import ImageStageOutputs
from card_reviewer.review.models import CandidateInput, ResolvedCandidate
from card_reviewer.review.role_context import RoleContext, resolve_role_context
from card_reviewer.review.roles import ImageRole


def _candidate(title="2023 Topps Chrome Julio Rodriguez", roles=("front", "back")):
    from card_reviewer.review.models import ResolvedImage

    return ResolvedCandidate(
        candidate_id="c1", source="manual", title=title,
        images=[ResolvedImage(image_hash=f"h{i}", supplied_role=r, ordering=i)
                for i, r in enumerate(roles)])


def _outputs(features):
    return [ImageStageOutputs(image_hash=f"h{i}",
                              geometry={"boundary_confidence": 0.9},
                              role_features=f)
            for i, f in enumerate(features)]


_FRONT = {"text_density": 0.05, "has_central_image_region": True,
          "layout_confidence": 0.9}
_BACK = {"text_density": 0.7, "has_central_image_region": False,
         "layout_confidence": 0.9}


def test_it_pairs_resolved_roles_with_normalized_card_context():
    """One cached stage, because both resolve from the same inputs."""
    rc = resolve_role_context(_candidate(), _outputs([_FRONT, _BACK]))
    assert rc.roles["h0"].role is ImageRole.FRONT
    assert rc.roles["h1"].role is ImageRole.BACK
    assert rc.card_context.canonical_card_types == ["chrome"]


def test_roles_are_inferred_from_features_when_none_are_supplied():
    rc = resolve_role_context(_candidate(roles=(None, None)),
                              _outputs([_FRONT, _BACK]))
    assert rc.roles["h0"].role is ImageRole.FRONT
    assert rc.roles["h1"].role is ImageRole.BACK


def test_an_unidentifiable_title_leaves_context_unknown():
    rc = resolve_role_context(_candidate(title="mystery lot"),
                              _outputs([_FRONT, _BACK]))
    assert rc.card_context.canonical_card_types is None
    assert rc.card_context.is_known is False


def test_an_image_without_role_features_resolves_to_unknown():
    """A photograph whose geometry failed must not be assigned a face."""
    outputs = [ImageStageOutputs(image_hash="h0")]
    rc = resolve_role_context(_candidate(roles=(None,)), outputs)
    assert rc.roles["h0"].role is ImageRole.UNKNOWN


def test_the_output_round_trips_through_json():
    rc = resolve_role_context(_candidate(), _outputs([_FRONT, _BACK]))
    assert RoleContext.model_validate(json.loads(rc.model_dump_json())) == rc


def test_the_output_canonicalizes_for_a_fingerprint():
    from card_reviewer.review.canonical import canonicalize

    rc = resolve_role_context(_candidate(), _outputs([_FRONT, _BACK]))
    assert canonicalize(rc.model_dump(mode="json"))


def test_a_supplied_role_wins_over_ambiguous_layout_features():
    """Isolates the supplied path.

    The main test gives supplied roles AND matching features, so inference
    reaches the same answer and dropping `supplied_role` changes nothing.
    Here the features are deliberately ambiguous — inference alone would
    return UNKNOWN — so only the supplied value can produce a face.
    """
    from card_reviewer.review.enums import Provenance

    ambiguous = {"text_density": 0.30, "has_central_image_region": True,
                 "layout_confidence": 0.9}
    rc = resolve_role_context(_candidate(roles=("back", "front")),
                              _outputs([ambiguous, ambiguous]))
    assert rc.roles["h0"].role is ImageRole.BACK
    assert rc.roles["h1"].role is ImageRole.FRONT
    assert all(r.provenance is Provenance.SUPPLIED for r in rc.roles.values())


def test_the_same_ambiguous_features_without_a_supplied_role_are_unknown():
    """The converse, proving the previous test depends on the supplied path."""
    ambiguous = {"text_density": 0.30, "has_central_image_region": True,
                 "layout_confidence": 0.9}
    rc = resolve_role_context(_candidate(roles=(None, None)),
                              _outputs([ambiguous, ambiguous]))
    assert all(r.role is ImageRole.UNKNOWN for r in rc.roles.values())
