import json

import pytest

from card_reviewer.review.enums import Provenance
from card_reviewer.review.roles import (
    ImageRole,
    ResolvedRole,
    RoleInput,
    resolve_roles,
)


def _img(h="h1", supplied=None, text_density=0.08, has_central_image=True):
    return RoleInput(
        image_hash=h, supplied_role=supplied, text_density=text_density,
        has_central_image_region=has_central_image,
    )


def test_a_supplied_role_outranks_inference():
    out = resolve_roles([_img(supplied="back", text_density=0.05)])
    assert out["h1"].role is ImageRole.BACK
    assert out["h1"].provenance is Provenance.SUPPLIED
    assert out["h1"].confidence == 1.0


def test_high_text_density_without_a_central_image_infers_a_back():
    out = resolve_roles([_img(text_density=0.55, has_central_image=False)])
    assert out["h1"].role is ImageRole.BACK
    assert out["h1"].provenance is Provenance.INFERRED


def test_low_text_density_with_a_central_image_infers_a_front():
    out = resolve_roles([_img(text_density=0.08, has_central_image=True)])
    assert out["h1"].role is ImageRole.FRONT


def test_ambiguous_signatures_yield_unknown_rather_than_a_guess():
    """Guessing a face wrong silently mis-assigns every measurement taken
    from that photograph."""
    out = resolve_roles([_img(text_density=0.30, has_central_image=True)])
    assert out["h1"].role is ImageRole.UNKNOWN
    assert out["h1"].provenance is Provenance.UNKNOWN
    assert out["h1"].confidence == 0.0


def test_an_unrecognized_supplied_role_falls_back_to_inference():
    """A caller typo must not become a confident face assignment."""
    out = resolve_roles([_img(supplied="frnt", text_density=0.08)])
    assert out["h1"].provenance is not Provenance.SUPPLIED


def test_every_image_gets_a_resolution_even_when_unknown():
    out = resolve_roles([_img("h1", text_density=0.30), _img("h2", supplied="front")])
    assert set(out) == {"h1", "h2"}


def test_a_resolved_role_round_trips_through_json():
    """role_context is a cached stage carrying these."""
    original = resolve_roles([_img(supplied="front")])["h1"]
    assert ResolvedRole.model_validate(json.loads(original.model_dump_json())) == (
        original
    )


def test_an_inferred_role_is_less_confident_than_a_supplied_one():
    inferred = resolve_roles([_img(text_density=0.55, has_central_image=False)])["h1"]
    supplied = resolve_roles([_img(supplied="back")])["h1"]
    assert inferred.confidence < supplied.confidence


@pytest.mark.parametrize("density", [0.0, 1.0])
def test_extreme_densities_are_accepted(density):
    assert resolve_roles([_img(text_density=density)])["h1"] is not None


def test_a_density_outside_the_unit_interval_is_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RoleInput(image_hash="h1", text_density=1.5)
