"""The Assembled model itself — Task 28 fills in `assemble`, but the model is
the heuristic's input contract and must exist and be cacheable first."""

import json

from card_reviewer.review.assembly import Assembled
from card_reviewer.review.enums import Scale
from card_reviewer.review.roles import ImageRole


def test_detectability_is_stored_flat_and_read_as_tuples():
    """evidence_assembly is a JSON-cached stage, so tuple keys cannot be
    stored — but every consumer wants the tuple view."""
    a = Assembled(detectability_flat={
        Assembled.key(ImageRole.FRONT, "corners", "rounding"): "high"})
    assert a.detectability[(ImageRole.FRONT, "corners", "rounding")] is Scale.HIGH


def test_the_model_round_trips_through_json_with_keys_intact():
    a = Assembled(
        detectability_flat={Assembled.key(ImageRole.FRONT, "corners", "whitening"): "low"},
        reason_codes_flat={Assembled.key(ImageRole.FRONT, "corners", "whitening"): "WHITE_BORDER"},
        faces_present=["front"])
    revived = Assembled.model_validate(json.loads(a.model_dump_json()))
    assert revived.detectability == a.detectability
    assert revived.reason_codes == a.reason_codes
    assert revived.faces == (ImageRole.FRONT,)


def test_a_dump_of_the_model_canonicalizes_for_a_fingerprint():
    """The cache boundary and the fingerprint boundary must agree — a tuple
    key would crash one of them."""
    from card_reviewer.review.canonical import canonicalize

    a = Assembled(detectability_flat={
        Assembled.key(ImageRole.BACK, "surface", "scratches"): "moderate"})
    assert canonicalize(a.model_dump(mode="json"))


def test_faces_present_is_a_list_of_labels_not_a_tuple_of_enums():
    a = Assembled(faces_present=["front", "back"])
    assert a.faces_present == ["front", "back"]
    assert a.faces == (ImageRole.FRONT, ImageRole.BACK)
