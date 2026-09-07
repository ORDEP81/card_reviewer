"""Producer-signature canonicalization is exact, not semantic.

Evidence fingerprints quantize by declared measurement precision: two
centering readings inside the same bucket are the same observation. A
producer signature is the opposite kind of value — it identifies the
implementation and configuration that ran, so `temperature=0.2` and
`temperature=0.204` are different behaviour and must never collide.

Using the measurement quantizer for both is what these tests forbid.
"""

import math

import pytest

from card_reviewer.review.canonical import canonicalize, canonicalize_config
from card_reviewer.review.fingerprint import fingerprint, signature_for


def _vision(**params):
    return {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "prompt_version": "1.0.0",
        "inference_params": params,
    }


# --- configuration floats need no measurement registration -----------------


def test_a_provider_signature_carrying_temperature_needs_no_precision_entry():
    """temperature is configuration, not a measurement. Requiring it in
    PRECISION_MAP would mean declaring a measurement precision for a value
    that measures nothing."""
    assert signature_for("vision", _vision(temperature=0.2))


def test_changing_temperature_changes_the_producer_signature():
    a = signature_for("vision", _vision(temperature=0.2))
    b = signature_for("vision", _vision(temperature=0.7))
    assert a != b


def test_two_config_floats_in_one_measurement_bucket_stay_distinct():
    """0.9500 and 0.9503 are the same *observation* under the declared
    confidence precision. As weights they are different behaviour."""
    a = signature_for("heuristic", {"scorer_version": "1.0.0",
                                    "taxonomy_version": "1.0.0",
                                    "weights": {"confidence": 0.9500}})
    b = signature_for("heuristic", {"scorer_version": "1.0.0",
                                    "taxonomy_version": "1.0.0",
                                    "weights": {"confidence": 0.9503}})
    assert a != b


def test_the_same_two_values_still_collapse_as_evidence():
    """The semantic boundary is correct and stays unchanged."""
    assert fingerprint({"heuristic_output": {"findings": [{"confidence": 0.9500}]}}) == \
        fingerprint({"heuristic_output": {"findings": [{"confidence": 0.9503}]}})


# --- determinism -----------------------------------------------------------


def test_dictionary_ordering_does_not_change_a_producer_signature():
    a = signature_for("vision", _vision(temperature=0.2, top_p=0.9))
    b = signature_for("vision", {
        "inference_params": {"top_p": 0.9, "temperature": 0.2},
        "prompt_version": "1.0.0", "model": "claude-sonnet-5",
        "provider": "anthropic",
    })
    assert a == b


def test_nested_config_ordering_does_not_change_a_producer_signature():
    a = canonicalize_config({"weights": {"a": 1, "b": {"x": 1, "y": 2}}})
    b = canonicalize_config({"weights": {"b": {"y": 2, "x": 1}, "a": 1}})
    assert a == b


# --- strictness ------------------------------------------------------------


def test_a_non_finite_config_float_is_rejected():
    """NaN and Infinity are not JSON, and NaN never equals itself — a
    signature containing one could never match its own cache row."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            canonicalize_config({"weights": {"scale": bad}})


def test_an_unsupported_config_object_is_rejected():
    with pytest.raises(TypeError):
        canonicalize_config({"weights": {"values": {"a", "b"}}})


def test_non_string_config_keys_are_rejected():
    with pytest.raises(TypeError, match="string keys"):
        canonicalize_config({("a", "b"): 1})


def test_config_floats_are_preserved_exactly():
    assert "0.123456789" in canonicalize_config({"weights": {"w": 0.123456789}})


# --- the two boundaries stay separate --------------------------------------


def test_evidence_canonicalization_still_rejects_unregistered_floats():
    with pytest.raises(ValueError, match="no declared precision"):
        canonicalize({"stage": {"some_new_float": 1.234}})


def test_the_signature_scheme_is_versioned_independently():
    from card_reviewer.review.canonical import SIGNATURE_SCHEME_VERSION

    assert SIGNATURE_SCHEME_VERSION


def test_a_signature_and_a_fingerprint_of_the_same_payload_differ():
    """They answer different questions and must not be interchangeable."""
    payload = {"weights": {"confidence": 0.95}}
    assert canonicalize_config(payload) != canonicalize(
        {"heuristic_output": {"findings": [{"confidence": 0.95}]}}
    )


def test_math_isfinite_guard_covers_ints_without_complaint():
    assert canonicalize_config({"weights": {"n": 4096, "flag": True, "s": "x",
                                            "nothing": None}})
    assert math.isfinite(1)
