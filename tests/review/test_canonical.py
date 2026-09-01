from card_reviewer.review.canonical import (
    CANON_SCHEME_VERSION,
    canonicalize,
    precision_for,
    quantize,
)


def test_there_is_no_single_global_float_precision():
    """A centering ratio measured to +/-1.5pp must quantize far more coarsely
    than a normalized coordinate; one rounding for both either discards real
    signal or manufactures spurious cache misses.

    Note the claim being made: values falling in the SAME declared bucket
    canonicalize identically. A fixed bucket cannot promise that for every
    pair less than one step apart — two such values may straddle a boundary.
    """
    assert quantize("centering.horizontal", 54.03) == quantize(
        "centering.horizontal", 54.4
    )
    assert quantize("region.x0", 0.5001) != quantize("region.x0", 0.5099)


def test_key_order_does_not_change_the_canonical_form():
    assert canonicalize({"b": 1, "a": 2}) == canonicalize({"a": 2, "b": 1})


def test_non_semantic_fields_are_excluded():
    """Timestamps must not make identical work look different."""
    a = canonicalize({"value": 1, "computed_at": "2026-08-30T10:00:00Z",
                      "elapsed_ms": 12})
    b = canonicalize({"value": 1, "computed_at": "2026-08-30T11:00:00Z",
                      "elapsed_ms": 99})
    assert a == b


def test_quantization_applies_inside_nested_structures():
    assert canonicalize({"centering": {"horizontal": 54.03}}) == canonicalize(
        {"centering": {"horizontal": 54.40}}
    )


def test_precision_resolves_by_semantic_suffix_not_absolute_path():
    """The same meaning at any depth resolves to the same precision."""
    assert precision_for("centering.horizontal") == precision_for(
        "assembled_evidence.centering.horizontal"
    )
    assert precision_for("heuristic_output.findings.confidence") == 0.01


def test_a_semantic_change_still_changes_the_canonical_form():
    assert canonicalize({"a": 1}) != canonicalize({"a": 2})


def test_booleans_are_not_treated_as_numbers():
    assert canonicalize({"a": True}) != canonicalize({"a": 1})


def test_non_string_dict_keys_are_rejected_rather_than_silently_coerced():
    """Tuple keys crash json.dumps at the cache boundary. Catching it here
    names the offending field instead of failing deep inside SQLite."""
    import pytest

    with pytest.raises(TypeError, match="string keys"):
        canonicalize({("front", "corners", "whitening"): "high"})


def test_the_scheme_version_is_declared_and_matches_the_versions_table():
    from card_reviewer.review.versions import SUPPORTING_VERSIONS

    assert CANON_SCHEME_VERSION == SUPPORTING_VERSIONS["canonicalization"]


# --- semantic precision on realistically nested payloads --------------------


def test_centering_precision_applies_under_a_stage_wrapper():
    """Real payloads nest under names like `assembled_evidence`. Precision is
    a property of the value's meaning, not of the outer wrapper."""
    a = canonicalize({"assembled_evidence": {"centering": {"horizontal": 54.03}}})
    b = canonicalize({"assembled_evidence": {"centering": {"horizontal": 54.40}}})
    assert a == b


def test_finding_confidence_precision_applies_inside_a_list_of_findings():
    a = canonicalize({"heuristic_output": {"findings": [{"confidence": 0.9500}]}})
    b = canonicalize({"heuristic_output": {"findings": [{"confidence": 0.9503}]}})
    assert a == b


def test_normalized_coordinates_keep_their_precision_wherever_they_appear():
    """EvidenceRef.region and Finding.location are both NormalizedBox, at
    different depths and under different field names."""
    deep = {"combine": {"findings": [
        {"evidence": [{"region": {"x0": 0.50001, "y0": 0.1, "x1": 0.9, "y1": 0.9}}],
         "location": {"x0": 0.50002, "y0": 0.1, "x1": 0.9, "y1": 0.9}}]}}
    other = {"combine": {"findings": [
        {"evidence": [{"region": {"x0": 0.50004, "y0": 0.1, "x1": 0.9, "y1": 0.9}}],
         "location": {"x0": 0.50003, "y0": 0.1, "x1": 0.9, "y1": 0.9}}]}}
    assert canonicalize(deep) == canonicalize(other)


def test_coordinates_a_full_step_apart_still_differ():
    a = canonicalize({"region": {"x0": 0.100, "y0": 0.1, "x1": 0.9, "y1": 0.9}})
    b = canonicalize({"region": {"x0": 0.105, "y0": 0.1, "x1": 0.9, "y1": 0.9}})
    assert a != b


# --- strictness ------------------------------------------------------------


def test_an_unregistered_float_field_raises_rather_than_guessing():
    """Spec §4 says precision is DECLARED. A generic fallback silently invents
    a precision for a value nobody reasoned about."""
    import pytest

    with pytest.raises(ValueError, match="no declared precision"):
        canonicalize({"stage": {"some_new_float": 1.234}})


def test_an_unsupported_object_cannot_silently_enter_a_fingerprint():
    """`default=str` would stringify a set into the cache key, so two
    different sets could collide or the same set could hash unstably."""
    import pytest

    with pytest.raises(TypeError):
        canonicalize({"stage": {"values": {"a", "b"}}})


def test_an_arbitrary_object_cannot_silently_enter_a_fingerprint():
    import pytest

    class Opaque:
        pass

    with pytest.raises(TypeError):
        canonicalize({"stage": {"thing": Opaque()}})
