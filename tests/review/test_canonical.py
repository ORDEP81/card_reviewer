from card_reviewer.review.canonical import CANON_SCHEME_VERSION, canonicalize, quantize


def test_there_is_no_single_global_float_precision():
    """A centering ratio measured to +/-1.5pp must quantize far more coarsely
    than a normalized coordinate; one rounding for both either discards real
    signal or manufactures spurious cache misses."""
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


def test_unknown_float_fields_use_the_declared_default_precision():
    assert quantize("some.new.field", 1.0 / 3.0) == quantize(
        "some.new.field", 0.33334
    )


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
