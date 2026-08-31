from card_reviewer.review.enums import Authority, Coverage, Mode, Scale, Verdict


def test_scale_is_ordered_so_thresholds_can_be_compared():
    assert Scale.NONE < Scale.LOW < Scale.MODERATE < Scale.HIGH
    assert Scale.MODERATE >= Scale.MODERATE


def test_scale_parses_from_its_string_value():
    assert Scale("moderate") is Scale.MODERATE


def test_scale_exposes_a_label_for_json_storage():
    assert Scale.MODERATE.label == "moderate"


def test_verdict_has_exactly_the_four_spec_states():
    assert {v.value for v in Verdict} == {
        "PASS", "REVIEW", "REJECT", "INSUFFICIENT_IMAGES"
    }


def test_coverage_has_exactly_three_outcomes():
    assert {c.value for c in Coverage} == {"SUFFICIENT", "PARTIAL", "INADEQUATE"}


def test_mode_has_three_values_and_smart_is_the_default():
    assert {m.value for m in Mode} == {"off", "smart", "deep"}
    assert Mode.default() is Mode.SMART


def test_authority_orders_binding_above_advisory_above_inert():
    assert Authority.BINDING > Authority.ADVISORY > Authority.INERT
