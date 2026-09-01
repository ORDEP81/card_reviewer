import json

from card_reviewer.review.assembly import Assembled
from card_reviewer.review.enums import Coverage, FindingState, Mode, Scale
from card_reviewer.review.findings import Finding, FindingProducer
from card_reviewer.review.policies.routing_v1 import RoutingDecision, decide_routing
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef
from card_reviewer.review.roles import ImageRole


def _f(state=FindingState.SUSPECTED, category="surface", defect="print_lines"):
    return Finding(
        defect_type=defect, category=category, state=state,
        producer=FindingProducer.HEURISTIC, confidence=0.6, psa10_relevant=True,
        evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                              origin=EvidenceOrigin.NORMALIZED, view="v")])


def _det(scale=Scale.HIGH, category="surface", defect="print_lines"):
    return {(ImageRole.FRONT, category, defect): scale}


def test_off_never_calls():
    assert decide_routing(Mode.OFF, [_f()], Coverage.PARTIAL, _det()).call_vision \
        is False


def test_deep_always_calls():
    assert decide_routing(Mode.DEEP, [], Coverage.SUFFICIENT, {}).call_vision is True


def test_smart_calls_on_a_resolvable_ambiguity():
    d = decide_routing(Mode.SMART, [_f()], Coverage.SUFFICIENT, _det())
    assert d.call_vision is True
    assert "suspected" in " ".join(d.trigger_reasons)


def test_smart_does_not_call_when_the_evidence_cannot_settle_it():
    """A provider cannot recover information absent from the pixels. Sending
    an occluded corner buys insufficient_evidence at cost."""
    d = decide_routing(Mode.SMART, [_f()], Coverage.PARTIAL, _det(Scale.LOW))
    assert d.call_vision is False


def test_smart_does_not_call_when_detectability_is_unknown():
    """No reason to believe the pixels carry the answer."""
    d = decide_routing(Mode.SMART, [_f()], Coverage.PARTIAL, {})
    assert d.call_vision is False


def test_smart_does_not_call_when_provisional_coverage_is_inadequate():
    d = decide_routing(Mode.SMART, [_f()], Coverage.INADEQUATE, _det())
    assert d.call_vision is False


def test_deep_still_calls_on_inadequate_coverage():
    """The owner asked for maximum evidence explicitly."""
    assert decide_routing(Mode.DEEP, [], Coverage.INADEQUATE, {}).call_vision is True


def test_smart_calls_to_confirm_a_strong_gem_candidate():
    d = decide_routing(Mode.SMART, [], Coverage.SUFFICIENT, {})
    assert d.call_vision is True
    assert "confirm" in " ".join(d.trigger_reasons)


def test_smart_does_not_confirm_a_gem_when_coverage_is_only_partial():
    """Nothing to confirm: the card cannot pass regardless."""
    d = decide_routing(Mode.SMART, [], Coverage.PARTIAL, {})
    assert d.call_vision is False


def test_the_decision_records_the_mode_it_was_made_under():
    assert decide_routing(Mode.SMART, [], Coverage.SUFFICIENT, {}).mode is Mode.SMART


def test_an_already_observed_finding_does_not_trigger_a_call():
    """Nothing ambiguous about it — the call would buy no new information."""
    d = decide_routing(Mode.SMART, [_f(FindingState.OBSERVED)],
                       Coverage.PARTIAL, _det())
    assert d.call_vision is False


def test_the_decision_round_trips_and_canonicalizes():
    from card_reviewer.review.canonical import canonicalize

    d = decide_routing(Mode.SMART, [_f()], Coverage.SUFFICIENT, _det())
    assert RoutingDecision.model_validate(json.loads(d.model_dump_json())) == d
    assert canonicalize(d.model_dump(mode="json"))
