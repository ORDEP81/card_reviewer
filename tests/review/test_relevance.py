from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.context import CardContext
from card_reviewer.review.enums import Authority, FindingState, Provenance
from card_reviewer.review.evaluability import scope_rules
from card_reviewer.review.findings import Finding, FindingProducer
from card_reviewer.review.policies.authority_v1 import authority_of
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef
from card_reviewer.review.relevance import resolve_relevance


def _f(category="corners", defect="rounding", relevant=True):
    return Finding(
        defect_type=defect, category=category, state=FindingState.OBSERVED,
        producer=FindingProducer.HEURISTIC, confidence=0.9,
        psa10_relevant=relevant,
        evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                              origin=EvidenceOrigin.ORIGINAL, view="v")])


def _scoped(card_types=None):
    ctx = CardContext(canonical_card_types=card_types,
                      provenance=Provenance.SUPPLIED if card_types
                      else Provenance.UNKNOWN)
    return scope_rules(load_active_rubric().for_card(card_types, None), ctx)


def test_a_finding_matches_only_rules_in_its_own_category():
    resolved = resolve_relevance([_f(category="corners")], _scoped(["chrome"]))[0]
    rubric = {r.id: r for r in load_active_rubric().rules}
    assert resolved.rule_ids
    assert all(rubric[rid].category.value == "corners" for rid in resolved.rule_ids)


def test_a_corner_finding_never_inherits_a_centering_rules_authority():
    """Cross-category leakage is the failure worth preventing: an unrelated
    high-authority rule must not lend weight to a different defect."""
    resolved = resolve_relevance([_f(category="corners")], _scoped(["chrome"]))[0]
    rubric = {r.id: r for r in load_active_rubric().rules}
    assert not any(rubric[rid].category.value == "centering"
                   for rid in resolved.rule_ids)


def test_authority_is_the_maximum_among_matched_rules():
    resolved = resolve_relevance([_f(category="surface")], _scoped(["chrome"]))[0]
    rubric = {r.id: r for r in load_active_rubric().rules}
    expected = max(authority_of(rubric[rid]) for rid in resolved.rule_ids)
    assert resolved.authority is expected


def test_with_no_rules_at_all_authority_falls_back_to_advisory():
    """The safety net, tested against a constructed empty rule set.

    At v4.0.0 every grading category has active rules, so under category
    matching this default cannot fire against live data. Defaulting to
    BINDING would let any unrecognized anomaly reject a card.
    """
    resolved = resolve_relevance([_f(category="corners")], [])[0]
    assert resolved.rule_ids == []
    assert resolved.authority is Authority.ADVISORY
    assert resolved.psa10_relevant is True


def test_a_finding_outside_the_grading_taxonomy_is_not_psa10_relevant():
    assert resolve_relevance([_f(category="handling", defect="x")],
                             _scoped(["chrome"]))[0].psa10_relevant is False


def test_a_provider_claim_of_relevance_is_overridden_by_our_policy():
    """Claude may describe a defect; whether it disqualifies a 10 is ours."""
    claimed = _f(category="handling", defect="looks_bad", relevant=True)
    assert resolve_relevance([claimed], _scoped(["chrome"]))[0].psa10_relevant is False


def test_a_grading_category_finding_stays_relevant_even_when_odd():
    """Gating relevance on rule matching would let an unexplained corner
    defect ship as a clean gem candidate."""
    odd = _f(category="corners", defect="unrecognized_thing")
    assert resolve_relevance([odd], _scoped(["chrome"]))[0].psa10_relevant is True


def test_unevaluable_rules_are_never_matched():
    """SURFACE_SHINY_001 is product-scoped; with product unknown it must not
    lend its authority to a surface finding."""
    resolved = resolve_relevance([_f(category="surface", defect="scratches")],
                                 _scoped(None))[0]
    assert "SURFACE_SHINY_001" not in resolved.rule_ids


def test_the_finding_carries_the_matched_rule_ids_forward():
    resolved = resolve_relevance([_f(category="corners")], _scoped(["chrome"]))[0]
    assert resolved.finding.rule_ids == resolved.rule_ids


def test_resolution_preserves_input_order():
    findings = [_f(category="corners"), _f(category="surface"),
                _f(category="edges")]
    out = resolve_relevance(findings, _scoped(["chrome"]))
    assert [r.finding.category for r in out] == ["corners", "surface", "edges"]
