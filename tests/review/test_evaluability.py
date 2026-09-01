from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.context import CardContext
from card_reviewer.review.enums import Provenance, RuleEvaluability
from card_reviewer.review.evaluability import (
    UNKNOWN_PRODUCT_CONTEXT,
    applicable,
    rule_content,
    scope_rules,
    unevaluable_reasons,
)


def _ctx(card_types=None):
    return CardContext(
        canonical_card_types=card_types,
        provenance=Provenance.SUPPLIED if card_types else Provenance.UNKNOWN,
    )


def _scoped(card_types=None):
    rubric = load_active_rubric()
    return scope_rules(rubric.for_card(card_types, None), _ctx(card_types))


def test_unscoped_rules_are_always_applicable():
    scoped = _scoped(None)
    unscoped = [s for s in scoped if not s.rule.applies_to.card_types]
    assert unscoped
    assert all(s.evaluability is RuleEvaluability.APPLICABLE for s in unscoped)


def test_a_product_scoped_rule_is_unevaluable_when_context_is_unknown():
    """SURFACE_SHINY_001 is scoped to chrome/refractor/foil. With the product
    unknown it must not silently apply."""
    scoped = {s.rule.id: s for s in _scoped(None)}
    assert scoped["SURFACE_SHINY_001"].evaluability is RuleEvaluability.UNEVALUABLE


def test_the_same_rule_is_applicable_once_the_product_is_known():
    scoped = {s.rule.id: s for s in _scoped(["chrome"])}
    assert scoped["SURFACE_SHINY_001"].evaluability is RuleEvaluability.APPLICABLE


def test_a_scoped_rule_whose_scope_excludes_the_card_is_not_returned_at_all():
    """for_card already filters this: a paper card never sees SURFACE_SHINY_001."""
    paper = load_active_rubric().for_card(card_types=["paper"], sets=None)
    assert "SURFACE_SHINY_001" not in {r.id for r in paper}


def test_unknown_context_produces_a_metadata_resolvable_reason_code():
    assert UNKNOWN_PRODUCT_CONTEXT in unevaluable_reasons(_scoped(None))


def test_that_reason_code_is_classed_metadata_resolvable_not_circumstantial():
    """An identity problem must never become a photograph problem."""
    from card_reviewer.review.enums import UndetectabilityClass
    from card_reviewer.review.taxonomy import class_of

    assert class_of(UNKNOWN_PRODUCT_CONTEXT) is (
        UndetectabilityClass.METADATA_RESOLVABLE
    )


def test_known_context_produces_no_unevaluable_reasons():
    assert unevaluable_reasons(_scoped(["chrome"])) == []


def test_unknown_context_still_returns_every_rule():
    """Unknown context must not narrow the rubric — the rules are all there,
    they are merely marked unevaluable."""
    assert len(_scoped(None)) == len(load_active_rubric().rules)


def test_applicable_excludes_unevaluable_rules():
    ids = {r.id for r in applicable(_scoped(None))}
    assert "SURFACE_SHINY_001" not in ids


def test_rule_content_carries_what_the_rules_say_not_a_version_string():
    """Stages fingerprint rubric CONTENT, so a release leaving the applicable
    rules unchanged for this card does not invalidate their results."""
    content = rule_content(_scoped(["chrome"]))
    assert content and "statement" in content[0]
    assert {"id", "category", "statement", "evidence_type", "confidence"} <= set(
        content[0]
    )


def test_rule_content_excludes_unevaluable_rules():
    ids = {r["id"] for r in rule_content(_scoped(None))}
    assert "SURFACE_SHINY_001" not in ids


def test_rule_content_is_deterministic_for_the_same_scoping():
    assert rule_content(_scoped(["chrome"])) == rule_content(_scoped(["chrome"]))


def test_rule_content_canonicalizes_for_a_fingerprint():
    """It is a fingerprint input, so it must survive the strict evidence
    canonicalizer — no floats, no odd keys."""
    from card_reviewer.review.canonical import canonicalize

    assert canonicalize({"applicable_rubric_rules": rule_content(_scoped(["chrome"]))})
