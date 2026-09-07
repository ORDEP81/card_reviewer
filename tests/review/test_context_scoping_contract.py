"""Contract: CardContextNormalizer -> Rubric.for_card -> scope_rules.

Three real components, wired as the pipeline wires them. Hand-built
CardContext fixtures cannot catch the failure that matters here — the
normalizer emitting a shape `for_card` matches differently than intended.
"""

import json

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.context import CardContext
from card_reviewer.review.enums import RuleEvaluability
from card_reviewer.review.evaluability import (
    UNKNOWN_PRODUCT_CONTEXT,
    applicable,
    rule_content,
    scope_rules,
    unevaluable_reasons,
)
from card_reviewer.review.normalize import CardContextNormalizer


def _pipeline(title=None, supplied_card_type=None):
    """Exactly the call chain Task 36 will make."""
    context = CardContextNormalizer().normalize(
        raw_title=title, supplied_card_type=supplied_card_type
    )
    rules = load_active_rubric().for_card(
        context.canonical_card_types, context.canonical_sets
    )
    return context, scope_rules(rules, context)


def test_a_chrome_title_reaches_the_product_scoped_rule_as_applicable():
    """The end-to-end path that must work: free text in, scoped rule out."""
    _, scoped = _pipeline(title="2023 Topps Chrome Julio Rodriguez #150")
    by_id = {s.rule.id: s for s in scoped}
    assert by_id["SURFACE_SHINY_001"].evaluability is RuleEvaluability.APPLICABLE


def test_an_unrecognized_title_leaves_that_rule_unevaluable_not_applied():
    _, scoped = _pipeline(title="2023 Some Unknown Parallel /25")
    by_id = {s.rule.id: s for s in scoped}
    assert by_id["SURFACE_SHINY_001"].evaluability is RuleEvaluability.UNEVALUABLE
    assert UNKNOWN_PRODUCT_CONTEXT in unevaluable_reasons(scoped)


def test_the_normalizers_none_reaches_for_card_as_unconstrained():
    """The None-vs-[] distinction has to survive the hand-off, or every
    scoped rule silently disappears from the rubric."""
    context, scoped = _pipeline(title="no product named here")
    assert context.canonical_card_types is None
    assert len(scoped) == len(load_active_rubric().rules)


def test_a_paper_card_never_sees_the_chrome_rule_at_all():
    """for_card filters it out upstream, so it is not merely unevaluable —
    it is absent, and must not appear in rule_content either."""
    context = CardContext(canonical_card_types=["paper"])
    scoped = scope_rules(
        load_active_rubric().for_card(["paper"], None), context
    )
    assert "SURFACE_SHINY_001" not in {s.rule.id for s in scoped}
    assert "SURFACE_SHINY_001" not in {r["id"] for r in rule_content(scoped)}


def test_unknown_context_yields_fewer_applicable_rules_than_known_context():
    """The gate has to actually withhold something, or it is decoration."""
    _, unknown = _pipeline(title="no product named here")
    _, known = _pipeline(supplied_card_type="chrome")
    assert len(applicable(unknown)) < len(applicable(known))


def test_rule_content_from_a_real_context_survives_a_json_round_trip():
    """It is a cached fingerprint input, so it crosses SQLite."""
    _, scoped = _pipeline(supplied_card_type="chrome")
    content = rule_content(scoped)
    assert json.loads(json.dumps(content)) == content


def test_a_context_from_the_normalizer_round_trips_before_scoping():
    """role_context caches the context, then a later run scopes from the
    revived value — the revived one must scope identically."""
    context = CardContextNormalizer().normalize(raw_title="2023 Topps Chrome")
    revived = CardContext.model_validate(json.loads(context.model_dump_json()))
    rubric = load_active_rubric()
    fresh = scope_rules(
        rubric.for_card(context.canonical_card_types, context.canonical_sets),
        context,
    )
    after = scope_rules(
        rubric.for_card(revived.canonical_card_types, revived.canonical_sets),
        revived,
    )
    assert [(s.rule.id, s.evaluability) for s in fresh] == [
        (s.rule.id, s.evaluability) for s in after
    ]
