"""Whether a returned rubric rule may actually be applied.

`for_card(None, None)` returns every rule, deliberately — unknown context
must not narrow the rubric (spec §8). But an unscoped rule set is not a
satisfied one. A rule scoped to a product we cannot identify is tagged
UNEVALUABLE: it never fires a finding and never contributes to a verdict.
It instead raises an UNKNOWN_PRODUCT_CONTEXT coverage gap, which is
metadata-resolvable — an identity problem, never a photograph problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from card_reviewer.knowledge.models import Rule

from .context import CardContext
from .enums import RuleEvaluability

__all__ = [
    "UNKNOWN_PRODUCT_CONTEXT",
    "ScopedRule",
    "applicable",
    "rule_content",
    "scope_rules",
    "unevaluable_reasons",
]

#: Declared in the detectability taxonomy as METADATA_RESOLVABLE, so it
#: requests card identification rather than a better photograph.
UNKNOWN_PRODUCT_CONTEXT = "UNKNOWN_PRODUCT_CONTEXT"


@dataclass(frozen=True)
class ScopedRule:
    rule: Rule
    evaluability: RuleEvaluability
    reason: str = ""


def scope_rules(rules: list[Rule], context: CardContext) -> list[ScopedRule]:
    """Tag each rule with whether the context it needs is actually known.

    Every rule is returned either way — narrowing the set here would be the
    silent rubric-narrowing this gate exists to prevent.
    """
    out: list[ScopedRule] = []
    for rule in rules:
        type_unknown = (
            bool(rule.applies_to.card_types) and context.canonical_card_types is None
        )
        set_unknown = bool(rule.applies_to.sets) and context.canonical_sets is None
        if type_unknown or set_unknown:
            out.append(
                ScopedRule(rule, RuleEvaluability.UNEVALUABLE, UNKNOWN_PRODUCT_CONTEXT)
            )
        else:
            out.append(ScopedRule(rule, RuleEvaluability.APPLICABLE))
    return out


def unevaluable_reasons(scoped: list[ScopedRule]) -> list[str]:
    return sorted(
        {
            s.reason
            for s in scoped
            if s.evaluability is RuleEvaluability.UNEVALUABLE and s.reason
        }
    )


def applicable(scoped: list[ScopedRule]) -> list[Rule]:
    return [s.rule for s in scoped if s.evaluability is RuleEvaluability.APPLICABLE]


def rule_content(scoped: list[ScopedRule]) -> list[dict[str, str]]:
    """The applicable rules as fingerprintable CONTENT, not a version string.

    Stages that consume the rubric fingerprint what the rules actually say,
    so a rubric release leaving the applicable rules unchanged for this card
    does not invalidate their results (spec §4, values not signatures).
    """
    return [
        {
            "id": r.id,
            "category": r.category.value,
            "statement": r.statement,
            "evidence_type": r.evidence_type.value,
            "confidence": r.confidence.value,
        }
        for r in applicable(scoped)
    ]
