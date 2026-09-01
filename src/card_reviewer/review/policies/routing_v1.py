"""SMART routing: fires on resolvable ambiguity, not missing information.

A provider cannot recover information absent from the pixels. Sending an
occluded corner buys `insufficient_evidence` at cost, so SMART calls only
when the evidence could actually settle the question.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..enums import Coverage, FindingState, Mode, Scale
from ..versions import ROUTING_POLICY_VERSION

__all__ = [
    "MIN_DETECTABILITY_TO_RESOLVE",
    "ROUTING_POLICY_VERSION",
    "RoutingDecision",
    "decide_routing",
]

MIN_DETECTABILITY_TO_RESOLVE = Scale.MODERATE


class RoutingDecision(BaseModel):
    mode: Mode
    call_vision: bool
    trigger_reasons: list[str] = Field(default_factory=list)
    policy_version: str = ROUTING_POLICY_VERSION


def decide_routing(
    mode: Mode,
    findings: list,
    provisional: Coverage,
    detectability: dict,
) -> RoutingDecision:
    if mode is Mode.OFF:
        return RoutingDecision(mode=mode, call_vision=False,
                               trigger_reasons=["mode is OFF"])
    if mode is Mode.DEEP:
        return RoutingDecision(mode=mode, call_vision=True,
                               trigger_reasons=["mode is DEEP"])

    # The provisional gate: a card whose photographs cannot support an
    # assessment at all has nothing for the provider to resolve, and its
    # verdict cannot become PASS regardless.
    if provisional is Coverage.INADEQUATE:
        return RoutingDecision(
            mode=mode, call_vision=False,
            trigger_reasons=[
                "provisional coverage INADEQUATE — a call cannot recover "
                "information absent from the pixels"
            ],
        )

    reasons: list[str] = []
    for finding in findings:
        if finding.state is not FindingState.SUSPECTED:
            # An already-observed finding is not ambiguous; a call would buy
            # no new information.
            continue
        resolvable = any(
            value >= MIN_DETECTABILITY_TO_RESOLVE
            for (_face, _region, category, defect_type), value
            in detectability.items()
            if category == finding.category and defect_type == finding.defect_type
        )
        if resolvable:
            reasons.append(
                f"{finding.category}/{finding.defect_type} suspected and resolvable"
            )

    # `not reasons`, not `not findings`. A real card produces many
    # NOT_OBSERVED findings — the reviewer measured 18 — so requiring an
    # empty finding list made this branch effectively dead and SMART never
    # took the confirming call on a clean card. What qualifies a gem
    # candidate is that nothing is WRONG, which is what `reasons` records.
    if not reasons and provisional is Coverage.SUFFICIENT:
        reasons.append("strong gem candidate worth confirming")

    return RoutingDecision(mode=mode, call_vision=bool(reasons),
                           trigger_reasons=reasons)
