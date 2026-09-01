"""Rubric evaluation against assembled CV evidence (spec §10).

Emits findings in the shared §9 vocabulary. The promotion limit is the rule
that matters: measurement-establishable defect types may reach `observed`,
interpretive ones may not, no matter how confident the pixel evidence looks.
That is what stops OFF mode manufacturing a confirmed defect out of
high-contrast pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .enums import FindingState, Scale
from .evaluability import ScopedRule, applicable, unevaluable_reasons
from .findings import Finding, FindingProducer, Severity
from .provenance import EvidenceRef, NormalizedBox
from .taxonomy import CATEGORIES, Promotion, promotion_of
from .versions import SCORER_VERSION

if TYPE_CHECKING:
    from .assembly import Assembled

__all__ = ["HeuristicResult", "best_detectability", "evaluate"]

MIN_DETECTABILITY_FOR_OBSERVED = Scale.MODERATE
MIN_CONFIDENCE_FOR_OBSERVED = 0.8

# PSA's own tolerance is "approximately 55/45" — five percentage points off
# centre — and explicitly not a hard cutoff, so a breach is only reported
# past it, and severely past it grades worse.
CENTERING_TOLERANCE_PP = 5.0
CENTERING_SEVERE_PP = 15.0
CENTERING_CONFIDENCE = 0.9

# The whole card: a centering breach is not localized to a region.
_WHOLE_CARD = NormalizedBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)


class HeuristicResult(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    unevaluable_reasons: list[str] = Field(default_factory=list)
    scorer_version: str = SCORER_VERSION


def best_detectability(
    detectability: dict[tuple[Any, str, str], Scale], category: str, defect_type: str
) -> Scale:
    """Max over faces for one (category, defect_type).

    Callers hold a 3-tuple-keyed map; looking it up with a 2-tuple would miss
    every time and return the default, which is how an invariant quietly
    stops binding. NONE when nothing is registered — absent evidence must
    never read as adequate evidence.
    """
    values = [
        v for (_face, c, d), v in detectability.items()
        if c == category and d == defect_type
    ]
    return Scale(max(values)) if values else Scale.NONE


def _state_for(
    category: str, defect_type: str, confidence: float, detectability: Scale
) -> FindingState:
    """Certainty, and only certainty. Rule authority plays no part here."""
    if detectability < MIN_DETECTABILITY_FOR_OBSERVED:
        return FindingState.SUSPECTED
    if promotion_of(category, defect_type) is Promotion.INTERPRETIVE:
        return FindingState.SUSPECTED
    if confidence < MIN_CONFIDENCE_FOR_OBSERVED:
        return FindingState.SUSPECTED
    return FindingState.OBSERVED


def evaluate(assembled: Assembled, scoped_rules: list[ScopedRule]) -> HeuristicResult:
    rules_by_category: dict[str, list[str]] = {}
    for rule in applicable(scoped_rules):
        rules_by_category.setdefault(rule.category.value, []).append(rule.id)

    detectability = assembled.detectability
    findings: list[Finding] = []

    for anomaly in assembled.anomalies:
        category = anomaly["category"]
        defect_type = anomaly["defect_type"]
        refs = _refs_for(assembled, category, defect_type, anomaly.get("region"))
        if not refs:
            # A finding with no evidence cannot support anything downstream.
            continue
        confidence = float(anomaly.get("confidence", 0.0))
        findings.append(
            Finding(
                defect_type=defect_type,
                category=category,
                state=_state_for(
                    category, defect_type, confidence,
                    best_detectability(detectability, category, defect_type),
                ),
                producer=FindingProducer.HEURISTIC,
                confidence=confidence,
                psa10_relevant=category in CATEGORIES,
                evidence=refs,
                severity=(
                    Severity(anomaly["severity"]) if anomaly.get("severity") else None
                ),
                # A location is REQUIRED: fusion correlates by overlapping
                # region, and a finding without one can never fuse, so the
                # same physical defect seen by both producers would be
                # penalized twice.
                location=_location_of(anomaly, refs),
                rule_ids=rules_by_category.get(category, []),
                explanation=f"CV anomaly candidate in {category}/{defect_type}",
            )
        )

    findings.extend(_centering_findings(assembled, detectability, rules_by_category))
    return HeuristicResult(
        findings=findings, unevaluable_reasons=unevaluable_reasons(scoped_rules)
    )


def _refs_for(
    assembled: Assembled, category: str, defect_type: str, region: str | None
) -> list[EvidenceRef]:
    """Prefer the refs for the anomaly's own region, so its location is that
    region rather than the union of every region in the category."""
    key = f"{category}:{defect_type}"
    if region:
        scoped = assembled.evidence_refs.get(f"{key}:{region}")
        if scoped:
            return scoped
    return assembled.evidence_refs.get(key) or []


def _location_of(
    anomaly: dict[str, Any], refs: list[EvidenceRef]
) -> NormalizedBox | None:
    if anomaly.get("region_box"):
        return NormalizedBox.model_validate(anomaly["region_box"])
    boxes = [r.region for r in refs if r.region is not None]
    if not boxes:
        return None
    return NormalizedBox(
        x0=min(b.x0 for b in boxes), y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes), y1=max(b.y1 for b in boxes),
    )


def _centering_findings(
    assembled: Assembled,
    detectability: dict[tuple[Any, str, str], Scale],
    rules_by_category: dict[str, list[str]],
) -> list[Finding]:
    """Centering is a measurement, not an anomaly candidate.

    It never appears in `assembled.anomalies`, so without its own evaluation
    a grossly miscut card would produce no finding at all.
    """
    centering = assembled.centering
    if not centering.get("measurable"):
        return []
    refs = assembled.evidence_refs.get("centering:border_ratio") or []
    if not refs:
        return []

    worst = max(
        abs(float(centering.get("horizontal", 50.0)) - 50.0),
        abs(float(centering.get("vertical", 50.0)) - 50.0),
    )
    if worst <= CENTERING_TOLERANCE_PP:
        return []

    return [
        Finding(
            defect_type="border_ratio",
            category="centering",
            state=_state_for(
                "centering", "border_ratio", CENTERING_CONFIDENCE,
                best_detectability(detectability, "centering", "border_ratio"),
            ),
            producer=FindingProducer.HEURISTIC,
            confidence=CENTERING_CONFIDENCE,
            psa10_relevant=True,
            severity=(
                Severity.SEVERE if worst > CENTERING_SEVERE_PP else Severity.MODERATE
            ),
            location=_WHOLE_CARD,
            evidence=refs,
            rule_ids=rules_by_category.get("centering", []),
            explanation=(
                f"centering off-centre by {worst:.1f} percentage points, "
                "beyond the PSA 10 tolerance"
            ),
        )
    ]
