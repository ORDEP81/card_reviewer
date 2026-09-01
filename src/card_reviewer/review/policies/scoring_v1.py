"""The three summary values (Decision 2).

They answer three different questions and must never collapse into one:

  psa10_rank_score     — how should this card sort against others?
  estimated_psa_grade  — what coarse grade does the evidence support?
  review_confidence    — how much do we trust this assessment at all?

Every weight and threshold in the system's scoring lives here. No magic
number appears in any other module.
"""

from __future__ import annotations

from ..enums import Authority, Coverage, FindingState, ReviewConfidence
from ..findings import Finding, Severity
from ..versions import SCORING_POLICY_VERSION

__all__ = [
    "COVERAGE_PENALTY",
    "PENALTIES",
    "SCORING_POLICY_VERSION",
    "estimated_grade",
    "rank_score",
    "review_confidence",
]

MAX_SCORE = 100
MIN_SCORE = 0

# Keyed by (state, authority, i1_satisfied). Penalties are non-negative and
# monotone in state, which is what makes the monotonicity properties hold by
# construction rather than by luck.
#
# Only a binding disqualifier that actually satisfies I1 floors the score. An
# observed-but-unresolved finding routes to REVIEW and must stay meaningfully
# rankable there, or it sorts identically to a confirmed reject and destroys
# the triage ordering the score exists to provide.
PENALTIES: dict[tuple[FindingState, Authority, bool], int] = {
    (FindingState.OBSERVED, Authority.BINDING, True): 100,
    (FindingState.OBSERVED, Authority.BINDING, False): 35,
    (FindingState.OBSERVED, Authority.ADVISORY, True): 25,
    (FindingState.OBSERVED, Authority.ADVISORY, False): 25,
    (FindingState.SUSPECTED, Authority.BINDING, True): 15,
    (FindingState.SUSPECTED, Authority.BINDING, False): 15,
    (FindingState.SUSPECTED, Authority.ADVISORY, True): 6,
    (FindingState.SUSPECTED, Authority.ADVISORY, False): 6,
}

COVERAGE_PENALTY: dict[Coverage, int] = {
    Coverage.SUFFICIENT: 0,
    Coverage.PARTIAL: 10,
}

SEVERITY_GRADE: dict[Severity, str] = {
    Severity.MINOR: "9",
    Severity.MODERATE: "8-9",
    Severity.SEVERE: "<=8",
}


def _triples(findings) -> list[tuple[Finding, Authority, bool]]:
    """Normalize to (finding, authority, i1_satisfied).

    Authority defaults to ADVISORY, never BINDING: an unmapped finding must
    not be able to reject a card (Decision 4).
    """
    out: list[tuple[Finding, Authority, bool]] = []
    for item in findings:
        if isinstance(item, tuple):
            out.append(item if len(item) == 3 else (item[0], item[1], False))
        else:
            out.append((item, Authority.ADVISORY, False))
    return out


def rank_score(findings, coverage: Coverage) -> int | None:
    """0-100 ranking heuristic. Explicitly NOT a probability.

    Expects FUSED findings (Decision 5): one physical defect penalizes once,
    however many producers saw it.
    """
    if coverage is Coverage.INADEQUATE:
        # Never manufacture a neutral-looking number for an unrankable card.
        return None
    score = MAX_SCORE - COVERAGE_PENALTY.get(coverage, 0)
    for finding, authority, i1 in _triples(findings):
        if not finding.psa10_relevant or authority is Authority.INERT:
            continue
        # not_observed and not_assessable cost nothing here by design: missing
        # evidence is already paid for in coverage and confidence, and
        # charging it again would double-count absence as a defect.
        score -= PENALTIES.get((finding.state, authority, i1), 0)
    return max(MIN_SCORE, min(MAX_SCORE, score))


def estimated_grade(findings, coverage: Coverage) -> str | None:
    """A coarse estimate from the worst CONFIRMED defect.

    Deliberately not a function of rank_score: deriving it from the score
    would imply the score is calibrated, which it is not. "Confirmed" means
    confirmed — an observed finding that fails I1 is an unresolved concern,
    not an established defect, so it must not drag the grade down.
    """
    if coverage is Coverage.INADEQUATE:
        return None
    observed = [
        f for f, _, i1 in _triples(findings)
        if f.state is FindingState.OBSERVED and f.psa10_relevant and i1
    ]
    if not observed:
        # A suspicion is not a confirmed defect and must not drag the grade
        # down as if it were. But a card the engine has an open question
        # about is not an outright 10 either: reporting one identically to a
        # pristine card is the assumption rule 2 forbids. "9-10" says what is
        # actually known — it might gem, and something unresolved says it
        # might not.
        unresolved = any(
            f.state is FindingState.SUSPECTED and f.psa10_relevant
            for f, _, _ in _triples(findings)
        )
        if coverage is Coverage.SUFFICIENT and not unresolved:
            return "10"
        return "9-10"
    severities = [f.severity for f in observed if f.severity]
    if not severities:
        return "9"
    if Severity.SEVERE in severities:
        return "<=8"
    if severities.count(Severity.MODERATE) >= 2:
        return "<=8"
    if Severity.MODERATE in severities:
        return "8-9"
    return "9"


def review_confidence(
    coverage: Coverage,
    contradictions: list,
    producers_disagreed: bool,
    card_context_known: bool,
    *,
    required_face_missing: bool = False,
) -> ReviewConfidence:
    """Confidence in the ASSESSMENT, never the probability of a PSA 10.

    `required_face_missing` is explicit because it is not inferrable from the
    coverage outcome: a front-only card is PARTIAL — rankable and forwarded —
    yet its confidence is LOW, because half the card was never seen. That
    combination is intended, not a contradiction.
    """
    if coverage is Coverage.INADEQUATE or contradictions or required_face_missing:
        return ReviewConfidence.LOW
    if coverage is Coverage.PARTIAL or producers_disagreed or (
        not card_context_known
    ):
        return ReviewConfidence.MEDIUM
    return ReviewConfidence.HIGH
