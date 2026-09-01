"""Human-readable rendering of a CardReview (spec §16).

Limitations are always shown, never elided — non-negotiable rule 3. The rank
score is labelled as a ranking score and never as a probability.
"""

from __future__ import annotations

from .models import CardReview

__all__ = ["render"]


def render(review: CardReview) -> str:
    lines: list[str] = [
        f"Candidate      {review.candidate_id}",
        f"Title          {review.title or '(untitled)'}",
        f"Verdict        {review.verdict}",
        f"PSA 10 chance  {review.psa10_candidate}",
        f"Rank score     {_score(review)}  (ranking score, not a probability)",
        f"Rough grade    {review.estimated_psa_grade or 'not estimable'}",
        f"Confidence     {review.review_confidence}  (in the review, not the card)",
        f"Coverage       {review.coverage}",
    ]

    if review.defects_found:
        lines.append("")
        lines.append("Findings")
        for finding in review.defects_found:
            lines.append(
                f"  [{finding['state']:14}] {finding['category']}/"
                f"{finding['defect_type']}  ({finding['producer']})"
            )

    # Always rendered, even when empty: silence would read as "nothing was
    # limited", which is a claim the photographs may not support.
    lines.append("")
    lines.append(f"Limitations ({len(review.limitations)})")
    if not review.limitations:
        lines.append("  none recorded")
    for limitation in review.limitations:
        lines.append(
            f"  {limitation['reason_code']:24} "
            f"{limitation['undetectability_class']:18} "
            f"{limitation['face']}/{limitation['category']}"
        )

    if review.recommended_additional_photos:
        lines.append("")
        lines.append("Better photographs would help")
        for request in review.recommended_additional_photos:
            lines.append(f"  - {request}")

    if review.card_identification_request:
        lines.append("")
        lines.append("Identify the card: a product-scoped rule could not be "
                     "applied without knowing what this is.")

    if review.reasoning:
        lines.append("")
        lines.append(f"Why  {review.reasoning}")

    return "\n".join(lines)


def _score(review: CardReview) -> str:
    return "unrankable" if review.psa10_rank_score is None else str(
        review.psa10_rank_score)
