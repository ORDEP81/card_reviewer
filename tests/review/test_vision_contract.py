"""What the provider returns is data, not truth, and must be checked.

`parse_assessment` already validates the assessability keys and rejects a
finding citing an artifact that was never sent. Finding CATEGORIES went
unchecked, and `resolve_relevance` sets psa10_relevant from
`category in CATEGORIES` — so a category we do not recognise silently
vanished from the verdict, the score AND the grade.

Reproduced by the reviewer: an observed, confidence-0.97 crease reported
under the category "Surface" instead of "surface" yields verdict PASS, score
100, grade 10. A capitalisation from the model turns a confirmed crease into
a gem candidate.
"""

import pytest

from card_reviewer.review.taxonomy import CATEGORIES
from card_reviewer.review.vision.provider import (
    Assessment, GemView, ProviderContractError, VisionFinding, parse_assessment,
)


def _payload(category="surface", **finding):
    base = {
        "defect_type": "crease", "category": category, "state": "observed",
        "confidence": 0.97, "severity": "severe", "psa10_relevant": True,
        "evidence_artifact_ids": ["art-1"],
    }
    base.update(finding)
    return {
        "category_assessability": {c: True for c in CATEGORIES},
        "gem_view": GemView.VISIBLE_DISQUALIFIER.value,
        "findings": [base],
    }


ALLOWED = {"art-1"}


def test_a_known_category_parses():
    assessment = parse_assessment(_payload(), ALLOWED)
    assert assessment.findings[0].category == "surface"


@pytest.mark.parametrize("category", ["Surface", "SURFACE", "surfaces",
                                      "surface_damage", "corner", ""])
def test_an_unrecognised_category_is_a_contract_error(category):
    """Refusing is what the pipeline already does for an unknown artifact id,
    and it routes to a provider failure -> categories unassessed -> PARTIAL ->
    no PASS. Silently dropping the finding is the one outcome that must not
    happen, because it drops a defect rather than the response."""
    with pytest.raises(ProviderContractError) as raised:
        parse_assessment(_payload(category), ALLOWED)
    assert category in str(raised.value) or "categor" in str(raised.value)


def test_the_error_names_the_categories_we_do_know():
    with pytest.raises(ProviderContractError) as raised:
        parse_assessment(_payload("Surface"), ALLOWED)
    assert "surface" in str(raised.value)


def test_a_finding_is_never_silently_dropped_for_its_category():
    """The property behind all of the above: whatever comes back, either it
    reaches relevance resolution or the whole response is refused."""
    from card_reviewer.review.relevance import resolve_relevance

    assessment = parse_assessment(_payload(), ALLOWED)
    assert all(f.category in CATEGORIES for f in assessment.findings)
