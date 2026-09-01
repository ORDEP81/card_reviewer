import pytest

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.context import CardContext
from card_reviewer.review.enums import Provenance
from card_reviewer.review.evaluability import applicable, scope_rules


@pytest.fixture(scope="session")
def rubric():
    return load_active_rubric()


@pytest.fixture
def rubric_rules(rubric):
    """The applicable rules for a known-chrome card, as the pipeline passes
    them to the manifest builder."""
    context = CardContext(canonical_card_types=["chrome"],
                          provenance=Provenance.SUPPLIED, confidence=1.0)
    return applicable(scope_rules(rubric.for_card(["chrome"], None), context))


@pytest.fixture
def rubric_scoped(rubric):
    """Scoped rules for a known-chrome card, as the pipeline passes them to
    relevance resolution inside combine."""
    context = CardContext(canonical_card_types=["chrome"],
                          provenance=Provenance.SUPPLIED, confidence=1.0)
    return scope_rules(rubric.for_card(["chrome"], None), context)
