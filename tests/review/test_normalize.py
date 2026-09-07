import pytest

from card_reviewer.review.enums import Provenance
from card_reviewer.review.normalize import CardContextNormalizer
from card_reviewer.review.vocabulary import CARD_TYPE_VOCABULARY, SET_VOCABULARY


@pytest.fixture
def norm():
    return CardContextNormalizer()


def test_the_vocabulary_matches_what_subsystem_b_actually_scopes_on():
    """Inspected from the live rubric, not invented: SURFACE_SHINY_001 is the
    only scoped rule and it uses chrome/refractor/foil."""
    assert set(CARD_TYPE_VOCABULARY.values()) == {"chrome", "refractor", "foil"}
    assert SET_VOCABULARY == {}


@pytest.mark.parametrize("raw,expected", [
    ("Chrome", "chrome"), ("  CHROME  ", "chrome"),
    ("Topps Chrome", "chrome"), ("Bowman Chrome", "chrome"),
    ("Refractor", "refractor"), ("refractors", "refractor"),
    ("Prizm", "refractor"), ("holo", "foil"), ("Holofoil", "foil"),
])
def test_aliases_and_capitalization_normalize_to_canonical_values(
        norm, raw, expected):
    assert norm.normalize(supplied_card_type=raw).canonical_card_types == [expected]


def test_unrecognized_values_become_unknown_never_the_nearest_neighbour(norm):
    """Guessing 'Prizm Silver Mojo' into chrome would apply a rule the owner
    never sanctioned. No fuzzy matching."""
    ctx = norm.normalize(supplied_card_type="Prizm Silver Mojo /25")
    assert ctx.canonical_card_types is None
    assert ctx.provenance is Provenance.UNKNOWN


def test_raw_values_are_preserved_alongside_canonical_ones(norm):
    ctx = norm.normalize(supplied_card_type="Topps Chrome")
    assert ctx.raw_card_type == "Topps Chrome"
    assert ctx.canonical_card_types == ["chrome"]


def test_unknown_context_yields_none_never_an_empty_list(norm):
    """Subsystem B distinguishes them: None means unconstrained, [] means
    'known to be empty' and would drop every scoped rule."""
    ctx = norm.normalize()
    assert ctx.canonical_card_types is None
    assert ctx.canonical_sets is None
    assert ctx.canonical_card_types != []


def test_supplied_metadata_outranks_title_inference(norm):
    ctx = norm.normalize(raw_title="2023 Bowman Chrome Auto",
                         supplied_card_type="foil")
    assert ctx.canonical_card_types == ["foil"]
    assert ctx.provenance is Provenance.SUPPLIED


def test_title_inference_is_marked_inferred_with_a_confidence(norm):
    ctx = norm.normalize(raw_title="2023 Topps Chrome Julio Rodriguez #150")
    assert ctx.canonical_card_types == ["chrome"]
    assert ctx.provenance is Provenance.INFERRED
    assert 0.0 < ctx.confidence < 1.0


def test_a_supplied_value_carries_full_confidence(norm):
    assert norm.normalize(supplied_card_type="chrome").confidence == 1.0


def test_the_set_axis_is_unknown_because_no_rule_is_set_scoped(norm):
    ctx = norm.normalize(raw_title="2023 Topps Chrome", supplied_set="Topps Chrome")
    assert ctx.canonical_sets is None


def test_a_title_naming_two_products_reports_both(norm):
    ctx = norm.normalize(raw_title="Topps Chrome Refractor parallel")
    assert ctx.canonical_card_types == ["chrome", "refractor"]


def test_a_substring_inside_a_longer_word_does_not_match(norm):
    """'Chromed' is not 'Chrome'. Matching a fragment would silently scope
    the rubric on a product the listing never named."""
    assert norm.normalize(raw_title="Chromedome promo").canonical_card_types is None


def test_is_known_reflects_whether_the_rubric_can_be_scoped(norm):
    assert norm.normalize(supplied_card_type="chrome").is_known is True
    assert norm.normalize().is_known is False


def test_a_card_context_round_trips_through_json(norm):
    """role_context is a cached stage; its output must survive SQLite."""
    import json

    from card_reviewer.review.context import CardContext

    ctx = norm.normalize(raw_title="2023 Topps Chrome")
    assert CardContext.model_validate(json.loads(ctx.model_dump_json())) == ctx
