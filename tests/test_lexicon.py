import pytest

from card_reviewer.knowledge import lexicon


@pytest.fixture
def lex(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text(
        """
version: "1"
demonstration_weight: 2.0
categories:
  corners:
    corner: 1.0
    soft corner: 2.0
  surface:
    print line: 3.0
  demonstration:
    look right here: 1.0
    you can see: 1.0
"""
    )
    return lexicon.load(path)


def test_scores_sum_matched_term_weights(lex):
    result = lex.score("There is a soft corner and a corner ding here.")
    # "soft corner" (2.0) + "corner" (1.0) = 3.0
    assert result.score == pytest.approx(3.0)
    assert set(result.matched_terms) == {"soft corner", "corner"}


def test_matching_is_case_insensitive(lex):
    assert lex.score("PRINT LINE across the front").score == pytest.approx(3.0)


def test_a_term_counts_once_per_cue(lex):
    once = lex.score("print line")
    thrice = lex.score("print line print line print line")
    assert once.score == thrice.score


def test_categories_exclude_demonstration(lex):
    result = lex.score("Look right here at the corner.")
    assert result.categories == ["corners"]
    assert result.visual_cue is True


def test_visual_cue_false_without_demonstration_terms(lex):
    result = lex.score("Centering matters a lot on this one corner.")
    assert result.visual_cue is False


def test_unrelated_text_scores_zero(lex):
    result = lex.score("Welcome back to the channel, smash that like button.")
    assert result.score == 0.0
    assert result.categories == []


def test_real_lexicon_file_loads_and_has_every_category():
    """The shipped lexicon must cover every category the spec names."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    lex = lexicon.load(repo / "knowledge" / "segmentation_lexicon.yaml")
    for expected in ("centering", "corners", "edges", "surface", "outcomes", "demonstration"):
        assert expected in lex.categories, f"lexicon missing category: {expected}"
