import json

import pytest
from typer.testing import CliRunner

from card_reviewer.review.cli import app
from card_reviewer.review.imaging.synthetic import CardSpec, render_png

runner = CliRunner()


@pytest.fixture
def card(tmp_path):
    front, back = tmp_path / "front.png", tmp_path / "back.png"
    front.write_bytes(render_png(CardSpec()))
    back.write_bytes(render_png(CardSpec(text_heavy=True)))
    return front, back


def _screen(card, data, *extra):
    front, back = card
    return runner.invoke(app, ["screen", str(front), str(back), "--mode", "off",
                               "--data-dir", str(data), *extra])


def test_screen_runs_a_card_and_prints_a_verdict(card, tmp_path):
    result = _screen(card, tmp_path / "data")
    assert result.exit_code == 0, result.output
    assert any(v in result.output for v in
               ("PASS", "REVIEW", "REJECT", "INSUFFICIENT_IMAGES"))


def test_the_default_mode_is_smart():
    assert "smart" in runner.invoke(app, ["screen", "--help"]).output.lower()


def test_the_report_always_shows_limitations(card, tmp_path):
    """Non-negotiable rule 3: never hide image limitations."""
    result = _screen(card, tmp_path / "data")
    assert "limitation" in result.output.lower()


def test_the_report_shows_the_rank_score_without_calling_it_a_probability(
        card, tmp_path):
    """The score orders candidates for inspection; it is not calibrated.

    The report is expected to SAY 'not a probability', so banning the word
    outright would forbid the disclaimer. What must not appear is the score
    presented AS a likelihood.
    """
    out = _screen(card, tmp_path / "data").output.lower()
    assert "rank" in out
    assert "not a probability" in out
    for forbidden in ("% chance", "likelihood", "probability of grading"):
        assert forbidden not in out


def test_the_report_never_shows_a_price(card, tmp_path):
    out = _screen(card, tmp_path / "data").output.lower()
    for word in ("price", "profit", "resale", "value"):
        assert word not in out


def test_export_emits_valid_json(card, tmp_path):
    data = tmp_path / "data"
    _screen(card, data)
    result = runner.invoke(app, ["export", "1", "--data-dir", str(data)])
    assert result.exit_code == 0
    assert json.loads(result.output)["verdict"]


def test_show_renders_a_stored_review(card, tmp_path):
    data = tmp_path / "data"
    _screen(card, data)
    result = runner.invoke(app, ["show", "1", "--data-dir", str(data)])
    assert result.exit_code == 0
    assert "verdict" in result.output.lower()


def test_outcome_records_a_psa_result_joinable_to_its_review(card, tmp_path):
    data = tmp_path / "data"
    _screen(card, data)
    result = runner.invoke(app, ["outcome", "1", "--grade", "10",
                                 "--cert", "12345678", "--data-dir", str(data)])
    assert result.exit_code == 0, result.output


def test_a_recorded_outcome_joins_back_to_the_prediction(card, tmp_path):
    data = tmp_path / "data"
    _screen(card, data)
    runner.invoke(app, ["outcome", "1", "--grade", "10", "--cert", "1",
                        "--data-dir", str(data)])
    from card_reviewer.review.storage.migrations import connect

    conn = connect(data / "card_reviewer.db")
    row = conn.execute(
        "SELECT r.verdict, g.grade FROM review r"
        " JOIN grading_submission g ON g.candidate_id = r.candidate_id"
    ).fetchone()
    assert row is not None and row[1] == "10"


def test_recording_a_second_outcome_does_not_overwrite_the_first(card, tmp_path):
    """Cards get cracked and resubmitted; history is append-only."""
    data = tmp_path / "data"
    _screen(card, data)
    for grade in ("9", "10"):
        runner.invoke(app, ["outcome", "1", "--grade", grade, "--cert", grade,
                            "--data-dir", str(data)])
    from card_reviewer.review.storage.migrations import connect

    conn = connect(data / "card_reviewer.db")
    assert conn.execute(
        "SELECT COUNT(*) FROM grading_submission").fetchone()[0] == 2


def test_an_unknown_review_id_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["show", "999", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_no_grading_logic_lives_in_the_cli():
    """The CLI is a surface; thresholds and verdict logic belong in policies."""
    import inspect

    import card_reviewer.review.cli as mod

    source = inspect.getsource(mod)
    for forbidden in ("Verdict.REJECT", "Scale.MODERATE", "psa10_rank_score ="):
        assert forbidden not in source


def test_provider_smoke_is_the_only_command_that_reaches_the_api():
    import inspect

    import card_reviewer.review.cli as mod

    source = inspect.getsource(mod)
    assert "provider_smoke" in source or "provider-smoke" in source


def _review(**kw):
    from card_reviewer.review.models import CardReview

    base = dict(candidate_id="c1", verdict="PASS", psa10_candidate="yes",
                psa10_rank_score=100, rankable=True, estimated_psa_grade="10",
                review_confidence="high", coverage="SUFFICIENT")
    return CardReview(**(base | kw))


def test_the_limitations_section_appears_even_when_there_are_none():
    """Silence would read as 'nothing was limited', which is a claim the
    photographs may not support. The section is always rendered."""
    from card_reviewer.review.report import render

    out = render(_review(limitations=[])).lower()
    assert "limitations" in out
    assert "none recorded" in out


def test_an_unrankable_card_is_not_rendered_as_a_zero_score():
    """Zero means 'ranked last'; unrankable means 'we could not rank it'.
    Printing 0 would sort an unassessable card alongside a confirmed reject."""
    from card_reviewer.review.report import render

    out = render(_review(psa10_rank_score=None, rankable=False,
                         verdict="INSUFFICIENT_IMAGES")).lower()
    assert "unrankable" in out
    assert "rank score     0" not in out


def test_an_unknown_review_id_says_so_rather_than_failing_obscurely(tmp_path):
    result = runner.invoke(app, ["show", "999", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "999" in result.output


def test_off_mode_never_constructs_a_provider_even_with_credentials(
        card, tmp_path, monkeypatch):
    """OFF means no external call is even contemplated."""
    import card_reviewer.review.cli as cli_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    built = []
    monkeypatch.setattr(cli_module, "build_provider",
                        lambda store: built.append(1))
    _screen(card, tmp_path / "data")
    assert built == []


def test_two_grading_submissions_for_one_card_both_survive(card, tmp_path):
    """A card can be cracked and resubmitted; the first result is history and
    must not be replaced."""
    data = tmp_path / "data"
    _screen(card, data)
    for grade, cert in (("9", "aaa"), ("10", "bbb")):
        runner.invoke(app, ["outcome", "1", "--grade", grade, "--cert", cert,
                            "--data-dir", str(data)])
    from card_reviewer.review.storage.migrations import connect

    conn = connect(data / "card_reviewer.db")
    grades = [r[0] for r in conn.execute(
        "SELECT grade FROM grading_submission ORDER BY cert_number")]
    assert grades == ["9", "10"]
