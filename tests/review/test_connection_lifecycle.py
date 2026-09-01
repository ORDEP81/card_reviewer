"""Connections are a finite resource; a screening run must not hoard them.

The suite raises ResourceWarning to an error, which is how the leak these
tests close was found in the first place.
"""

import sqlite3

import pytest

from card_reviewer.review import service
from card_reviewer.review.enums import Mode
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.models import CandidateInput


@pytest.fixture
def card(tmp_path):
    path = tmp_path / "card.png"
    path.write_bytes(render_png(CardSpec()))
    return CandidateInput(source="manual", title="2023 Topps Chrome",
                          image_paths=[path])


@pytest.fixture
def counted(monkeypatch):
    """Count every connection the service opens and every one it closes."""
    opened: list[sqlite3.Connection] = []
    real = service.connect

    def counting(path):
        conn = real(path)
        opened.append(conn)
        return conn

    monkeypatch.setattr(service, "connect", counting)
    return opened


def _is_closed(conn):
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def test_a_context_closes_its_connection(tmp_path, counted):
    context = service.open_context(tmp_path)
    context.close()
    assert _is_closed(counted[0])


def test_a_context_closes_on_exit_from_a_with_block(tmp_path, counted):
    with service.open_context(tmp_path):
        pass
    assert _is_closed(counted[0])


def test_closing_twice_is_not_an_error(tmp_path):
    context = service.open_context(tmp_path)
    context.close()
    context.close()


def test_a_review_leaves_no_connection_open(tmp_path, card, counted):
    service.review_card(card, Mode.OFF, tmp_path)
    assert counted, "the service opened no connection at all"
    assert all(_is_closed(c) for c in counted)


def test_a_review_through_a_caller_supplied_context_opens_nothing_new(
        tmp_path, card, counted):
    with service.open_context(tmp_path) as context:
        service.review_card(card, Mode.OFF, context=context)
        assert len(counted) == 1


def test_a_caller_supplied_context_survives_the_review(tmp_path, card):
    """review_card must not close a connection it did not open."""
    with service.open_context(tmp_path) as context:
        service.review_card(card, Mode.OFF, context=context)
        assert not _is_closed(context.repo._conn)
