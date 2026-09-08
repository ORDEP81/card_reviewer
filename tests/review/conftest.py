"""Shared fixtures — and the enforcement half of the no-live-API rule."""

import socket

import pytest

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.context import CardContext
from card_reviewer.review.enums import Provenance
from card_reviewer.review.evaluability import applicable, scope_rules


class LiveApiCallAttempted(RuntimeError):
    """A test tried to open a network connection."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test opens a socket. Ever.

    `test_no_live_api.py` scans the source for constructor SPELLINGS, which
    catches the shapes it can name and cannot catch an indirect one — bind
    the provider to a variable, call `.assess()` on it later, and the regex
    sees nothing while the call bills. That route is one line away now that
    a test constructs the real provider.

    A scan says what the code looks like; this says what it may DO. Both
    are wanted: the scan names the mistake at the point it is written, this
    stops it whatever it looks like.
    """
    def refuse(*args, **kwargs):
        raise LiveApiCallAttempted(
            "a test tried to open a network connection. Automated tests must "
            "never reach a provider — use FakeProvider or a saved fixture. "
            "The real-provider smoke test is manual and explicit."
        )

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


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

