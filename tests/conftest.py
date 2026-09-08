"""No test in this repository opens a network connection.

`tests/review/test_no_live_api.py` scans the source for constructor
SPELLINGS. That catches the shapes it can name and cannot express an
indirect one — bind a provider to a variable and call `.assess()` later and
the regex sees nothing while the call bills. A scan says what the code
looks like; this says what it may DO, and both are wanted.

Patched in `pytest_configure`, which runs BEFORE collection, so it also
covers a connection made at module import time or from a session-scoped
fixture. An autouse function-scoped fixture — the obvious first attempt —
is installed too late for either, and a `provider` fixture is exactly the
sort of thing written at session scope.

It patches `socket.socket`, which every library in this tree reaches the
network through, verified against `http.client`, `urllib`, `httpx` sync
and async, `asyncio.open_connection`, `ssl`, and the real `anthropic` SDK.
It does NOT patch the C-level `_socket.socket`, so code deliberately
importing that could still connect; nothing here does, and this stops
mistakes rather than evasion.
"""

import socket

import pytest


class LiveApiCallAttempted(RuntimeError):
    """A test tried to open a network connection."""


def _refuse(*args, **kwargs):
    raise LiveApiCallAttempted(
        "a test tried to open a network connection. Automated tests must "
        "never reach a provider — use FakeProvider or a saved fixture. The "
        "real-provider smoke test is manual and explicit."
    )


def pytest_configure(config):
    socket.socket.connect = _refuse
    socket.socket.connect_ex = _refuse
    socket.create_connection = _refuse


@pytest.fixture
def live_api_error():
    """The exception the block raises, for tests that assert on it."""
    return LiveApiCallAttempted
