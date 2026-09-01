import pytest
from pydantic import BaseModel

from card_reviewer.review.pipeline import StageRunner, StageValidationError
from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository


class Out(BaseModel):
    n: int


@pytest.fixture
def runner(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return StageRunner(SqliteRepository(conn))


_PRE = {"preflight_version": "1.0.0", "config": {}}


def test_a_second_identical_run_reuses_the_cached_result(runner):
    calls = []
    runner.run("preflight", {"image_hash": "h"}, _PRE,
               lambda: calls.append(1) or {"n": 1}, image_hash="h")
    runner.run("preflight", {"image_hash": "h"}, _PRE,
               lambda: calls.append(1) or {"n": 1}, image_hash="h")
    assert len(calls) == 1


def test_changed_inputs_recompute(runner):
    calls = []
    for h in ("h1", "h2"):
        runner.run("preflight", {"image_hash": h}, _PRE,
                   lambda: calls.append(1) or {"n": 1}, image_hash=h)
    assert len(calls) == 2


def test_a_bumped_producer_version_recomputes(runner):
    calls = []
    for version in ("1.0.0", "1.0.1"):
        runner.run("preflight", {"image_hash": "h"},
                   {"preflight_version": version, "config": {}},
                   lambda: calls.append(1) or {"n": 1}, image_hash="h")
    assert len(calls) == 2


def test_run_with_id_returns_the_row_the_output_was_stored_in(runner):
    """review carries foreign keys to exact rows, so the id must travel with
    the output rather than being re-derived afterwards."""
    out, row_id = runner.run_with_id("preflight", {"image_hash": "h"}, _PRE,
                                     lambda: {"n": 1}, image_hash="h")
    again, again_id = runner.run_with_id("preflight", {"image_hash": "h"}, _PRE,
                                         lambda: {"n": 2}, image_hash="h")
    assert out == again and row_id == again_id
    assert isinstance(row_id, int)


def test_a_failure_is_recorded_and_never_cached(runner):
    def boom():
        raise RuntimeError("provider exploded")

    versions = {"provider": "anthropic", "model": "m", "prompt_version": "1",
                "inference_params": {}}
    with pytest.raises(RuntimeError):
        runner.run("vision", {"provider_evidence_payload": {}}, versions, boom,
                   candidate_id="c1")
    calls = []
    out = runner.run("vision", {"provider_evidence_payload": {}}, versions,
                     lambda: calls.append(1) or {"ok": True}, candidate_id="c1")
    assert out == {"ok": True} and len(calls) == 1


def test_output_failing_schema_validation_is_never_cached(runner):
    """A row exists only for an output that ran to completion AND validated."""
    with pytest.raises(StageValidationError):
        runner.run("preflight", {"image_hash": "h"}, _PRE,
                   lambda: {"n": "not an int"}, schema=Out, image_hash="h")
    calls = []
    out = runner.run("preflight", {"image_hash": "h"}, _PRE,
                     lambda: calls.append(1) or {"n": 1}, schema=Out,
                     image_hash="h")
    assert out == {"n": 1} and len(calls) == 1


def test_a_validation_failure_is_recorded_as_an_attempt(runner):
    with pytest.raises(StageValidationError):
        runner.run("preflight", {"image_hash": "h"}, _PRE,
                   lambda: {"wrong": True}, schema=Out, image_hash="h")
    rows = runner._repo._conn.execute(
        "SELECT error_kind FROM stage_attempt").fetchall()
    assert any("Validation" in r[0] for r in rows)


def test_an_exception_is_recorded_once_not_twice(runner):
    """Double-recording would double-count every failure in cost accounting."""
    with pytest.raises(RuntimeError):
        runner.run("preflight", {"image_hash": "h"}, _PRE,
                   lambda: (_ for _ in ()).throw(RuntimeError("x")),
                   image_hash="h")
    count = runner._repo._conn.execute(
        "SELECT COUNT(*) FROM stage_attempt").fetchone()[0]
    assert count == 1


def test_an_off_run_never_satisfies_a_deep_lookup(runner):
    """The routing cache bug this design exists to prevent."""
    calls = []
    versions = {"routing_policy_version": "1.0.0"}
    for mode in ("off", "deep"):
        runner.run("routing", {"mode": mode, "heuristic_output": {}}, versions,
                   lambda: calls.append(1) or {"call_vision": mode == "deep"},
                   candidate_id="c1")
    assert len(calls) == 2


def test_an_unknown_stage_is_rejected_rather_than_cached_untyped(runner):
    with pytest.raises(KeyError):
        runner.run("not_a_stage", {}, {}, lambda: {}, candidate_id="c1")
