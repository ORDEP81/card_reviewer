import sqlite3

import pytest

from card_reviewer.review.storage.migrations import SCHEMA_VERSION, connect, migrate


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    migrate(c)
    return c


def test_migrate_creates_every_table_the_spec_declares(conn):
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "candidate", "image", "candidate_image", "stage_result",
        "stage_attempt", "routing_decision", "review", "candidate_outcome",
        "grading_submission",
    } <= names


def test_migrate_is_idempotent(tmp_path):
    c = connect(tmp_path / "t.db")
    assert migrate(c) == SCHEMA_VERSION
    assert migrate(c) == SCHEMA_VERSION


def test_stage_result_is_unique_on_the_cache_identity(conn):
    row = ("preflight", "fp1", "sig1", "{}", "{}", "2026-08-31T00:00:00Z",
           "h1", None)
    sql = (
        "INSERT INTO stage_result(stage, input_fingerprint, producer_signature,"
        " output_json, versions_json, created_at, image_hash, candidate_id)"
        " VALUES(?,?,?,?,?,?,?,?)"
    )
    conn.execute(sql, row)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, row)


def test_candidate_outcome_has_no_price_or_purchase_column(conn):
    """Non-negotiable rule 14: storing price beside returned grades puts ROI
    analysis one join away."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(candidate_outcome)")}
    assert not (cols & {"price", "purchased", "cost", "paid", "value"})


def test_multiple_grading_submissions_per_candidate_are_allowed(conn):
    """Cards get returned ungraded, resubmitted, cracked, or crossed."""
    conn.execute(
        "INSERT INTO candidate(id, source, created_at) VALUES('c1','manual','t')")
    for n in ("s1", "s2"):
        conn.execute(
            "INSERT INTO grading_submission(id, candidate_id, grader, status)"
            " VALUES(?, 'c1', 'PSA', 'submitted')", (n,))
    assert conn.execute(
        "SELECT COUNT(*) FROM grading_submission WHERE candidate_id='c1'"
    ).fetchone()[0] == 2


def test_review_carries_the_foreign_keys_the_pipeline_needs(conn):
    names = {c[1] for c in conn.execute("PRAGMA table_info(review)")}
    assert {
        "routing_decision_id", "coverage_provisional_result_id",
        "coverage_result_id", "combine_result_id", "vision_result_id",
        "heuristic_result_id", "review_confidence", "rubric_version",
    } <= names


def test_foreign_keys_are_enforced(conn):
    """Without this pragma the FK columns are decorative and a review could
    reference a candidate that does not exist."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO routing_decision(candidate_id, policy_version, mode,"
            " call_vision, input_fingerprint, created_at)"
            " VALUES('nonexistent','1.0.0','off',0,'fp','t')")


def test_a_vision_result_id_may_be_null_which_is_how_off_is_represented(conn):
    cols = {c[1]: c for c in conn.execute("PRAGMA table_info(review)")}
    notnull = cols["vision_result_id"][3]
    assert notnull == 0
