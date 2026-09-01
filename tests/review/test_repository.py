import pytest

from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    r = SqliteRepository(conn)
    r.save_candidate(id="c1", source="manual", title="t")
    return r


def test_a_stored_result_is_returned_for_the_same_cache_identity(repo):
    repo.put_stage_result("preflight", "fp1", "sig1", {"ok": True}, {"v": "1"},
                          image_hash="h1")
    got = repo.get_stage_result("preflight", "fp1", "sig1")
    assert got is not None and got.output == {"ok": True}


def test_a_different_producer_signature_is_a_cache_miss(repo):
    repo.put_stage_result("preflight", "fp1", "sig1", {"ok": True}, {},
                          image_hash="h1")
    assert repo.get_stage_result("preflight", "fp1", "sig2") is None


def test_a_recorded_failure_never_satisfies_a_cache_lookup(repo):
    """Spec §4: a failed vision call must not suppress a later successful one."""
    repo.record_attempt("vision", "fp1", "sig1", error_kind="timeout",
                        error_detail="504", candidate_id="c1")
    assert repo.get_stage_result("vision", "fp1", "sig1") is None


def test_a_success_after_a_failure_is_cached_normally(repo):
    repo.record_attempt("vision", "fp1", "sig1", error_kind="timeout",
                        candidate_id="c1")
    repo.put_stage_result("vision", "fp1", "sig1", {"findings": []}, {},
                          candidate_id="c1")
    assert repo.get_stage_result("vision", "fp1", "sig1") is not None


def test_putting_the_same_identity_twice_returns_the_existing_row(repo):
    a = repo.put_stage_result("preflight", "fp1", "sig1", {"n": 1}, {},
                              image_hash="h1")
    b = repo.put_stage_result("preflight", "fp1", "sig1", {"n": 1}, {},
                              image_hash="h1")
    assert a == b


def test_put_stage_result_returns_the_row_id_the_output_lives_in(repo):
    """review carries foreign keys to exact rows, so the id must come back
    from the write rather than being re-derived afterwards."""
    row_id = repo.put_stage_result("preflight", "fp1", "sig1", {"n": 1}, {},
                                   image_hash="h1")
    assert isinstance(row_id, int)
    assert repo.get_stage_result("preflight", "fp1", "sig1").id == row_id


def test_an_image_can_be_linked_to_two_candidates(repo):
    """The same photograph across two listings is stored once and linked
    twice — and the foreign keys the pipeline relies on must resolve."""
    repo.save_candidate(id="c2", source="manual", title="t2")
    repo.save_image("h1", "/tmp/a.png")
    repo.link_image("c1", "h1")
    repo.link_image("c2", "h1")
    assert repo._conn.execute(
        "SELECT COUNT(*) FROM candidate_image WHERE image_hash='h1'"
    ).fetchone()[0] == 2


def test_saving_the_same_candidate_twice_is_idempotent(repo):
    repo.save_candidate(id="c1", source="manual", title="t")
    assert repo._conn.execute(
        "SELECT COUNT(*) FROM candidate WHERE id='c1'").fetchone()[0] == 1


def test_reviews_are_append_only_so_history_survives(repo):
    rd = repo.save_routing_decision(candidate_id="c1", policy_version="1.0.0",
                                    mode="off", call_vision=False,
                                    trigger_reasons=[], input_fingerprint="fp")
    for verdict in ("REVIEW", "PASS"):
        repo.save_review(candidate_id="c1", mode="off", routing_decision_id=rd,
                         verdict=verdict, psa10_candidate="uncertain",
                         psa10_rank_score=50, rankable=True,
                         estimated_psa_grade="9", review_confidence="medium",
                         coverage="PARTIAL", rubric_version="4.0.0", output={})
    assert len(repo.reviews_for("c1")) == 2


def test_a_review_can_reference_the_exact_stage_rows_that_produced_it(repo):
    rd = repo.save_routing_decision(candidate_id="c1", policy_version="1.0.0",
                                    mode="off", call_vision=False,
                                    trigger_reasons=[], input_fingerprint="fp")
    combine_id = repo.put_stage_result("combine", "fp1", "sig1", {"v": "PASS"},
                                       {}, candidate_id="c1")
    review_id = repo.save_review(
        candidate_id="c1", mode="off", routing_decision_id=rd, verdict="PASS",
        psa10_candidate="yes", psa10_rank_score=100, rankable=True,
        estimated_psa_grade="10", review_confidence="high",
        coverage="SUFFICIENT", rubric_version="4.0.0",
        combine_result_id=combine_id, output={})
    row = repo._conn.execute("SELECT combine_result_id FROM review WHERE id=?",
                             (review_id,)).fetchone()
    assert row[0] == combine_id


def test_a_review_referencing_a_missing_stage_row_is_rejected(repo):
    """The FK is live, so a dangling reference fails at write rather than
    surfacing as a broken join months later."""
    import sqlite3

    rd = repo.save_routing_decision(candidate_id="c1", policy_version="1.0.0",
                                    mode="off", call_vision=False,
                                    trigger_reasons=[], input_fingerprint="fp")
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_review(
            candidate_id="c1", mode="off", routing_decision_id=rd,
            verdict="PASS", psa10_candidate="yes", psa10_rank_score=100,
            rankable=True, estimated_psa_grade="10", review_confidence="high",
            coverage="SUFFICIENT", rubric_version="4.0.0",
            combine_result_id=999999, output={})


def test_an_attempt_records_its_cost_and_latency_for_accounting(repo):
    repo.record_attempt("vision", "fp", "sig", error_kind="rate_limit",
                        cost_usd=0.0, latency_ms=1200, candidate_id="c1")
    row = repo._conn.execute(
        "SELECT error_kind, latency_ms FROM stage_attempt").fetchone()
    assert row[0] == "rate_limit" and row[1] == 1200
