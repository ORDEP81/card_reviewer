import pytest

from card_reviewer.review.enums import Mode, Verdict
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.ingest.adapter import ManualAdapter
from card_reviewer.review.models import CandidateInput
from card_reviewer.review.pipeline import ReviewPipeline
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository
from card_reviewer.review.vision.provider import Assessment, FakeProvider, GemView

DECLARED_STAGES = {
    "preflight", "geometry", "observability", "cv_measurements",
    "role_features", "role_context", "evidence_assembly", "heuristic",
    "coverage_provisional", "routing", "coverage", "combine",
}


def _provider(model="fake-model", prompt="1.0.0"):
    return FakeProvider(
        Assessment(category_assessability={"centering": True, "corners": True,
                                           "edges": True, "surface": True},
                   gem_view=GemView.NO_DISQUALIFIER),
        model=model, prompt_version=prompt)


@pytest.fixture
def rig(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    store = ArtifactStore(tmp_path / "store")
    yield ReviewPipeline(SqliteRepository(conn), store), store, SqliteRepository(conn)
    conn.close()


def _candidate(tmp_path, store, specs, title="2023 Topps Chrome test",
               roles=("front", "back")):
    paths = []
    for i, spec in enumerate(specs):
        p = tmp_path / f"img{i}_{abs(hash(str(spec)))}.png"
        p.write_bytes(render_png(spec))
        paths.append(p)
    supplied = {str(p): r for p, r in zip(paths, roles)}
    return ManualAdapter(store).resolve(CandidateInput(
        source="manual", title=title, image_paths=paths,
        supplied_roles=supplied))


def _rows_by_stage(repo):
    return dict(repo._conn.execute(
        "SELECT stage, COUNT(*) FROM stage_result GROUP BY stage"))


# --- end to end ------------------------------------------------------------

def test_a_front_and_back_card_runs_end_to_end_in_off_mode(rig, tmp_path):
    pipeline, store, repo = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    review = pipeline.review(resolved, Mode.OFF)
    assert review.verdict in {v.value for v in Verdict}
    assert repo.reviews_for(resolved.candidate_id)


def test_every_declared_stage_produces_a_cached_result(rig, tmp_path):
    """A stage the tables claim to cache but that runs directly is not
    cached. Counting one stage's rows would never catch that."""
    pipeline, store, repo = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    pipeline.review(resolved, Mode.OFF)
    assert DECLARED_STAGES <= set(_rows_by_stage(repo))


def test_rerunning_reuses_every_stage_not_just_one(rig, tmp_path):
    pipeline, store, repo = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    pipeline.review(resolved, Mode.OFF)
    before = _rows_by_stage(repo)
    pipeline.review(resolved, Mode.OFF)
    assert _rows_by_stage(repo) == before


def test_the_same_image_in_two_candidates_is_analyzed_once(rig, tmp_path):
    pipeline, store, repo = rig
    a = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    pipeline.review(a, Mode.OFF)
    before = _rows_by_stage(repo)["preflight"]
    b = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)],
                   title="a different listing")
    pipeline.review(b, Mode.OFF)
    assert _rows_by_stage(repo)["preflight"] == before


# --- vision --------------------------------------------------------------

def test_off_never_calls_and_deep_always_does(rig, tmp_path):
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    provider = _provider()
    pipeline.review(resolved, Mode.OFF, provider)
    assert provider.calls == 0
    pipeline.review(resolved, Mode.DEEP, provider)
    assert provider.calls == 1


def test_deep_reviewed_twice_calls_the_provider_exactly_once(rig, tmp_path):
    """The cache lookup must happen BEFORE assess(), or every re-review of an
    unchanged card bills again."""
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    provider = _provider()
    pipeline.review(resolved, Mode.DEEP, provider)
    pipeline.review(resolved, Mode.DEEP, provider)
    assert provider.calls == 1


def test_an_off_run_does_not_poison_the_deep_cache(rig, tmp_path):
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    provider = _provider()
    pipeline.review(resolved, Mode.OFF, provider)
    pipeline.review(resolved, Mode.DEEP, provider)
    assert provider.calls == 1


def test_a_model_change_forces_a_new_provider_call(rig, tmp_path):
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    a, b = _provider(model="m1"), _provider(model="m2")
    pipeline.review(resolved, Mode.DEEP, a)
    pipeline.review(resolved, Mode.DEEP, b)
    assert a.calls == 1 and b.calls == 1


def test_a_prompt_version_change_forces_a_new_provider_call(rig, tmp_path):
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    a, b = _provider(prompt="1.0.0"), _provider(prompt="1.1.0")
    pipeline.review(resolved, Mode.DEEP, a)
    pipeline.review(resolved, Mode.DEEP, b)
    assert a.calls == 1 and b.calls == 1


def test_smart_wanting_vision_without_a_provider_does_not_behave_as_off(
        rig, tmp_path):
    """Silently degrading to OFF would report a card as fully assessed when
    the layer that judges surface never ran."""
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    review = pipeline.review(resolved, Mode.SMART, None)
    assert review.verdict != Verdict.PASS.value
    assert any(l["reason_code"] == "VISION_UNAVAILABLE" for l in review.limitations)


def test_a_provider_failure_removes_evidence_without_failing_the_review(
        rig, tmp_path):
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])

    class Broken(FakeProvider):
        def assess(self, manifest):
            raise RuntimeError("provider down")

    review = pipeline.review(resolved, Mode.DEEP, Broken(
        Assessment(category_assessability={}, gem_view=GemView.NO_DISQUALIFIER)))
    assert review.verdict != Verdict.PASS.value
    assert any(l["reason_code"] == "VISION_FAILED" for l in review.limitations)


# --- policy behaviour end to end ------------------------------------------

def test_a_clean_front_only_card_is_partial_rankable_and_reviewed(rig, tmp_path):
    """A missing back bars PASS, not REJECT."""
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec()], roles=("front",))
    review = pipeline.review(resolved, Mode.OFF)
    assert review.verdict == Verdict.REVIEW.value
    assert review.rankable is True
    assert review.review_confidence == "low"


def test_an_unidentifiable_listing_asks_for_identity_not_photographs(
        rig, tmp_path):
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)],
                          title="mystery lot of four")
    review = pipeline.review(resolved, Mode.OFF)
    assert review.card_identification_request is True
    assert not any("photograph" in p for p in
                   review.recommended_additional_photos
                   if "identif" in p.lower())


def test_the_review_is_stamped_with_the_versions_that_actually_ran(rig, tmp_path):
    from card_reviewer.review.versions import VISION_PLACEHOLDER

    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    off = pipeline.review(resolved, Mode.OFF)
    assert off.versions["vision"] == "not_run"
    deep = pipeline.review(resolved, Mode.DEEP, _provider(model="m1"))
    assert "m1" in deep.versions["vision"]
    for review in (off, deep):
        assert VISION_PLACEHOLDER not in review.versions.values()


def test_the_review_references_the_exact_stage_rows_that_produced_it(
        rig, tmp_path):
    pipeline, store, repo = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    pipeline.review(resolved, Mode.OFF)
    row = repo.reviews_for(resolved.candidate_id)[0]
    assert row["combine_result_id"] is not None
    assert repo._conn.execute(
        "SELECT COUNT(*) FROM stage_result WHERE id=?",
        (row["combine_result_id"],)).fetchone()[0] == 1


def test_cv_and_vision_assessments_stay_separately_recoverable(rig, tmp_path):
    """Required for later calibration against actual PSA results."""
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    review = pipeline.review(resolved, Mode.DEEP, _provider())

    # Not merely "is not None": replacing the payload with {} passed that,
    # so the assertion held nothing. The CV side must carry real
    # measurements, and the vision side must be DISTINGUISHABLE from a run
    # where vision never happened — calibration compares the layers against
    # each other, which needs to know which layers spoke.
    assert review.cv_assessment
    assert any(measurements for measurements in review.cv_assessment.values())
    assert review.vision_assessment is not None
    assert "fused" in review.vision_assessment

    off = pipeline.review(resolved, Mode.OFF)
    assert off.vision_assessment is None, (
        "a run without vision is indistinguishable from one with it")
    assert off.cv_assessment, "the CV side vanished when vision did"


def test_a_crash_after_the_provider_response_does_not_rebill(rig, tmp_path):
    """The vision row is committed before combine runs, so a restart reuses
    it rather than paying again."""
    pipeline, store, repo = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    provider = _provider()
    pipeline.review(resolved, Mode.DEEP, provider)
    fresh = ReviewPipeline(repo, store)
    fresh.review(resolved, Mode.DEEP, provider)
    assert provider.calls == 1


def test_no_price_field_reaches_the_review(rig, tmp_path):
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    review = pipeline.review(resolved, Mode.OFF)
    blob = review.model_dump_json().lower()
    for word in ("price", "asking", "profit", "resale"):
        assert word not in blob


def test_an_error_in_our_own_code_is_not_reported_as_a_provider_outage(
        rig, tmp_path, monkeypatch):
    """The boundary is the VisionProvider protocol.

    Catching every exception around the vision stage would report a bug in
    the canonicalizer or the cache as 'the provider is down' and silently
    degrade the review — which is exactly how one such bug first surfaced.
    """
    from card_reviewer.review.pipeline import StageRunner

    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])

    original = StageRunner.run_with_id

    def sabotage(self, stage, *args, **kwargs):
        if stage == "vision":
            raise ValueError("a bug in our own code, not the provider")
        return original(self, stage, *args, **kwargs)

    monkeypatch.setattr(StageRunner, "run_with_id", sabotage)
    with pytest.raises(ValueError, match="bug in our own code"):
        pipeline.review(resolved, Mode.DEEP, _provider())


def test_an_unavailable_vision_layer_stops_coverage_reaching_sufficient(
        rig, tmp_path):
    """The veto is what actually prevents PASS. Recording only a limitation
    would let the card pass as though it had been fully assessed."""
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(text_heavy=True)])
    with_provider = pipeline.review(resolved, Mode.DEEP, _provider())
    without = pipeline.review(resolved, Mode.SMART, None)
    assert without.coverage != "SUFFICIENT"
    assert without.coverage != with_provider.coverage or (
        with_provider.coverage != "SUFFICIENT")


def test_an_unusable_image_is_dropped_before_any_geometry_runs(rig, tmp_path):
    """One bad photograph out of several must not fail the card, and must not
    cost the geometry stage either."""
    import cv2
    import numpy as np

    pipeline, store, repo = rig
    blurred = cv2.GaussianBlur(
        cv2.imdecode(np.frombuffer(render_png(CardSpec()), np.uint8),
                     cv2.IMREAD_COLOR), (99, 99), 0)
    bad = tmp_path / "blurred.png"
    bad.write_bytes(cv2.imencode(".png", blurred)[1].tobytes())
    good_front = tmp_path / "front.png"
    good_front.write_bytes(render_png(CardSpec()))
    good_back = tmp_path / "back.png"
    good_back.write_bytes(render_png(CardSpec(text_heavy=True)))

    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="2023 Topps Chrome", 
        image_paths=[good_front, good_back, bad],
        supplied_roles={str(good_front): "front", str(good_back): "back"}))
    pipeline.review(resolved, Mode.OFF)

    rows = _rows_by_stage(repo)
    assert rows["preflight"] == 3
    assert rows["geometry"] == 2, "geometry ran on an image preflight rejected"
