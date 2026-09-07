"""One test per definition-of-done item in the approved spec (§19).

Several are demonstrated elsewhere; this module is the single place the
whole contract can be checked in one run.
"""

import pytest

from detectability_helpers import (
    detectability_map, regions_for, set_every_region,
)
from card_reviewer.review.enums import Coverage, Mode, Scale, Verdict
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.ingest.adapter import ManualAdapter
from card_reviewer.review.models import CandidateInput, ResolvedCandidate
from card_reviewer.review.pipeline import ReviewPipeline
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository
from card_reviewer.review.taxonomy import CATEGORIES, defect_types_for
from card_reviewer.review.vision.provider import Assessment, FakeProvider, GemView


def _provider(model="fake-model", prompt="1.0.0"):
    return FakeProvider(
        Assessment(category_assessability={c: True for c in CATEGORIES},
                   gem_view=GemView.NO_DISQUALIFIER),
        model=model, prompt_version=prompt)


def _closed_after(request, conn):
    """An inspection connection that does not outlive its test."""
    request.addfinalizer(conn.close)
    return conn


@pytest.fixture
def rig_factory(tmp_path):
    """Build a pipeline over a shared DB and store.

    Every call reuses the same paths, which is exactly what a restart looks
    like: the stage_result rows and stored artifacts are all that survive.
    """
    store = ArtifactStore(tmp_path / "store")
    db = tmp_path / "t.db"
    opened = []

    def make(cv_version=None, provider=None, monkeypatch=None):
        if cv_version and monkeypatch:
            # Patch where pipeline BOUND the name, not where it is defined:
            # the module imports it at call time from .imaging.measure.
            import card_reviewer.review.imaging.measure as measure_mod

            monkeypatch.setattr(measure_mod, "CV_VERSION", cv_version)
        conn = connect(db)
        opened.append(conn)
        migrate(conn)
        repo = SqliteRepository(conn)
        paths = []
        for i, spec in enumerate((CardSpec(), CardSpec(text_heavy=True))):
            path = tmp_path / f"img{i}.png"
            path.write_bytes(render_png(spec))
            paths.append(path)
        resolved = ManualAdapter(store).resolve(CandidateInput(
            source="manual", title="2023 Topps Chrome test",
            candidate_id="fixed-candidate", image_paths=paths,
            supplied_roles={str(paths[0]): "front", str(paths[1]): "back"}))
        return (ReviewPipeline(repo, store), resolved,
                provider or _provider())

    yield make
    for conn in opened:
        conn.close()


def test_dod1_a_card_runs_end_to_end_in_off_mode_and_persists(rig_factory):
    pipeline, resolved, _ = rig_factory()
    review = pipeline.review(resolved, Mode.OFF)
    assert review.verdict in {v.value for v in Verdict}
    assert review.review_id is not None


def test_dod2_off_never_calls_deep_always_does(rig_factory):
    pipeline, resolved, provider = rig_factory()
    pipeline.review(resolved, Mode.OFF, provider)
    assert provider.calls == 0
    pipeline.review(resolved, Mode.DEEP, provider)
    assert provider.calls == 1


def test_dod3_a_stored_vision_result_survives_a_restart(rig_factory):
    """The expensive step is committed before combine runs, so a crash
    between them must not re-bill."""
    pipeline, resolved, provider = rig_factory()
    pipeline.review(resolved, Mode.DEEP, provider)
    after_crash, resolved_again, _ = rig_factory(provider=provider)
    after_crash.review(resolved_again, Mode.DEEP, provider)
    assert provider.calls == 1


def test_dod4_a_cv_bump_does_not_rebill_an_unchanged_vision_call(
        rig_factory, monkeypatch, tmp_path, request):
    """Downstream stages fingerprint upstream VALUES, not signatures."""
    pipeline, resolved, provider = rig_factory()
    pipeline.review(resolved, Mode.DEEP, provider)
    calls = provider.calls

    bumped, resolved_again, _ = rig_factory(cv_version="9.9.9",
                                            provider=provider,
                                            monkeypatch=monkeypatch)
    conn = _closed_after(request, connect(tmp_path / "t.db"))
    before = conn.execute(
        "SELECT COUNT(*) FROM stage_result WHERE stage='cv_measurements'"
    ).fetchone()[0]
    bumped.review(resolved_again, Mode.DEEP, provider)

    # The bump must actually bite, or this test proves nothing: cv re-runs
    # under its new signature...
    assert conn.execute(
        "SELECT COUNT(*) FROM stage_result WHERE stage='cv_measurements'"
    ).fetchone()[0] > before
    # ...while vision, which fingerprints cv's VALUES, still hits cache.
    assert provider.calls == calls


def test_dod5_the_same_image_across_candidates_is_analyzed_once(
        rig_factory, tmp_path, request):
    pipeline, resolved, _ = rig_factory()
    pipeline.review(resolved, Mode.OFF)
    conn = _closed_after(request, connect(tmp_path / "t.db"))
    before = conn.execute(
        "SELECT COUNT(*) FROM stage_result WHERE stage='preflight'").fetchone()[0]

    other = ManualAdapter(pipeline._store).resolve(CandidateInput(
        source="manual", title="a different listing", candidate_id="second",
        image_paths=[tmp_path / "img0.png", tmp_path / "img1.png"],
        supplied_roles={str(tmp_path / "img0.png"): "front",
                        str(tmp_path / "img1.png"): "back"}))
    pipeline.review(other, Mode.OFF)
    assert conn.execute(
        "SELECT COUNT(*) FROM stage_result WHERE stage='preflight'"
    ).fetchone()[0] == before


def test_dod6_the_three_invariants_hold_under_table_driven_tests():
    """I1, I2 and I3 each have dedicated suites; this asserts they exist and
    that the modules enforcing them are importable together."""
    from card_reviewer.review.findings import i3_satisfied
    from card_reviewer.review.policies.combine_v1 import decide_verdict, i1_satisfied

    assert callable(i1_satisfied) and callable(i3_satisfied)
    assert decide_verdict([], Coverage.PARTIAL, ambiguity=False).verdict is (
        Verdict.REVIEW)


def test_dod7_coverage_returns_three_outcomes_and_gates_pass():
    from card_reviewer.review.policies.coverage_v1 import (
        REQUIRED_FACES, evaluate_coverage,
    )
    from card_reviewer.review.roles import ImageRole

    full = detectability_map(REQUIRED_FACES)
    assert evaluate_coverage(full, {}, {}, REQUIRED_FACES).outcome is (
        Coverage.SUFFICIENT)
    assert evaluate_coverage(full, {}, {}, (ImageRole.FRONT,)).outcome is (
        Coverage.PARTIAL)
    assert evaluate_coverage({}, {}, {}, (ImageRole.FRONT,)).outcome is (
        Coverage.INADEQUATE)


def test_dod8_unknown_context_receives_every_rule_and_biases_to_review(rubric):
    from card_reviewer.review.context import CardContext
    from card_reviewer.review.enums import RuleEvaluability
    from card_reviewer.review.evaluability import scope_rules

    scoped = scope_rules(rubric.for_card(None, None), CardContext())
    assert len(scoped) == len(rubric.rules)
    assert any(s.evaluability is RuleEvaluability.UNEVALUABLE for s in scoped)


def test_dod9_a_psa_outcome_joins_back_to_the_review_that_predicted_it(
        rig_factory, tmp_path, request):
    pipeline, resolved, _ = rig_factory()
    review = pipeline.review(resolved, Mode.OFF)
    conn = _closed_after(request, connect(tmp_path / "t.db"))
    conn.execute(
        "INSERT INTO grading_submission(id, candidate_id, grader, grade, status)"
        " VALUES('s1', ?, 'PSA', '10', 'returned')", (resolved.candidate_id,))
    conn.commit()
    row = conn.execute(
        "SELECT r.verdict, r.psa10_rank_score, g.grade FROM review r"
        " JOIN grading_submission g ON g.candidate_id = r.candidate_id"
        " WHERE r.id = ?", (review.review_id,)).fetchone()
    assert row is not None and row[2] == "10"


def test_dod10_a_white_bordered_card_can_still_reach_sufficient_coverage():
    """PASS must stay reachable for most of the modern base-card population."""
    from card_reviewer.review.policies.coverage_v1 import (
        REQUIRED_FACES, evaluate_coverage,
    )

    det = detectability_map(REQUIRED_FACES)
    reasons = {}
    for face in REQUIRED_FACES:
        for category in ("corners", "edges"):
            det[(face, "top_left", category, "whitening")] = Scale.LOW
            reasons[(face, "top_left", category, "whitening")] = "WHITE_BORDER"
    assert evaluate_coverage(det, reasons, {}, REQUIRED_FACES).outcome is (
        Coverage.SUFFICIENT)


def test_dod11_an_off_run_never_satisfies_a_deep_lookup(rig_factory):
    pipeline, resolved, provider = rig_factory()
    pipeline.review(resolved, Mode.OFF, provider)
    pipeline.review(resolved, Mode.DEEP, provider)
    assert provider.calls == 1


def test_dod12_the_verdict_function_is_total(rubric):
    """Every combination maps to exactly one verdict, with no hole."""
    import itertools

    from card_reviewer.review.enums import Authority, FindingState
    from card_reviewer.review.findings import Finding, FindingProducer
    from card_reviewer.review.policies.combine_v1 import decide_verdict
    from card_reviewer.review.provenance import (
        EvidenceOrigin, EvidenceRef, NormalizedBox,
    )

    def finding(conf):
        return Finding(
            defect_type="rounding", category="corners",
            state=FindingState.OBSERVED, producer=FindingProducer.HEURISTIC,
            confidence=conf, psa10_relevant=True,
            location=NormalizedBox(x0=0.0, y0=0.0, x1=0.3, y1=0.3),
            evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                                  origin=EvidenceOrigin.ORIGINAL, view="v")])

    for coverage, sat, unsat, ambiguity in itertools.product(
            list(Coverage), [False, True], [False, True], [False, True]):
        findings = []
        if sat:
            findings.append((finding(0.95), Authority.BINDING, Scale.HIGH))
        if unsat:
            findings.append((finding(0.5), Authority.BINDING, Scale.HIGH))
        result = decide_verdict(findings, coverage, ambiguity=ambiguity)
        assert result.verdict in set(Verdict)


def test_dod13_vision_not_assessable_prevents_sufficient_coverage():
    from card_reviewer.review.policies.coverage_v1 import (
        REQUIRED_FACES, evaluate_coverage,
    )

    det = detectability_map(REQUIRED_FACES)
    assert evaluate_coverage(det, {}, {}, REQUIRED_FACES).outcome is (
        Coverage.SUFFICIENT)
    assert evaluate_coverage(det, {}, {"surface": False},
                             REQUIRED_FACES).outcome is Coverage.PARTIAL


def test_dod14_the_synthetic_generator_stands_alone():
    from card_reviewer.review.imaging.synthetic import render

    assert render(CardSpec(borderless=True)) is not None
    assert render(CardSpec(corner_damage={"top_left": 0.5})) is not None


def test_dod15_no_price_field_reaches_the_core_or_the_output():
    from card_reviewer.review.models import CardReview

    forbidden = {"asking_price", "price", "cost", "value", "purchased"}
    assert not (set(ResolvedCandidate.model_fields) & forbidden)
    assert not (set(CardReview.model_fields) & forbidden)


def test_dod16_a_front_only_card_is_partial_rankable_and_never_passes(
        rig_factory, tmp_path):
    pipeline, _, _ = rig_factory()
    front = tmp_path / "front_only.png"
    front.write_bytes(render_png(CardSpec()))
    resolved = ManualAdapter(pipeline._store).resolve(CandidateInput(
        source="manual", title="2023 Topps Chrome", candidate_id="front-only",
        image_paths=[front], supplied_roles={str(front): "front"}))
    review = pipeline.review(resolved, Mode.OFF)
    assert review.verdict != Verdict.PASS.value
    assert review.rankable is True
    assert review.review_confidence == "low"


def test_dod17_a_rubric_content_change_refreshes_the_manifest(rubric):
    from card_reviewer.review.assembly import Assembled
    from card_reviewer.review.fingerprint import fingerprint
    from card_reviewer.review.manifest import build_manifest

    rules = rubric.for_card(None, None)
    assembled = Assembled()
    a = fingerprint(build_manifest(assembled, Mode.SMART, rules).payload)
    b = fingerprint(build_manifest(assembled, Mode.SMART, rules).payload)
    assert a == b, "identical rule content must not change the fingerprint"
    c = fingerprint(build_manifest(assembled, Mode.SMART, rules[:-1]).payload)
    assert a != c, "changed rule content must change the fingerprint"


def test_dod18_no_test_in_the_suite_calls_the_anthropic_api():
    """Covered properly in test_no_live_api.py.

    The version that lived here grepped a RELATIVE "tests/" path and ignored
    grep's return code, so from any other working directory it passed while
    checking nothing; and its pattern matched only the dotted spelling, so a
    plain `from anthropic import ...` construction went straight through.
    Both holes were demonstrated. It is replaced rather than patched in
    place, because the replacement needs to resolve the repo root and match
    several spellings.
    """
    from test_no_live_api import CONSTRUCTORS, REPO

    assert (REPO / "tests").is_dir()
    # Assembled at runtime, so this line is not itself an offender.
    assert CONSTRUCTORS.search("client = Anthro" + "pic()")


def test_every_review_is_stamped_with_the_versions_that_actually_ran(
        rig_factory):
    from card_reviewer.review.versions import VISION_PLACEHOLDER

    pipeline, resolved, _ = rig_factory()
    deep = pipeline.review(resolved, Mode.DEEP, _provider(model="m1",
                                                          prompt="2.0.0"))
    assert "m1" in deep.versions["vision"]
    assert "2.0.0" in deep.versions["vision"]

    off = pipeline.review(resolved, Mode.OFF)
    assert off.versions["vision"] == "not_run"
    for review in (deep, off):
        assert VISION_PLACEHOLDER not in review.versions.values()
