"""The report must state what the engine concluded, not what it first saw.

`defects_found` was `combined.findings` — the RAW producer findings, before
I3 demotion and before fusion. The adjudicated state and its demotion_reason
lived only in `combined.fused`, which is persisted under vision_assessment
and never rendered. So a vision finding citing only a CLAHE view was
correctly demoted to `suspected` internally, and the report printed:

    [observed      ] surface/crease  (vision)
    ...
    Why  surface/crease suspected

The human-facing artifact asserted `observed` for a finding the engine itself
refused to treat as observed — non-negotiable rules 3 and 6. The raw findings
must stay recoverable for calibration; what changes is what is DISPLAYED.
"""

import pytest

from card_reviewer.review.enums import FindingState
from card_reviewer.review.models import CardReview
from card_reviewer.review.report import render


def _finding(state, demotion_reason=None, **kw):
    base = {
        "category": "surface", "defect_type": "crease", "state": state,
        "producer": "vision", "confidence": 0.9, "psa10_relevant": True,
    }
    if demotion_reason:
        base["demotion_reason"] = demotion_reason
    base.update(kw)
    return base


def _review(**kw):
    base = dict(
        candidate_id="c1", verdict="REVIEW", coverage="SUFFICIENT",
        review_confidence="medium", psa10_candidate="unlikely",
        estimated_psa_grade="9-10", limitations=[], defects_found=[],
    )
    base.update(kw)
    return CardReview(**base)


def test_the_rendered_state_is_the_adjudicated_one():
    text = render(_review(defects_found=[
        _finding("suspected", demotion_reason="enhancement-only evidence")]))
    assert "suspected" in text
    assert "[observed" not in text


def test_a_demotion_says_why_it_was_demoted():
    """I3's whole point is that enhancement-only evidence cannot establish a
    confirmed defect. A reader must be able to see that is what happened."""
    text = render(_review(defects_found=[
        _finding("suspected", demotion_reason="enhancement-only evidence")]))
    assert "enhancement-only evidence" in text


def test_an_undemoted_finding_renders_without_a_reason():
    text = render(_review(defects_found=[_finding("observed")]))
    assert "observed" in text
    assert "demoted" not in text.lower()


def test_the_report_never_calls_the_rank_score_a_chance():
    """`psa10_candidate` is an enum, not a number, and "chance" is
    probability language the spec asks not to use for it."""
    text = render(_review())
    assert "chance" not in text.lower()


def test_the_report_still_disclaims_the_rank_score():
    assert "not a probability" in render(_review())


def test_the_pipeline_hands_the_report_the_adjudicated_findings(tmp_path):
    """The report can only be as honest as what it is given. `defects_found`
    was `combined.findings` — raw, pre-I3, pre-fusion — so the rendering
    above would still have printed the producer's original claim."""
    from card_reviewer.review.findings import Finding, FindingProducer
    from card_reviewer.review.heuristic import HeuristicResult
    from card_reviewer.review.policies.combine_v1 import combine
    from card_reviewer.review.policies.coverage_v1 import CoverageResult
    from card_reviewer.review.provenance import (
        EvidenceOrigin, EvidenceRef, NormalizedBox,
    )
    from card_reviewer.review.enums import Coverage

    box = NormalizedBox(x0=0.1, y0=0.1, x1=0.4, y1=0.4)
    enhancement_only = Finding(
        defect_type="crease", category="surface",
        state=FindingState.OBSERVED, producer=FindingProducer.VISION,
        confidence=0.95, psa10_relevant=True, location=box,
        evidence=[EvidenceRef(artifact_id="clahe-1", image_hash="h1",
                              origin=EvidenceOrigin.ENHANCED,
                              enhancement="clahe", view="surface_clahe",
                              region=box)])

    # Routed in as a heuristic finding: this test is about I3 demotion and
    # what the report is handed, not about the provider adapter.
    result = combine(
        HeuristicResult(findings=[enhancement_only]), None,
        CoverageResult(outcome=Coverage.SUFFICIENT, rankable=True),
        card_context_known=True, scoped_rules=[],
    )

    adjudicated = [f for f in result.fused]
    assert adjudicated, "the finding vanished instead of being demoted"
    assert adjudicated[0].state is not FindingState.OBSERVED, (
        "I3 did not demote an enhancement-only finding")
    assert adjudicated[0].demotion_reason


def test_the_raw_producer_findings_survive_beside_the_adjudicated_ones(tmp_path):
    """Rendering the adjudicated view must not cost the raw one. Calibrating
    OpenCV against Claude against the eventual PSA result needs both sides
    intact, and fusion alone cannot support that."""
    from card_reviewer.review.enums import Mode
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.ingest.adapter import ManualAdapter
    from card_reviewer.review.models import CandidateInput
    from card_reviewer.review.pipeline import ReviewPipeline
    from card_reviewer.review.storage.artifacts import ArtifactStore
    from card_reviewer.review.storage.migrations import connect, migrate
    from card_reviewer.review.storage.repository import SqliteRepository

    store = ArtifactStore(tmp_path / "store")
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    paths = []
    for i, spec in enumerate((CardSpec(corner_damage={"bottom_left": 0.9},
                                       border_color=(20, 20, 20)),
                              CardSpec(text_heavy=True))):
        path = tmp_path / f"{i}.png"
        path.write_bytes(render_png(spec))
        paths.append(path)
    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="2023 Topps Chrome", candidate_id="c",
        image_paths=paths,
        supplied_roles={str(paths[0]): "front", str(paths[1]): "back"}))

    review = ReviewPipeline(SqliteRepository(conn), store).review(
        resolved, Mode.OFF)
    conn.close()

    assert review.raw_findings, "the producers' own findings were not kept"
    for finding in review.raw_findings:
        assert "producer" in finding


def test_defects_found_is_the_fused_view_not_the_raw_one(tmp_path):
    """Two photographs of the same damaged corner produce two raw findings
    and ONE fused defect. If `defects_found` were still the raw list the
    report would show the same physical flaw twice, and the reader would
    count two problems where the engine counted one."""
    from card_reviewer.review.enums import Mode
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.ingest.adapter import ManualAdapter
    from card_reviewer.review.models import CandidateInput
    from card_reviewer.review.pipeline import ReviewPipeline
    from card_reviewer.review.storage.artifacts import ArtifactStore
    from card_reviewer.review.storage.migrations import connect, migrate
    from card_reviewer.review.storage.repository import SqliteRepository

    store = ArtifactStore(tmp_path / "store")
    conn = connect(tmp_path / "t.db")
    migrate(conn)

    damaged = CardSpec(border_color=(20, 20, 20),
                       corner_damage={"bottom_left": 0.9})
    paths = []
    for i, spec in enumerate((damaged, damaged.model_copy(update={"seed": 7}),
                              CardSpec(text_heavy=True))):
        path = tmp_path / f"{i}.png"
        path.write_bytes(render_png(spec))
        paths.append(path)
    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="2023 Topps Chrome", candidate_id="c",
        image_paths=paths,
        supplied_roles={str(paths[0]): "front", str(paths[1]): "front",
                        str(paths[2]): "back"}))

    review = ReviewPipeline(SqliteRepository(conn), store).review(
        resolved, Mode.OFF)
    conn.close()

    assert len(review.raw_findings) > len(review.defects_found), (
        "two photographs of one corner were reported as two separate defects")
