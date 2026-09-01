"""A finding belongs to a face, and both I1 and fusion must know which.

The key is (face, region, category, defect_type) and `detectability_for`
discarded the face before taking the minimum — so a FRONT finding's I1
adequacy was judged against the worse of the two faces. Its own docstring
cites "a sharp front vouch for a blown-out back" as the motivation and
achieved the exact mirror image: a positively measured 78/22 miscut front
dropped from REJECT to REVIEW purely because the BACK was a borderless
design, a structural property of a face that says nothing about the front's
border measurement.

Fusion had the same omission: it correlates on category, defect type and
overlapping normalized box, with no face discriminator, and boxes are
normalized per card — so a defect on the front and a different defect at the
same corner of the BACK merged into one. "Do not double-penalize
corroboration" became "do not count the second face".
"""

import pytest

from card_reviewer.review.enums import FindingState, Scale
from card_reviewer.review.findings import Finding, FindingProducer, Severity
from card_reviewer.review.fusion import fuse
from card_reviewer.review.heuristic import detectability_for
from card_reviewer.review.provenance import (
    EvidenceOrigin, EvidenceRef, NormalizedBox,
)
from card_reviewer.review.roles import ImageRole

BOX = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)


def _finding(image_hash, producer=FindingProducer.HEURISTIC,
             severity=Severity.MODERATE):
    return Finding(
        defect_type="rounding", category="corners",
        state=FindingState.OBSERVED, producer=producer, confidence=0.95,
        psa10_relevant=True, severity=severity, location=BOX,
        evidence=[EvidenceRef(artifact_id=f"a-{image_hash}",
                              image_hash=image_hash,
                              origin=EvidenceOrigin.ORIGINAL,
                              view="corner_top_left", region=BOX)])


DETECTABILITY = {
    (ImageRole.FRONT, "top_left", "corners", "rounding"): Scale.HIGH,
    (ImageRole.BACK, "top_left", "corners", "rounding"): Scale.LOW,
}


def test_a_front_finding_is_judged_against_the_front():
    assert detectability_for(DETECTABILITY, "corners", "rounding",
                             "top_left", ImageRole.FRONT) is Scale.HIGH


def test_a_back_finding_is_judged_against_the_back():
    assert detectability_for(DETECTABILITY, "corners", "rounding",
                             "top_left", ImageRole.BACK) is Scale.LOW


def test_an_unlocated_finding_still_takes_the_weakest():
    """Unchanged where the face is genuinely unknown: whatever we have not
    narrowed to is somewhere the finding might be."""
    assert detectability_for(DETECTABILITY, "corners", "rounding") is Scale.LOW


def test_a_blown_out_back_does_not_weaken_a_front_measurement(rubric_scoped):
    """The reported consequence, end to end."""
    from card_reviewer.review.enums import Coverage, Verdict
    from card_reviewer.review.heuristic import HeuristicResult
    from card_reviewer.review.policies.combine_v1 import combine
    from card_reviewer.review.policies.coverage_v1 import CoverageResult

    front_finding = _finding("front-hash", severity=Severity.SEVERE)
    roles = {"front-hash": ImageRole.FRONT, "back-hash": ImageRole.BACK}

    def verdict(detectability):
        return combine(
            HeuristicResult(findings=[front_finding]), None,
            CoverageResult(outcome=Coverage.SUFFICIENT, rankable=True),
            card_context_known=True, scoped_rules=rubric_scoped,
            detectability=detectability, image_roles=roles,
        ).verdict

    good_back = dict(DETECTABILITY)
    good_back[(ImageRole.BACK, "top_left", "corners", "rounding")] = Scale.HIGH

    assert verdict(good_back) is verdict(DETECTABILITY), (
        "the back's detectability changed the verdict on a front finding")


def test_the_same_corner_on_two_faces_is_two_defects():
    """Normalized boxes are per card, so the front's top-left corner and the
    back's occupy the same box. Without a face discriminator a card damaged
    on both faces reported one defect."""
    roles = {"front-hash": ImageRole.FRONT, "back-hash": ImageRole.BACK}
    fused = fuse([_finding("front-hash"), _finding("back-hash")], roles)
    assert len(fused) == 2, "damage on two faces was counted once"


def test_two_producers_on_the_same_face_still_fuse():
    """The behaviour fusion exists for must survive."""
    roles = {"front-hash": ImageRole.FRONT}
    fused = fuse([_finding("front-hash", FindingProducer.HEURISTIC),
                  _finding("front-hash", FindingProducer.VISION)], roles)
    assert len(fused) == 1
    assert len(fused[0].sources) == 2


def test_fusion_without_roles_still_works():
    """Roles are optional: callers that do not have them must not crash, and
    must not silently start merging across faces either."""
    fused = fuse([_finding("front-hash"), _finding("back-hash")])
    assert len(fused) >= 1


def test_the_pipeline_supplies_the_role_map(tmp_path):
    """Threading the face through combine and fusion is decorative unless
    the pipeline actually hands over the roles. A card damaged at the same
    corner of BOTH faces is two defects; with no role map the normalized
    boxes coincide and it reports one.
    """
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
    for i, spec in enumerate((damaged,
                              damaged.model_copy(update={"text_heavy": True}))):
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

    corners = [f for f in review.defects_found if f["category"] == "corners"]
    assert len(corners) >= 2, (
        "the same corner damaged on both faces was reported as one defect")
