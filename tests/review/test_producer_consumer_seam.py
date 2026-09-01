"""The seam between the evidence producer and the finding consumer.

The mutation reviewer deleted `_refs_for`'s region preference and all 640
tests still passed, because the two sides were tested separately and nothing
tested the join: `test_assembly.py` checks that `to_image_evidence` EMITS
region-scoped keys, `test_fusion.py` checks that fusion separates hand-built
boxes, and `test_heuristic.py` asserted only `location is not None` — which
the mutant satisfies.

Run end to end through the real producers, the mutant collapses two damaged
corners onto the whole card, they overlap, and fusion merges them into one
defect. The card is penalised once for two flaws — exactly what
`_correlates`' own docstring says must not happen. CLAUDE.md makes this class
of test mandatory for precisely this reason.
"""

import numpy as np
import pytest

from card_reviewer.review.assembly import (
    ImageStageOutputs, assemble, to_image_evidence,
)
from card_reviewer.review.enums import Provenance
from card_reviewer.review.fusion import fuse
from card_reviewer.review.heuristic import evaluate
from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
from card_reviewer.review.imaging.measure import measure_all
from card_reviewer.review.imaging.observability import analyze as observability_analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.roles import ImageRole, ResolvedRole
from card_reviewer.review.storage.artifacts import ArtifactStore

TWO_CORNERS = {"bottom_left": 0.9, "top_right": 0.9}


@pytest.fixture
def assembled(tmp_path):
    """Real geometry -> real measurement -> real assembly. No fixtures."""
    store = ArtifactStore(tmp_path / "store")
    data = render_png(CardSpec(border_color=(20, 20, 20),
                               corner_damage=TWO_CORNERS))
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    outputs = ImageStageOutputs(
        image_hash=image_hash, preflight={"global_sharpness": 120.0},
        geometry=geometry.model_dump(),
        observability=observability_analyze(
            geometry, store, image_hash).model_dump(),
        cv_measurements=measure_all(geometry, store, image_hash).model_dump(),
    )
    roles = {image_hash: ResolvedRole(image_hash=image_hash,
                                      role=ImageRole.FRONT,
                                      provenance=Provenance.SUPPLIED,
                                      confidence=1.0)}
    return assemble(to_image_evidence([outputs]), roles)


def _corner_findings(assembled, rubric_scoped):
    result = evaluate(assembled, rubric_scoped)
    return [f for f in result.findings if f.category == "corners"]


def test_two_damaged_corners_reach_the_consumer_as_two_findings(
        assembled, rubric_scoped):
    findings = _corner_findings(assembled, rubric_scoped)
    assert len(findings) >= 2, "the real producers found fewer than two corners"


def test_two_damaged_corners_have_different_locations(assembled, rubric_scoped):
    """Not merely `location is not None` — that is what let the region
    preference be deleted unnoticed."""
    findings = _corner_findings(assembled, rubric_scoped)
    boxes = {(f.location.x0, f.location.y0, f.location.x1, f.location.y1)
             for f in findings if f.location}
    assert len(boxes) >= 2, f"both corners share one box: {boxes}"
    for box in boxes:
        assert box != (0.0, 0.0, 1.0, 1.0), (
            "a corner finding was located on the whole card")


def test_two_damaged_corners_survive_fusion_as_two_defects(
        assembled, rubric_scoped):
    """The consequence. Collapsed onto the whole card the boxes overlap,
    fusion merges them, and a card with two chewed corners is charged for
    one."""
    findings = _corner_findings(assembled, rubric_scoped)
    fused = fuse(findings)
    assert len(fused) >= 2, (
        f"{len(findings)} corner findings fused down to {len(fused)}")


def test_the_regions_the_producer_emits_are_the_ones_the_consumer_looks_up(
        assembled):
    """The join itself: keys written by `to_image_evidence` must be the keys
    `_refs_for` asks for, or the region preference silently never applies."""
    from card_reviewer.review.heuristic import _refs_for

    scoped = [k for k in assembled.evidence_refs if k.count(":") == 2]
    assert scoped, "the producer emitted no region-scoped keys at all"

    for key in scoped:
        category, defect_type, region = key.split(":")
        refs = _refs_for(assembled, category, defect_type, region)
        assert refs, f"the consumer could not resolve {key}"
        assert refs == assembled.evidence_refs[key], (
            f"{key} resolved to a different ref set than the producer wrote")
