"""More damage must never produce a better verdict.

The glare test measured each region's clipping against the MEDIAN of its
siblings. Once half of them clip, the median moves with them and the glared
regions stop standing out — so a white-bordered card blown out on three or
four corners was reported fully assessable, and the engine returned:

    PASS / SUFFICIENT / rank 100 / grade "10" / confidence high

with zero limitations and zero photo requests, while the same card with ONE
glared corner correctly returned REVIEW. That is I2 — unassessable evidence
producing a PASS — and non-negotiable rule 3, hiding an image limitation.

The fix is the MINIMUM sibling rather than the median: the cleanest
comparable region is the only honest reference, and it cannot be dragged up
by the regions being tested. It introduces no new threshold — the corpus
shows glare separates from clean at 0-12% recall at zero false positives,
so a tuned absolute cut-off is exactly what must NOT be invented here.
Measured over the 40 labelled clean and glare photographs, the minimum and
the median flag identical corners, so the robustness is free.
"""

import tempfile
from pathlib import Path

import pytest

from card_reviewer.review.enums import Mode
from card_reviewer.review.imaging.observability import REGIONS_FOR_CATEGORY
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.ingest.adapter import ManualAdapter
from card_reviewer.review.models import CandidateInput
from card_reviewer.review.pipeline import ReviewPipeline
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository

CORNERS = list(REGIONS_FOR_CATEGORY["corners"])


def _review(tmp_path, glared):
    store = ArtifactStore(tmp_path / f"store{glared}")
    conn = connect(tmp_path / f"t{glared}.db")
    migrate(conn)
    front = tmp_path / f"f{glared}.png"
    back = tmp_path / f"b{glared}.png"
    front.write_bytes(render_png(CardSpec(border_color=(255, 255, 255),
                                          glare_regions=CORNERS[:glared])))
    back.write_bytes(render_png(CardSpec(border_color=(255, 255, 255),
                                         text_heavy=True)))
    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="2023 Topps Chrome test",
        image_paths=[front, back],
        supplied_roles={str(front): "front", str(back): "back"}))
    try:
        return ReviewPipeline(SqliteRepository(conn), store).review(
            resolved, Mode.OFF)
    finally:
        conn.close()


@pytest.mark.parametrize("glared", [1, 2, 3])
def test_a_card_blown_out_on_most_corners_cannot_pass(tmp_path, glared):
    review = _review(tmp_path, glared)
    assert review.verdict != "PASS", (
        f"{glared} of four corners blown out returned {review.verdict} with "
        f"{len(review.limitations)} limitations — not seeing a defect became "
        f"evidence of a clean card")


def test_all_four_corners_blown_is_an_open_design_question(tmp_path):
    """A RECORDED GAP, and a question for the owner rather than a bug to
    fix quietly.

    With every corner blown there is no unglared sibling to reference, and
    the engine cannot distinguish a uniformly glared white card from a
    plain white-bordered one. Two things that are both required now
    conflict:

      I2 says an unassessable card must not PASS.
      Spec section 19 item 10 says a white-bordered card CAN pass.

    On this input they are the same photograph as far as any measurement
    here can tell. Resolving it means either an absolute clipping
    threshold or a policy that a fully-clipped border is unassessable —
    both change product behaviour, so neither belongs in an
    implementation commit.

    Two references were measured and rejected. An absolute cut-off at 0.60
    costs about 10% false positives on the miscut population this tool
    exists to screen. Comparing corners against the EDGES — the same
    border material — separates cleanly on synthetic cards (0.146 against
    0.334) and gives ZERO separation on the real corpus, where clean cards
    reach 0.229 and every glared one sits at 0.000. Calibrating on the
    synthetic number is the exact mistake that put the wrong framing
    assumption in this engine for ninety commits.

    This asserts today's behaviour so the day it changes is deliberate.
    """
    review = _review(tmp_path, 4)
    assert review.verdict == "PASS", (
        f"the all-corners-blown case now returns {review.verdict}. If that "
        f"was intended, this recorded gap is closed and the test should "
        f"assert the new behaviour and say which way the question was "
        f"answered.")


def test_more_glare_never_improves_the_verdict(tmp_path):
    """The property, stated as itself. One glared corner was handled
    correctly and three were not, so the engine's answer improved as the
    photograph got worse — which no amount of threshold tuning would have
    surfaced, because each case looked defensible alone."""
    scores = {}
    for glared in (0, 1, 2, 3, 4):
        review = _review(tmp_path, glared)
        scores[glared] = (review.psa10_rank_score, review.verdict)

    best = scores[0][0]
    for glared in (1, 2, 3, 4):
        score = scores[glared][0]
        assert score is None or best is None or score <= best, (
            f"{glared} glared corners scored {score}, better than the clean "
            f"card's {best}: {scores}")
