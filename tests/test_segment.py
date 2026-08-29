import json
import pathlib

import pytest

from card_reviewer.knowledge import lexicon, segment
from card_reviewer.knowledge.models import Cue

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def lex():
    return lexicon.load(REPO / "knowledge" / "segmentation_lexicon.yaml")


def cue(start, end, text):
    return Cue(start_s=start, end_s=end, text=text)


def test_cold_transcript_yields_no_segments(lex):
    cues = [cue(0, 5, "Welcome back everyone."), cue(5, 10, "Subscribe below.")]
    assert segment.build(cues, lex) == []


def test_consecutive_hot_cues_merge_into_one_window(lex):
    cues = [
        cue(10, 15, "Look right here at this corner."),
        cue(15, 20, "You can see the whitening on the edge."),
    ]
    segments = segment.build(cues, lex, pad_s=0.0)
    assert len(segments) == 1
    assert segments[0].start_s == 10.0
    assert segments[0].end_s == 20.0


def test_single_cold_cue_does_not_break_a_run(lex):
    cues = [
        cue(0, 5, "Look right here at this corner."),
        cue(5, 10, "Anyway."),
        cue(10, 15, "You can see the print line on the surface."),
    ]
    segments = segment.build(cues, lex, pad_s=0.0)
    assert len(segments) == 1
    assert segments[0].end_s == 15.0


def test_two_cold_cues_end_a_window(lex):
    cues = [
        cue(0, 5, "Look right here at this corner."),
        cue(5, 10, "Anyway."),
        cue(10, 15, "So."),
        cue(15, 20, "You can see the print line on the surface."),
    ]
    segments = segment.build(cues, lex, pad_s=0.0)
    assert len(segments) == 2


def test_padding_is_applied_and_clamped_at_zero(lex):
    cues = [cue(2, 6, "Look right here at this corner.")]
    segments = segment.build(cues, lex, pad_s=5.0)
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 11.0


def test_long_window_is_split_to_max_length(lex):
    cues = [cue(i * 10, (i + 1) * 10, "print line right here") for i in range(20)]
    segments = segment.build(cues, lex, pad_s=0.0, max_len_s=90.0)
    assert len(segments) > 1
    assert all(s.end_s - s.start_s <= 90.0 + 1e-6 for s in segments)


def test_ids_are_assigned_in_time_order(lex):
    cues = [
        cue(0, 5, "Look at this corner."),
        cue(5, 10, "Anyway."),
        cue(10, 15, "So."),
        cue(30, 35, "Look right here, you can see the print line and the whitening."),
    ]
    segments = segment.build(cues, lex, pad_s=0.0)
    by_id = {s.id: s for s in segments}
    assert by_id["seg_001"].start_s < by_id["seg_002"].start_s


def test_large_time_gap_ends_a_window_even_with_no_cues_between(lex):
    cues = [
        cue(0, 5, "Look at this corner."),
        cue(5 + segment.MAX_GAP_S + 1, 5 + segment.MAX_GAP_S + 6, "Look right here, you can see the print line and the whitening."),
    ]
    segments = segment.build(cues, lex, pad_s=0.0)
    assert len(segments) == 2


def test_small_time_gap_still_merges_into_one_window(lex):
    cues = [
        cue(0, 5, "Look at this corner."),
        cue(5 + segment.MAX_GAP_S - 1, 5 + segment.MAX_GAP_S + 4, "Look right here, you can see the print line and the whitening."),
    ]
    segments = segment.build(cues, lex, pad_s=0.0)
    assert len(segments) == 1


def test_segment_records_categories_and_terms(lex):
    cues = [cue(0, 5, "Look right here at the print line on the surface.")]
    seg = segment.build(cues, lex, pad_s=0.0)[0]
    assert "surface" in seg.categories
    assert "print line" in seg.matched_terms
    assert seg.visual_cue is True


def test_golden_fixture_matches_expected_segments(lex):
    """Regression guard on the lexicon. If you tune the lexicon and this fails,
    inspect the diff and update the fixture deliberately — do not auto-accept."""
    cues = [Cue(**c) for c in json.loads((FIXTURES / "transcript_sample.json").read_text())["cues"]]
    got = [s.model_dump() for s in segment.build(cues, lex)]
    expected = json.loads((FIXTURES / "expected_segments.json").read_text())
    assert got == expected
