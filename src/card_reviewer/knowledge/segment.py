"""Stage 3: turn a transcript into ranked windows worth watching.

A 90-minute course video contains perhaps 8 minutes of card inspection. This
module finds those minutes so Pass 2 stays affordable.
"""

from __future__ import annotations

import json
import math

from . import lexicon as lex_mod
from . import manifest as mf
from .models import Cue, Segment, Transcript
from .paths import ProjectPaths

MIN_SCORE = 2.0
PAD_S = 5.0
MAX_LEN_S = 90.0
GAP_TOLERANCE = 1  # consecutive cold cues allowed inside a run
MAX_GAP_S = 30.0  # elapsed time since the previous cue that ends a run outright


def _split(start: float, end: float, max_len_s: float) -> list[tuple[float, float]]:
    span = end - start
    if span <= max_len_s:
        return [(start, end)]
    parts = math.ceil(span / max_len_s)
    width = span / parts
    return [(start + i * width, start + (i + 1) * width) for i in range(parts)]


def build(
    cues: list[Cue],
    lex: lex_mod.Lexicon,
    min_score: float = MIN_SCORE,
    pad_s: float = PAD_S,
    max_len_s: float = MAX_LEN_S,
    max_gap_s: float = MAX_GAP_S,
) -> list[Segment]:
    scored = [(c, lex.score(c.text)) for c in cues]

    runs: list[list[tuple[Cue, lex_mod.CueScore]]] = []
    current: list[tuple[Cue, lex_mod.CueScore]] = []
    cold_streak = 0
    prev_end: float | None = None

    for cue, score in scored:
        # A long silence (no cues at all, hot or cold) always ends a run,
        # independent of the cold-cue-count bridging below: one rule
        # tolerates a brief verbal pause, the other refuses to bridge a
        # real gap in the transcript (silence, music, a cut edit).
        if current and prev_end is not None and (cue.start_s - prev_end) > max_gap_s:
            runs.append(current)
            current, cold_streak = [], 0
        prev_end = cue.end_s

        if score.score >= min_score:
            current.append((cue, score))
            cold_streak = 0
            continue
        if not current:
            continue
        cold_streak += 1
        if cold_streak > GAP_TOLERANCE:
            runs.append(current)
            current, cold_streak = [], 0
        else:
            current.append((cue, score))

    if current:
        runs.append(current)

    # Trim trailing cold cues that were only kept to bridge a gap.
    trimmed: list[list[tuple[Cue, lex_mod.CueScore]]] = []
    for run in runs:
        while run and run[-1][1].score < min_score:
            run = run[:-1]
        if run:
            trimmed.append(run)

    segments: list[Segment] = []
    for run in trimmed:
        start = max(0.0, run[0][0].start_s - pad_s)
        end = run[-1][0].end_s + pad_s
        total = sum(s.score for _, s in run)
        categories = sorted({c for _, s in run for c in s.categories})
        terms = sorted({t for _, s in run for t in s.matched_terms})
        visual = any(s.visual_cue for _, s in run)
        text = " ".join(c.text for c, _ in run)

        pieces = _split(start, end, max_len_s)
        for piece_start, piece_end in pieces:
            segments.append(
                Segment(
                    id="",
                    start_s=round(piece_start, 3),
                    end_s=round(piece_end, 3),
                    score=round(total / len(pieces), 3),
                    categories=categories,
                    matched_terms=terms,
                    text=text,
                    visual_cue=visual,
                )
            )

    # Ids in time order so an id always names the same moment; ranking after.
    segments.sort(key=lambda s: s.start_s)
    for index, seg in enumerate(segments, start=1):
        seg.id = f"seg_{index:03d}"
    segments.sort(key=lambda s: (-s.score, s.start_s))
    return segments


def run(paths: ProjectPaths, video_id: str, lex: lex_mod.Lexicon | None = None) -> list[Segment]:
    m = mf.load(paths, video_id)
    mf.require_ready(m, "segment")

    transcript = Transcript.model_validate_json(paths.transcript(video_id).read_text())
    lex = lex or lex_mod.load(paths.lexicon_file)
    segments = build(transcript.cues, lex)

    paths.segments(video_id).write_text(
        json.dumps(
            {
                "lexicon_version": lex.version,
                "total_cues": len(transcript.cues),
                "segments": [s.model_dump() for s in segments],
            },
            indent=2,
        )
        + "\n"
    )
    mf.finish(paths, m, "segment", n_segments=len(segments), lexicon_version=lex.version)
    return segments
