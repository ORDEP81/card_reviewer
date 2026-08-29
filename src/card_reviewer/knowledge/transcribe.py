"""Stage 2: produce a timestamped transcript.

Captions are free when they exist; Skool course video generally has none, so a
local Whisper fallback keeps Pass 1 available for every source.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from . import manifest as mf
from .models import Cue, Transcript
from .paths import ProjectPaths

Runner = Callable[..., subprocess.CompletedProcess]

WHISPER_MODEL = "mlx-community/whisper-medium-mlx"

_TIMING = re.compile(
    r"^(?P<start>[\d:.]+)\s*-->\s*(?P<end>[\d:.]+)"
)

# YouTube auto-captions embed per-word timing inside cue text, e.g.
# `to<00:00:04.000><c> buy</c><00:00:04.200><c> the</c>`. Match ONLY the two
# shapes YouTube actually emits -- a `<HH:MM:SS.mmm>` timestamp, or a
# `<c>`/`</c>`/`<c.classname>` span tag -- never a bare `<[^>]*>`, which would
# also swallow a literal angle bracket used in ordinary speech (e.g. a
# centering ratio like "<20/80 or >15/85").
_INLINE_TAG = re.compile(
    r"<(?:\d{2}:\d{2}:\d{2}\.\d{3}|/?c(?:\.[\w-]+)?)>"
)
_WHITESPACE = re.compile(r"\s+")

# YouTube's rolling caption window is strictly contiguous: a settle cue's
# end_s equals the next cue's start_s (real data shows a gap of 0-10ms).
# Two cues separated by more than this are two distinct moments in the
# video, even if their text happens to match or overlap.
_CONTIGUITY_EPSILON_S = 0.05


def _to_seconds(stamp: str) -> float:
    """Accept HH:MM:SS.mmm or MM:SS.mmm."""
    parts = stamp.strip().split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def _strip_inline_tags(line: str) -> str:
    """Remove YouTube's inline `<...>` word-timing and `<c>` markup."""
    return _INLINE_TAG.sub("", line)


def parse_vtt(text: str) -> list[Cue]:
    """Parse WebVTT into cues.

    Handles two YouTube auto-caption quirks on top of plain WebVTT:
    - inline per-word timing tags (`<00:00:04.000>`, `<c>`/`</c>`) in cue text
    - a rolling caption window that repeats the previous cue's settled text
      as the first line of the next cue, interspersed with ~10ms "settle"
      cues that add nothing new

    Rolling-window collapsing only fires when ALL of the following hold, so
    that it can only ever suppress text that is truly a re-display of what
    was already emitted -- never fabricate or discard genuine speech:
    - this cue starts within `_CONTIGUITY_EPSILON_S` of the previously
      *emitted* cue's end (YouTube's window is strictly contiguous; a real
      gap -- e.g. the same phrase spoken again later -- means these are two
      distinct utterances, not one rolling window, so both are kept);
    - this cue's normalized text begins with the exact text of the
      previously emitted cue;
    - that match ends at a word boundary (the previous text is the whole
      cue, or is followed by a space) -- otherwise a merely-contiguous
      prefix like "I see" of "I seem" would slice a real word in half and
      emit a fabricated fragment.

    Two more normalizations apply to every cue, not just collapsed ones:
    whitespace is collapsed to single spaces (this also tidies up doubled
    spaces in hand-written VTT, not only tag-stripping artifacts -- a
    deliberate, not incidental, behavior change from the pre-fix parser).
    A block separator is an exactly empty line; a line that is merely
    whitespace does not end the block (see the loop below) -- so a
    hand-written VTT that pads its separator lines with trailing spaces
    would behave differently here than a truly blank line.
    """
    cues: list[Cue] = []
    block_lines: list[str] = []
    timing: tuple[float, float] | None = None
    previous_text = ""
    previous_end_s: float | None = None

    def flush() -> None:
        nonlocal timing, block_lines, previous_text, previous_end_s
        if timing and block_lines:
            joined = " ".join(_strip_inline_tags(l) for l in block_lines)
            cue_text = _WHITESPACE.sub(" ", joined).strip()
            if cue_text:
                start, end = timing
                contiguous = (
                    previous_end_s is not None
                    and abs(start - previous_end_s) <= _CONTIGUITY_EPSILON_S
                )
                at_word_boundary = len(cue_text) == len(
                    previous_text
                ) or cue_text[len(previous_text) :].startswith(" ")
                collapsible = (
                    contiguous
                    and previous_text != ""
                    and cue_text.startswith(previous_text)
                    and at_word_boundary
                )
                remainder = (
                    cue_text[len(previous_text) :].strip()
                    if collapsible
                    else cue_text
                )
                if remainder:
                    cues.append(Cue(start_s=start, end_s=end, text=remainder))
                    previous_text = remainder
                previous_end_s = end
        timing, block_lines = None, []

    for raw in text.splitlines():
        if raw == "":
            # A truly empty line separates cue blocks. A line that is only
            # whitespace (YouTube uses a lone space as a placeholder for a
            # not-yet-populated caption line) is NOT a separator -- it must
            # not end the block early, or the cue's real content line(s)
            # that follow would be orphaned with no active timing.
            flush()
            continue
        line = raw.strip()
        if not line:
            continue
        if line == "WEBVTT" or line.startswith("NOTE"):
            continue
        match = _TIMING.match(line)
        if match:
            timing = (_to_seconds(match["start"]), _to_seconds(match["end"]))
            block_lines = []
            continue
        if timing is None:
            # A cue identifier line preceding the timing line; ignore it.
            continue
        block_lines.append(line)

    flush()
    return cues


def fetch_captions(
    url: str | None,
    dest_dir: Path,
    browser: str | None = None,
    runner: Runner = subprocess.run,
) -> list[Cue] | None:
    """Return caption cues, or None when the source has no usable captions."""
    if not url:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Clear any caption files left by an earlier run first. yt-dlp can exit 0
    # while writing nothing this invocation (expired cookies, a re-upload
    # that lost its captions, a subtitle-specific network failure); without
    # this, the glob below could pick up a stale file and this run would
    # falsely report method="captions" sourced from a prior transcript.
    for stale in dest_dir.glob("captions*.vtt"):
        stale.unlink()
    cookie = ["--cookies-from-browser", browser] if browser else []
    proc = runner(
        [
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--write-subs",
            "--sub-langs", "en.*",
            "--sub-format", "vtt",
            *cookie,
            "-o", str(dest_dir / "captions.%(ext)s"),
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    files = sorted(dest_dir.glob("captions*.vtt"))
    if not files:
        return None
    cues = parse_vtt(files[0].read_text())
    return cues or None


def whisper_transcribe(
    video_path: Path, transcriber: Callable | None = None
) -> tuple[list[Cue], str, str]:
    """Transcribe locally. Returns (cues, model_name, language)."""
    if transcriber is None:  # pragma: no cover - exercised only with a real model
        import mlx_whisper

        def transcriber(path, **kwargs):
            return mlx_whisper.transcribe(str(path), path_or_hf_repo=WHISPER_MODEL)

    result = transcriber(video_path)
    cues = [
        Cue(
            start_s=float(seg["start"]),
            end_s=float(seg["end"]),
            text=str(seg["text"]).strip(),
        )
        for seg in result.get("segments", [])
    ]
    return cues, WHISPER_MODEL, result.get("language", "en")


def run(
    paths: ProjectPaths,
    video_id: str,
    browser: str | None = None,
    runner: Runner = subprocess.run,
    transcriber: Callable | None = None,
) -> Transcript:
    m = mf.load(paths, video_id)
    mf.require_ready(m, "transcribe")

    cues = fetch_captions(
        m.source.url, paths.source_dir(video_id), browser=browser, runner=runner
    )
    if cues:
        transcript = Transcript(method="captions", model=None, language="en", cues=cues)
    else:
        videos = sorted(paths.source_dir(video_id).glob("video.*"))
        if not videos:
            raise FileNotFoundError(f"no media in {paths.source_dir(video_id)}")
        cues, model, language = whisper_transcribe(videos[0], transcriber)
        transcript = Transcript(
            method="mlx-whisper", model=model, language=language, cues=cues
        )

    paths.transcript(video_id).write_text(transcript.model_dump_json(indent=2) + "\n")
    mf.finish(
        paths,
        m,
        "transcribe",
        method=transcript.method,
        model=transcript.model,
        cues=len(transcript.cues),
    )
    return transcript
