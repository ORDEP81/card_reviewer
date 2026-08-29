"""Advance a packet through every deterministic stage, then hand off to Claude.

`analyze` is deliberately absent from DETERMINISTIC_STAGES: this module stops
where judgment begins.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import manifest as mf
from .models import Manifest
from .paths import ProjectPaths

DETERMINISTIC_STAGES = ("acquire", "transcribe", "segment", "extract_frames")


def _default_steps() -> dict[str, Callable]:
    from . import acquire, frames, segment, transcribe

    def do_acquire(paths, url=None, file=None, browser=None, rubric_version="0.1.0"):
        if file:
            return acquire.from_file(paths, file, rubric_version)
        return acquire.from_url(paths, url, rubric_version, browser=browser)

    def do_transcribe(paths, video_id, browser=None, **_):
        transcribe.run(paths, video_id, browser=browser)

    def do_segment(paths, video_id, **_):
        segment.run(paths, video_id)

    def do_frames(paths, video_id, top_n=12, **_):
        frames.run(paths, video_id, top_n=top_n)

    return {
        "acquire": do_acquire,
        "transcribe": do_transcribe,
        "segment": do_segment,
        "extract_frames": do_frames,
    }


def run_all(
    paths: ProjectPaths,
    url: str | None = None,
    file: Path | str | None = None,
    browser: str | None = None,
    top_n: int = 12,
    force: bool = False,
    steps: dict[str, Callable] | None = None,
) -> Manifest:
    """Advance a packet through DETERMINISTIC_STAGES, skipping stages already done.

    Contract for a custom `steps["acquire"]`: it must behave like
    `acquire.from_url`/`acquire.from_file` and preserve `stages` and
    `lesson_id` when a manifest for this `video_id` already exists, updating
    only `source`/`file` in place. The skip decision below trusts the
    manifest `steps["acquire"]` returns; an acquire step that rebuilds the
    manifest from scratch will make every downstream stage look pending and
    rerun regardless of prior completion.
    """
    if not url and not file:
        raise ValueError("run_all requires either url or file")

    from . import version

    steps = steps or _default_steps()

    m = steps["acquire"](
        paths, url=url, file=file, browser=browser, rubric_version=version.read(paths)
    )

    for stage in DETERMINISTIC_STAGES[1:]:
        if mf.is_done(m, stage) and not force:
            continue
        steps[stage](paths, m.video_id, browser=browser, top_n=top_n)

    return mf.load(paths, m.video_id)


def status(paths: ProjectPaths, video_id: str | None = None) -> list[Manifest]:
    if video_id:
        return [mf.load(paths, video_id)]
    if not paths.work.exists():
        return []
    return [
        Manifest.model_validate_json(p.read_text())
        for p in sorted(paths.work.glob("*/manifest.json"))
    ]
