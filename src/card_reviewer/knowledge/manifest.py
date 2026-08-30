"""Work packet persistence and the stage state machine.

A stage may only run when every stage before it in STAGES is `done`. This is
what makes the pipeline resumable: state lives on disk, not in a process.
"""

from __future__ import annotations

import datetime
from typing import Any

from .models import STAGES, Manifest, StageState, StageStatus
from .paths import ProjectPaths


class PacketNotFound(Exception):
    """No manifest exists for this video_id."""


class StageNotReady(Exception):
    """A prerequisite stage has not completed."""


class UnknownStage(Exception):
    """`require_ready` was asked about a stage name not in STAGES."""


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def load(paths: ProjectPaths, video_id: str) -> Manifest:
    path = paths.manifest(video_id)
    if not path.exists():
        raise PacketNotFound(
            f"no work packet for {video_id!r} at {path}. Run `acquire` first."
        )
    return Manifest.model_validate_json(path.read_text())


def save(paths: ProjectPaths, m: Manifest) -> None:
    path = paths.manifest(m.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(m.model_dump_json(indent=2) + "\n")


def is_done(m: Manifest, stage: str) -> bool:
    return m.stages[stage].status is StageStatus.DONE


def start(paths: ProjectPaths, m: Manifest, stage: str) -> Manifest:
    m.stages[stage] = StageState(status=StageStatus.RUNNING, at=_now())
    save(paths, m)
    return m


def finish(paths: ProjectPaths, m: Manifest, stage: str, **detail: Any) -> Manifest:
    m.stages[stage] = StageState(
        status=StageStatus.DONE, at=_now(), detail=dict(detail)
    )
    save(paths, m)
    return m


def fail(paths: ProjectPaths, m: Manifest, stage: str, error: str) -> Manifest:
    m.stages[stage] = StageState(status=StageStatus.FAILED, at=_now(), error=error)
    save(paths, m)
    return m


def require_ready(m: Manifest, stage: str) -> None:
    """Raise StageNotReady if any earlier stage has not completed."""
    if stage not in STAGES:
        raise UnknownStage(
            f"unknown stage {stage!r}; valid stages are {', '.join(STAGES)}"
        )
    for earlier in STAGES[: STAGES.index(stage)]:
        if not is_done(m, earlier):
            raise StageNotReady(
                f"cannot run {stage!r}: stage {earlier!r} is "
                f"{m.stages[earlier].status.value}, expected 'done'"
            )
