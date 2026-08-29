"""Every filesystem path in the project, derived from one root.

No other module computes paths. Tests pass a tmp_path root; the CLI passes the
repository root.
"""

from __future__ import annotations

from pathlib import Path


class ProjectPaths:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def work(self) -> Path:
        return self.training / "work"

    @property
    def lessons(self) -> Path:
        return self.training / "lessons"

    @property
    def knowledge(self) -> Path:
        return self.root / "knowledge"

    @property
    def pending_rules(self) -> Path:
        return self.knowledge / "pending_rules"

    @property
    def rules(self) -> Path:
        return self.knowledge / "rules"

    @property
    def rubric_file(self) -> Path:
        return self.knowledge / "ACTIVE_RUBRIC.md"

    @property
    def version_file(self) -> Path:
        return self.knowledge / "RUBRIC_VERSION"

    @property
    def lexicon_file(self) -> Path:
        return self.knowledge / "segmentation_lexicon.yaml"

    def packet(self, video_id: str) -> Path:
        return self.work / video_id

    def manifest(self, video_id: str) -> Path:
        return self.packet(video_id) / "manifest.json"

    def source_dir(self, video_id: str) -> Path:
        return self.packet(video_id) / "source"

    def transcript(self, video_id: str) -> Path:
        return self.packet(video_id) / "transcript.json"

    def segments(self, video_id: str) -> Path:
        return self.packet(video_id) / "segments.json"

    def frames(self, video_id: str) -> Path:
        return self.packet(video_id) / "frames"

    def lesson(self, lesson_id: str) -> Path:
        return self.lessons / f"{lesson_id}.md"
