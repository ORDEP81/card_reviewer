# Video Learning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable pipeline that turns sports-card grading videos into a versioned, auditable knowledge base of grading rules.

**Architecture:** Each video becomes a work packet directory whose `manifest.json` holds a stage state machine. Python CLI subcommands advance the deterministic stages (acquire, transcribe, segment, extract-frames, validate); Claude performs the one judgment stage (`analyze`) in an interactive session driven by a skill, writing a lesson record and candidate rule files. Rules are mechanically validated, then approved by the user one at a time before being promoted into a semver-versioned rubric that a future card review engine consumes.

**Tech Stack:** Python 3.14, uv, Pydantic v2, Typer, Rich, PyYAML, imagehash/Pillow, pytest. External binaries: yt-dlp, ffmpeg. Local transcription via mlx-whisper.

**Spec:** `docs/superpowers/specs/2026-08-28-video-learning-pipeline-design.md`

## Global Constraints

- **Python `>=3.14`.** Verified: `mlx-whisper==0.4.3`, `mlx==0.32.2`, `pydantic==2.13.5`, `typer==0.27.2`, `imagehash==4.3.2` all resolve on CPython 3.14.6 / macOS arm64. No version pin needed.
- **No OpenCV, no grading logic, no probability logic, no card review schema** beyond `load_active_rubric()`. Those belong to subsystem A's separate spec.
- **No pricing, EV, market-value, or buy/sell logic anywhere in this repository.** (Plan §30 rule 14.)
- **Never delete a rule.** Rejected and superseded rules stay on disk with `status` changed. (Plan §30 rule 9.)
- **Never circumvent DRM or platform protections.** Authenticated acquisition failure is recorded and reported; the only fallback is the user supplying a local file. (Plan §4, §30 rule 13.)
- **Never write cookies, tokens, or credentials into the repository.** `--cookies-from-browser` reads the live session at run time only.
- **`evidence_type` on a rule is required and has no default.** (Plan §30 rule 11.)
- **`ACTIVE_RUBRIC.md` is generated, never hand-edited.** YAML rule files under `knowledge/rules/` are the source of truth.
- **Stage keys use underscores** (`extract_frames`); **CLI subcommands use hyphens** (`extract-frames`). Do not mix them.
- **All timestamps in manifests are UTC ISO-8601.** All durations and cue boundaries are float seconds.

---

## File Structure

Subsystem B lives entirely under `src/card_reviewer/knowledge/`. Each module has one responsibility and is independently testable.

| File | Responsibility |
|---|---|
| `models.py` | All Pydantic models: `Manifest`, `StageState`, `Rule`, `Segment`, `Transcript`. No I/O, no logic beyond validation. |
| `paths.py` | Every filesystem path in the project, derived from a project root. Nothing else computes paths. |
| `manifest.py` | Load/save a manifest; the stage state machine and its idempotency rules. |
| `doctor.py` | Preflight checks for yt-dlp, ffmpeg, mlx-whisper. |
| `acquire.py` | yt-dlp invocation, local file adoption, `video_id` derivation, failure capture. |
| `transcribe.py` | Caption extraction and mlx-whisper fallback, normalized to one `Transcript` shape. |
| `lexicon.py` | Load `segmentation_lexicon.yaml`; score a single cue. |
| `segment.py` | Merge scored cues into ranked windows; write `segments.json`. |
| `frames.py` | ffmpeg frame sampling and perceptual-hash deduplication. |
| `validate.py` | Schema, ID uniqueness, lesson existence, and timestamp-bounds checks. |
| `dedup.py` | Duplicate and contradiction flagging against active rules. |
| `version.py` | Read, write, and bump `RUBRIC_VERSION`. |
| `promote.py` | Rule status transitions: accept, reject, supersede. |
| `rubric.py` | `build_rubric()` renderer and the `load_active_rubric()` contract for subsystem A. |
| `cli.py` | Typer wiring only. No business logic — every command delegates to a module above. |

`skills/learn-video/SKILL.md` drives the `analyze` stage. `knowledge/segmentation_lexicon.yaml` is data, not code.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `CLAUDE.md`
- Create: `src/card_reviewer/__init__.py`, `src/card_reviewer/knowledge/__init__.py`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable package `card_reviewer.knowledge`; a working `pytest` run; the `card-knowledge` console script name (wired in Task 4)

- [ ] **Step 1: Write the failing test**

Create `tests/test_scaffolding.py`:

```python
"""The package must be importable and the repo must refuse to track secrets."""
import pathlib
import tomllib


REPO = pathlib.Path(__file__).resolve().parents[1]


def test_package_is_importable():
    import card_reviewer.knowledge  # noqa: F401


def test_python_floor_is_3_14():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert data["project"]["requires-python"] == ">=3.14"


def test_gitignore_covers_secrets_and_media():
    body = (REPO / ".gitignore").read_text()
    for pattern in (
        "cookies.txt",
        "training/work/*/source/",
        ".env",
        "*.mp4",
    ):
        assert pattern in body, f"missing .gitignore entry: {pattern}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scaffolding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer'`

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "card-reviewer"
version = "0.1.0"
description = "Sports card grading knowledge pipeline and review engine"
requires-python = ">=3.14"
dependencies = [
    "pydantic>=2.13",
    "typer>=0.27",
    "rich>=14",
    "pyyaml>=6",
    "imagehash>=4.3",
    "pillow>=11",
    "mlx-whisper>=0.4.3",
]

[project.scripts]
card-knowledge = "card_reviewer.knowledge.cli:app"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/card_reviewer"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create `.gitignore`**

```gitignore
# Secrets — never commit (spec §9, plan §4)
cookies.txt
*.cookies
.env
.env.*
credentials.json
*token*.json

# Downloaded media — large, and possibly licensed course material
training/work/*/source/
*.mp4
*.mov
*.mkv
*.webm
*.m4a

# Extracted frames — regenerable from source
training/work/*/frames/

# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
*.egg-info/
dist/
build/
```

- [ ] **Step 5: Create the package**

```bash
mkdir -p src/card_reviewer/knowledge tests
touch src/card_reviewer/__init__.py
printf '"""Video learning pipeline: videos to versioned grading rules."""\n' > src/card_reviewer/knowledge/__init__.py
```

- [ ] **Step 6: Create the knowledge and training directories**

Empty directories are invisible to git, so each gets a `.gitkeep`. The pipeline
writes into these on the first real run.

```bash
mkdir -p knowledge/pending_rules knowledge/rules knowledge/sets \
         knowledge/psa knowledge/card_types \
         training/work training/lessons \
         skills/learn-video docs/superpowers/plans
for d in knowledge/pending_rules knowledge/rules knowledge/sets \
         knowledge/psa knowledge/card_types training/work training/lessons; do
  touch "$d/.gitkeep"
done
echo "0.1.0" > knowledge/RUBRIC_VERSION
```

- [ ] **Step 7: Create `CLAUDE.md`**

```markdown
# Card Reviewer

Two subsystems joined only by `knowledge/`:

- **A — Card Review Engine** (`src/card_reviewer/`, not yet built): images to a
  structured PSA grade estimate.
- **B — Video Learning Pipeline** (`src/card_reviewer/knowledge/`): training
  videos to versioned grading rules.

**Build order is B before A**, reversing the phase numbering in
`CARD_REVIEWER_BUILD_PLAN.md`. Do not start OpenCV or grading work.

Spec: `docs/superpowers/specs/2026-08-28-video-learning-pipeline-design.md`
Plan: `docs/superpowers/plans/2026-08-28-video-learning-pipeline.md`

## Non-negotiable rules

1. Never use card value when judging condition.
2. Never automatically assume PSA 10; search for reasons a card will not gem.
3. Never hide image limitations or claim certainty photographs cannot support.
4. OpenCV measurements are evidence, not the grader. Claude assessment is
   evidence, not measurement.
5. Distinguish definite defects from suspected defects.
6. Preserve original images, knowledge provenance, and rubric version.
7. Do not blindly learn every claim in a training video — classify each by
   `evidence_type`.
8. Never circumvent DRM or access material the user is not authorized to view.
9. No pricing, EV, buying, or selling logic in this repository.
10. Never delete a rule. Change its `status`.

## Commands

    uv sync                    # install dependencies
    uv run pytest              # run tests
    uv run card-knowledge --help
```

- [ ] **Step 8: Install and run the tests**

Run:
```bash
uv sync
uv run pytest tests/test_scaffolding.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore CLAUDE.md src tests uv.lock knowledge training
git commit -m "feat: project scaffolding for video learning pipeline"
```

---

## Task 2: Core models

**Files:**
- Create: `src/card_reviewer/knowledge/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `STAGES: tuple[str, ...]` = `("acquire", "transcribe", "segment", "extract_frames", "analyze", "validate")`
  - `StageStatus`, `EvidenceType`, `Confidence`, `RuleStatus`, `Category` — str enums
  - `StageState(status, at, error, detail)`
  - `SourceInfo(type, url, title, uploader, duration_s)`
  - `FileInfo(path, sha256, bytes)`
  - `Manifest(video_id, source, file, stages, lesson_id, rubric_version_at_ingest)`
  - `Cue(start_s, end_s, text)`, `Transcript(method, model, language, cues)`
  - `Segment(id, start_s, end_s, score, categories, matched_terms, text, visual_cue)`
  - `AppliesTo(card_types, sets)`, `RuleSource(lesson, video_id, timestamps, quote)`
  - `Rule(id, category, statement, evidence_type, confidence, applies_to, sources, status, supersedes, created, rubric_version_added, notes)` — `notes` carries rejection reasons written by Task 13

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
"""Models enforce the spec's non-negotiables at the type level."""
import datetime

import pytest
from pydantic import ValidationError

from card_reviewer.knowledge.models import (
    STAGES,
    AppliesTo,
    Confidence,
    EvidenceType,
    Manifest,
    Rule,
    RuleSource,
    RuleStatus,
    SourceInfo,
    StageState,
    StageStatus,
)


def a_source(**over):
    base = dict(
        lesson="lesson_014",
        video_id="yt_abc123",
        timestamps=["12:04-12:38"],
        quote="I've never seen one gem with a line like that.",
    )
    return RuleSource(**(base | over))


def a_rule(**over):
    base = dict(
        id="SURFACE_PRINT_LINE_001",
        category="surface",
        statement="Vertical print lines commonly prevent a PSA 10.",
        evidence_type=EvidenceType.EXPERIENCE_BASED,
        confidence=Confidence.HIGH,
        sources=[a_source()],
        created=datetime.date(2026, 8, 28),
    )
    return Rule(**(base | over))


def test_stage_order_is_the_spec_order():
    assert STAGES == (
        "acquire",
        "transcribe",
        "segment",
        "extract_frames",
        "analyze",
        "validate",
    )


def test_rule_requires_evidence_type():
    """Spec §7: evidence_type is required with no default (plan §30 rule 11)."""
    with pytest.raises(ValidationError):
        Rule(
            id="SURFACE_001",
            category="surface",
            statement="x",
            confidence=Confidence.HIGH,
            sources=[a_source()],
            created=datetime.date(2026, 8, 28),
        )


def test_rule_requires_at_least_one_source():
    with pytest.raises(ValidationError):
        a_rule(sources=[])


def test_rule_rejects_empty_statement():
    with pytest.raises(ValidationError):
        a_rule(statement="   ")


def test_rule_defaults_to_pending_and_unversioned():
    rule = a_rule()
    assert rule.status is RuleStatus.PENDING
    assert rule.rubric_version_added is None
    assert rule.applies_to == AppliesTo()


def test_rule_id_must_be_uppercase_slug():
    with pytest.raises(ValidationError):
        a_rule(id="surface print line 1")


def test_manifest_starts_every_stage_pending():
    manifest = Manifest(
        video_id="yt_abc123",
        source=SourceInfo(
            type="youtube",
            url="https://youtube.com/watch?v=abc123",
            title="Grading 101",
            uploader="Someone",
            duration_s=3120.0,
        ),
        rubric_version_at_ingest="0.1.0",
    )
    assert set(manifest.stages) == set(STAGES)
    assert all(s.status is StageStatus.PENDING for s in manifest.stages.values())


def test_rule_notes_default_to_none():
    """notes carries rejection reasons; Task 13 writes it."""
    assert a_rule().notes is None


def test_rule_accepts_notes():
    assert a_rule(notes="rejected: opinion").notes == "rejected: opinion"


def test_stage_state_carries_error_on_failure():
    state = StageState(status=StageStatus.FAILED, error="yt-dlp exited 1")
    assert state.error == "yt-dlp exited 1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.models'`

- [ ] **Step 3: Write the implementation**

Create `src/card_reviewer/knowledge/models.py`:

```python
"""Pydantic models for the video learning pipeline.

These types carry the spec's non-negotiables. Validation lives here so that no
downstream module has to remember them.
"""

from __future__ import annotations

import datetime
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

STAGES: tuple[str, ...] = (
    "acquire",
    "transcribe",
    "segment",
    "extract_frames",
    "analyze",
    "validate",
)

RULE_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]*[0-9]{3}$")


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvidenceType(StrEnum):
    OBJECTIVE = "objective"
    EXPERIENCE_BASED = "experience_based"
    OPINION = "opinion"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RuleStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class Category(StrEnum):
    CENTERING = "centering"
    CORNERS = "corners"
    EDGES = "edges"
    SURFACE = "surface"
    PRINT = "print"
    HANDLING = "handling"
    IMAGE_LIMITATIONS = "image_limitations"
    PROCESS = "process"


class StageState(BaseModel):
    status: StageStatus = StageStatus.PENDING
    at: datetime.datetime | None = None
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class SourceInfo(BaseModel):
    type: Literal["youtube", "skool", "local"]
    url: str | None = None
    title: str
    uploader: str | None = None
    duration_s: float = Field(ge=0)


class FileInfo(BaseModel):
    path: str
    sha256: str
    bytes: int = Field(ge=0)


def _default_stages() -> dict[str, StageState]:
    return {name: StageState() for name in STAGES}


class Manifest(BaseModel):
    video_id: str
    source: SourceInfo
    file: FileInfo | None = None
    stages: dict[str, StageState] = Field(default_factory=_default_stages)
    lesson_id: str | None = None
    rubric_version_at_ingest: str


class Cue(BaseModel):
    """One timestamped utterance from a transcript."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    text: str


class Transcript(BaseModel):
    method: Literal["captions", "mlx-whisper"]
    model: str | None = None
    language: str = "en"
    cues: list[Cue] = Field(default_factory=list)


class Segment(BaseModel):
    """A ranked candidate window worth visual inspection."""

    id: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    score: float
    categories: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    text: str = ""
    visual_cue: bool = False


class AppliesTo(BaseModel):
    card_types: list[str] = Field(default_factory=list)
    sets: list[str] = Field(default_factory=list)


class RuleSource(BaseModel):
    lesson: str
    video_id: str
    timestamps: list[str] = Field(min_length=1)
    quote: str = ""


class Rule(BaseModel):
    id: str
    category: Category
    statement: str
    evidence_type: EvidenceType
    confidence: Confidence
    applies_to: AppliesTo = Field(default_factory=AppliesTo)
    sources: list[RuleSource] = Field(min_length=1)
    status: RuleStatus = RuleStatus.PENDING
    supersedes: str | None = None
    created: datetime.date
    rubric_version_added: str | None = None
    notes: str | None = None  # rejection reasons and review annotations

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, value: str) -> str:
        if not RULE_ID_RE.match(value):
            raise ValueError(
                f"rule id {value!r} must be an uppercase slug ending in three "
                "digits, e.g. SURFACE_PRINT_LINE_001"
            )
        return value

    @field_validator("statement")
    @classmethod
    def _statement_is_meaningful(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("statement must not be empty")
        return value.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/knowledge/models.py tests/test_models.py
git commit -m "feat: core Pydantic models for pipeline and rules"
```

---

## Task 3: Paths and the manifest state machine

**Files:**
- Create: `src/card_reviewer/knowledge/paths.py`
- Create: `src/card_reviewer/knowledge/manifest.py`
- Test: `tests/test_paths.py`, `tests/test_manifest.py`

**Interfaces:**
- Consumes: `models.Manifest`, `models.StageState`, `models.StageStatus`, `models.STAGES`
- Produces:
  - `paths.ProjectPaths(root)` with attributes `root`, `work`, `lessons`, `knowledge`, `pending_rules`, `rules`, `rubric_file`, `version_file`, `lexicon_file`; and methods `packet(video_id) -> Path`, `manifest(video_id) -> Path`, `transcript(video_id) -> Path`, `segments(video_id) -> Path`, `frames(video_id) -> Path`, `source_dir(video_id) -> Path`, `lesson(lesson_id) -> Path`
  - `manifest.load(paths, video_id) -> Manifest`
  - `manifest.save(paths, m) -> None`
  - `manifest.start(paths, m, stage) -> Manifest`
  - `manifest.finish(paths, m, stage, **detail) -> Manifest`
  - `manifest.fail(paths, m, stage, error) -> Manifest`
  - `manifest.is_done(m, stage) -> bool`
  - `manifest.require_ready(m, stage) -> None` — raises `StageNotReady` if a prerequisite stage is not `done`
  - Exceptions `StageNotReady`, `PacketNotFound`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paths.py`:

```python
from card_reviewer.knowledge.paths import ProjectPaths


def test_packet_paths_are_derived_from_root(tmp_path):
    p = ProjectPaths(tmp_path)
    assert p.packet("yt_abc") == tmp_path / "training" / "work" / "yt_abc"
    assert p.manifest("yt_abc") == p.packet("yt_abc") / "manifest.json"
    assert p.transcript("yt_abc") == p.packet("yt_abc") / "transcript.json"
    assert p.segments("yt_abc") == p.packet("yt_abc") / "segments.json"
    assert p.frames("yt_abc") == p.packet("yt_abc") / "frames"
    assert p.source_dir("yt_abc") == p.packet("yt_abc") / "source"


def test_knowledge_paths(tmp_path):
    p = ProjectPaths(tmp_path)
    assert p.pending_rules == tmp_path / "knowledge" / "pending_rules"
    assert p.rules == tmp_path / "knowledge" / "rules"
    assert p.rubric_file == tmp_path / "knowledge" / "ACTIVE_RUBRIC.md"
    assert p.version_file == tmp_path / "knowledge" / "RUBRIC_VERSION"
    assert p.lexicon_file == tmp_path / "knowledge" / "segmentation_lexicon.yaml"
    assert p.lesson("lesson_001") == tmp_path / "training" / "lessons" / "lesson_001.md"
```

Create `tests/test_manifest.py`:

```python
import pytest

from card_reviewer.knowledge import manifest as mf
from card_reviewer.knowledge.models import Manifest, SourceInfo, StageStatus
from card_reviewer.knowledge.paths import ProjectPaths


@pytest.fixture
def paths(tmp_path):
    return ProjectPaths(tmp_path)


@pytest.fixture
def packet(paths):
    m = Manifest(
        video_id="yt_abc",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=100.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(paths, m)
    return m


def test_save_then_load_roundtrips(paths, packet):
    loaded = mf.load(paths, "yt_abc")
    assert loaded.video_id == "yt_abc"
    assert loaded.source.duration_s == 100.0


def test_load_missing_packet_raises(paths):
    with pytest.raises(mf.PacketNotFound):
        mf.load(paths, "nope")


def test_finish_marks_done_and_records_detail(paths, packet):
    m = mf.finish(paths, packet, "acquire", tool="yt-dlp 2026.1.1")
    assert m.stages["acquire"].status is StageStatus.DONE
    assert m.stages["acquire"].at is not None
    assert m.stages["acquire"].detail["tool"] == "yt-dlp 2026.1.1"
    assert mf.is_done(mf.load(paths, "yt_abc"), "acquire")


def test_fail_records_error_and_is_not_done(paths, packet):
    m = mf.fail(paths, packet, "acquire", "yt-dlp exited 1: sign in required")
    assert m.stages["acquire"].status is StageStatus.FAILED
    assert "sign in required" in m.stages["acquire"].error
    assert not mf.is_done(m, "acquire")


def test_require_ready_blocks_when_prerequisite_incomplete(paths, packet):
    """transcribe cannot run before acquire is done."""
    with pytest.raises(mf.StageNotReady, match="acquire"):
        mf.require_ready(packet, "transcribe")


def test_require_ready_passes_when_prerequisites_done(paths, packet):
    m = mf.finish(paths, packet, "acquire")
    mf.require_ready(m, "transcribe")  # must not raise


def test_first_stage_has_no_prerequisites(packet):
    mf.require_ready(packet, "acquire")  # must not raise


def test_failed_prerequisite_also_blocks(paths, packet):
    m = mf.fail(paths, packet, "acquire", "boom")
    with pytest.raises(mf.StageNotReady):
        mf.require_ready(m, "transcribe")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_paths.py tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError` for `paths` and `manifest`

- [ ] **Step 3: Write `paths.py`**

```python
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
```

- [ ] **Step 4: Write `manifest.py`**

```python
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
    for earlier in STAGES[: STAGES.index(stage)]:
        if not is_done(m, earlier):
            raise StageNotReady(
                f"cannot run {stage!r}: stage {earlier!r} is "
                f"{m.stages[earlier].status.value}, expected 'done'"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_paths.py tests/test_manifest.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add src/card_reviewer/knowledge/paths.py src/card_reviewer/knowledge/manifest.py tests/test_paths.py tests/test_manifest.py
git commit -m "feat: project paths and resumable stage state machine"
```

---

## Task 4: CLI skeleton and `doctor`

**Files:**
- Create: `src/card_reviewer/knowledge/doctor.py`
- Create: `src/card_reviewer/knowledge/cli.py`
- Test: `tests/test_doctor.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `paths.ProjectPaths`
- Produces:
  - `doctor.ToolCheck(name, found, version, install_hint)` — a dataclass
  - `doctor.check_all(runner=subprocess.run, has_module=...) -> list[ToolCheck]`
  - `cli.app` — the Typer application, the console-script entry point
  - `cli.project_root() -> Path` — resolves the repo root for all commands

Dependency injection note: `check_all` takes `runner` and `has_module` as parameters so tests never invoke real binaries. Every later task that shells out follows this same pattern.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_doctor.py`:

```python
import subprocess

from card_reviewer.knowledge import doctor


def fake_runner(found: set[str]):
    def run(cmd, **kwargs):
        if cmd[0] not in found:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, stdout="2026.01.01\n", stderr="")

    return run


def test_reports_missing_binary_with_install_hint():
    checks = doctor.check_all(
        runner=fake_runner({"ffmpeg"}), has_module=lambda n: True
    )
    by_name = {c.name: c for c in checks}
    assert by_name["yt-dlp"].found is False
    assert "brew install yt-dlp" in by_name["yt-dlp"].install_hint
    assert by_name["ffmpeg"].found is True
    assert by_name["ffmpeg"].version == "2026.01.01"


def test_reports_missing_python_module():
    checks = doctor.check_all(
        runner=fake_runner({"yt-dlp", "ffmpeg"}), has_module=lambda n: False
    )
    by_name = {c.name: c for c in checks}
    assert by_name["mlx-whisper"].found is False
    assert "uv sync" in by_name["mlx-whisper"].install_hint
```

Create `tests/test_cli.py`:

```python
from typer.testing import CliRunner

from card_reviewer.knowledge.cli import app

runner = CliRunner()


def test_help_lists_the_doctor_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    # Exit code 1 is correct when tools are genuinely missing on this machine.
    assert result.exit_code in (0, 1)
    assert "yt-dlp" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_doctor.py tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError` for `doctor` and `cli`

- [ ] **Step 3: Write `doctor.py`**

```python
"""Preflight checks for the external tools the pipeline shells out to."""

from __future__ import annotations

import importlib.util
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

EXTERNAL_TOOLS = [
    ("yt-dlp", ["--version"], "brew install yt-dlp"),
    ("ffmpeg", ["-version"], "brew install ffmpeg"),
]

PYTHON_MODULES = [("mlx-whisper", "mlx_whisper", "uv sync")]


@dataclass
class ToolCheck:
    name: str
    found: bool
    version: str | None
    install_hint: str


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_all(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    has_module: Callable[[str], bool] = _has_module,
) -> list[ToolCheck]:
    checks: list[ToolCheck] = []

    for name, args, hint in EXTERNAL_TOOLS:
        try:
            proc = runner([name, *args], capture_output=True, text=True, check=False)
        except (FileNotFoundError, OSError):
            checks.append(ToolCheck(name, False, None, hint))
            continue
        first_line = (proc.stdout or "").strip().splitlines()
        checks.append(
            ToolCheck(name, True, first_line[0] if first_line else None, hint)
        )

    for display, module, hint in PYTHON_MODULES:
        checks.append(ToolCheck(display, has_module(module), None, hint))

    return checks
```

- [ ] **Step 4: Write `cli.py`**

```python
"""Typer wiring. No business logic lives here."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import doctor
from .paths import ProjectPaths

app = typer.Typer(
    help="Turn grading training videos into a versioned knowledge base.",
    no_args_is_help=True,
)
console = Console()


def project_root() -> Path:
    """The repository root: three parents up from this file."""
    return Path(__file__).resolve().parents[3]


def paths() -> ProjectPaths:
    return ProjectPaths(project_root())


def doctor_cmd() -> None:
    """Check that yt-dlp, ffmpeg, and mlx-whisper are available."""
    checks = doctor.check_all()
    table = Table("tool", "status", "version", "install")
    for c in checks:
        table.add_row(
            c.name,
            "[green]ok[/green]" if c.found else "[red]missing[/red]",
            c.version or "",
            "" if c.found else c.install_hint,
        )
    console.print(table)
    if not all(c.found for c in checks):
        raise typer.Exit(code=1)


# Typer would derive "doctor-cmd" from the function name, so register the
# command explicitly. Every later command uses the @app.command(name="...")
# decorator form instead; this one is written out to show the equivalence.
app.command(name="doctor")(doctor_cmd)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_doctor.py tests/test_cli.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Verify the console script works**

Run: `uv run card-knowledge doctor`
Expected: a table listing yt-dlp, ffmpeg, mlx-whisper. On a fresh machine yt-dlp and ffmpeg show `missing` with install hints — that is correct output, not a failure.

- [ ] **Step 7: Commit**

```bash
git add src/card_reviewer/knowledge/doctor.py src/card_reviewer/knowledge/cli.py tests/test_doctor.py tests/test_cli.py
git commit -m "feat: CLI skeleton and external tool preflight"
```

---

## Task 5: Acquisition

**Files:**
- Create: `src/card_reviewer/knowledge/acquire.py`
- Modify: `src/card_reviewer/knowledge/cli.py` (add the `acquire` command)
- Test: `tests/test_acquire.py`

**Interfaces:**
- Consumes: `models.Manifest`, `models.SourceInfo`, `models.FileInfo`, `manifest.save/finish/fail`, `paths.ProjectPaths`
- Produces:
  - `acquire.derive_video_id(url=None, file=None) -> str`
  - `acquire.AcquisitionFailed(Exception)` with attribute `guidance: str`
  - `acquire.from_url(paths, url, rubric_version, browser=None, runner=subprocess.run) -> Manifest`
  - `acquire.from_file(paths, file, rubric_version, runner=subprocess.run) -> Manifest`
  - `acquire.MANUAL_FALLBACK: str` — the message shown when authenticated download fails

**Safety requirement for this task:** on failure there is exactly one code path — record, raise, stop. No retry against a different endpoint, no alternate extractor, no player workaround.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_acquire.py`:

```python
import json
import subprocess

import pytest

from card_reviewer.knowledge import acquire, manifest as mf
from card_reviewer.knowledge.models import StageStatus
from card_reviewer.knowledge.paths import ProjectPaths

METADATA = {
    "id": "abc123",
    "title": "Grading 101",
    "uploader": "Someone",
    "duration": 3120,
    "ext": "mp4",
}


@pytest.fixture
def paths(tmp_path):
    return ProjectPaths(tmp_path)


def recording_runner(calls, *, fail_on=None, stdout=""):
    def run(cmd, **kwargs):
        calls.append(cmd)
        if fail_on and fail_on in " ".join(cmd):
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="ERROR: sign in to confirm"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return run


def test_derive_video_id_from_youtube_watch_url():
    assert acquire.derive_video_id(url="https://www.youtube.com/watch?v=abc123") == "yt_abc123"


def test_derive_video_id_from_youtube_short_url():
    assert acquire.derive_video_id(url="https://youtu.be/abc123") == "yt_abc123"


def test_derive_video_id_from_skool_url_is_stable_hash():
    url = "https://www.skool.com/mlp/classroom/xyz"
    first = acquire.derive_video_id(url=url)
    assert first.startswith("skool_")
    assert first == acquire.derive_video_id(url=url)


def test_derive_video_id_from_local_file_hashes_content(tmp_path):
    f = tmp_path / "lesson.mp4"
    f.write_bytes(b"pretend video bytes")
    vid = acquire.derive_video_id(file=f)
    assert vid.startswith("local_")
    assert vid == acquire.derive_video_id(file=f)


def test_from_url_uses_cookies_from_browser_when_requested(paths):
    calls = []
    runner = recording_runner(calls, stdout=json.dumps(METADATA))
    with pytest.raises(acquire.AcquisitionFailed):
        # No real file lands on disk, so this fails at the verify step. We only
        # care that the cookie flag was passed correctly.
        acquire.from_url(
            paths,
            "https://www.skool.com/mlp/x",
            rubric_version="0.1.0",
            browser="chrome",
            runner=runner,
        )
    flat = [" ".join(c) for c in calls]
    assert any("--cookies-from-browser chrome" in f for f in flat)


def test_from_url_never_writes_a_cookie_file(paths):
    calls = []
    runner = recording_runner(calls, stdout=json.dumps(METADATA))
    with pytest.raises(acquire.AcquisitionFailed):
        acquire.from_url(
            paths, "https://www.skool.com/mlp/x", "0.1.0", browser="chrome", runner=runner
        )
    flat = " ".join(" ".join(c) for c in calls)
    assert "--cookies " not in flat
    assert "cookies.txt" not in flat


def test_failed_download_records_failure_and_gives_manual_path(paths):
    calls = []
    runner = recording_runner(calls, fail_on="--dump-json", stdout="")
    with pytest.raises(acquire.AcquisitionFailed) as exc:
        acquire.from_url(paths, "https://www.skool.com/mlp/x", "0.1.0", runner=runner)
    assert "sign in" in str(exc.value)
    assert "--file" in exc.value.guidance


def test_failed_download_does_not_retry(paths):
    """Spec §9: failure is terminal. Exactly one metadata attempt."""
    calls = []
    runner = recording_runner(calls, fail_on="--dump-json")
    with pytest.raises(acquire.AcquisitionFailed):
        acquire.from_url(paths, "https://www.skool.com/mlp/x", "0.1.0", runner=runner)
    dump_calls = [c for c in calls if "--dump-json" in c]
    assert len(dump_calls) == 1


def test_from_file_adopts_local_video(paths, tmp_path):
    src = tmp_path / "lesson.mp4"
    src.write_bytes(b"pretend video bytes")
    runner = recording_runner([], stdout="42.5")
    m = acquire.from_file(paths, src, rubric_version="0.1.0", runner=runner)
    assert m.source.type == "local"
    assert m.source.duration_s == 42.5
    assert m.file is not None
    assert m.stages["acquire"].status is StageStatus.DONE
    assert paths.manifest(m.video_id).exists()
    # Original is copied into the packet, not moved.
    assert src.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_acquire.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.acquire'`

- [ ] **Step 3: Write the implementation**

Create `src/card_reviewer/knowledge/acquire.py`:

```python
"""Stage 1: get the media onto disk and open a work packet.

Three sources, one stage. Authenticated failure is terminal by design: see
spec §9 and CARD_REVIEWER_BUILD_PLAN §4 and §30 rule 13. There is deliberately
no retry, no alternate extractor, and no player workaround in this module.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.parse
from collections.abc import Callable
from pathlib import Path

from . import manifest as mf
from .models import FileInfo, Manifest, SourceInfo
from .paths import ProjectPaths

Runner = Callable[..., subprocess.CompletedProcess]

MANUAL_FALLBACK = (
    "If this is protected course material you have access to, play the lesson "
    "in your browser, save the video yourself, then run:\n"
    "  card-knowledge acquire --file /path/to/lesson.mp4"
)


class AcquisitionFailed(Exception):
    def __init__(self, message: str, guidance: str = MANUAL_FALLBACK) -> None:
        super().__init__(message)
        self.guidance = guidance


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_video_id(url: str | None = None, file: Path | None = None) -> str:
    """A stable, filesystem-safe id so re-running never duplicates work."""
    if file is not None:
        return f"local_{_sha256_file(Path(file))[:12]}"
    if url is None:
        raise ValueError("derive_video_id requires either url or file")

    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if "youtube.com" in host:
        qs = urllib.parse.parse_qs(parsed.query)
        if "v" in qs:
            return f"yt_{qs['v'][0]}"
    if "youtu.be" in host:
        return f"yt_{parsed.path.lstrip('/')}"

    prefix = "skool" if "skool.com" in host else "web"
    return f"{prefix}_{hashlib.sha256(url.encode()).hexdigest()[:12]}"


def _probe_duration(path: Path, runner: Runner) -> float:
    proc = runner(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float((proc.stdout or "0").strip())
    except ValueError:
        return 0.0


def _cookie_args(browser: str | None) -> list[str]:
    return ["--cookies-from-browser", browser] if browser else []


def from_url(
    paths: ProjectPaths,
    url: str,
    rubric_version: str,
    browser: str | None = None,
    runner: Runner = subprocess.run,
) -> Manifest:
    video_id = derive_video_id(url=url)
    dest = paths.source_dir(video_id)
    dest.mkdir(parents=True, exist_ok=True)

    meta_proc = runner(
        ["yt-dlp", "--dump-json", "--no-warnings", *_cookie_args(browser), url],
        capture_output=True,
        text=True,
        check=False,
    )
    if meta_proc.returncode != 0:
        raise AcquisitionFailed(
            f"yt-dlp could not read {url}: {(meta_proc.stderr or '').strip()}"
        )

    meta = json.loads(meta_proc.stdout)
    source = SourceInfo(
        type="skool" if video_id.startswith("skool_") else "youtube",
        url=url,
        title=meta.get("title", video_id),
        uploader=meta.get("uploader"),
        duration_s=float(meta.get("duration") or 0),
    )

    m = Manifest(video_id=video_id, source=source, rubric_version_at_ingest=rubric_version)
    mf.save(paths, m)
    mf.start(paths, m, "acquire")

    out_template = str(dest / "video.%(ext)s")
    dl = runner(
        [
            "yt-dlp",
            "-f", "bv*+ba/b",
            *_cookie_args(browser),
            "-o", out_template,
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    downloaded = sorted(dest.glob("video.*"))
    if dl.returncode != 0 or not downloaded:
        error = (dl.stderr or "yt-dlp produced no file").strip()
        mf.fail(paths, m, "acquire", error)
        raise AcquisitionFailed(f"download failed for {url}: {error}")

    path = downloaded[0]
    m.file = FileInfo(
        path=str(path.relative_to(paths.packet(video_id))),
        sha256=_sha256_file(path),
        bytes=path.stat().st_size,
    )
    return mf.finish(paths, m, "acquire", tool="yt-dlp", browser=browser)


def from_file(
    paths: ProjectPaths,
    file: Path | str,
    rubric_version: str,
    runner: Runner = subprocess.run,
) -> Manifest:
    src = Path(file)
    if not src.exists():
        raise AcquisitionFailed(f"no such file: {src}", guidance="Check the path.")

    video_id = derive_video_id(file=src)
    dest_dir = paths.source_dir(video_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"video{src.suffix}"
    if not dest.exists():
        shutil.copy2(src, dest)

    source = SourceInfo(
        type="local",
        url=None,
        title=src.stem,
        uploader=None,
        duration_s=_probe_duration(dest, runner),
    )
    m = Manifest(
        video_id=video_id,
        source=source,
        file=FileInfo(
            path=str(dest.relative_to(paths.packet(video_id))),
            sha256=_sha256_file(dest),
            bytes=dest.stat().st_size,
        ),
        rubric_version_at_ingest=rubric_version,
    )
    mf.save(paths, m)
    return mf.finish(paths, m, "acquire", tool="local-copy", original=str(src))
```

- [ ] **Step 4: Add the CLI command**

Append to `src/card_reviewer/knowledge/cli.py`:

```python
@app.command(name="acquire")
def acquire_cmd(
    url: str | None = typer.Argument(None, help="Video URL"),
    file: Path | None = typer.Option(None, "--file", help="Local video file"),
    browser: str | None = typer.Option(
        None,
        "--browser",
        help="Read cookies from this browser: chrome, brave, edge, firefox, safari",
    ),
) -> None:
    """Download a video (or adopt a local file) and open its work packet."""
    from . import acquire as acq
    from . import version as ver

    p = paths()
    rubric_version = ver.read(p)
    try:
        m = acq.from_file(p, file, rubric_version) if file else acq.from_url(
            p, url, rubric_version, browser=browser
        )
    except acq.AcquisitionFailed as exc:
        console.print(f"[red]Acquisition failed:[/red] {exc}")
        console.print(exc.guidance)
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Packet ready:[/green] {m.video_id} — {m.source.title}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_acquire.py -v`
Expected: PASS (9 tests)

Note: the `acquire` CLI command imports `version`, which Task 12 creates. Until then, run the test file rather than the CLI command. If you are executing tasks strictly in order, that is expected — the module tests are what gate this task.

- [ ] **Step 6: Commit**

```bash
git add src/card_reviewer/knowledge/acquire.py src/card_reviewer/knowledge/cli.py tests/test_acquire.py
git commit -m "feat: video acquisition with terminal failure on protected sources"
```

---

## Task 6: Transcription

**Files:**
- Create: `src/card_reviewer/knowledge/transcribe.py`
- Modify: `src/card_reviewer/knowledge/cli.py` (add the `transcribe` command)
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Consumes: `models.Cue`, `models.Transcript`, `manifest.load/require_ready/finish`, `paths.ProjectPaths`
- Produces:
  - `transcribe.parse_vtt(text: str) -> list[Cue]`
  - `transcribe.fetch_captions(url, dest_dir, browser=None, runner=subprocess.run) -> list[Cue] | None`
  - `transcribe.whisper_transcribe(video_path, transcriber=None) -> tuple[list[Cue], str]` — returns cues and the model name
  - `transcribe.run(paths, video_id, browser=None, runner=subprocess.run, transcriber=None) -> Transcript`

Design note: captions are tried first and cost nothing. `whisper_transcribe` takes a `transcriber` callable so tests never load a model; production passes `None` and the real `mlx_whisper.transcribe` is imported lazily.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transcribe.py`:

```python
import pytest

from card_reviewer.knowledge import manifest as mf, transcribe
from card_reviewer.knowledge.models import Manifest, SourceInfo
from card_reviewer.knowledge.paths import ProjectPaths

VTT = """WEBVTT

00:00:01.000 --> 00:00:04.500
Look right here at this corner.

00:00:04.500 --> 00:00:08.000
You can see the whitening on the edge.
"""


def test_parse_vtt_extracts_cues_with_seconds():
    cues = transcribe.parse_vtt(VTT)
    assert len(cues) == 2
    assert cues[0].start_s == 1.0
    assert cues[0].end_s == 4.5
    assert cues[0].text == "Look right here at this corner."
    assert cues[1].start_s == 4.5


def test_parse_vtt_handles_hourless_timestamps():
    cues = transcribe.parse_vtt("WEBVTT\n\n01:02.000 --> 01:05.000\nHello.\n")
    assert cues[0].start_s == 62.0
    assert cues[0].end_s == 65.0


def test_parse_vtt_joins_multiline_cue_text():
    cues = transcribe.parse_vtt(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nfirst line\nsecond line\n"
    )
    assert cues[0].text == "first line second line"


def test_parse_vtt_ignores_cue_identifiers_and_notes():
    body = "WEBVTT\n\nNOTE something\n\ncue-7\n00:00:01.000 --> 00:00:02.000\nText.\n"
    cues = transcribe.parse_vtt(body)
    assert len(cues) == 1
    assert cues[0].text == "Text."


def test_parse_vtt_on_empty_input_returns_no_cues():
    assert transcribe.parse_vtt("WEBVTT\n") == []


@pytest.fixture
def packet(tmp_path):
    p = ProjectPaths(tmp_path)
    m = Manifest(
        video_id="yt_abc",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=100.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    mf.finish(p, m, "acquire")
    (p.source_dir("yt_abc")).mkdir(parents=True, exist_ok=True)
    (p.source_dir("yt_abc") / "video.mp4").write_bytes(b"x")
    m.file = None
    return p, m


def test_run_prefers_captions_when_available(packet, monkeypatch):
    p, _ = packet
    monkeypatch.setattr(
        transcribe, "fetch_captions", lambda *a, **k: transcribe.parse_vtt(VTT)
    )

    def explode(*a, **k):
        raise AssertionError("whisper must not run when captions exist")

    t = transcribe.run(p, "yt_abc", transcriber=explode)
    assert t.method == "captions"
    assert len(t.cues) == 2
    assert p.transcript("yt_abc").exists()


def test_run_falls_back_to_whisper_when_no_captions(packet, monkeypatch):
    p, _ = packet
    monkeypatch.setattr(transcribe, "fetch_captions", lambda *a, **k: None)

    def fake_transcriber(path, **kwargs):
        return {
            "language": "en",
            "segments": [{"start": 0.0, "end": 2.0, "text": " Whisper output."}],
        }

    t = transcribe.run(p, "yt_abc", transcriber=fake_transcriber)
    assert t.method == "mlx-whisper"
    assert t.cues[0].text == "Whisper output."


def test_run_blocks_when_acquire_not_done(tmp_path):
    p = ProjectPaths(tmp_path)
    m = Manifest(
        video_id="yt_zzz",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=1.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    with pytest.raises(mf.StageNotReady):
        transcribe.run(p, "yt_zzz")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.transcribe'`

- [ ] **Step 3: Write the implementation**

Create `src/card_reviewer/knowledge/transcribe.py`:

```python
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


def _to_seconds(stamp: str) -> float:
    """Accept HH:MM:SS.mmm or MM:SS.mmm."""
    parts = stamp.strip().split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def parse_vtt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    block_lines: list[str] = []
    timing: tuple[float, float] | None = None

    def flush() -> None:
        nonlocal timing, block_lines
        if timing and block_lines:
            cues.append(
                Cue(start_s=timing[0], end_s=timing[1], text=" ".join(block_lines))
            )
        timing, block_lines = None, []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
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
```

- [ ] **Step 4: Add the CLI command**

Append to `src/card_reviewer/knowledge/cli.py`:

```python
@app.command(name="transcribe")
def transcribe_cmd(
    video_id: str,
    browser: str | None = typer.Option(None, "--browser"),
) -> None:
    """Produce a timestamped transcript for a work packet."""
    from . import transcribe as tr

    t = tr.run(paths(), video_id, browser=browser)
    console.print(
        f"[green]Transcript:[/green] {len(t.cues)} cues via {t.method}"
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add src/card_reviewer/knowledge/transcribe.py src/card_reviewer/knowledge/cli.py tests/test_transcribe.py
git commit -m "feat: transcription via captions with local whisper fallback"
```

---

## Task 7: Segmentation lexicon

**Files:**
- Create: `knowledge/segmentation_lexicon.yaml`
- Create: `src/card_reviewer/knowledge/lexicon.py`
- Test: `tests/test_lexicon.py`

**Interfaces:**
- Consumes: `paths.ProjectPaths`
- Produces:
  - `lexicon.CueScore(score, categories, matched_terms, visual_cue)` — a dataclass
  - `lexicon.Lexicon(version, categories, demonstration_weight)` — a dataclass with method `score(text: str) -> CueScore`
  - `lexicon.load(path: Path) -> Lexicon`
  - `lexicon.DEMONSTRATION = "demonstration"` — the category name whose hits set `visual_cue`

Scoring rules, fixed here so later tasks can rely on them:
- Matching is case-insensitive and phrase-based (a multi-word term matches as a contiguous phrase).
- A term is counted **once per cue** no matter how many times it appears — one mention and five mentions both indicate the same thing: this cue is about corners.
- `score` is the sum of matched term weights.
- `visual_cue` is `True` when any `demonstration` term matched.
- `categories` excludes `demonstration`; it names the grading topics only.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lexicon.py`:

```python
import pytest

from card_reviewer.knowledge import lexicon


@pytest.fixture
def lex(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text(
        """
version: "1"
demonstration_weight: 2.0
categories:
  corners:
    corner: 1.0
    soft corner: 2.0
  surface:
    print line: 3.0
  demonstration:
    look right here: 1.0
    you can see: 1.0
"""
    )
    return lexicon.load(path)


def test_scores_sum_matched_term_weights(lex):
    result = lex.score("There is a soft corner and a corner ding here.")
    # "soft corner" (2.0) + "corner" (1.0) = 3.0
    assert result.score == pytest.approx(3.0)
    assert set(result.matched_terms) == {"soft corner", "corner"}


def test_matching_is_case_insensitive(lex):
    assert lex.score("PRINT LINE across the front").score == pytest.approx(3.0)


def test_a_term_counts_once_per_cue(lex):
    once = lex.score("print line")
    thrice = lex.score("print line print line print line")
    assert once.score == thrice.score


def test_categories_exclude_demonstration(lex):
    result = lex.score("Look right here at the corner.")
    assert result.categories == ["corners"]
    assert result.visual_cue is True


def test_visual_cue_false_without_demonstration_terms(lex):
    result = lex.score("Centering matters a lot on this one corner.")
    assert result.visual_cue is False


def test_unrelated_text_scores_zero(lex):
    result = lex.score("Welcome back to the channel, smash that like button.")
    assert result.score == 0.0
    assert result.categories == []


def test_real_lexicon_file_loads_and_has_every_category():
    """The shipped lexicon must cover every category the spec names."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    lex = lexicon.load(repo / "knowledge" / "segmentation_lexicon.yaml")
    for expected in ("centering", "corners", "edges", "surface", "outcomes", "demonstration"):
        assert expected in lex.categories, f"lexicon missing category: {expected}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lexicon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.lexicon'`

- [ ] **Step 3: Create the lexicon data file**

Create `knowledge/segmentation_lexicon.yaml`:

```yaml
# Segmentation lexicon — scores transcript cues for grading relevance.
# This is DATA, not code. Tune it freely; tests/fixtures/expected_segments.json
# is the regression guard that shows what your tuning changed.
version: "1"
demonstration_weight: 2.0

categories:
  centering:
    centering: 2.0
    center: 1.0
    borders: 1.5
    border: 1.0
    off-center: 3.0
    off center: 3.0
    left to right: 1.5
    top to bottom: 1.5
    "60/40": 3.0
    "55/45": 3.0
    diamond cut: 3.0

  corners:
    corner: 1.5
    corners: 1.5
    soft corner: 3.0
    ding: 2.5
    dinged: 2.5
    fraying: 2.5
    frayed: 2.5
    rounded: 2.0
    corner wear: 3.0

  edges:
    edge: 1.5
    edges: 1.5
    chipping: 3.0
    chipped: 3.0
    rough cut: 2.5
    edge wear: 3.0
    whitening: 3.0
    silvering: 2.5

  surface:
    surface: 1.5
    print line: 3.5
    print lines: 3.5
    scratch: 2.5
    scratches: 2.5
    roller mark: 3.0
    dimple: 2.5
    orange peel: 3.0
    gloss: 1.5
    fisheye: 3.0
    indentation: 2.5
    crease: 3.0

  print:
    print defect: 3.0
    print dot: 2.5
    registration: 2.0
    miscut: 3.0

  handling:
    top loader: 1.5
    toploader: 1.5
    one touch: 1.5
    penny sleeve: 1.5
    handling: 1.5

  outcomes:
    psa 10: 3.0
    psa 9: 3.0
    psa 8: 2.5
    gem mint: 3.0
    gem: 2.0
    came back: 2.5
    bumped: 2.5
    qualifier: 2.5
    submission: 1.5

  demonstration:
    look right here: 2.0
    look at this: 2.0
    you can see: 2.0
    right here: 1.5
    under the light: 2.5
    angle the light: 2.5
    tilt it: 2.0
    zoom in: 2.0
    flip it over: 2.0
    take a look: 1.5
    notice how: 2.0
```

- [ ] **Step 4: Write `lexicon.py`**

```python
"""Load the segmentation lexicon and score a single transcript cue.

Scoring is deliberately simple and inspectable: phrase matching with weights.
A cue's score answers one question — how likely is it that the instructor is
inspecting a card right now?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEMONSTRATION = "demonstration"


@dataclass
class CueScore:
    score: float = 0.0
    categories: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    visual_cue: bool = False


@dataclass
class Lexicon:
    version: str
    categories: dict[str, dict[str, float]]
    demonstration_weight: float = 1.0

    def score(self, text: str) -> CueScore:
        haystack = text.lower()
        total = 0.0
        matched: list[str] = []
        hit_categories: list[str] = []
        visual = False

        for category, terms in self.categories.items():
            category_hit = False
            for term, weight in terms.items():
                if term.lower() in haystack:
                    # Counted once per cue: repetition does not add information.
                    total += float(weight)
                    matched.append(term)
                    category_hit = True
            if not category_hit:
                continue
            if category == DEMONSTRATION:
                visual = True
            else:
                hit_categories.append(category)

        return CueScore(
            score=total,
            categories=hit_categories,
            matched_terms=matched,
            visual_cue=visual,
        )


def load(path: Path | str) -> Lexicon:
    data = yaml.safe_load(Path(path).read_text())
    return Lexicon(
        version=str(data.get("version", "0")),
        categories=data.get("categories", {}),
        demonstration_weight=float(data.get("demonstration_weight", 1.0)),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_lexicon.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add knowledge/segmentation_lexicon.yaml src/card_reviewer/knowledge/lexicon.py tests/test_lexicon.py
git commit -m "feat: grading lexicon and cue scoring"
```

---

## Task 8: Segment building

**Files:**
- Create: `src/card_reviewer/knowledge/segment.py`
- Create: `tests/fixtures/transcript_sample.json`
- Create: `tests/fixtures/expected_segments.json`
- Modify: `src/card_reviewer/knowledge/cli.py` (add the `segment` command)
- Test: `tests/test_segment.py`

**Interfaces:**
- Consumes: `models.Cue`, `models.Segment`, `models.Transcript`, `lexicon.Lexicon`, `manifest`, `paths.ProjectPaths`
- Produces:
  - `segment.build(cues, lex, min_score=2.0, pad_s=5.0, max_len_s=90.0) -> list[Segment]`
  - `segment.run(paths, video_id, lex=None) -> list[Segment]`
  - Constants `MIN_SCORE = 2.0`, `PAD_S = 5.0`, `MAX_LEN_S = 90.0`

Merging rules, fixed here:
- A cue is "hot" when its score is `>= min_score`.
- Consecutive hot cues merge into one window. A single non-hot cue between two hot cues does **not** break the run — instructors pause mid-demonstration.
- Two non-hot cues in a row end the window.
- The window is padded by `pad_s` on each side, clamped at zero.
- A window longer than `max_len_s` is split into equal parts each no longer than `max_len_s`.
- Segments are returned sorted by score descending, and ids are assigned `seg_001`, `seg_002`, ... **in time order** before sorting, so an id always refers to the same moment.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_segment.py`:

```python
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


def test_segments_are_sorted_by_score_descending(lex):
    cues = [
        cue(0, 5, "There is a corner."),
        cue(30, 35, "Look right here, you can see the print line and the whitening."),
    ]
    segments = segment.build(cues, lex, pad_s=0.0)
    assert segments[0].score >= segments[1].score


def test_ids_are_assigned_in_time_order(lex):
    cues = [
        cue(0, 5, "There is a corner."),
        cue(30, 35, "Look right here, you can see the print line and the whitening."),
    ]
    segments = segment.build(cues, lex, pad_s=0.0)
    by_id = {s.id: s for s in segments}
    assert by_id["seg_001"].start_s < by_id["seg_002"].start_s


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
```

- [ ] **Step 2: Create the transcript fixture**

Create `tests/fixtures/transcript_sample.json`:

```json
{
  "method": "captions",
  "model": null,
  "language": "en",
  "cues": [
    {"start_s": 0.0,   "end_s": 6.0,   "text": "What is going on everybody, welcome back to the channel."},
    {"start_s": 6.0,   "end_s": 12.0,  "text": "Today we are going through a stack of chrome."},
    {"start_s": 12.0,  "end_s": 18.0,  "text": "Before we start, hit subscribe."},
    {"start_s": 120.0, "end_s": 126.0, "text": "Okay so look right here at this corner."},
    {"start_s": 126.0, "end_s": 132.0, "text": "You can see the whitening along that edge, that is corner wear."},
    {"start_s": 132.0, "end_s": 138.0, "text": "That is never coming back a PSA 10."},
    {"start_s": 138.0, "end_s": 144.0, "text": "Alright."},
    {"start_s": 144.0, "end_s": 150.0, "text": "Moving on."},
    {"start_s": 300.0, "end_s": 306.0, "text": "Now tilt it under the light and you can see the print line."},
    {"start_s": 306.0, "end_s": 312.0, "text": "That print line runs the whole surface, that is a gem killer."},
    {"start_s": 312.0, "end_s": 318.0, "text": "I have had these come back a PSA 9 every single time."},
    {"start_s": 400.0, "end_s": 406.0, "text": "Centering on this one is about 60/40 left to right."},
    {"start_s": 406.0, "end_s": 412.0, "text": "PSA will still gem that, borders are within tolerance."},
    {"start_s": 500.0, "end_s": 506.0, "text": "Anyway that is the video, thanks for watching."}
  ]
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_segment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.segment'`

- [ ] **Step 4: Write `segment.py`**

```python
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
) -> list[Segment]:
    scored = [(c, lex.score(c.text)) for c in cues]

    runs: list[list[tuple[Cue, lex_mod.CueScore]]] = []
    current: list[tuple[Cue, lex_mod.CueScore]] = []
    cold_streak = 0

    for cue, score in scored:
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
```

- [ ] **Step 5: Generate the golden fixture and inspect it**

The expected fixture must be generated once and then read by a human before being committed. Run:

```bash
uv run python -c "
import json, pathlib
from card_reviewer.knowledge import lexicon, segment
from card_reviewer.knowledge.models import Cue
repo = pathlib.Path('.')
lex = lexicon.load(repo / 'knowledge' / 'segmentation_lexicon.yaml')
cues = [Cue(**c) for c in json.loads((repo / 'tests/fixtures/transcript_sample.json').read_text())['cues']]
out = [s.model_dump() for s in segment.build(cues, lex)]
(repo / 'tests/fixtures/expected_segments.json').write_text(json.dumps(out, indent=2) + chr(10))
print(json.dumps(out, indent=2))
"
```

**Read the printed output before continuing.** It must show roughly three segments: the corner-wear demonstration near 120s, the print-line demonstration near 300s, and the centering discussion near 400s. The intro and outro must be absent. If the intro appears, the lexicon is too loose — fix the lexicon, not the fixture.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_segment.py -v`
Expected: PASS (10 tests)

- [ ] **Step 7: Add the CLI command**

Append to `src/card_reviewer/knowledge/cli.py`:

```python
@app.command(name="segment")
def segment_cmd(video_id: str) -> None:
    """Rank the transcript into candidate windows worth inspecting."""
    from . import segment as seg

    segments = seg.run(paths(), video_id)
    console.print(f"[green]{len(segments)} segments[/green] written")
    for s in segments[:12]:
        console.print(
            f"  {s.id}  {s.start_s:8.1f}s-{s.end_s:8.1f}s  "
            f"score {s.score:6.1f}  {','.join(s.categories) or '-'}"
        )
```

- [ ] **Step 8: Commit**

```bash
git add src/card_reviewer/knowledge/segment.py src/card_reviewer/knowledge/cli.py tests/test_segment.py tests/fixtures/
git commit -m "feat: transcript segmentation with golden regression fixture"
```

---

## Task 9: Frame extraction

**Files:**
- Create: `src/card_reviewer/knowledge/frames.py`
- Modify: `src/card_reviewer/knowledge/cli.py` (add the `extract-frames` command)
- Test: `tests/test_frames.py`

**Interfaces:**
- Consumes: `models.Segment`, `manifest`, `paths.ProjectPaths`
- Produces:
  - `frames.sample(video, out_dir, start_s, end_s, fps=1.0, cap=20, runner=subprocess.run) -> list[Path]`
  - `frames.dedupe(image_paths, threshold=5, hasher=None) -> list[Path]` — deletes near-duplicates, returns survivors
  - `frames.run(paths, video_id, top_n=12, uniform=False, at=None, window_s=30.0, fps=1.0, runner=subprocess.run) -> int`
  - `frames.TOP_N = 12`, `frames.FPS = 1.0`, `frames.CAP_PER_SEGMENT = 20`

`dedupe` takes a `hasher` so tests can supply deterministic fake hashes instead of real images. In production it uses `imagehash.phash` on a `PIL.Image`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_frames.py`:

```python
import subprocess

import pytest

from card_reviewer.knowledge import frames, manifest as mf
from card_reviewer.knowledge.models import Manifest, SourceInfo
from card_reviewer.knowledge.paths import ProjectPaths


def fake_ffmpeg(produced_names):
    """Simulate ffmpeg by writing the files it would have produced."""

    def run(cmd, **kwargs):
        out_pattern = cmd[-1]
        out_dir = __import__("pathlib").Path(out_pattern).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in produced_names:
            (out_dir / name).write_bytes(b"fake-jpeg")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return run


def test_sample_invokes_ffmpeg_with_the_right_window(tmp_path):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        (tmp_path / "out").mkdir(exist_ok=True)
        (tmp_path / "out" / "frame_0001.jpg").write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    out = frames.sample(
        tmp_path / "video.mp4", tmp_path / "out", 120.0, 150.0, fps=1.0, runner=run
    )
    flat = " ".join(calls[0])
    assert "-ss 120.0" in flat
    assert "-t 30.0" in flat
    assert "fps=1.0" in flat
    assert len(out) == 1


def test_sample_caps_frame_count(tmp_path):
    run = fake_ffmpeg([f"frame_{i:04d}.jpg" for i in range(1, 51)])
    out = frames.sample(
        tmp_path / "v.mp4", tmp_path / "out", 0.0, 100.0, cap=20, runner=run
    )
    assert len(out) == 20
    # Frames beyond the cap are deleted, not merely hidden.
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 20


def test_dedupe_removes_near_identical_frames(tmp_path):
    paths_ = []
    for i in range(4):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"x")
        paths_.append(p)

    # f0, f1, f2 are near-identical; f3 differs.
    fake_hashes = {paths_[0]: 0, paths_[1]: 1, paths_[2]: 2, paths_[3]: 999}
    survivors = frames.dedupe(
        paths_, threshold=5, hasher=lambda p: fake_hashes[p]
    )
    assert len(survivors) == 2
    assert paths_[0] in survivors
    assert paths_[3] in survivors
    assert not paths_[1].exists()


def test_dedupe_keeps_everything_when_all_differ(tmp_path):
    paths_ = []
    for i in range(3):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"x")
        paths_.append(p)
    survivors = frames.dedupe(paths_, threshold=5, hasher=lambda p: hash(p.name) % 10000)
    assert len(survivors) == 3


def test_run_blocks_before_segment_stage(tmp_path):
    p = ProjectPaths(tmp_path)
    m = Manifest(
        video_id="yt_abc",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=100.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    with pytest.raises(mf.StageNotReady):
        frames.run(p, "yt_abc")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_frames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.frames'`

- [ ] **Step 3: Write the implementation**

Create `src/card_reviewer/knowledge/frames.py`:

```python
"""Stage 4: pull frames for the top-ranked segments.

Deduplication matters more than it sounds: a static talking head at 1 fps
produces twenty copies of one image and buries the frames that show a card.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from . import manifest as mf
from .models import Segment
from .paths import ProjectPaths

Runner = Callable[..., subprocess.CompletedProcess]

TOP_N = 12
FPS = 1.0
CAP_PER_SEGMENT = 20
PHASH_THRESHOLD = 5


def _default_hasher(path: Path):  # pragma: no cover - needs a real image
    import imagehash
    from PIL import Image

    return imagehash.phash(Image.open(path))


def sample(
    video: Path,
    out_dir: Path,
    start_s: float,
    end_s: float,
    fps: float = FPS,
    cap: int = CAP_PER_SEGMENT,
    runner: Runner = subprocess.run,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_s - start_s)
    proc = runner(
        [
            "ffmpeg", "-v", "error", "-y",
            "-ss", str(start_s),
            "-t", str(duration),
            "-i", str(video),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(out_dir / "frame_%04d.jpg"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {(proc.stderr or '').strip()}")

    produced = sorted(out_dir.glob("frame_*.jpg"))
    for extra in produced[cap:]:
        extra.unlink()
    return produced[:cap]


def dedupe(
    image_paths: list[Path],
    threshold: int = PHASH_THRESHOLD,
    hasher: Callable[[Path], object] | None = None,
) -> list[Path]:
    """Delete frames within `threshold` perceptual distance of a kept frame."""
    hasher = hasher or _default_hasher
    survivors: list[Path] = []
    kept_hashes: list[object] = []

    for path in image_paths:
        digest = hasher(path)
        if any(abs(digest - kept) <= threshold for kept in kept_hashes):
            path.unlink(missing_ok=True)
            continue
        survivors.append(path)
        kept_hashes.append(digest)

    return survivors


def run(
    paths: ProjectPaths,
    video_id: str,
    top_n: int = TOP_N,
    uniform: bool = False,
    at: float | None = None,
    window_s: float = 30.0,
    fps: float = FPS,
    runner: Runner = subprocess.run,
) -> int:
    m = mf.load(paths, video_id)
    mf.require_ready(m, "extract_frames")

    videos = sorted(paths.source_dir(video_id).glob("video.*"))
    if not videos:
        raise FileNotFoundError(f"no media in {paths.source_dir(video_id)}")
    video = videos[0]

    if at is not None:
        targets = [Segment(id="seg_adhoc", start_s=at, end_s=at + window_s, score=0.0)]
    elif uniform:
        step = max(30.0, m.source.duration_s / max(top_n, 1))
        targets = [
            Segment(id=f"seg_u{i:03d}", start_s=i * step, end_s=i * step + window_s, score=0.0)
            for i in range(top_n)
        ]
    else:
        data = json.loads(paths.segments(video_id).read_text())
        targets = [Segment(**s) for s in data["segments"]][:top_n]

    total = 0
    for seg in targets:
        out_dir = paths.frames(video_id) / seg.id
        produced = sample(video, out_dir, seg.start_s, seg.end_s, fps=fps, runner=runner)
        total += len(dedupe(produced))

    if at is None:
        mf.finish(paths, m, "extract_frames", n_frames=total, uniform=uniform, fps=fps)
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_frames.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Add the CLI command**

Append to `src/card_reviewer/knowledge/cli.py`:

```python
@app.command(name="extract-frames")
def extract_frames_cmd(
    video_id: str,
    top_n: int = typer.Option(12, "--top-n"),
    uniform: bool = typer.Option(False, "--uniform", help="Ignore ranking; sample the whole video"),
    at: float | None = typer.Option(None, "--at", help="Ad-hoc window start, in seconds"),
    window: float = typer.Option(30.0, "--window", help="Ad-hoc window length, in seconds"),
) -> None:
    """Pull frames for the top-ranked segments, or for an ad-hoc window."""
    from . import frames as fr

    count = fr.run(paths(), video_id, top_n=top_n, uniform=uniform, at=at, window_s=window)
    console.print(f"[green]{count} frames[/green] kept after deduplication")
```

- [ ] **Step 6: Verify against a real video**

Generate a tiny clip and run the real ffmpeg path once:

```bash
ffmpeg -v error -y -f lavfi -i testsrc=size=320x240:rate=10 -t 5 /tmp/testclip.mp4
uv run python -c "
from pathlib import Path
from card_reviewer.knowledge import frames
out = frames.sample(Path('/tmp/testclip.mp4'), Path('/tmp/testframes'), 0.0, 5.0, fps=1.0)
print(f'{len(out)} frames'); print(f'{len(frames.dedupe(out))} after dedupe')
"
```
Expected: about 5 frames sampled. Because `testsrc` animates, dedupe should keep more than one. If ffmpeg is not installed, this step is blocked — run `card-knowledge doctor` and install it.

- [ ] **Step 7: Commit**

```bash
git add src/card_reviewer/knowledge/frames.py src/card_reviewer/knowledge/cli.py tests/test_frames.py
git commit -m "feat: frame extraction with perceptual dedupe and escape hatches"
```

---

## Task 10: Rule validation

**Files:**
- Create: `src/card_reviewer/knowledge/validate.py`
- Modify: `src/card_reviewer/knowledge/cli.py` (add the `validate` command)
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `models.Rule`, `models.Manifest`, `paths.ProjectPaths`
- Produces:
  - `validate.parse_timestamp(text: str) -> tuple[float, float]` — accepts `"12:04-12:38"`, `"12:04"`, `"1:02:03-1:02:30"`
  - `validate.load_pending(paths) -> list[tuple[Path, Rule]]`
  - `validate.load_active(paths) -> list[Rule]`
  - `validate.video_durations(paths) -> dict[str, float]`
  - `validate.check_rule(rule, active_ids, durations, paths) -> list[str]` — returns error strings
  - `validate.run(paths) -> ValidationReport`
  - `validate.ValidationReport(ok, errors, checked)` — a dataclass; `errors` maps rule id to error list
  - `validate.BadTimestamp(Exception)`

The timestamp-bounds check is the load-bearing one: it catches a fabricated citation mechanically.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validate.py`:

```python
import datetime

import pytest
import yaml

from card_reviewer.knowledge import manifest as mf, validate
from card_reviewer.knowledge.models import Manifest, SourceInfo
from card_reviewer.knowledge.paths import ProjectPaths


def rule_dict(**over):
    base = {
        "id": "SURFACE_PRINT_LINE_001",
        "category": "surface",
        "statement": "Vertical print lines commonly prevent a PSA 10.",
        "evidence_type": "experience_based",
        "confidence": "high",
        "applies_to": {"card_types": ["chrome"], "sets": []},
        "sources": [
            {
                "lesson": "lesson_001",
                "video_id": "yt_abc",
                "timestamps": ["05:00-05:30"],
                "quote": "look at that line",
            }
        ],
        "status": "pending",
        "supersedes": None,
        "created": datetime.date(2026, 8, 28),
        "rubric_version_added": None,
    }
    return base | over


@pytest.fixture
def project(tmp_path):
    p = ProjectPaths(tmp_path)
    p.pending_rules.mkdir(parents=True)
    p.rules.mkdir(parents=True)
    p.lessons.mkdir(parents=True)
    p.lesson("lesson_001").write_text("# Lesson 1\n")
    m = Manifest(
        video_id="yt_abc",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=600.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    return p


def write_pending(p, data, name=None):
    path = p.pending_rules / f"{name or data['id']}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_parse_timestamp_range():
    assert validate.parse_timestamp("12:04-12:38") == (724.0, 758.0)


def test_parse_timestamp_single_point():
    assert validate.parse_timestamp("12:04") == (724.0, 724.0)


def test_parse_timestamp_with_hours():
    assert validate.parse_timestamp("1:02:03-1:02:30") == (3723.0, 3750.0)


def test_parse_timestamp_rejects_garbage():
    with pytest.raises(validate.BadTimestamp):
        validate.parse_timestamp("sometime near the end")


def test_valid_rule_passes(project):
    write_pending(project, rule_dict())
    report = validate.run(project)
    assert report.ok
    assert report.errors == {}
    assert report.checked == 1


def test_timestamp_beyond_video_duration_is_rejected(project):
    """The load-bearing check: a citation past the end of the video is fabricated."""
    data = rule_dict()
    data["sources"][0]["timestamps"] = ["59:00-59:30"]  # video is 600s
    write_pending(project, data)
    report = validate.run(project)
    assert not report.ok
    assert any("exceeds" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_missing_lesson_is_rejected(project):
    data = rule_dict()
    data["sources"][0]["lesson"] = "lesson_999"
    write_pending(project, data)
    report = validate.run(project)
    assert any("lesson_999" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_unknown_video_id_is_rejected(project):
    data = rule_dict()
    data["sources"][0]["video_id"] = "yt_nope"
    write_pending(project, data)
    report = validate.run(project)
    assert any("yt_nope" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_id_colliding_with_an_active_rule_is_rejected(project):
    active = rule_dict(status="active", rubric_version_added="0.1.0")
    (project.rules / "surface").mkdir(parents=True)
    (project.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").write_text(
        yaml.safe_dump(active, sort_keys=False)
    )
    write_pending(project, rule_dict())
    report = validate.run(project)
    assert any("already active" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_pending_rule_marked_active_is_rejected(project):
    write_pending(project, rule_dict(status="active"))
    report = validate.run(project)
    assert any("status" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_malformed_yaml_is_reported_not_raised(project):
    (project.pending_rules / "broken.yaml").write_text("id: [unclosed\n")
    report = validate.run(project)
    assert not report.ok
    assert "broken.yaml" in report.errors


def test_missing_evidence_type_is_reported(project):
    data = rule_dict()
    del data["evidence_type"]
    write_pending(project, data)
    report = validate.run(project)
    assert not report.ok
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.validate'`

- [ ] **Step 3: Write the implementation**

Create `src/card_reviewer/knowledge/validate.py`:

```python
"""Stage 6: mechanically check what Claude wrote before a human ever sees it.

The timestamp-bounds check is the reason rules are structured rather than
prose: a citation pointing past the end of the video is caught by arithmetic,
which prompting alone does not reliably achieve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import Manifest, Rule, RuleStatus
from .paths import ProjectPaths

TIMESTAMP_RE = re.compile(r"^\d{1,2}(:\d{2}){1,2}$")


class BadTimestamp(Exception):
    """A timestamp string that is not HH:MM:SS or MM:SS."""


@dataclass
class ValidationReport:
    ok: bool = True
    errors: dict[str, list[str]] = field(default_factory=dict)
    checked: int = 0


def _to_seconds(stamp: str) -> float:
    if not TIMESTAMP_RE.match(stamp.strip()):
        raise BadTimestamp(f"not a timestamp: {stamp!r} (expected MM:SS or HH:MM:SS)")
    seconds = 0.0
    for part in stamp.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def parse_timestamp(text: str) -> tuple[float, float]:
    """'12:04-12:38' -> (724.0, 758.0); '12:04' -> (724.0, 724.0)."""
    parts = [p for p in text.split("-") if p.strip()]
    if len(parts) == 1:
        point = _to_seconds(parts[0])
        return point, point
    if len(parts) == 2:
        return _to_seconds(parts[0]), _to_seconds(parts[1])
    raise BadTimestamp(f"cannot parse timestamp range: {text!r}")


def load_pending(paths: ProjectPaths) -> list[tuple[Path, Rule]]:
    out: list[tuple[Path, Rule]] = []
    if not paths.pending_rules.exists():
        return out
    for path in sorted(paths.pending_rules.glob("*.yaml")):
        out.append((path, Rule.model_validate(yaml.safe_load(path.read_text()))))
    return out


def load_active(paths: ProjectPaths) -> list[Rule]:
    out: list[Rule] = []
    if not paths.rules.exists():
        return out
    for path in sorted(paths.rules.rglob("*.yaml")):
        rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        if rule.status is RuleStatus.ACTIVE:
            out.append(rule)
    return out


def video_durations(paths: ProjectPaths) -> dict[str, float]:
    durations: dict[str, float] = {}
    if not paths.work.exists():
        return durations
    for manifest_path in paths.work.glob("*/manifest.json"):
        m = Manifest.model_validate_json(manifest_path.read_text())
        durations[m.video_id] = m.source.duration_s
    return durations


def check_rule(
    rule: Rule,
    active_ids: set[str],
    durations: dict[str, float],
    paths: ProjectPaths,
) -> list[str]:
    errors: list[str] = []

    if rule.status is not RuleStatus.PENDING:
        errors.append(
            f"status is {rule.status.value!r}; rules in pending_rules/ must be 'pending'"
        )

    if rule.id in active_ids:
        errors.append(f"id {rule.id} is already active; choose a new id")

    for source in rule.sources:
        if not paths.lesson(source.lesson).exists():
            errors.append(f"cited lesson {source.lesson} does not exist")

        if source.video_id not in durations:
            errors.append(f"cited video_id {source.video_id} has no work packet")
            continue

        duration = durations[source.video_id]
        for stamp in source.timestamps:
            try:
                start, end = parse_timestamp(stamp)
            except BadTimestamp as exc:
                errors.append(str(exc))
                continue
            if end > duration:
                errors.append(
                    f"timestamp {stamp} exceeds the {duration:.0f}s duration of "
                    f"{source.video_id} — the citation cannot be real"
                )
            if start > end:
                errors.append(f"timestamp {stamp} starts after it ends")

    return errors


def run(paths: ProjectPaths) -> ValidationReport:
    report = ValidationReport()
    durations = video_durations(paths)

    try:
        active_ids = {r.id for r in load_active(paths)}
    except (ValidationError, yaml.YAMLError) as exc:
        report.ok = False
        report.errors["knowledge/rules"] = [f"active rules are unreadable: {exc}"]
        return report

    if not paths.pending_rules.exists():
        return report

    for path in sorted(paths.pending_rules.glob("*.yaml")):
        report.checked += 1
        try:
            rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        except (ValidationError, yaml.YAMLError) as exc:
            report.ok = False
            report.errors[path.name] = [f"does not parse as a Rule: {exc}"]
            continue

        errors = check_rule(rule, active_ids, durations, paths)
        if errors:
            report.ok = False
            report.errors[rule.id] = errors

    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validate.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Add the CLI command**

Append to `src/card_reviewer/knowledge/cli.py`:

```python
@app.command(name="validate")
def validate_cmd() -> None:
    """Check every pending rule for schema, citation, and status errors."""
    from . import validate as val

    report = val.run(paths())
    if report.ok:
        console.print(f"[green]{report.checked} pending rules valid[/green]")
        return
    for rule_id, errors in report.errors.items():
        console.print(f"[red]{rule_id}[/red]")
        for error in errors:
            console.print(f"  - {error}")
    raise typer.Exit(code=1)
```

- [ ] **Step 6: Commit**

```bash
git add src/card_reviewer/knowledge/validate.py src/card_reviewer/knowledge/cli.py tests/test_validate.py
git commit -m "feat: rule validation with mechanical citation bounds checking"
```

---

## Task 11: Duplicate and contradiction detection

**Files:**
- Create: `src/card_reviewer/knowledge/dedup.py`
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `models.Rule`
- Produces:
  - `dedup.Flag(kind, other_id, score, other_statement)` — a dataclass; `kind` is `"duplicate"` or `"contradiction"`
  - `dedup.normalize(text: str) -> str`
  - `dedup.similarity(a: str, b: str) -> float`
  - `dedup.is_negated(text: str) -> bool`
  - `dedup.flags_for(rule, active_rules, threshold=0.72) -> list[Flag]`
  - `dedup.THRESHOLD = 0.72`

Rule for classification: two rules in the same category whose normalized statements are at least `threshold` similar are related. If their negation parity **differs**, that is a `contradiction`; otherwise a `duplicate`. Flags are advisory — this module never changes a rule.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dedup.py`:

```python
import datetime

from card_reviewer.knowledge import dedup
from card_reviewer.knowledge.models import Rule, RuleSource


def make(rule_id, statement, category="surface"):
    return Rule(
        id=rule_id,
        category=category,
        statement=statement,
        evidence_type="experience_based",
        confidence="high",
        sources=[RuleSource(lesson="lesson_001", video_id="yt_a", timestamps=["01:00"])],
        created=datetime.date(2026, 8, 28),
    )


def test_normalize_strips_case_and_punctuation():
    assert dedup.normalize("A print LINE, running!") == "a print line running"


def test_identical_statements_are_flagged_duplicate():
    new = make("SURFACE_002", "Vertical print lines prevent a PSA 10.")
    active = [make("SURFACE_001", "Vertical print lines prevent a PSA 10.")]
    flags = dedup.flags_for(new, active)
    assert len(flags) == 1
    assert flags[0].kind == "duplicate"
    assert flags[0].other_id == "SURFACE_001"


def test_negation_mismatch_is_flagged_contradiction():
    new = make("SURFACE_002", "Vertical print lines do not prevent a PSA 10.")
    active = [make("SURFACE_001", "Vertical print lines prevent a PSA 10.")]
    flags = dedup.flags_for(new, active)
    assert flags[0].kind == "contradiction"


def test_different_categories_are_never_compared():
    new = make("CORNERS_001", "Vertical print lines prevent a PSA 10.", category="corners")
    active = [make("SURFACE_001", "Vertical print lines prevent a PSA 10.")]
    assert dedup.flags_for(new, active) == []


def test_unrelated_statements_produce_no_flags():
    new = make("SURFACE_002", "Refractor lines are a factory characteristic, not damage.")
    active = [make("SURFACE_001", "Vertical print lines prevent a PSA 10.")]
    assert dedup.flags_for(new, active) == []


def test_is_negated_detects_common_negations():
    assert dedup.is_negated("This will never gem.")
    assert dedup.is_negated("This does not gem.")
    assert dedup.is_negated("No card with this gems.")
    assert not dedup.is_negated("This card gems reliably.")


def test_flags_are_sorted_most_similar_first():
    new = make("SURFACE_003", "Vertical print lines prevent a PSA 10.")
    active = [
        make("SURFACE_001", "Vertical print lines on chrome prevent a high grade."),
        make("SURFACE_002", "Vertical print lines prevent a PSA 10."),
    ]
    flags = dedup.flags_for(new, active, threshold=0.5)
    assert flags[0].other_id == "SURFACE_002"
    assert flags[0].score >= flags[1].score
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.dedup'`

- [ ] **Step 3: Write the implementation**

Create `src/card_reviewer/knowledge/dedup.py`:

```python
"""Flag pending rules that duplicate or contradict active ones.

This module only ever raises flags. Resolution belongs to the user during
`card-knowledge review` — an automated merge would silently rewrite what the
grader believes.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .models import Rule

THRESHOLD = 0.72

NEGATIONS = (
    "not",
    "never",
    "no",
    "cannot",
    "won't",
    "doesn't",
    "does not",
    "will not",
    "rarely",
    "unlikely",
)


@dataclass
class Flag:
    kind: str  # "duplicate" | "contradiction"
    other_id: str
    score: float
    other_statement: str


def normalize(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def is_negated(text: str) -> bool:
    words = set(normalize(text).split())
    phrase = normalize(text)
    return any(
        (n in words) if " " not in n else (n in phrase) for n in NEGATIONS
    )


def flags_for(
    rule: Rule, active_rules: list[Rule], threshold: float = THRESHOLD
) -> list[Flag]:
    flags: list[Flag] = []
    for other in active_rules:
        if other.category is not rule.category or other.id == rule.id:
            continue
        score = similarity(rule.statement, other.statement)
        if score < threshold:
            continue
        kind = (
            "contradiction"
            if is_negated(rule.statement) != is_negated(other.statement)
            else "duplicate"
        )
        flags.append(Flag(kind, other.id, round(score, 3), other.statement))

    flags.sort(key=lambda f: -f.score)
    return flags
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/knowledge/dedup.py tests/test_dedup.py
git commit -m "feat: duplicate and contradiction flagging for pending rules"
```

---

## Task 12: Rubric versioning

**Files:**
- Create: `src/card_reviewer/knowledge/version.py`
- Test: `tests/test_version.py`

**Interfaces:**
- Consumes: `paths.ProjectPaths`
- Produces:
  - `version.read(paths) -> str` — returns `"0.1.0"` when the file does not exist
  - `version.write(paths, value: str) -> None`
  - `version.bump(current: str, level: str) -> str` — level is `"patch" | "minor" | "major"`
  - `version.INITIAL = "0.1.0"`

Semantics, per the spec: **patch** = wording only; **minor** = rules added; **major** = an active rule changed meaning, was superseded, or was retracted. The distinction is whether an already-issued review would come out differently.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_version.py`:

```python
import pytest

from card_reviewer.knowledge import version
from card_reviewer.knowledge.paths import ProjectPaths


@pytest.fixture
def paths(tmp_path):
    p = ProjectPaths(tmp_path)
    p.knowledge.mkdir(parents=True)
    return p


def test_read_defaults_to_initial_when_file_absent(paths):
    assert version.read(paths) == "0.1.0"


def test_write_then_read_roundtrips(paths):
    version.write(paths, "1.2.3")
    assert version.read(paths) == "1.2.3"


def test_read_tolerates_trailing_whitespace(paths):
    paths.version_file.write_text("0.4.2\n\n")
    assert version.read(paths) == "0.4.2"


def test_bump_patch():
    assert version.bump("0.4.2", "patch") == "0.4.3"


def test_bump_minor_resets_patch():
    assert version.bump("0.4.2", "minor") == "0.5.0"


def test_bump_major_resets_minor_and_patch():
    assert version.bump("0.4.2", "major") == "1.0.0"


def test_bump_rejects_unknown_level():
    with pytest.raises(ValueError):
        version.bump("0.4.2", "sideways")


def test_bump_rejects_malformed_version():
    with pytest.raises(ValueError):
        version.bump("nope", "patch")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_version.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.version'`

- [ ] **Step 3: Write the implementation**

Create `src/card_reviewer/knowledge/version.py`:

```python
"""The rubric's semver, stamped onto every review subsystem A emits.

The bump level answers one question: would an already-issued review come out
differently under this change? If yes, it is at least minor, and a changed or
retracted active rule is major.
"""

from __future__ import annotations

import re

from .paths import ProjectPaths

INITIAL = "0.1.0"
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
LEVELS = ("patch", "minor", "major")


def read(paths: ProjectPaths) -> str:
    if not paths.version_file.exists():
        return INITIAL
    return paths.version_file.read_text().strip() or INITIAL


def write(paths: ProjectPaths, value: str) -> None:
    if not SEMVER_RE.match(value):
        raise ValueError(f"not a semver string: {value!r}")
    paths.version_file.parent.mkdir(parents=True, exist_ok=True)
    paths.version_file.write_text(value + "\n")


def bump(current: str, level: str) -> str:
    match = SEMVER_RE.match(current.strip())
    if not match:
        raise ValueError(f"not a semver string: {current!r}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    major, minor, patch = (int(g) for g in match.groups())
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_version.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/knowledge/version.py tests/test_version.py
git commit -m "feat: semver rubric versioning"
```

---

## Task 13: Promotion

**Files:**
- Create: `src/card_reviewer/knowledge/promote.py`
- Modify: `src/card_reviewer/knowledge/cli.py` (add the `review` command)
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `models.Rule`, `models.RuleStatus`, `validate.load_pending/load_active`, `dedup.flags_for`, `version`, `paths.ProjectPaths`
- Produces:
  - `promote.accept(paths, rule, rubric_version) -> Path`
  - `promote.reject(paths, rule, reason) -> Path`
  - `promote.supersede(paths, new_rule, old_id, rubric_version) -> tuple[Path, Path]`
  - `promote.rule_path(paths, rule) -> Path` — `knowledge/rules/<category>/<id>.yaml`
  - `promote.write_rule(path, rule) -> None`
  - `promote.UnknownRule(Exception)`

Test design note: the interactive prompt loop is not unit-tested. Test the **transitions** — what lands on disk, with what status and version — because that is where correctness lives. A test that drives a prompt library tests the prompt library.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_promote.py`:

```python
import datetime

import pytest
import yaml

from card_reviewer.knowledge import promote, validate
from card_reviewer.knowledge.models import Rule, RuleSource, RuleStatus
from card_reviewer.knowledge.paths import ProjectPaths


def make(rule_id="SURFACE_PRINT_LINE_001", **over):
    base = dict(
        id=rule_id,
        category="surface",
        statement="Vertical print lines commonly prevent a PSA 10.",
        evidence_type="experience_based",
        confidence="high",
        sources=[RuleSource(lesson="lesson_001", video_id="yt_a", timestamps=["01:00"])],
        created=datetime.date(2026, 8, 28),
    )
    return Rule(**(base | over))


@pytest.fixture
def paths(tmp_path):
    p = ProjectPaths(tmp_path)
    p.pending_rules.mkdir(parents=True)
    p.rules.mkdir(parents=True)
    return p


def write_pending(paths, rule):
    path = paths.pending_rules / f"{rule.id}.yaml"
    path.write_text(yaml.safe_dump(rule.model_dump(mode="json"), sort_keys=False))
    return path


def test_accept_moves_rule_to_category_directory(paths):
    rule = make()
    pending = write_pending(paths, rule)
    dest = promote.accept(paths, rule, "0.2.0")
    assert dest == paths.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml"
    assert dest.exists()
    assert not pending.exists()


def test_accept_sets_status_and_stamps_version(paths):
    rule = make()
    write_pending(paths, rule)
    dest = promote.accept(paths, rule, "0.2.0")
    stored = Rule.model_validate(yaml.safe_load(dest.read_text()))
    assert stored.status is RuleStatus.ACTIVE
    assert stored.rubric_version_added == "0.2.0"


def test_accepted_rule_is_visible_to_load_active(paths):
    rule = make()
    write_pending(paths, rule)
    promote.accept(paths, rule, "0.2.0")
    assert [r.id for r in validate.load_active(paths)] == ["SURFACE_PRINT_LINE_001"]


def test_reject_keeps_the_rule_on_disk_with_reason(paths):
    """Spec §7 and plan §30 rule 9: nothing is ever deleted."""
    rule = make()
    pending = write_pending(paths, rule)
    dest = promote.reject(paths, rule, "instructor opinion, contradicted by lesson_003")
    assert dest.exists()
    assert not pending.exists()
    stored = Rule.model_validate(yaml.safe_load(dest.read_text()))
    assert stored.status is RuleStatus.REJECTED
    assert "contradicted by lesson_003" in stored.notes


def test_rejected_rule_is_not_active(paths):
    rule = make()
    write_pending(paths, rule)
    promote.reject(paths, rule, "opinion")
    assert validate.load_active(paths) == []


def test_supersede_activates_new_and_retires_old(paths):
    old = make("SURFACE_PRINT_LINE_001")
    write_pending(paths, old)
    promote.accept(paths, old, "0.1.0")

    new = make("SURFACE_PRINT_LINE_002", statement="Print lines under 1cm may still gem.")
    write_pending(paths, new)
    new_path, old_path = promote.supersede(paths, new, "SURFACE_PRINT_LINE_001", "1.0.0")

    stored_new = Rule.model_validate(yaml.safe_load(new_path.read_text()))
    stored_old = Rule.model_validate(yaml.safe_load(old_path.read_text()))
    assert stored_new.status is RuleStatus.ACTIVE
    assert stored_new.supersedes == "SURFACE_PRINT_LINE_001"
    assert stored_old.status is RuleStatus.SUPERSEDED
    assert stored_old.id == "SURFACE_PRINT_LINE_001"  # id is never reused
    assert [r.id for r in validate.load_active(paths)] == ["SURFACE_PRINT_LINE_002"]


def test_supersede_unknown_rule_raises(paths):
    new = make("SURFACE_PRINT_LINE_002")
    write_pending(paths, new)
    with pytest.raises(promote.UnknownRule):
        promote.supersede(paths, new, "SURFACE_NOPE_999", "1.0.0")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_promote.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.promote'`

- [ ] **Step 3: Write the implementation**

Create `src/card_reviewer/knowledge/promote.py`:

```python
"""Rule status transitions.

Every transition preserves the file. A rejected rule is a rejected rule on
disk forever, not a deleted one — the provenance of what the grader was told
and chose not to believe is part of the audit trail.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from . import validate
from .models import Rule, RuleStatus
from .paths import ProjectPaths


class UnknownRule(Exception):
    """No rule with this id exists in knowledge/rules/."""


def rule_path(paths: ProjectPaths, rule: Rule) -> Path:
    return paths.rules / rule.category.value / f"{rule.id}.yaml"


def write_rule(path: Path, rule: Rule) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rule.model_dump(mode="json"), sort_keys=False))


def _drop_pending(paths: ProjectPaths, rule: Rule) -> None:
    (paths.pending_rules / f"{rule.id}.yaml").unlink(missing_ok=True)


def _find_active(paths: ProjectPaths, rule_id: str) -> tuple[Path, Rule]:
    for path in sorted(paths.rules.rglob("*.yaml")):
        rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        if rule.id == rule_id:
            return path, rule
    raise UnknownRule(f"no rule with id {rule_id!r} in {paths.rules}")


def accept(paths: ProjectPaths, rule: Rule, rubric_version: str) -> Path:
    promoted = rule.model_copy(
        update={"status": RuleStatus.ACTIVE, "rubric_version_added": rubric_version}
    )
    dest = rule_path(paths, promoted)
    write_rule(dest, promoted)
    _drop_pending(paths, rule)
    return dest


def reject(paths: ProjectPaths, rule: Rule, reason: str) -> Path:
    existing = f"{rule.notes}\n" if rule.notes else ""
    rejected = rule.model_copy(
        update={"status": RuleStatus.REJECTED, "notes": f"{existing}rejected: {reason}"}
    )
    dest = rule_path(paths, rejected)
    write_rule(dest, rejected)
    _drop_pending(paths, rule)
    return dest


def supersede(
    paths: ProjectPaths, new_rule: Rule, old_id: str, rubric_version: str
) -> tuple[Path, Path]:
    old_path, old_rule = _find_active(paths, old_id)

    retired = old_rule.model_copy(update={"status": RuleStatus.SUPERSEDED})
    write_rule(old_path, retired)

    promoted = new_rule.model_copy(
        update={
            "status": RuleStatus.ACTIVE,
            "rubric_version_added": rubric_version,
            "supersedes": old_id,
        }
    )
    new_path = rule_path(paths, promoted)
    write_rule(new_path, promoted)
    _drop_pending(paths, new_rule)
    return new_path, old_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_promote.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Add the interactive `review` command**

Append to `src/card_reviewer/knowledge/cli.py`:

```python
@app.command(name="review")
def review_cmd() -> None:
    """Walk pending rules one at a time and decide each one."""
    from . import dedup, promote as pr, validate as val, version as ver

    p = paths()
    report = val.run(p)
    if not report.ok:
        console.print("[red]Fix validation errors before reviewing:[/red]")
        for rule_id, errors in report.errors.items():
            console.print(f"  {rule_id}: {'; '.join(errors)}")
        raise typer.Exit(code=1)

    pending = val.load_pending(p)
    if not pending:
        console.print("No pending rules.")
        return

    active = val.load_active(p)
    accepted = superseded = 0

    for _, rule in pending:
        console.rule(f"[bold]{rule.id}[/bold]  ({rule.category.value})")
        console.print(f"[bold]{rule.statement}[/bold]")
        console.print(
            f"evidence: {rule.evidence_type.value}   confidence: {rule.confidence.value}"
        )
        if rule.applies_to.card_types or rule.applies_to.sets:
            console.print(
                f"applies to: {rule.applies_to.card_types or '-'} / {rule.applies_to.sets or '-'}"
            )
        for source in rule.sources:
            console.print(
                f"  [dim]{source.lesson} {source.video_id} {','.join(source.timestamps)}[/dim]"
            )
            if source.quote:
                console.print(f'    "{source.quote}"')

        for flag in dedup.flags_for(rule, active):
            colour = "red" if flag.kind == "contradiction" else "yellow"
            console.print(
                f"  [{colour}]{flag.kind}[/{colour}] vs {flag.other_id} "
                f"({flag.score}): {flag.other_statement}"
            )

        choice = typer.prompt(
            "accept / reject / supersede <ID> / defer", default="defer"
        ).strip()

        if choice.startswith("accept"):
            pr.accept(p, rule, ver.bump(ver.read(p), "minor"))
            accepted += 1
        elif choice.startswith("reject"):
            reason = typer.prompt("reason")
            pr.reject(p, rule, reason)
        elif choice.startswith("supersede"):
            parts = choice.split(maxsplit=1)
            if len(parts) != 2:
                console.print("[red]supersede needs a rule id; deferring[/red]")
                continue
            pr.supersede(p, rule, parts[1].strip(), ver.bump(ver.read(p), "major"))
            superseded += 1

    if accepted:
        ver.write(p, ver.bump(ver.read(p), "minor"))
    if superseded:
        ver.write(p, ver.bump(ver.read(p), "major"))
    console.print(
        f"[green]{accepted} accepted, {superseded} superseded.[/green] "
        f"Rubric now {ver.read(p)}. Run `card-knowledge build-rubric`."
    )
```

- [ ] **Step 6: Commit**

```bash
git add src/card_reviewer/knowledge/promote.py src/card_reviewer/knowledge/cli.py tests/test_promote.py
git commit -m "feat: rule promotion with preserved provenance"
```

---

## Task 14: Rubric rendering and the subsystem A contract

**Files:**
- Create: `src/card_reviewer/knowledge/rubric.py`
- Modify: `src/card_reviewer/knowledge/cli.py` (add the `build-rubric` command)
- Test: `tests/test_rubric.py`

**Interfaces:**
- Consumes: `models.Rule`, `validate.load_active`, `version.read`, `paths.ProjectPaths`
- Produces:
  - `rubric.Rubric(version, rules)` with methods `by_category(category) -> list[Rule]` and `for_card(card_types=None, sets=None) -> list[Rule]`
  - `rubric.load_active_rubric(root: Path | str | None = None) -> Rubric` — **the contract subsystem A imports**
  - `rubric.render(rubric) -> str`
  - `rubric.build(paths) -> Path`

`for_card` matching rule, fixed here: a rule matches when *(its `applies_to.card_types` is empty **or** intersects the given card types)* **and** *(its `applies_to.sets` is empty **or** intersects the given sets)*. An unscoped rule therefore applies to every card, which is what "vertical print lines hurt" should do.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rubric.py`:

```python
import datetime

import pytest
import yaml

from card_reviewer.knowledge import rubric, version
from card_reviewer.knowledge.models import Rule, RuleSource
from card_reviewer.knowledge.paths import ProjectPaths


def make(rule_id, category="surface", card_types=None, sets=None, status="active"):
    return Rule(
        id=rule_id,
        category=category,
        statement=f"Statement for {rule_id}.",
        evidence_type="objective",
        confidence="high",
        applies_to={"card_types": card_types or [], "sets": sets or []},
        sources=[RuleSource(lesson="lesson_001", video_id="yt_a", timestamps=["01:00"])],
        status=status,
        created=datetime.date(2026, 8, 28),
        rubric_version_added="0.1.0",
    )


@pytest.fixture
def project(tmp_path):
    p = ProjectPaths(tmp_path)
    for rule in [
        make("SURFACE_001"),
        make("SURFACE_002", card_types=["chrome", "refractor"]),
        make("CORNERS_001", category="corners"),
        make("SURFACE_003", status="rejected"),
    ]:
        path = p.rules / rule.category.value / f"{rule.id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(rule.model_dump(mode="json"), sort_keys=False))
    version.write(p, "0.3.0")
    return p


def test_load_active_rubric_carries_the_version(project):
    r = rubric.load_active_rubric(project.root)
    assert r.version == "0.3.0"


def test_load_active_rubric_excludes_non_active_rules(project):
    r = rubric.load_active_rubric(project.root)
    assert "SURFACE_003" not in {rule.id for rule in r.rules}


def test_by_category_filters(project):
    r = rubric.load_active_rubric(project.root)
    assert {rule.id for rule in r.by_category("surface")} == {"SURFACE_001", "SURFACE_002"}


def test_for_card_includes_unscoped_rules(project):
    r = rubric.load_active_rubric(project.root)
    ids = {rule.id for rule in r.for_card(card_types=["paper"])}
    assert "SURFACE_001" in ids  # unscoped applies to everything
    assert "SURFACE_002" not in ids  # scoped to chrome/refractor


def test_for_card_includes_matching_scoped_rules(project):
    r = rubric.load_active_rubric(project.root)
    ids = {rule.id for rule in r.for_card(card_types=["chrome"])}
    assert {"SURFACE_001", "SURFACE_002", "CORNERS_001"} <= ids


def test_for_card_with_no_arguments_returns_everything(project):
    r = rubric.load_active_rubric(project.root)
    assert len(r.for_card()) == 3


def test_render_includes_version_and_rule_count(project):
    r = rubric.load_active_rubric(project.root)
    text = rubric.render(r)
    assert "0.3.0" in text
    assert "3 active rules" in text


def test_render_groups_by_category(project):
    text = rubric.render(rubric.load_active_rubric(project.root))
    assert "## corners" in text
    assert "## surface" in text


def test_render_warns_against_hand_editing(project):
    """Spec §8: the markdown is a view, the YAML is the source of truth."""
    text = rubric.render(rubric.load_active_rubric(project.root))
    assert "generated" in text.lower()
    assert "do not edit" in text.lower()


def test_build_writes_the_rubric_file(project):
    path = rubric.build(project)
    assert path == project.rubric_file
    assert "0.3.0" in path.read_text()


def test_empty_knowledge_base_renders_without_error(tmp_path):
    p = ProjectPaths(tmp_path)
    p.knowledge.mkdir(parents=True)
    r = rubric.load_active_rubric(tmp_path)
    assert r.rules == []
    assert "0 active rules" in rubric.render(r)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rubric.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.rubric'`

- [ ] **Step 3: Write the implementation**

Create `src/card_reviewer/knowledge/rubric.py`:

```python
"""The rubric: what the grader currently believes, and the contract that
subsystem A imports to read it.

`ACTIVE_RUBRIC.md` is a rendered view. The YAML files under knowledge/rules/
are the source of truth. Never parse the markdown.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

from . import validate, version
from .models import Rule
from .paths import ProjectPaths


@dataclass
class Rubric:
    version: str
    rules: list[Rule]

    def by_category(self, category: str) -> list[Rule]:
        return [r for r in self.rules if r.category.value == str(category)]

    def for_card(
        self,
        card_types: list[str] | None = None,
        sets: list[str] | None = None,
    ) -> list[Rule]:
        """Rules relevant to one card.

        An unscoped rule applies to every card. A scoped rule applies only when
        its scope intersects what the caller asked for.
        """
        if card_types is None and sets is None:
            return list(self.rules)

        wanted_types = set(card_types or [])
        wanted_sets = set(sets or [])

        def matches(rule: Rule) -> bool:
            type_ok = not rule.applies_to.card_types or bool(
                set(rule.applies_to.card_types) & wanted_types
            )
            set_ok = not rule.applies_to.sets or bool(
                set(rule.applies_to.sets) & wanted_sets
            )
            return type_ok and set_ok

        return [r for r in self.rules if matches(r)]


def load_active_rubric(root: Path | str | None = None) -> Rubric:
    """The contract subsystem A calls. `version` is stamped onto every review."""
    paths = ProjectPaths(root or Path(__file__).resolve().parents[3])
    return Rubric(version=version.read(paths), rules=validate.load_active(paths))


def render(r: Rubric) -> str:
    lines = [
        "# Active Grading Rubric",
        "",
        "<!-- GENERATED FILE — DO NOT EDIT.",
        "     Source of truth is knowledge/rules/. Regenerate with:",
        "       card-knowledge build-rubric -->",
        "",
        f"**Version:** {r.version}  ",
        f"**Rules:** {len(r.rules)} active rules  ",
        f"**Built:** {datetime.date.today().isoformat()}",
        "",
    ]

    by_category: dict[str, list[Rule]] = {}
    for rule in r.rules:
        by_category.setdefault(rule.category.value, []).append(rule)

    for category in sorted(by_category):
        lines += [f"## {category}", ""]
        for rule in sorted(by_category[category], key=lambda x: x.id):
            scope = ""
            if rule.applies_to.card_types or rule.applies_to.sets:
                bits = []
                if rule.applies_to.card_types:
                    bits.append("card types: " + ", ".join(rule.applies_to.card_types))
                if rule.applies_to.sets:
                    bits.append("sets: " + ", ".join(rule.applies_to.sets))
                scope = f" _({'; '.join(bits)})_"
            citations = ", ".join(
                f"{s.lesson} {'/'.join(s.timestamps)}" for s in rule.sources
            )
            lines += [
                f"### {rule.id}{scope}",
                "",
                rule.statement,
                "",
                f"- evidence: `{rule.evidence_type.value}`  ",
                f"- confidence: `{rule.confidence.value}`  ",
                f"- sources: {citations}",
                "",
            ]

    return "\n".join(lines) + "\n"


def build(paths: ProjectPaths) -> Path:
    r = Rubric(version=version.read(paths), rules=validate.load_active(paths))
    paths.rubric_file.parent.mkdir(parents=True, exist_ok=True)
    paths.rubric_file.write_text(render(r))
    return paths.rubric_file
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rubric.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Add the CLI command**

Append to `src/card_reviewer/knowledge/cli.py`:

```python
@app.command(name="build-rubric")
def build_rubric_cmd() -> None:
    """Render ACTIVE_RUBRIC.md from the active rule files."""
    from . import rubric as rb

    path = rb.build(paths())
    r = rb.load_active_rubric(project_root())
    console.print(f"[green]Wrote[/green] {path} — v{r.version}, {len(r.rules)} rules")
```

- [ ] **Step 6: Commit**

```bash
git add src/card_reviewer/knowledge/rubric.py src/card_reviewer/knowledge/cli.py tests/test_rubric.py
git commit -m "feat: rubric rendering and load_active_rubric contract"
```

---

## Task 15: The learn-video skill

**Files:**
- Create: `skills/learn-video/SKILL.md`
- Create: `training/lessons/TEMPLATE.md`
- Test: `tests/test_skill_contract.py`

**Interfaces:**
- Consumes: the CLI commands from Tasks 5–9 and the `Rule` schema from Task 2
- Produces: `skills/learn-video/SKILL.md` — the `analyze` stage's instructions

This is the one stage Python does not perform. The skill is tested for its **contract**, not its prose: that it exists, names the right commands, and states the prohibitions. Per plan §22, it calls the Python utilities rather than restating their algorithms.

- [ ] **Step 1: Write the failing test**

Create `tests/test_skill_contract.py`:

```python
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
SKILL = REPO / "skills" / "learn-video" / "SKILL.md"


def test_skill_file_exists():
    assert SKILL.exists()


def test_skill_has_frontmatter_with_name_and_description():
    body = SKILL.read_text()
    assert body.startswith("---")
    assert "name: learn-video" in body
    assert "description:" in body


def test_skill_names_the_commands_it_drives():
    body = SKILL.read_text()
    for command in ("card-knowledge extract-frames", "card-knowledge validate"):
        assert command in body, f"skill does not mention: {command}"


def test_skill_states_the_prohibitions():
    """Spec §5: the skill may not promote its own output."""
    body = SKILL.read_text().lower()
    assert "knowledge/rules/" in body
    assert "active_rubric.md" in body
    assert "status: active" in body


def test_skill_requires_evidence_type_classification():
    body = SKILL.read_text()
    for value in ("objective", "experience_based", "opinion", "unverified", "contradicted"):
        assert value in body


def test_lesson_template_exists_and_covers_the_plan_sections():
    template = (REPO / "training" / "lessons" / "TEMPLATE.md").read_text()
    for heading in (
        "RULES TAUGHT",
        "DEFECTS SHOWN",
        "PSA GRADE EXAMPLES",
        "INSTRUCTOR OPINIONS",
        "POTENTIAL CONTRADICTIONS",
        "SOURCE TIMESTAMPS",
    ):
        assert heading in template
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_skill_contract.py -v`
Expected: FAIL — `FileNotFoundError` / assertion error, the skill does not exist

- [ ] **Step 3: Create the lesson template**

Create `training/lessons/TEMPLATE.md`:

```markdown
---
lesson_id: lesson_NNN
video_id: <video_id>
source: <url or local path>
title: <video title>
date_processed: YYYY-MM-DD
file_sha256: <hash from manifest.json>
duration_s: <from manifest.json>
topics: [centering, corners, edges, surface]
segments_reviewed: [seg_001, seg_004, seg_009]
---

# <Video title>

## RULES TAUGHT

<Each claim the instructor makes about grading, one per bullet, with a timestamp.>

## VISUAL EXAMPLES

<What was actually shown on screen, and in which frames.>

## DEFECTS SHOWN

<Specific defects visible in the footage, with timestamps.>

## PSA GRADE EXAMPLES

<Cards with a stated PSA grade, and the reason given.>

## INSTRUCTOR OPINIONS

<Claims presented as judgment rather than fact. Say who is speaking.>

## OBJECTIVE OBSERVATIONS

<Facts visible in the footage independent of the instructor's commentary.>

## POTENTIAL CONTRADICTIONS

<Anything conflicting with an existing active rule, or with itself.>

## SOURCE TIMESTAMPS

<Every timestamp cited above, so citations can be spot-checked.>

## CANDIDATE KNOWLEDGE-BASE UPDATES

<The rule ids written to knowledge/pending_rules/, one per line, with a
one-line justification each.>
```

- [ ] **Step 4: Create the skill**

Create `skills/learn-video/SKILL.md`:

```markdown
---
name: learn-video
description: Use when extracting grading knowledge from a prepared training-video work packet - performs the analyze stage of the card reviewer's video learning pipeline, producing a lesson record and candidate rules.
---

# Learning From a Grading Video

You are performing the `analyze` stage of the video learning pipeline. Python
has already downloaded the video, transcribed it, ranked the segments worth
watching, and extracted frames. Your job is judgment: read what is there and
write down what it teaches, with citations.

## Before you start

Confirm the packet is ready:

    card-knowledge status <video_id>

`acquire`, `transcribe`, `segment`, and `extract_frames` must all be `done`.
If they are not, stop and run the missing stage. Do not analyze a partial packet.

## What you are given

    training/work/<video_id>/
      manifest.json    source, duration, file hash
      transcript.json  every cue with timestamps
      segments.json    ranked windows, with matched terms and categories
      frames/seg_NNN/  extracted frames for the top segments

## Process

1. **Read `segments.json` first.** Work through segments in rank order. Each
   one names the categories and terms that made it rank.
2. **Look at the frames for each segment.** Read the image files directly.
   The transcript says what was claimed; the frames show what was true.
3. **Read the surrounding transcript** for context that the window clipped.
4. **Go get more when you need it.** If the transcript hints at something the
   ranker skipped:

       card-knowledge extract-frames <video_id> --at 754 --window 30

   The ranking is a starting point, not a limit.
5. **Write the lesson record** to `training/lessons/lesson_NNN.md`, following
   `training/lessons/TEMPLATE.md`. Use the next free number.
6. **Write candidate rules** to `knowledge/pending_rules/<RULE_ID>.yaml`.
7. **Validate your own output:**

       card-knowledge validate

   Fix every error before you report finished.

## Writing a rule

One rule, one testable claim. If a sentence contains "and" joining two
different defects, it is two rules.

```yaml
id: SURFACE_PRINT_LINE_001      # CATEGORY_TOPIC_NNN, uppercase, three digits
category: surface               # centering|corners|edges|surface|print|handling|image_limitations|process
statement: "Vertical print lines running the length of the card commonly prevent a PSA 10."
evidence_type: experience_based
confidence: high
applies_to:
  card_types: [chrome, refractor]   # omit entries to mean "all cards"
  sets: []
sources:
  - lesson: lesson_014
    video_id: yt_abc123
    timestamps: ["12:04-12:38"]
    quote: "I've never seen one of these gem with a line like that."
status: pending
supersedes: null
created: 2026-08-28
rubric_version_added: null
```

### Classifying evidence — this is the point of the exercise

`evidence_type` is required and you must choose it deliberately. An instructor
being confident is not evidence.

| Value | Use when |
|---|---|
| `objective` | Verifiable independent of the speaker — PSA's published centering tolerances, what is visibly on screen |
| `experience_based` | The instructor's submission history — "these come back a 9 for me" |
| `opinion` | Judgment or preference presented without support |
| `unverified` | Asserted, plausible, nothing shown to support it |
| `contradicted` | Conflicts with an existing active rule, or with the footage |

When in doubt, choose the weaker classification. A rule graded too weak gets
promoted anyway after review; a rule graded too strong quietly biases every
future grade.

### Citations

Every source needs a real timestamp from `transcript.json`, in `MM:SS` or
`HH:MM:SS`. `card-knowledge validate` checks each one against the video's
actual duration and will reject a citation that cannot exist. Do not
approximate — look it up.

## Prohibitions

- **Never write to `knowledge/rules/`.** That directory holds promoted rules
  only. Yours go to `knowledge/pending_rules/`.
- **Never edit `ACTIVE_RUBRIC.md`.** It is generated by `card-knowledge
  build-rubric`.
- **Never set `status: active`.** Only the user's `card-knowledge review`
  promotes a rule.
- **Never delete or rewrite an existing rule.** If yours conflicts, say so in
  the lesson's POTENTIAL CONTRADICTIONS section and let review resolve it.
- **Never record pricing, value, or profitability.** Not even when the
  instructor talks about it — this repository grades condition only.
- **Do not treat everything the instructor says as fact.** Plan §30 rule 11.

## When you are done

Report: the lesson file written, the rule ids created with their evidence
types, anything you could not verify, and anything in the video that
contradicts an active rule. Then tell the user to run `card-knowledge review`.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_skill_contract.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add skills/learn-video/SKILL.md training/lessons/TEMPLATE.md tests/test_skill_contract.py
git commit -m "feat: learn-video skill for the analyze stage"
```

---

## Task 16: Orchestration — `run` and `status`

**Files:**
- Modify: `src/card_reviewer/knowledge/cli.py` (add `run` and `status`)
- Create: `src/card_reviewer/knowledge/pipeline.py`
- Create: `README.md`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `acquire`, `transcribe`, `segment`, `frames`, `manifest`, `paths.ProjectPaths`
- Produces:
  - `pipeline.DETERMINISTIC_STAGES = ("acquire", "transcribe", "segment", "extract_frames")`
  - `pipeline.run_all(paths, url=None, file=None, browser=None, top_n=12, force=False, steps=None) -> Manifest`
  - `pipeline.status(paths, video_id=None) -> list[Manifest]`

`steps` is a dict mapping stage name to callable, defaulting to the real stage functions. This keeps `run_all` testable without any external binary.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
import pytest

from card_reviewer.knowledge import manifest as mf, pipeline
from card_reviewer.knowledge.models import Manifest, SourceInfo
from card_reviewer.knowledge.paths import ProjectPaths


@pytest.fixture
def paths(tmp_path):
    return ProjectPaths(tmp_path)


def fake_steps(paths, calls):
    def make_packet(*a, **k):
        m = Manifest(
            video_id="yt_abc",
            source=SourceInfo(type="youtube", url="u", title="t", duration_s=100.0),
            rubric_version_at_ingest="0.1.0",
        )
        mf.save(paths, m)
        calls.append("acquire")
        return mf.finish(paths, m, "acquire")

    def stage(name):
        def go(p, video_id, **kwargs):
            calls.append(name)
            m = mf.load(p, video_id)
            mf.finish(p, m, name)

        return go

    return {
        "acquire": make_packet,
        "transcribe": stage("transcribe"),
        "segment": stage("segment"),
        "extract_frames": stage("extract_frames"),
    }


def test_run_all_advances_every_deterministic_stage(paths):
    calls = []
    m = pipeline.run_all(paths, url="https://youtu.be/abc", steps=fake_steps(paths, calls))
    assert calls == ["acquire", "transcribe", "segment", "extract_frames"]
    assert all(mf.is_done(m, s) for s in pipeline.DETERMINISTIC_STAGES)


def test_run_all_stops_before_analyze(paths):
    calls = []
    m = pipeline.run_all(paths, url="https://youtu.be/abc", steps=fake_steps(paths, calls))
    assert "analyze" not in calls
    assert not mf.is_done(m, "analyze")


def test_completed_stages_are_skipped_on_rerun(paths):
    calls = []
    steps = fake_steps(paths, calls)
    pipeline.run_all(paths, url="https://youtu.be/abc", steps=steps)
    calls.clear()
    pipeline.run_all(paths, url="https://youtu.be/abc", steps=steps)
    assert calls == ["acquire"]  # acquire re-runs to locate the packet; rest skipped


def test_force_reruns_completed_stages(paths):
    calls = []
    steps = fake_steps(paths, calls)
    pipeline.run_all(paths, url="https://youtu.be/abc", steps=steps)
    calls.clear()
    pipeline.run_all(paths, url="https://youtu.be/abc", steps=steps, force=True)
    assert calls == ["acquire", "transcribe", "segment", "extract_frames"]


def test_status_lists_all_packets(paths):
    for vid in ("yt_a", "yt_b"):
        mf.save(
            paths,
            Manifest(
                video_id=vid,
                source=SourceInfo(type="youtube", url="u", title=vid, duration_s=1.0),
                rubric_version_at_ingest="0.1.0",
            ),
        )
    assert {m.video_id for m in pipeline.status(paths)} == {"yt_a", "yt_b"}


def test_status_for_one_packet(paths):
    mf.save(
        paths,
        Manifest(
            video_id="yt_a",
            source=SourceInfo(type="youtube", url="u", title="t", duration_s=1.0),
            rubric_version_at_ingest="0.1.0",
        ),
    )
    assert len(pipeline.status(paths, "yt_a")) == 1


def test_run_all_requires_a_source(paths):
    with pytest.raises(ValueError):
        pipeline.run_all(paths)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_reviewer.knowledge.pipeline'`

- [ ] **Step 3: Write `pipeline.py`**

```python
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
    if not url and not file:
        raise ValueError("run_all requires either url or file")

    from . import version

    steps = steps or _default_steps()
    m = steps["acquire"](
        paths, url=url, file=file, browser=browser, rubric_version=version.read(paths)
    )

    for stage in DETERMINISTIC_STAGES[1:]:
        current = mf.load(paths, m.video_id)
        if mf.is_done(current, stage) and not force:
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
```

- [ ] **Step 4: Add the CLI commands**

Append to `src/card_reviewer/knowledge/cli.py`:

```python
@app.command(name="run")
def run_cmd(
    url: str | None = typer.Argument(None),
    file: Path | None = typer.Option(None, "--file"),
    browser: str | None = typer.Option(None, "--browser"),
    top_n: int = typer.Option(12, "--top-n"),
    force: bool = typer.Option(False, "--force", help="Re-run stages already done"),
) -> None:
    """Advance a video through every deterministic stage, stopping at analyze."""
    from . import acquire as acq
    from . import pipeline as pl

    try:
        m = pl.run_all(paths(), url=url, file=file, browser=browser, top_n=top_n, force=force)
    except acq.AcquisitionFailed as exc:
        console.print(f"[red]Acquisition failed:[/red] {exc}")
        console.print(exc.guidance)
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Packet ready for analysis:[/green] {m.video_id}")
    console.print(
        "Next: start an interactive Claude Code session and invoke the "
        f"learn-video skill on {m.video_id}."
    )


@app.command(name="status")
def status_cmd(video_id: str | None = typer.Argument(None)) -> None:
    """Show the stage state of one packet, or of every packet."""
    from . import pipeline as pl
    from .models import STAGES

    manifests = pl.status(paths(), video_id)
    if not manifests:
        console.print("No work packets.")
        return

    table = Table("video_id", "title", *STAGES)
    for m in manifests:
        marks = []
        for stage in STAGES:
            state = m.stages[stage].status.value
            marks.append(
                {"done": "[green]done[/green]", "failed": "[red]failed[/red]"}.get(
                    state, state
                )
            )
        table.add_row(m.video_id, m.source.title[:30], *marks)
    console.print(table)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Write the README**

Create `README.md`:

```markdown
# Card Reviewer

Two subsystems joined only by `knowledge/`. Currently building **subsystem B**,
the video learning pipeline. See `CLAUDE.md` for the rules and
`docs/superpowers/specs/` for the design.

## Setup

    brew install yt-dlp ffmpeg
    uv sync
    uv run card-knowledge doctor

## Ingesting a training video

    # Public YouTube
    uv run card-knowledge run "https://youtube.com/watch?v=..."

    # Authenticated course material you have access to
    uv run card-knowledge run "https://www.skool.com/..." --browser chrome

    # A video already on disk
    uv run card-knowledge run --file ~/Downloads/lesson.mp4

This stops at the `analyze` stage. Open an interactive Claude Code session and
invoke the `learn-video` skill on the packet. Claude writes a lesson record and
candidate rules, then:

    uv run card-knowledge validate      # mechanical checks
    uv run card-knowledge review        # you approve each rule
    uv run card-knowledge build-rubric  # regenerate ACTIVE_RUBRIC.md

Commit the result — the git history of `knowledge/` is the record of what the
grader believes.

## If a download fails

Authenticated platforms may refuse yt-dlp. That failure is terminal by design;
nothing here works around platform protections. Save the video from your
browser and use `--file`.

## Testing

    uv run pytest
```

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS — every test from Tasks 1–16.

- [ ] **Step 8: Commit**

```bash
git add src/card_reviewer/knowledge/pipeline.py src/card_reviewer/knowledge/cli.py README.md tests/test_pipeline.py
git commit -m "feat: pipeline orchestration, status reporting, and README"
```

---

## Definition of Done

From spec §11. Verify each with a real run, not by inspection:

1. **A public YouTube grading video runs end to end** and produces
   `training/lessons/lesson_001.md`, reviewed rules under `knowledge/rules/`,
   and `knowledge/ACTIVE_RUBRIC.md` at a version above `0.1.0`.
2. **A Skool video is attempted.** Either it processes, or it fails cleanly and
   completes via `--file`. Record which happened in the README's limitations.
3. **`load_active_rubric()` returns a versioned rubric** — check with
   `uv run python -c "from card_reviewer.knowledge.rubric import load_active_rubric; r = load_active_rubric(); print(r.version, len(r.rules))"`
4. **`uv run pytest` passes.**
5. **Known limitations are written down** in the README.

Out of scope, and must not appear in the repository: OpenCV, grading logic,
probability estimates, pricing, EV, or buy/sell recommendations.
