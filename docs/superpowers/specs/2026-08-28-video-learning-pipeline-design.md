# Video Learning Pipeline — Design

**Date:** 2026-08-28
**Status:** Approved for planning
**Scope:** Subsystem B of the Card Reviewer project
**Parent plan:** `CARD_REVIEWER_BUILD_PLAN.md` (§3, §6, §7, §27, §30)

---

## 1. Purpose

Turn sports-card grading training videos into a persistent, auditable, versioned
knowledge base that a separate card review engine can consume.

The pipeline's output is `knowledge/ACTIVE_RUBRIC.md` plus the structured rule
files behind it. Its input is video from YouTube, Skool/MLP, or local disk.

This subsystem performs no grading, no image analysis, and no pricing logic.

## 2. Scope decomposition

The parent build plan describes two subsystems joined only by files in
`knowledge/`:

- **A — Card Review Engine**: images + metadata to structured grade estimate
  (plan §8–§21, §23, §24)
- **B — Video Learning Pipeline**: videos to versioned grading rules
  (plan §3, §6, §7)

They are specified and built separately. This document covers **B only**.
Subsystem A gets its own spec, plan, and implementation cycle.

### Out of scope for this spec

- Any OpenCV code
- Any grading or probability logic
- Any card review schema beyond the `load_active_rubric()` contract in §8
- Pricing, EV, or buy/sell logic anywhere in the repository (plan §30 rule 14)

## 3. Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Build order | B before A | User direction |
| Sources for v1 | Public YouTube + authenticated Skool/MLP | User's available material |
| Extraction execution | Claude Code in-session via skill | Plan §23; no API key or per-video cost |
| Transcripts | yt-dlp captions, mlx-whisper fallback | Skool videos generally lack captions |
| Promotion | User approves, CLI applies | Plan §30 rule 11; keeps user as authority |
| Orchestration | Staged, resumable work packets | Acquisition is the least reliable stage; isolate its failures |

## 4. Repository skeleton

The full directory structure from plan §5 is created up front; only subsystem B
is populated. Subsystem B code lives under `src/card_reviewer/knowledge/` so it
never tangles with the future review modules.

- Dependency management: `uv`
- Two console scripts, one package: `card-knowledge` (this spec),
  `card-review` (subsystem A, later)
- Git initialized; `.gitignore` covers `training/work/*/source/`, `cookies.txt`,
  browser session data, tokens, credentials, and API keys (plan §4)

### Relationship to the plan §5 knowledge layout

Plan §5 lists `knowledge/psa/`, `knowledge/card_types/`, and `knowledge/sets/`.
Those hold hand-written topical notes and set templates. This spec **adds**
`knowledge/rules/<category>/` for generated, promoted rule files. The two
coexist: `ACTIVE_RUBRIC.md` is built from `knowledge/rules/`, while the topical
directories remain human-authored reference material. No plan §5 directory is
removed or repurposed.

### CLI surface

```
card-knowledge doctor                          # preflight external tools
card-knowledge acquire <url> | --file <path>   # stage 1
card-knowledge transcribe <video_id>           # stage 2
card-knowledge segment <video_id>              # stage 3
card-knowledge extract-frames <video_id> [--uniform] [--at TS --window N]
card-knowledge run <url>                       # stages 1-4, stops at analyze
card-knowledge validate <video_id>             # stage 6
card-knowledge review                          # interactive promotion
card-knowledge build-rubric                    # render ACTIVE_RUBRIC.md
card-knowledge status [<video_id>]             # show packet states
```

## 5. Work packets and the pipeline state machine

Every video becomes a self-describing directory:

```
training/work/<video_id>/
├── manifest.json        # state machine + provenance  (committed)
├── source/video.mp4     # gitignored
├── transcript.json      # timestamped cues
├── segments.json        # ranked candidate windows
└── frames/seg_003/00-12-30.jpg
```

`video_id` is stable and derived from the source: YouTube's own video ID, or a
content-hash slug for Skool and local files. Re-running never duplicates work.

### Stages

Each stage is a CLI subcommand that advances exactly one step and records its
result in `manifest.json`.

| Stage | Action | Recorded |
|---|---|---|
| `acquire` | yt-dlp download or local file adoption | URL, title, uploader, duration, sha256, tool version |
| `transcribe` | captions, else mlx-whisper | method, model, language |
| `segment` | rank grading-relevant windows | segment count, lexicon version |
| `extract-frames` | ffmpeg sampling for top segments | frame count, sampling rate |
| `analyze` | skill-driven, performed by Claude | lesson id written |
| `validate` | schema and citation checks | pass/fail, errors |

Stage status is one of `pending`, `running`, `done`, `failed`, `skipped`, with a
timestamp and, on failure, the captured error.

`card-knowledge run <url>` advances through the deterministic stages and stops
at `analyze`, instructing the user to start the skill. Re-running a `done` stage
is a no-op unless `--force` is passed.

The manifest carries the provenance required by plan §30: source, file hash,
tool versions, and the rubric version current at ingest time.

### The `analyze` stage and the learn-video skill

`analyze` is the one stage Python does not perform. It is executed by Claude in
an interactive Claude Code session, driven by `skills/learn-video/SKILL.md`
(a sibling of the plan §22 `skills/card-review/SKILL.md`).

The skill's contract:

- **Input**: a work packet whose `segment` and `extract-frames` stages are
  `done` — transcript, ranked segments, and extracted frames on disk
- **Instructions**: read the ranked segments in order; inspect the frames for
  each; consult the transcript for context; request additional frames via
  `extract-frames --at` when the transcript suggests the ranker missed
  something; classify every extracted claim by `evidence_type`; cite a
  timestamp for every rule
- **Output**: `training/lessons/lesson_NNN.md` and one or more
  `knowledge/pending_rules/<rule_id>.yaml` files
- **Prohibited**: writing directly to `knowledge/rules/`, editing
  `ACTIVE_RUBRIC.md`, or setting `status: active`

Per plan §22, the skill calls the Python utilities rather than restating their
algorithms. Its output is not trusted: the `validate` stage checks it, and the
user approves it in `review` before any of it reaches the rubric.

### Note on `/watch`

The `watch@claude-video` plugin is not installed and its interface is unverified.
The design does not depend on it: since the pipeline controls frame extraction,
extracted frames are read directly. Treat `/watch` as an optional enhancement to
spike during implementation, not a load-bearing dependency.

## 6. Segmentation

Pass 1 of plan §7. A 90-minute course video contains perhaps 8 minutes of actual
card inspection; segmentation finds those minutes so visual analysis is
affordable.

Each timestamped transcript cue is scored against a grading lexicon held in
`knowledge/segmentation_lexicon.yaml` — versioned and editable, not hardcoded —
organized by category:

- **centering** — centering, borders, off-center, 60/40, diamond cut
- **corners** — corner, ding, soft, whitening, fray, rounded
- **edges** — edge, chipping, rough cut, edge wear
- **surface** — print line, scratch, roller mark, dimple, gloss, orange peel
- **outcomes** — PSA 10, gem, came back a nine, bumped, qualifier
- **demonstration cues** — look right here, you can see, under the light, zoom
  in, flip it over

Demonstration cues carry their own weight. Pass 2 seeks moments where the
instructor is *showing* something, and demonstration phrasing predicts that
better than topic vocabulary alone.

Contiguous scoring cues merge into windows with ±5s padding, capped at 90
seconds and split when longer. `segments.json` records each window's score,
matched terms, categories, transcript text, and bounds, so a rule written later
already has its citation available.

### Frames

For the top *N* segments (default 12, configurable), ffmpeg samples at 1 fps
with a per-segment cap. Perceptual-hash deduplication then drops near-identical
frames, preventing a static talking head from producing twenty copies of one
image.

### Escape hatches

A lexicon is a heuristic and will miss things.

- `--uniform` ignores ranking and samples across the whole video, for poor
  transcripts or unusual vocabulary.
- `extract-frames --at 12:04 --window 30s` pulls additional frames on demand
  during analysis.

Ranking is a starting point, not a constraint.

## 7. Knowledge artifacts

Each processed video produces two artifacts.

### Lesson record

`training/lessons/lesson_NNN.md` — the audit trail. Plan §6's template, with
YAML frontmatter for machine-readable fields (source, file hash, duration, date
processed, topics, segment references) and the narrative sections in the body:
rules taught, visual examples, defects shown, PSA grade examples, instructor
opinions, objective observations, potential contradictions, source timestamps,
candidate knowledge-base updates.

It preserves what the video said in context. Nothing consumes it
programmatically; it exists so any rule can be traced to its origin.

### Rules

`knowledge/pending_rules/<rule_id>.yaml` — one rule, one testable claim.

```yaml
id: SURFACE_PRINT_LINE_001
category: surface
statement: "Vertical print lines running the length of the card commonly prevent a PSA 10."
evidence_type: experience_based    # objective | experience_based | opinion | unverified | contradicted
confidence: high                   # high | medium | low
applies_to:
  card_types: [chrome, refractor]
  sets: []
sources:
  - lesson: lesson_014
    video_id: yt_abc123
    timestamps: ["12:04-12:38"]
    quote: "I've never seen one of these gem with a line like that."
status: pending                    # pending | active | rejected | superseded
supersedes: null
created: 2026-08-28
rubric_version_added: null
```

`evidence_type` is required and has no default. Plan §30 rule 11 forbids blindly
learning every claim; requiring the field makes classifying an instructor's
opinion an explicit decision rather than an omission.

`sources` is a list. A claim cited by three independent lessons is stronger than
one cited by a single lesson, and that strength is recorded on disk rather than
held in memory.

### Validation

The `validate` stage enforces:

1. YAML parses and conforms to the Pydantic schema
2. Rule IDs are unique and do not collide with active rules
3. Every cited lesson exists
4. **Every cited timestamp falls within the source video's actual duration**
5. `statement` is non-empty and `evidence_type` is set
6. No rule in `pending_rules/` carries `status: active`

Check 4 catches a fabricated citation mechanically, which prompting alone does
not reliably achieve.

Validation also runs duplicate and contradiction detection — normalized
similarity against active rules within the same category — but only raises
flags. Resolution belongs to the user during review.

Nothing is ever deleted. Rejected and superseded rules remain on disk with
`status` changed (plan §30 rule 9).

## 8. Promotion, versioning, and the subsystem A contract

### Review

`card-knowledge review` walks pending rules one at a time in the terminal
(typer + rich). Each rule is shown with its statement, category, evidence type,
confidence, every source with quote and timestamp, and any duplicate or
contradiction flags from validation.

Actions: `accept`, `reject`, `edit`, `defer`, `supersede <id>`.

On accept, the rule moves to `knowledge/rules/<category>/<id>.yaml` with
`status: active` and `rubric_version_added` stamped.

### Rubric generation

`ACTIVE_RUBRIC.md` is generated by `card-knowledge build-rubric` and never
hand-edited. It renders active rules grouped by category, with version, rule
count, and build date in the header.

The YAML rule files are the source of truth; the markdown is a view. Permitting
hand edits would allow the two to drift, leaving the grader's actual beliefs
unknown.

### Versioning

`knowledge/RUBRIC_VERSION` holds a semver string.

- **patch** — wording or clarification; no behavior change
- **minor** — new rules added
- **major** — an active rule changed meaning, was superseded, or was retracted

The distinction is whether an already-issued review would come out differently.
Plan §27 requires determining whether reviewer versions improve; that comparison
is only meaningful when bumps track behavior change rather than file churn.

Every promotion is a git commit, making the history of the grader's beliefs
reviewable.

### Contract with subsystem A

Defined now so B cannot produce something A is unable to consume:

```python
load_active_rubric() -> Rubric
    .version: str                                    # stamped into every review (plan §27)
    .rules: list[Rule]
    .by_category(cat) -> list[Rule]
    .for_card(card_types=[...], sets=[...]) -> list[Rule]
```

`for_card` is what gives `applies_to` its purpose: the reviewer receives
chrome and refractor rules for a chrome card and is not told about paper stock.
Subsystem A imports this and stamps `version` on every review it emits.

## 9. Acquisition and safety

One `acquire` stage, three paths:

| Source | Method |
|---|---|
| Public YouTube | `yt-dlp <url>` |
| Skool / MLP | `yt-dlp --cookies-from-browser <chrome\|brave\|edge\|firefox\|safari>` |
| Local file | `--file <path>` — skips download, hashes for a stable `video_id` |

`--cookies-from-browser` reads the live browser session at run time. Nothing is
written into the repository.

**Failure is loud and terminal.** When authenticated acquisition fails, the
stage records `failed` with yt-dlp's stderr and prints the manual path: play the
lesson in the browser, save it manually, then `acquire --file <path>`.

There is no fallback that scrapes, retries against a protected endpoint, or
works around a player. Plan §4 and §30 rule 13 are binding. The local-file path
exists so that a blocked platform is an inconvenience rather than a dead end.

`card-knowledge doctor` preflights yt-dlp, ffmpeg, and mlx-whisper and reports
what to install. None of the three are present on the target machine at time of
writing.

## 10. Testing

Most of the pipeline is deterministic and testable without media.

- **Unit** — lexicon scoring, window merging and capping, manifest state
  transitions and idempotency, rule schema validation, the timestamp-bounds
  check, duplicate detection, version bump logic, rubric rendering
- **Fixtures** — checked-in transcript JSON and manifests; no media required
- **Golden** — a fixture transcript with an expected `segments.json`, guarding
  against lexicon regressions: tuning the vocabulary immediately shows which
  segment selections changed
- **Mocked boundaries** — yt-dlp and whisper are wrapped and mocked. The tests
  cover the wrapper's parsing and error handling, not the external tool. One
  small ffmpeg-generated clip exercises the real frame-extraction path.

## 11. Definition of done

1. One public YouTube grading video runs end to end and produces
   `lesson_001.md`, reviewed rules, and `ACTIVE_RUBRIC.md` at v0.1.0
2. One Skool video is attempted, and either processes successfully or produces a
   clean documented failure and completes via the `--file` path
3. `load_active_rubric()` exists and returns a versioned rubric
4. Test suite passes; known limitations are documented

## 12. Environment

Verified present: Python 3.14.6, Homebrew 6.0.15, uv 0.11.32, Claude Code 2.1.220.

Not installed, required by this subsystem: `yt-dlp`, `ffmpeg`, `mlx-whisper`.

**Resolved 2026-08-28:** Python 3.14 wheel availability was verified by dry-run
resolution. `mlx-whisper==0.4.3`, `mlx==0.32.2`, `pydantic==2.13.5`,
`typer==0.27.2`, and `imagehash==4.3.2` all resolve on CPython 3.14.6 /
macOS arm64. No Python version pin is required and the transcription fallback
is retained.
