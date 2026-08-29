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

## Known limitations

`card-knowledge status` shows six stages (`acquire`, `transcribe`, `segment`,
`extract_frames`, `analyze`, `validate`), but only the first four are ever
written to a packet's manifest — they are the deterministic, automated
stages. `analyze` (Claude, via the `learn-video` skill) and `validate` (a
human running `card-knowledge validate`/`review`) are judgment calls with no
code that stamps a manifest when they happen, so `status` always renders
them `n/a` rather than a possibly-misleading `pending`. A packet that has
been fully analyzed and promoted looks identical, in `status`, to one that
has not been analyzed yet.

This does not mean that work is untracked — it means its provenance lives
elsewhere: which lesson a video produced is recorded in the lesson file
itself (`training/lessons/lesson_NNN.md`, which cites the `video_id`), and
what happened to every rule extracted from it — pending, accepted, rejected,
superseded, and why — is recorded permanently in `knowledge/rules/` and
`knowledge/pending_rules/` (rules are never deleted, only re-statused; see
`CLAUDE.md`).

## Testing

    uv run pytest
