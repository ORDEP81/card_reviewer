# Working agreement for this repo

The Superpowers skills are a **requirement for producing this application**, not
an optional aid. Invoke them; do not substitute judgement for them.

## Before writing code

**`superpowers:using-git-worktrees`** — branch first. Never start implementation
on `main`.

**`superpowers:test-driven-development`** — write the failing test, watch it
fail, then implement.

## Before merging

**`superpowers:requesting-code-review`** — dispatch an independent reviewer
subagent. **Self-review does not count.** Its own text makes this mandatory
before merge to main.

## When review comes back

**`superpowers:receiving-code-review`** — verify each finding before
implementing it. Push back with reasoning when a finding is wrong; do not
perform agreement.

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
