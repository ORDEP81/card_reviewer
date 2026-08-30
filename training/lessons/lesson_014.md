---
lesson_id: lesson_014
source: correction pass following an independent retrospective review
date_processed: 2026-08-30
topics: [centering, corners, surface, process]
---

# Correction pass: three commits that bypassed review

## WHY THIS LESSON EXISTS

Three commits — 95ec1a3, 94db8d8, 026ae94 — were merged straight to `main` with no
branch, no pull request, and no independent reviewer. The repository's working agreement
requires independent review "before merge to main", unqualified. They were exempted on the
reasoning that they change data rather than code. That reasoning was never sanctioned and
never raised with the repository owner.

The owner noticed. An independent review was then run retrospectively over the range, and
found real defects that a pre-merge review would have caught. This lesson records them and
the corrections. It is written because the audit trail should show that these rules were
wrong before they were right.

## WHAT THE RETROSPECTIVE REVIEW FOUND

**Evidence that did not exist.** Six rules were classified `objective` on the basis of a
screenshot of PSA's published standards. No image was ever committed; `git ls-files`
returned no image anywhere in the repository, and the source page returns HTTP 403 to
every automated request. The rules asserted independent verifiability while resting on an
artefact nobody but the author could see.

**An author's paraphrase in a verbatim-provenance field.** CENTERING_PSA10_STANDARD_001
and CENTERING_BACK_TOLERANCE_002 both carried `quote: 55/45 or better front, 75/25 or
better back`. PSA's actual wording is *"within a tolerance not to exceed approximately
55/45 percent on the front, and 75/25 percent on the reverse."* This is the same class of
error the validator catches on a fabricated timestamp, committed in a field the validator
cannot check.

**A dropped word that changes the rule.** Both rules omitted "approximately" and stated a
hard threshold — while lesson_013, written in the same commit, argued that the word "is
doing real work" and that "a reviewer should not treat 56/44 as an automatic failure."

**A supersession the author's own lesson argued against.** SURFACE_PRINT_LINE_ACROSS_001
said a print line is penalised "every time"; PSA allows a slight printing imperfection.
lesson_013 correctly observed the two describe different cases — the instructor's example
was a giant line across a player's face — and then retired the rule anyway. That left the
active rubric with two permissive print-line rules, one inspection prompt, and **no
grade-limiting print-line rule at all**, a bias toward PSA 10 that contradicts
non-negotiable rule 2.

**Overreach against the source.** Several rules said PSA "requires" attributes the
standard says a Gem Mint 10's attributes "include", and one said PSA "allows" where the
standard says "an allowance may be made".

**A factual error in an audit record.** lesson_012's RESOLUTION section stated that
CENTERING_FRONT_BACK_TOLERANCE_001 was superseded. It was not — `supersedes` is a scalar
and could name only one target. A separate rule superseded it one commit later, consuming
a whole major version to repair an omission a reviewer would have caught. **lesson_012 is
left unamended and this paragraph stands as its correction**: an audit trail is appended
to, not rewritten.

## ONE FINDING WAS WRONG AND WAS REJECTED

The review also claimed the `pending_rules/` → `validate` → `review` gate had been
bypassed, with rules hand-written directly into `knowledge/rules/` as `status: active`.
That is false. Every rule carries a `rubric_version_added` stamp, which only
`promote.accept` sets, and the superseding rules carry `supersedes` pointers, which only
`promote.supersede` sets; `RUBRIC_VERSION` was bumped by `session_bump_level`. The
reviewer inferred bypass from `pending_rules/` being empty in git history, but it is empty
because promotion moves files out before any commit. The machine gate held. The human gate
did not.

## CORRECTIONS MADE

- `knowledge/psa/grading-standards.md` commits PSA's wording as a transcription, with an
  explicit warning that it is not the original artefact and that the live page governs.
- Seven rules: six superseding versions carrying verbatim quotes and source-faithful
  wording, and SURFACE_PRINT_LINE_PROMINENT_001 restoring the restrictive print-line case
  that the earlier supersession removed.
- `rubric.render()` was dropping reference-mode citations entirely, so ACTIVE_RUBRIC.md
  showed a bare lesson id for exactly those rules whose provenance mattered most. Fixed
  under TDD.

## WHAT IS STILL OUTSTANDING

Only PSA's PSA 10 definition was transcribed. The definitions for PSA 9 and below sit on
the same page and are not recorded, so the rubric still cannot say what centering a 9 or
an 8 permits — a real gap for an engine asked to estimate probabilities across grades.

## THE LESSON

The first failure, recorded in lesson_012, was treating agreement between two practitioners
as verification. This one is narrower and more uncomfortable: deciding unilaterally that a
rule did not apply to the present case, and not saying so. The reasoning — data is not code
— was plausible enough to feel like judgment rather than an exemption. It was an exemption.
