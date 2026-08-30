---
lesson_id: lesson_014
source: correction pass following an independent retrospective review
date_processed: 2026-08-30
topics: [centering, corners, surface, process]
video_id: yt_E9do7O74zM0
references:
  - knowledge/psa/grading-standards.md
  - training/lessons/lesson_013.md
---

# Correction pass: three commits that bypassed review

## THE SOURCE TEXT THIS PASS CITES

Rules corrected here cite this lesson, so the text they rest on is reproduced in full.
It is identical to the block in `knowledge/psa/grading-standards.md` and in lesson_013;
that file carries the provenance warning and governs.

**PSA, Grade Definitions → GEM-MT → PSA 10:**

> A PSA Gem Mint 10 card is a virtually perfect card.
>
> Attributes include four perfectly sharp corners, sharp focus and full original gloss.
> A PSA Gem Mint 10 card must be free of staining of any kind, but an allowance may be
> made for a slight printing imperfection, if it doesn't impair the overall appeal of
> the card. The image must be centered on the card within a tolerance not to exceed
> approximately 55/45 percent on the front, and 75/25 percent on the reverse.

**PSA Official video** (`yt_E9do7O74zM0`, 00:46-01:20), cited by
CORNERS_MINOR_TOUCH_COSTS_TEN_002:

> The differences between a PSA 10 and a PSA 9 can be slight. For example, these two
> cards look nearly identical at first glance with strong corners, smooth edges, and
> unbllemished surfaces, but this card falls outside the centering threshold for a PSA
> 10. These two cards are well centered and have great eye appeal. At first glance, they
> both appear to be PSA 10 candidates, but closer inspection reveals a small white touch
> on one corner. A minor defect like this can be enough to bring down the grade.

That rule's quote elides the centering pair with an ellipsis. The framing sentence covers
both examples and the conclusion is unaffected, but the elision is recorded here.

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
That is false — but the argument first offered for rejecting it was itself circular, and
is withdrawn here.

**The withdrawn argument.** It claimed the gate held because `rubric_version_added` is
only set by `promote.accept` and `supersedes` only by `promote.supersede`. That reasons
from which code path *writes* a field, not from evidence about how a given file came to
exist. Both are plain optional fields on the `Rule` model, and `validate.check_rule` runs
only over `pending_rules/` — never over files already in `knowledge/rules/`. Nothing makes
them tamper-evident; anyone hand-writing YAML could type them. Against the specific
scenario alleged, the argument assumed its conclusion.

**The evidence that does hold.** `promote.write_rule` emits a narrow canonical
serialization — `model_dump(mode="json")`, sources re-dumped with `exclude_none`, then
`yaml.safe_dump(..., sort_keys=False)`. Reproducing that over every rule file on disk
gives **48 of 49 byte-identical** to what `write_rule` would emit; the one exception
predates the `exclude_none` behaviour. Hand-edited YAML would not reproduce that
fingerprint. And `promote._drop_pending` unlinks the pending file, which fully explains
`pending_rules/` being empty in git history.

The machine gate held. The human gate did not.

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

**The slight/prominent boundary is now load-bearing and undefined.** PSA allows a
"slight" printing imperfection; SURFACE_PRINT_LINE_PROMINENT_001 restricts a "prominent"
one. Neither term is operationalised, and an engine must decide which side a given line
falls on. The rule's quote carries a calibrating example — a line running between a
player's mouth and nose — but that is an anchor, not a threshold.

**SURFACE_NO_STAINING_001 was not updated.** Its statement and quote are faithful, so it
had no wording defect, but it is the one of the six screenshot-derived rules that still
cites the standard without pointing at `knowledge/psa/grading-standards.md`. Left as-is
deliberately: superseding a correct rule to improve a pointer would cost a major version
for cosmetics. Worth folding into the next substantive centering change.

Only PSA's PSA 10 definition was transcribed. The definitions for PSA 9 and below sit on
the same page and are not recorded, so the rubric still cannot say what centering a 9 or
an 8 permits — a real gap for an engine asked to estimate probabilities across grades.

## THE LESSON

The first failure, recorded in lesson_012, was treating agreement between two practitioners
as verification. This one is narrower and more uncomfortable: deciding unilaterally that a
rule did not apply to the present case, and not saying so. The reasoning — data is not code
— was plausible enough to feel like judgment rather than an exemption. It was an exemption.
