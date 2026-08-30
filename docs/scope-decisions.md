# Scope decisions

Durable record of what this repository deliberately does not cover, and why.
Each entry exists because the question came up against real material.

## Autographs are out of scope — 2026-08-29

**Decision:** autograph condition is not modelled. No `autograph` category is added to
`models.Category`, and no rules are extracted from autograph training material.

**Why:** the reviewer's stated job is to estimate the probability of each PSA grade.
PSA does not grade the autograph by default — it is a separately purchased grade. The
MLP course states this plainly: *"If you're submitting cards to PSA, PSA does not grade
the autograph."* So autograph quality does not move the number this system predicts.

The reason the course teaches it is resale value — a PSA 10 with a good autograph sells
for more than one with a bad autograph. That is market-value reasoning, which
CARD_REVIEWER_BUILD_PLAN.md §30 rule 14 forbids in this repository outright.

**What this costs:** the MLP "Autograph" video (`web_4ca7542ff114`, 5m01) is downloaded and
transcribed but deliberately unanalysed. Its five defect types — streaky, running off card,
running off sticker, white dots, fading — are not recorded anywhere.

**What would reverse it:** deciding the reviewer should also predict the optional PSA/DNA
autograph grade. That is a second prediction target with its own schema, not a category
bolted onto the existing one, and would need its own spec.

## Skool material is not ingested — 2026-08-29

**Decision:** the Skool classroom video was not processed. No further attempt will be made
to fetch Skool content automatically.

**Why:** yt-dlp has no Skool extractor (`ERROR: Unsupported URL`), and the classroom is a
JavaScript application — a request carrying the user's own valid session cookies returned an
HTTP 307 stub rather than the page. This is not an authentication problem that a flag can
solve, and CARD_REVIEWER_BUILD_PLAN.md §4 directs reporting such a failure rather than
working around the platform.

The `--file` path remains available if a Skool lesson is ever wanted: save the video locally
and run `card-knowledge acquire --file <path>`.

**What this costs:** nothing currently. The user judged the existing material sufficient.
