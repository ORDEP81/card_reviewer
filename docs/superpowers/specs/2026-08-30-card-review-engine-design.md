# Card Review Engine — Design

**Date:** 2026-08-30
**Status:** Awaiting owner approval
**Scope:** Subsystem A of the Card Reviewer project
**Parent plan:** `CARD_REVIEWER_BUILD_PLAN.md` §8–§24, §29, §30
**Consumes:** Subsystem B's rubric via `card_reviewer.knowledge.load_active_rubric()`

---

## 1. Purpose

Answer one question about a specific raw card, from listing photographs:

> **Does this particular copy have a realistic chance of grading PSA 10?**

It does not reproduce PSA grading, and does not need to. It produces a rough grade
estimate, a PSA-10 candidacy ranking, a defect report, an explicit account of what the
photographs could and could not support, and a screening verdict. The owner manually
inspects every card that survives.

### The governing asymmetry

**Missing a legitimate PSA-10 candidate is worse than forwarding a few extra cards.**
A false rejection loses money invisibly; a false pass costs a minute of inspection. Every
threshold, default and ambiguity in this system resolves toward recall.

Its mirror matters equally: **absence of visible damage is not evidence of a clean card.**
A category that could not be assessed must never be scored as if it passed.

### Out of scope

- Pricing, EV, market value, profitability, buy/sell recommendations — anywhere, including
  the prompt sent to the vision provider. (build plan §30 rule 14.) Asking price may be carried in
  candidate metadata but never reaches the grading path.
- Discovering or crawling listings. Candidates arrive from Flippah or manual entry.
- A calibrated probability. See §12 of this spec.
- Autograph condition. See `docs/scope-decisions.md`.

---

## 2. Decisions taken

| Decision | Choice |
|---|---|
| Usage shape | Two levels of analysis, one pipeline |
| Execution | Single pass per card; SQLite is the sole state authority |
| Resume granularity | Per stage, via content-addressed cache |
| Vision judgment | Anthropic API now, behind a provider interface |
| AI modes | `OFF` / `SMART` / `DEEP`; SMART the eventual default |
| Storage | SQLite for records; filesystem for image artifacts |
| Score semantics | Ranking heuristic, explicitly not a probability |
| Verdict | Four states: `PASS` / `REVIEW` / `REJECT` / `INSUFFICIENT_IMAGES` |
| Ingestion | Adapters resolve external input; the core never touches HTTP |

---

## 3. Pipeline

```
CandidateInput  →  ingestion/adapter  →  ResolvedCandidate
                                              │
        ┌─────────────────────────────────────┘
        ▼
    quality  →  opencv  →  heuristic  →  [routing]  →  [claude]  →  combine  →  verdict
                                                                                   │
                                                            PASS / REVIEW / REJECT / INSUFFICIENT_IMAGES
                                                                                   │
                                                                        owner's manual inspection
```

Each stage is a pure function from declared inputs to a stored output. Execution is one
card at a time, start to finish. There is no manifest, no packet directory and no parallel
workflow state: SQLite knows what exists.

---

## 4. Stage caching, versioning and history

### Cache identity

Every stage result is keyed by:

```
(stage, input_fingerprint, producer_signature)
```

- **`input_fingerprint`** — a canonical hash of *exactly the data the stage consumed*.
- **`producer_signature`** — the stage's own implementation and configuration, which can
  change its output for identical input.

### The rule that makes this work

**A downstream stage fingerprints upstream output *values*, never upstream producer
signatures.**

This is what keeps expensive work cheap. Bumping the OpenCV analyzer creates a new
`opencv` row (its own signature changed) — but if v1.3 and v1.4 produce identical
measurements, crops and evidence, the `claude` fingerprint is unchanged and the stored
assessment is reused. If the new version changes a measurement Claude actually received,
the fingerprint changes naturally and the call correctly re-runs.

| Stage | Fingerprints | Producer signature |
|---|---|---|
| `quality` | image hashes | quality analyzer version + config |
| `opencv` | image hashes | CV analyzer version + config |
| `heuristic` | measurement values, quality values, applicable rubric rules | scorer version + weights |
| `claude` | the canonical evidence manifest actually sent | provider + model + prompt version + material inference params |
| `combine` | heuristic output, optional Claude output, mode | combination/decision-policy version |
| `routing` | heuristic output, quality values, detectability | SMART policy version |

Routing is recorded separately from combination: *whether to call* and *what verdict given
what exists* are different questions. A routing-policy change does not invalidate a combine
whose actual inputs are unchanged — it may instead cause a card to acquire a Claude
assessment, at which point combine's inputs differ naturally.

### Canonical serialization

Fingerprints hash a canonical form: fixed key order, floats rounded to a declared
precision, non-semantic fields excluded. Without this, `52.0000001` versus `52.0` produces
a spurious cache miss and a needless API charge. **The rounding precision is itself a
versioned decision** and belongs to the canonicalizer's signature.

### History is never overwritten

Stage results are append-only. A *review* is a row recording which stage results produced
which verdict, under which versions. When an analyzer, rubric, prompt or model improves,
prior reviews remain exactly as they were — which is the entire basis for later asking
whether a change actually improved anything.

---

## 5. Data model

SQLite is authoritative. The filesystem holds only large artifacts, content-addressed.

```
data/
  card_reviewer.db
  images/<image_hash>/original.*
  crops/<image_hash>/{front,back}/{corners,edges,surface}/...
  artifacts/
```

`data/` is gitignored.

### Tables

**`candidate`** — source (`flippah` / `manual` / …), listing URL and ID when available,
title, asking price, card metadata, timestamps. Source is recorded and then never consulted
by the grading pipeline.

**`image`** — content hash, path, dimensions, intrinsic metadata. **Bytes only.** An image
is an artifact, not a listing's property.

**`candidate_image`** — join table: candidate, image, role (`front` / `back` / `unknown`),
source URL, ordering. Role is a relationship, not an intrinsic image property. The same
photo appearing in two listings is stored and analyzed once.

**`stage_result`** — append-only cache. Unique on `(stage, input_fingerprint,
producer_signature)`. Carries the output JSON, the full version set for reproducibility,
and a timestamp.

**`routing_decision`** — policy version, mode, `call_claude`, trigger reasons, the
fingerprint of the inputs the decision was made on.

**`review`** — one row per screening run. `candidate_id`, mode, verdict,
`psa10_rank_score`, `estimated_psa_grade`, `rankable`, and foreign keys to the exact
`stage_result` rows used — including the combine result, and a nullable `claude_result_id`
which is how `OFF` mode is represented rather than as a missing or failed stage.
Append-only.

**`candidate_outcome`** — purchased or not, purchase price, purchase date, status, notes.

**`grading_submission`** — candidate, grader, submission date, service tier, returned date,
grade, certification number, status. **Multiple rows per candidate are expected.** Cards get
returned ungraded, resubmitted, cracked and resubmitted, or sent to another grader. The
schema must not encode "a card is graded once".

### Calibration

A join, not a migration:

```
review → candidate → grading_submission
```

yields the heuristic score, the Claude assessment, the combined verdict and the actual
grade side by side, filterable by any version — which makes "did CV v1.4 improve anything?"
an answerable question.

### Storage abstraction

The pipeline never writes SQL. A repository interface (`find_stage_result`, `save_review`,
`record_outcome`, …) sits between, so moving to Postgres touches one module. An individual
review exports to JSON for debugging and sharing — interchange, not persistence.

---

## 6. Ingestion boundary

Three types, deliberately separated:

- **`CandidateInput`** — what a caller supplies: source, listing URL, listing ID, title,
  asking price, image URLs and/or uploaded images, optional card metadata.
- **`CandidateAdapter`** — resolves external input: fetches images, hashes them, writes
  them to the content-addressed store. **This is the only component permitted to touch the
  network.**
- **`ResolvedCandidate`** — stable metadata plus local content-addressed image references.

The core signature is `ReviewPipeline.review(ResolvedCandidate, mode) -> CardReview`. The
CV and grading core has no dependency on eBay, Flippah, HTTP or any external service.

A convenience application function `review_card(CandidateInput, mode) -> CardReview` wraps
adapter plus pipeline for callers who want one call. Per build plan §24, the reviewer never fetches a
listing on its own behalf; resolving a URL into images is the adapter's job.

If Flippah later exposes an API, that is a new adapter and nothing else changes.

---

## 7. Quality layer

Runs first, and reports **suitability per measurement and per region**, not a single global
verdict.

Emits: resolution, Laplacian-variance sharpness, clipped-highlight glare with masks,
occlusion masks, perspective severity, and per-purpose suitability —
`centering_suitability`, `corner_suitability`, `edge_suitability`, `surface_suitability`.

A glare spot, a finger, a sleeve or a top-loader must not automatically condemn the whole
card: a photo can be excellent for centering and useless for surface. Individual analyzers
decline only the measurements they cannot support. The *review* becomes
`INSUFFICIENT_IMAGES` when the available evidence cannot support a PSA-10 judgment — a
decision made at combination, not by any single analyzer.

Sleeve, top-loader and finger classification is not reliably solvable with traditional CV.
Report confidence, and `unknown` where warranted, rather than forcing a label.

---

## 8. OpenCV measurement layer

**Contract: emit observations, measurements, detectability, uncertainty, crops and anomaly
candidates. Never verdicts, never grading judgments, never product-specific leniency.**

### Geometry

Boundary detection → perspective correction → normalized card image → derived crops.
Everything downstream depends on the first step, so it reports `boundary_confidence`, and
**when detection is unreliable it declines geometry-dependent measurements** rather than
producing plausible numbers from a bad quad (build plan §11).

A single normalized card coordinate system is established after correction. Corners, edges,
defect locations and any future model output all refer to it.

### Centering

Reports measurement, not acceptability:

```
horizontal: 54/46
vertical:   51/49
method:     border_geometry
precision:  ±1.5 percentage points
detectability: high
```

Where no reliable reference exists:

```
measurable: false
reason: BORDERLESS_OR_NO_RELIABLE_REFERENCE
```

Two rubric rules bind here. `CENTERING_NO_OVERMEASURE_001` — do not claim precision the
method does not possess; if uncertainty makes 50.5/49.5 and 51/49 indistinguishable, say
so. `CENTERING_BORDERLESS_001` — never force a border ratio onto a design without borders.
`CENTERING_PRODUCT_LENIENCY_001` is **not** applied here: whether 54/46 passes on Prizm but
fails on Bowman Chrome is the heuristic layer's decision.

### Corners, edges, surface

Eight corner crops and eight edge strips across front and back where available, plus
high-contrast anomaly **candidates** — explicitly not defects. Distinguishing chipping from
glare is judgment, not measurement.

Surface produces deterministic enhanced views (CLAHE, sharpened, grayscale, edge-highlight)
**alongside the preserved original**, never replacing it, with reproducible parameters.
Enhancement must never manufacture a defect (build plan §15). **An anomaly surfaced only under
aggressive enhancement is tagged with the enhancement level that revealed it**, so any
downstream consumer can compare it against the original rather than assuming it is real.

### Detectability

Per region, the layer reports what could be *learned*, not only what was *found*:

```
bottom_left.whitening_detectability = LOW   reason = WHITE_BORDER
top_right.whitening_detectability   = HIGH  reason = DARK_PRINTED_BACKGROUND
```

This encodes `CORNERS_COLORED_001` and `EDGES_COLORED_001`: **a white corner cannot show
whitening.** "No whitening observed" with HIGH detectability is meaningful evidence; the
same observation with LOW detectability means almost nothing was learned. Downstream layers
decide the weight. This distinction is essential for honest calibration later.

### Provenance

Every measurement and crop retains: source image hash, detected quad, boundary confidence,
perspective transform, normalized coordinate system, analyzer version, measurement method,
uncertainty, crop coordinates and hashes, and relevant masks.

---

## 9. Heuristic layer

Combines measurements, quality and detectability against the rubric scoped by
`load_active_rubric().for_card(card_types, sets)` — which is where Subsystem B pays off: a
chrome card receives `SURFACE_SHINY_001`, a paper card does not.

This is where product-specific grading rules apply. `CENTERING_PRODUCT_LENIENCY_001` means
the same ratio is judged differently by product; `CENTERING_PSA10_STANDARD_002` supplies
PSA's own tolerance, stated as *approximately* 55/45 front and 75/25 back and therefore
**not a hard arithmetic cutoff** — 56/44 is not an automatic failure.

Outputs a per-category assessment with confidence, and its own PSA-10 view, stored
separately from anything the vision provider produces.

---

## 10. Vision judgment layer

### Provider abstraction

`VisionProvider.assess(evidence_manifest) -> Assessment`. Anthropic is the first
implementation. The provider owns model, prompt version and inference parameters; the
pipeline owns the evidence. Substituting a provider does not touch the pipeline.

### Search aggressively, conclude conservatively

The brief is adversarial (build plan §17): *find every visible reason this card might not receive a
10*, never "does this look like a 10". Optimism bias is the costly failure.

**But the evidence standard is conservative.** An adversarial prompt must never promote an
artifact into a confirmed defect. Every finding carries a state:

| State | Meaning |
|---|---|
| `observed` | visible in the evidence and confidently a real feature of the card |
| `suspected` | visible but could be glare, design, or artifact |
| `not_observed` | looked for, adequate evidence, not present |
| `not_assessable` | evidence insufficient to look |

plus category, location in normalized coordinates, evidence crop references, confidence,
severity where assessable, whether it is PSA-10-relevant, and an explanation. A suspected
print line stays suspected. Under a recall priority, a wrongly confirmed defect is the
expensive error.

### It defers to measurement

The provider does not restate centering. It interprets what CV cannot: chipping versus
glare, soft corners, print lines, dimples, stains, foil and refractor artifacts, and
whether a suspected defect is part of the card's design. Where it contradicts a
measurement it must state why, and **the disagreement is recorded rather than silently
resolved**.

### Insufficient evidence is a first-class answer

Four `image_limitations` rules — and PSA's own video — say surface often cannot be judged
from photographs. Any category may return `insufficient_evidence` without that implying a
defect.

### Independent gem assessment

The provider returns its own view, preserved separately so it can be evaluated against PSA
outcomes independently of the combined verdict:

```
no_visible_psa10_disqualifier | possible_psa10_disqualifier |
visible_psa10_disqualifier    | insufficient_evidence
```

It does **not** own the final verdict.

### Prohibited

No pricing, EV, profit or purchase information reaches the prompt or the output.

---

## 11. Routing and modes

Modes live at the pipeline, not inside the provider.

- **`OFF`** — never calls. `combine` cleanly accepts the absence of a Claude assessment;
  this is not a missing or failed stage.
- **`SMART`** — calls on resolvable ambiguity. The eventual default.
- **`DEEP`** — always calls, with a larger evidence budget.

### SMART fires on resolvable ambiguity, not on missing information

A provider cannot recover information that is not in the pixels. Sending an occluded corner
or a severely blurred surface buys `insufficient_evidence` at API cost.

**Worth a call:** an anomaly that could be glare or chipping with both visible; a possible
print line versus a refractor pattern; a questionable soft corner visible in the crop; a CV
candidate needing semantic interpretation; a strong gem candidate worth confirming before
it reaches the owner; meaningful disagreement between independent signals.

**Not worth a call on their own:** a completely occluded region; a severely blurred image;
surface hidden by extreme glare; a missing back photo. These propagate an evidence
limitation toward `REVIEW` / `INSUFFICIENT_IMAGES`.

### Evidence manifest

`DEEP` means *maximum useful evidence*, not mechanically every artifact. A deterministic
evidence manifest selects: useful original photos, normalized front and back, relevant
corner, edge and surface crops, enhanced views where they add information, measurements,
detectability and uncertainty, masks, and the applicable rubric — eliminating duplicates
and non-informative artifacts. `DEEP` carries a larger budget than `SMART`; both select.

Everything is preserved locally regardless. **The manifest actually sent is what the Claude
cache fingerprints.**

---

## 12. Combination, score and verdict

A pure, versioned function over the heuristic assessment and the optional provider
assessment. It owns two outputs neither input owns.

### `psa10_rank_score`

**A ranking heuristic. Not a probability.** Named `psa10_rank_score` so the semantics
cannot be misread; displayed as "PSA 10 Score". It exists to sort a batch so the best
candidates are inspected first.

If a genuinely calibrated probability is derived later from outcome data, it is stored as a
**separate field under a separate model** — never by silently redefining this historical
score.

### Non-rankable cards

When evidence is materially insufficient, the system does not manufacture a number:

```
psa10_rank_score:     null
estimated_psa_grade:  null
rankable:             false
verdict:              INSUFFICIENT_IMAGES
```

A blurred card must not receive an arbitrary 50/100 and then participate in ranking.
*Partial* evidence is different: where enough categories are observable to support a rough
score, the card is scored with its limitations and confidence carried alongside.

`estimated_psa_grade` is a coarse range — `10`, `9-10`, `9`, `8-9` — for the same reason.

### The four-state verdict

| Verdict | Meaning |
|---|---|
| `PASS` | no observed disqualifier, and adequate detectability in the categories that matter |
| `REVIEW` | ambiguity, suspicion, contradiction, or partial evidence |
| `REJECT` | at least one confidently observed PSA-10 disqualifier |
| `INSUFFICIENT_IMAGES` | the evidence cannot support a PSA-10 judgment at all |

`INSUFFICIENT_IMAGES` is distinct from `REVIEW`: it says *we could not evaluate*, not *this
looks questionable*. A downstream consumer needing three states may map it to `REVIEW`
operationally, but the authoritative record preserves four.

---

## 13. Invariants

Two properties are enforced by dedicated tests and must never regress.

**I1 — Ambiguity never rejects.** `REJECT` requires at least one confidently `observed`
PSA-10 disqualifier, adequate evidence for that finding, and no material unresolved
contradiction undermining it. Evidence that is `suspected`, `not_assessable`, conflicting
or merely low-confidence must never independently produce `REJECT`. **Poor photographs are
never evidence of card damage.**

**I2 — Unassessable never passes.** No card may receive `PASS` when a grading category
required by the active rubric is materially unassessable. "We could not see a problem" must
never become "the card is clean."

---

## 14. Output

```
candidate / listing identity
psa10_candidate:        yes | no | uncertain
psa10_rank_score:       0-100 | null
rankable:               true | false
estimated_psa_grade:    "10" | "9-10" | "9" | "8-9" | null
centering / corners / edges / surface   — assessment + confidence + detectability
image_quality           — per-region suitability, masks
defects_found           — state, category, normalized location, evidence refs, confidence
limitations             — what could not be assessed and why
opencv_assessment       — stored independently
claude_assessment       — stored independently, incl. independent gem view; null in OFF
final_recommendation:   PASS | REVIEW | REJECT | INSUFFICIENT_IMAGES
reasoning               — concise, evidence-anchored
versions                — CV, rubric, scorer, routing policy, provider/model, prompt
```

A human-readable report renders the same record (build plan §21).

---

## 15. CLI

```
card-review screen <candidates>   --mode off|smart|deep    batch screening
card-review deep <review_id>                                full review of one card
card-review show <review_id>                                human-readable report
card-review outcome <candidate_id> --grade 10 --cert ...    record a PSA result
card-review export <review_id>                              JSON interchange
card-review provider-smoke                                  manual, credentialed, never in CI
```

All wrap the same application service. No grading logic lives in the CLI.

---

## 16. Testing

**Synthetic** — generated cards with known geometry are the only honest way to test the
measurement engine. Known centering, border widths and corner damage; assert measured
values within *declared tolerance*, not exact floats. Must include rotation, perspective
distortion, uneven lighting, partial glare, white borders, dark borders, borderless
designs, front/back orientation and varying resolution.

**Golden** — a small set of committed real card photographs, guarding against regressions
on real-world noise. **These assert observations, never grades.** Boundary detection
succeeds or fails as expected; centering falls within tolerance; expected regions are
measurable or not; glare masks appear where expected; crops generate correctly; known
anomaly candidates remain detectable. A photograph is never labelled "this is a PSA 10" —
subjective grading opinion must not become fake ground truth. Actual PSA results live in the
calibration dataset, not the CV test suite.

**Pure logic** — fingerprinting, cache validity, routing policy, scoring and verdict logic
are pure functions with table-driven tests, including I1 and I2.

**Provider contract** — offline tests against saved fixture payloads and responses cover
evidence serialization, structured-response parsing, malformed responses, missing fields,
provider errors, timeouts, rate limits, schema validation, retry and cache behaviour.

**No automated test or CI run ever calls the Anthropic API.** `VisionProvider` is mocked in
every pipeline test. `card-review provider-smoke` is the only path to a real call, run
deliberately with credentials, never as part of the suite.

---

## 17. Definition of done

1. A candidate with front and back images runs end to end in `OFF` mode and produces a
   complete `CardReview` with a verdict, persisted to SQLite.
2. `SMART` invokes the provider only on resolvable ambiguity; `DEEP` always invokes it;
   `OFF` never does — each demonstrated.
3. Re-running a screened card reuses cached stage results; a stored Claude assessment
   survives a crash between the API call and combination.
4. Bumping the CV analyzer creates a new `opencv` result without re-billing a Claude call
   whose evidence is unchanged.
5. Invariants I1 and I2 hold under table-driven tests.
6. A PSA outcome can be recorded and joined back to the review that predicted it.
7. Test suite green; no test calls the API.
