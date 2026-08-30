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
- A calibrated probability. See §14 of this spec.
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
| Score semantics | Ranking heuristic, explicitly not a probability; `null` when unrankable |
| Verdict | Four states: `PASS` / `REVIEW` / `REJECT` / `INSUFFICIENT_IMAGES` |
| Ingestion | Adapters resolve external input; the core never touches HTTP |
| Stage tiers | Image-level stages cached by image hash; candidate-level stages fuse across images |
| Quality split | `preflight` before geometry; `observability` after it |
| Coverage | A versioned `EvidenceCoveragePolicy` decides `PASS` eligibility and the `REVIEW` / `INSUFFICIENT_IMAGES` boundary |
| Canonicalization | Per-field semantic quantization under a versioned scheme; no global float precision |
| Cacheability | Only validated successes; failed attempts recorded separately and never reused |

---

---

## 3. Pipeline

Stages divide into two tiers. **Image-level** stages depend only on the bytes of one image
and are cached by image hash, so the same photograph appearing in two listings is analyzed
once, ever. **Candidate-level** stages fuse across a candidate's images and depend on
which card it is.

```
CandidateInput → adapter → ResolvedCandidate
                                │
     ┌── per image, keyed by image hash ──────────────────────────┐
     │  preflight → geometry → observability → cv_measurements    │
     └────────────────────────────────────────────────────────────┘
                                │
     ┌── per candidate ──────────────────────────────────────────────────────┐
     │  role/context resolution → evidence assembly → heuristic              │
     │        → routing → [vision] → coverage evaluation → combine → verdict │
     └───────────────────────────────────────────────────────────────────────┘
                                │
              PASS / REVIEW / REJECT / INSUFFICIENT_IMAGES
                                │
                     owner's manual inspection
```

### Why preflight and observability are separate

Regional quality and perspective severity are *geometric* properties: "is the bottom-left
corner glared" has no meaning before the card's quad is known and the image is normalized.
Running a single quality stage before geometry would either be limited to whole-image
statistics or create a circular dependency.

So quality splits:

- **`preflight`** — raw-image properties needing no geometry: pixel dimensions, global
  sharpness, global exposure and clipping, file integrity, gross unusability. It can reject
  an image outright (a 200×150 thumbnail supports nothing), which saves the cost of
  attempting geometry.
- **`observability`** — post-geometry, in normalized card coordinates: per-region glare and
  occlusion masks, perspective severity, per-region resolution after rectification, and the
  per-purpose suitabilities (`centering_suitability`, `corner_suitability`,
  `edge_suitability`, `surface_suitability`) plus per-region **detectability**.

Execution remains one card at a time, start to finish. SQLite knows what exists; there is
no manifest and no parallel workflow state.

---

## 4. Stage caching, versioning and history

### Cache identity

```
(stage, input_fingerprint, producer_signature)
```

- **`input_fingerprint`** — canonical hash of *exactly the data the stage consumed*.
- **`producer_signature`** — the stage's own implementation and configuration, which can
  change its output for identical input.

### Downstream fingerprints consume values, not signatures

**A downstream stage fingerprints upstream output *values*, never upstream producer
signatures.** Bumping the CV analyzer creates a new `cv_measurements` row, but if the
measurements and crops a later stage received are unchanged, that stage's fingerprint is
unchanged and its stored result — including an expensive vision assessment — is reused.

| Tier | Stage | Fingerprints | Producer signature |
|---|---|---|---|
| image | `preflight` | image hash | preflight version + config |
| image | `geometry` | image hash, preflight output | geometry version + config |
| image | `observability` | image hash, geometry output | observability version + config |
| image | `cv_measurements` | image hash, geometry output, observability output | CV version + config |
| candidate | `role_context` | image hashes, per-image cv/geometry outputs, supplied metadata | resolver version |
| candidate | `evidence_assembly` | resolved roles/context, per-image outputs consumed | assembly version |
| candidate | `heuristic` | assembled evidence values, applicable rubric rules | scorer version + weights |
| candidate | `routing` | heuristic output, assembled observability, detectability | SMART policy version |
| candidate | `vision` | the canonical evidence manifest actually sent | provider + model + prompt version + material inference params |
| candidate | `coverage` | assembled detectability/observability, applicable rubric rules | coverage policy version |
| candidate | `combine` | heuristic output, optional vision output, coverage output | combination/decision-policy version |

**Mode is not part of `combine`'s fingerprint.** Combine consumes the heuristic assessment,
whatever vision assessment exists, and the coverage evaluation. Mode determined *whether* a
vision assessment exists; it is not itself an input. Two runs presenting combine with
identical inputs must reuse the same result regardless of the mode that produced them.
Mode is recorded on the `routing_decision` and on the `review`.

Routing is stored separately from combination: *whether to call* and *what verdict given
what exists* are different questions. A routing-policy change does not invalidate a combine
whose inputs are unchanged; it may instead cause a card to acquire a vision assessment, at
which point combine's inputs differ naturally.

### Only validated successes are cacheable

A `stage_result` row exists **only** for an output that ran to completion and passed schema
validation. Failures, timeouts, rate-limit rejections, malformed provider responses and
partial results are recorded as `stage_attempt` rows for diagnostics and cost accounting,
and **can never satisfy a cache lookup**. A failed vision call must not suppress a later
successful one.

### Canonicalization

Fingerprints hash a canonical form. **There is no single global float precision.** Each
value is quantized according to its own declared semantic precision before serialization —
a centering ratio measured to ±1.5 percentage points quantizes far more coarsely than a
normalized coordinate, and applying one arbitrary rounding to both either discards real
signal or manufactures spurious cache misses.

The canonicalizer therefore carries a **versioned scheme**: the field-to-precision map,
key ordering, and the set of non-semantic fields excluded. The scheme version participates
in every fingerprint, so changing quantization is a deliberate, traceable invalidation.

### History is never overwritten

Stage results are append-only. A *review* records which stage results produced which
verdict, under which versions. When an analyzer, rubric, prompt or model improves, prior
reviews remain exactly as they were — the basis for later asking whether a change actually
improved anything.

---

## 5. Data model

SQLite is authoritative. The filesystem holds only large artifacts, content-addressed.

```
data/
  card_reviewer.db
  images/<image_hash>/original.*
  crops/<image_hash>/{corners,edges,surface}/...
  artifacts/
```

`data/` is gitignored.

**`candidate`** — source (`flippah` / `manual` / …), listing URL and ID when available,
title, asking price, supplied card metadata, timestamps. Source is recorded and then never
consulted by the grading pipeline.

**`image`** — content hash, path, dimensions, intrinsic metadata. **Bytes only.**

**`candidate_image`** — join table: candidate, image, *supplied* role, source URL, ordering.
The same photo across two listings is stored and analyzed once.

**`stage_result`** — append-only cache of validated successes. Unique on
`(stage, input_fingerprint, producer_signature)`. Carries output JSON, the full version set,
and a timestamp. Image-tier rows reference an image hash; candidate-tier rows reference a
candidate.

**`stage_attempt`** — failures, timeouts, provider errors, malformed responses, with cost
and latency. Diagnostics only; never satisfies a cache lookup.

**`routing_decision`** — policy version, mode, `call_vision`, trigger reasons, and the
fingerprint of the inputs the decision was made on.

**`review`** — one row per screening run. `candidate_id`, mode, **`routing_decision_id`**,
verdict, `psa10_rank_score`, `rankable`, `estimated_psa_grade`, `psa10_candidate`, and
foreign keys to the exact `stage_result` rows used — including the combine and coverage
results, and a nullable `vision_result_id` which is how `OFF` mode is represented rather
than as a missing stage. Append-only.

**`candidate_outcome`** — purchased or not, price, date, status, notes.

**`grading_submission`** — candidate, grader, submission date, service tier, returned date,
grade, certification number, status. **Multiple rows per candidate are expected**: cards get
returned ungraded, resubmitted, cracked and resubmitted, or crossed to another grader.

### Calibration

```
review → candidate → grading_submission
```

yields the heuristic score, the vision assessment, the combined verdict and the actual
grade side by side, filterable by any version.

### Storage abstraction

The pipeline never writes SQL. A repository interface sits between, so moving to Postgres
touches one module. A review exports to JSON for debugging and sharing — interchange, not
persistence.

---

## 6. Ingestion boundary

- **`CandidateInput`** — source, listing URL, listing ID, title, asking price, image URLs
  and/or uploaded images, optional card metadata.
- **`CandidateAdapter`** — resolves external input: fetches images, hashes them, writes them
  to the content-addressed store. **The only component permitted to touch the network.**
- **`ResolvedCandidate`** — stable metadata plus local content-addressed image references.

Core signature: `ReviewPipeline.review(ResolvedCandidate, mode) -> CardReview`. The CV and
grading core has no dependency on eBay, Flippah, HTTP or any external service.

A convenience wrapper `review_card(CandidateInput, mode) -> CardReview` composes adapter and
pipeline for callers wanting one call. Per build plan §24 the reviewer never fetches a
listing on its own behalf. A future Flippah API is a new adapter and nothing else.

---

## 7. Image-tier stages

### 7.1 `preflight`

Raw-image properties requiring no geometry: pixel dimensions, global sharpness (Laplacian
variance), global exposure and clipping, file integrity. Emits a usability verdict that can
reject an image outright — a 200×150 thumbnail supports no grading measurement — saving the
cost of attempting geometry.

### 7.2 `geometry`

Boundary detection → perspective correction → normalized card image → derived crops.
Establishes **one normalized card coordinate system** that every later stage, defect
location and future model output refers to.

Reports `boundary_confidence`. **When detection is unreliable it declines geometry-dependent
work** rather than producing plausible numbers from a bad quad (build plan §11). Emits the
detected quad and the perspective transform as provenance.

### 7.3 `observability`

Post-geometry, in normalized coordinates. Per-region glare and occlusion masks, perspective
severity, per-region effective resolution after rectification, and:

- per-purpose suitability — `centering_suitability`, `corner_suitability`,
  `edge_suitability`, `surface_suitability`
- per-region **detectability**

```
bottom_left.whitening_detectability = LOW   reason = WHITE_BORDER
top_right.whitening_detectability   = HIGH  reason = DARK_PRINTED_BACKGROUND
```

Detectability encodes `CORNERS_COLORED_001` and `EDGES_COLORED_001`: **a white corner cannot
show whitening.** "No whitening observed" with HIGH detectability is meaningful evidence;
the same observation with LOW detectability means almost nothing was learned. This
distinction is what makes honest calibration possible later.

A glare spot, finger, sleeve or top-loader must not condemn a whole image: a photo can be
excellent for centering and useless for surface. Sleeve and top-loader classification is not
reliably solvable with traditional CV — report confidence and `unknown` rather than forcing
a label.

### 7.4 `cv_measurements`

**Contract: emit observations, measurements, detectability, uncertainty, crops and anomaly
candidates. Never verdicts, never grading judgments, never product-specific leniency.**

**Centering** reports measurement, not acceptability:

```
horizontal: 54/46      method: border_geometry
vertical:   51/49      precision: ±1.5 percentage points
```

or, where no reliable reference exists:

```
measurable: false      reason: BORDERLESS_OR_NO_RELIABLE_REFERENCE
```

`CENTERING_NO_OVERMEASURE_001` binds here — do not claim precision the method lacks. So does
`CENTERING_BORDERLESS_001` — never force a border ratio onto a borderless design.
`CENTERING_PRODUCT_LENIENCY_001` does **not** apply here; whether 54/46 passes on Prizm and
fails on Bowman Chrome is the heuristic layer's decision.

**Corners and edges** produce four corner crops and four edge strips per image face, plus
high-contrast anomaly **candidates** — explicitly not defects.

**Surface** produces deterministic enhanced views (CLAHE, sharpened, grayscale,
edge-highlight) **alongside the preserved original**, with reproducible parameters.

Every anomaly candidate records the **enhancement level that surfaced it**, and whether it
is visible in the unenhanced original.

### The enhancement rule

"Never manufacture a defect" is not enforceable — no test can assert it. The enforceable
form is:

> **An anomaly visible only under enhancement may become a *candidate*, but can never
> independently establish an `observed` defect.**

Confirming such a candidate requires corroboration: visibility in the original, a
consistent finding across independent enhancement paths, or an explicit vision judgment that
cites the original. This is mechanically testable and is tested.

---

## 8. Image-role and card-context resolution

`CandidateInput` often supplies neither which photo is the front nor what card it is. Both
are needed: roles drive evidence assembly; card type and set drive
`load_active_rubric().for_card(...)`.

### Role resolution

Precedence, highest first:

1. **Supplied** — an explicit role on `candidate_image`.
2. **Inferred** — from `cv_measurements`: card backs carry markedly different layout
   signatures from fronts (text density, absence of a large central image region, card
   number and copyright blocks).
3. **`unknown`** — when inference is not confident.

The resolved role carries a confidence and its provenance (`supplied` / `inferred` /
`unknown`). An `unknown` role is a first-class state, not an error: the image still
contributes to any measurement that does not depend on knowing the face, and is excluded
from those that do. A candidate with no confidently identified back cannot satisfy coverage
requirements that name the back — which routes it toward `REVIEW`, never toward `REJECT`.

### Card context resolution

Precedence, highest first:

1. **Supplied** — card type and/or set in `CandidateInput` metadata.
2. **Inferred** — from listing title text and, where feasible, visual set signatures.
3. **`unknown`**.

### Rubric behaviour under unknown context

`for_card(card_types=None, sets=None)` returns **every** active rule — the deliberate
`None`-means-unconstrained semantics established in subsystem B. That is the correct
default here: unknown context must not silently narrow the rules applied.

But an unscoped rule set is not the same as a *satisfied* one. Product-conditional rules
cannot be evaluated without context. `CENTERING_PRODUCT_LENIENCY_001` says the same ratio
passes on Prizm and fails on Bowman Chrome — so with product unknown, a borderline centering
measurement is **not resolvable**, and the coverage policy treats it as reduced evidence
rather than as a pass. Unknown context therefore biases toward `REVIEW`, which is the
correct direction under the recall priority.

Resolved roles and context are stored with their confidence and provenance, so a later
review can be compared against one made under better information.

---

## 9. Candidate-level evidence assembly

Fuses per-image results into one view of the card.

- Selects, per measurement purpose, the best available image — the sharpest front for
  surface, the least perspective-distorted front for centering — rather than assuming one
  photo serves everything.
- Merges detectability across images: a corner glared in one photo and clear in another is
  **observable**, and the assembly records which image established that.
- Reconciles conflicting measurements from different photos of the same face, preserving the
  disagreement rather than averaging it away.
- Produces the candidate-level detectability map the coverage policy consumes.

Every assembled value retains the image hash it came from, so any finding traces to a
specific photograph.

---

## 10. Heuristic layer

Consumes assembled evidence against the rubric scoped by
`load_active_rubric().for_card(card_types, sets)` — where subsystem B pays off: a chrome
card receives `SURFACE_SHINY_001`, a paper card does not.

Product-specific grading rules apply here. `CENTERING_PSA10_STANDARD_002` supplies PSA's own
tolerance — *approximately* 55/45 front and 75/25 back, explicitly **not a hard arithmetic
cutoff**, so 56/44 is not an automatic failure.

Outputs per-category assessments with confidence, and its own PSA-10 view, stored separately
from anything the vision provider produces.

---

## 11. Vision judgment layer

### Provider abstraction

`VisionProvider.assess(evidence_manifest) -> Assessment`. Anthropic is the first
implementation. The provider owns model, prompt version and inference parameters; the
pipeline owns the evidence.

### Search aggressively, conclude conservatively

The brief is adversarial (build plan §17): *find every visible reason this card might not
receive a 10*. **But the evidence standard is conservative** — an adversarial prompt must
never promote an artifact into a confirmed defect. Every finding carries a state:

| State | Meaning |
|---|---|
| `observed` | visible and confidently a real feature of the card |
| `suspected` | visible but could be glare, design, or artifact |
| `not_observed` | looked for, adequate evidence, not present |
| `not_assessable` | evidence insufficient to look |

plus category, normalized location, evidence crop references, confidence, severity where
assessable, PSA-10 relevance, and explanation. A suspected print line stays suspected. Under
a recall priority, a wrongly confirmed defect is the expensive error.

### Deference and disagreement

The provider does not restate centering. It interprets what CV cannot: chipping versus
glare, soft corners, print lines, dimples, stains, foil and refractor artifacts, and whether
a suspected defect is part of the card's design. Where it contradicts a measurement it must
say why, and **the disagreement is recorded rather than silently resolved**.

### Insufficient evidence is a first-class answer

Four `image_limitations` rules — and PSA's own video — say surface often cannot be judged
from photographs. Any category may return `insufficient_evidence` without implying a defect.

### Independent gem assessment

Preserved separately for calibration against PSA outcomes:

```
no_visible_psa10_disqualifier | possible_psa10_disqualifier |
visible_psa10_disqualifier    | insufficient_evidence
```

It does **not** own the final verdict.

### Prohibited

No pricing, EV, profit or purchase information reaches the prompt or the output.

---

## 12. Routing and modes

Modes live at the pipeline, not inside the provider.

- **`OFF`** — never calls. Combine cleanly accepts the absence of a vision assessment.
- **`SMART`** — calls on resolvable ambiguity. The eventual default.
- **`DEEP`** — always calls, with a larger evidence budget.

### SMART fires on resolvable ambiguity, not missing information

A provider cannot recover information absent from the pixels. Sending an occluded corner
buys `insufficient_evidence` at cost.

**Worth a call:** an anomaly that could be glare or chipping with both visible; a possible
print line versus a refractor pattern; a questionable soft corner visible in the crop; a CV
candidate needing semantic interpretation; a strong gem candidate worth confirming; genuine
disagreement between independent signals.

**Not worth a call alone:** a fully occluded region; a severely blurred image; surface
hidden by extreme glare; a missing back photo. These propagate an evidence limitation toward
`REVIEW` / `INSUFFICIENT_IMAGES`.

### Evidence manifest

`DEEP` means *maximum useful evidence*, not mechanically every artifact. A deterministic
manifest selects useful originals, normalized faces, relevant corner/edge/surface crops,
enhanced views that add information, measurements, detectability, uncertainty, masks and the
applicable rubric — eliminating duplicates and non-informative artifacts. `DEEP` carries a
larger budget than `SMART`; both select. Everything is preserved locally regardless. **The
manifest actually sent is what the vision cache fingerprints.**

---

## 13. EvidenceCoveragePolicy

Invariant I2 says a card must not `PASS` when a required category is "materially
unassessable". That phrase is not testable. This policy makes it so.

**A versioned, declarative artifact.** Its version participates in the `coverage` stage's
producer signature, so tightening it is a traceable invalidation rather than a silent
behaviour change.

### What it declares

For each grading category the active rubric requires — centering, corners, edges, surface —
the policy states the minimum evidence for that category to count as **assessed**:

| Category | Assessed when |
|---|---|
| `centering` | a face is measurable, or explicitly `BORDERLESS_OR_NO_RELIABLE_REFERENCE` with a resolved comparison basis |
| `corners` | every corner of every required face reaches at least the declared minimum detectability |
| `edges` | every edge of every required face reaches at least the declared minimum detectability |
| `surface` | required faces reach the declared minimum surface suitability |

The policy also declares **which faces are required**. Front-only evidence cannot establish
back condition; `SURFACE_TECHNICAL_DEFECT_001` records that a crease or paper loss on the
back is grade-limiting, so a missing back is a coverage failure, not an omission.

Detectability is not a proxy for absence. A corner with LOW `whitening_detectability` is
**not assessed for whitening**, regardless of whether whitening was found.

### The three outcomes

The policy returns one of:

- **`SUFFICIENT`** — every required category is assessed. `PASS` is permissible.
- **`PARTIAL`** — some categories assessed, at least one not. `PASS` is forbidden. A rough
  score may still be produced from what *was* assessed, carrying its limitations. Verdict
  floor is `REVIEW`.
- **`INADEQUATE`** — too little assessed to speak to PSA-10 candidacy at all. Verdict is
  `INSUFFICIENT_IMAGES`; the card is not rankable.

**This is the REVIEW / INSUFFICIENT_IMAGES boundary**, and it is a declared threshold rather
than a judgment call: `PARTIAL` means *we learned something but not enough to pass the card*;
`INADEQUATE` means *we could not evaluate*. Unknown card context (§8 of this spec) reduces coverage for
product-conditional categories and therefore biases toward `PARTIAL`.

Making this an artifact rather than scattered conditionals is what lets I2 be tested
directly: construct a detectability map, run the policy, assert the outcome.

---

## 14. Combination, score and verdict

A pure, versioned function over the heuristic assessment, the optional vision assessment,
and the coverage evaluation.

### `psa10_rank_score`

**A ranking heuristic. Not a probability.** Named so its semantics cannot be misread;
displayed as "PSA 10 Score". It exists to sort a batch so the best candidates are inspected
first. A calibrated probability, if ever derived from outcome data, becomes a **separate
field under a separate model** — never a silent redefinition of this one.

`null` when coverage is `INADEQUATE`:

```
psa10_rank_score:    null
estimated_psa_grade: null
rankable:            false
verdict:             INSUFFICIENT_IMAGES
```

A blurred card must not receive an arbitrary 50/100 and then participate in ranking. Under
`PARTIAL` coverage a score may be produced, with limitations carried alongside.

`estimated_psa_grade` is a coarse range: `10`, `9-10`, `9`, `8-9`, `<=8`, or `null`. The
`<=8` bucket exists because a card with an obvious crease is not an `8-9` candidate and the
scale must be able to say so.

### The four-state verdict

| Verdict | Meaning |
|---|---|
| `PASS` | no observed disqualifier **and** coverage is `SUFFICIENT` |
| `REVIEW` | ambiguity, suspicion, contradiction, or `PARTIAL` coverage |
| `REJECT` | at least one confidently observed PSA-10 disqualifier |
| `INSUFFICIENT_IMAGES` | coverage is `INADEQUATE` |

`INSUFFICIENT_IMAGES` says *we could not evaluate*, not *this looks questionable*. A consumer
needing three states may map it to `REVIEW`; the authoritative record preserves four.

`psa10_candidate` is **derived from the verdict**, never computed independently:

| Verdict | `psa10_candidate` |
|---|---|
| `PASS` | `yes` |
| `REVIEW` | `uncertain` |
| `REJECT` | `no` |
| `INSUFFICIENT_IMAGES` | `unknown` |

Deriving it removes any possibility of the two fields disagreeing.

---

## 15. Invariants

**I1 — Ambiguity never rejects.** `REJECT` requires at least one confidently `observed`
PSA-10 disqualifier, adequate evidence for that finding, and no material unresolved
contradiction undermining it. Evidence that is `suspected`, `not_assessable`, conflicting or
low-confidence must never independently produce `REJECT`. **Poor photographs are never
evidence of card damage.**

**I2 — Unassessable never passes.** `PASS` requires `EvidenceCoveragePolicy` to return
`SUFFICIENT`. "We could not see a problem" must never become "the card is clean."

**I3 — Enhancement alone never confirms.** An anomaly visible only under enhancement may be
a `suspected` candidate but can never independently reach `observed`.

---

## 16. Output

```
candidate / listing identity
psa10_candidate:      yes | no | uncertain | unknown        (derived from verdict)
psa10_rank_score:     0-100 | null
rankable:             true | false
estimated_psa_grade:  "10" | "9-10" | "9" | "8-9" | "<=8" | null
coverage:             SUFFICIENT | PARTIAL | INADEQUATE, per-category detail
centering / corners / edges / surface   — assessment, confidence, detectability
image_quality         — per-image preflight, per-region observability, masks
roles_and_context     — resolved values with confidence and provenance
defects_found         — state, category, normalized location, evidence refs,
                        enhancement level, confidence
limitations           — what could not be assessed and why
opencv_assessment     — stored independently
vision_assessment     — stored independently, incl. independent gem view; null in OFF
final_recommendation: PASS | REVIEW | REJECT | INSUFFICIENT_IMAGES
reasoning             — concise, evidence-anchored
versions              — CV, rubric, scorer, routing policy, coverage policy,
                        canonicalization scheme, provider/model, prompt
```

A human-readable report renders the same record (build plan §21).

---

## 17. CLI

```
card-review screen <candidates> --mode off|smart|deep    batch screening
card-review deep <review_id>                              full review of one card
card-review show <review_id>                              human-readable report
card-review outcome <candidate_id> --grade 10 --cert ...  record a PSA result
card-review export <review_id>                            JSON interchange
card-review provider-smoke                                manual, credentialed, never in CI
```

All wrap the same application service. No grading logic lives in the CLI.

---

## 18. Testing

**Synthetic image generation is its own implementation task**, not a subtask of the CV work.
It is the only honest way to test a measurement engine — real photographs have no ground
truth — and it is a substantial piece of software in its own right: rendering cards with
known centering, known border widths, known corner damage, controlled rotation, perspective
distortion, uneven lighting, partial glare, white and dark borders, borderless designs,
front/back orientation and varying resolution. It gets its own task, its own tests and its
own review.

**Synthetic tests** assert measured values within *declared tolerance*, never exact floats.

**Golden tests** — a small set of committed real photographs guarding against regressions on
real-world noise. **They assert observations, never grades**: boundary detection succeeds or
fails as expected, centering falls within tolerance, expected regions are measurable or not,
glare masks appear where expected, crops generate correctly, known anomaly candidates remain
detectable. A photograph is never labelled "this is a PSA 10" — subjective grading opinion
must not become fake ground truth. Actual PSA results live in the calibration dataset.

**Pure logic** — canonicalization and quantization, fingerprinting, cache validity,
validated-success-only caching, role and context resolution, evidence assembly, coverage
policy, routing policy, scoring and verdict logic. Table-driven, including I1, I2 and I3.

**Provider contract** — offline tests against saved fixture payloads and responses: evidence
serialization, structured-response parsing, malformed responses, missing fields, provider
errors, timeouts, rate limits, schema validation, retry, and that a failed attempt never
becomes a cache hit.

**No automated test or CI run ever calls the Anthropic API.** `VisionProvider` is mocked in
every pipeline test. `card-review provider-smoke` is the only path to a real call.

---

## 19. Definition of done

1. A candidate with front and back images runs end to end in `OFF` mode and produces a
   complete `CardReview` with a verdict, persisted to SQLite.
2. `SMART` invokes the provider only on resolvable ambiguity; `DEEP` always; `OFF` never —
   each demonstrated.
3. Re-running a screened card reuses cached stage results. A stored vision assessment
   survives a crash between the API call and combination. A *failed* provider attempt never
   satisfies a later cache lookup.
4. Bumping the CV analyzer creates a new `cv_measurements` result without re-billing a
   vision call whose evidence manifest is unchanged.
5. The same image supplied by two different candidates is analyzed once at the image tier.
6. Invariants I1, I2 and I3 hold under table-driven tests.
7. `EvidenceCoveragePolicy` returns `SUFFICIENT` / `PARTIAL` / `INADEQUATE` against
   constructed detectability maps, and `PASS` is unreachable without `SUFFICIENT`.
8. A card with unknown card context still receives every active rule and biases toward
   `REVIEW` rather than `PASS`.
9. A PSA outcome can be recorded and joined back to the review that predicted it.
10. Test suite green; no test calls the API.
