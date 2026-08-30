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
| Finding states | One vocabulary (`observed` / `suspected` / `not_observed` / `not_assessable`) shared by the heuristic and vision layers |
| Verdict resolution | Strict precedence, first match wins; the four states are mutually exclusive |
| Detectability | Per defect type, and split into structural (no photo could show it) vs circumstantial (a better photo would) |
| Coverage timing | Provisional before routing to gate spend; authoritative after vision |

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
     │  role/context resolution → evidence assembly → heuristic               │
     │        → provisional coverage → routing → [manifest → vision]          │
     │        → authoritative coverage → combine → verdict                    │
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
  sharpness, global exposure and clipping, file integrity, gross unusability. It can mark an
  image **unusable** (a 200×150 thumbnail supports nothing), which saves the cost of
  attempting geometry. An unusable *image* never implies a `REJECT` *verdict*.
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
| candidate | `coverage_provisional` | assembled detectability/observability, applicable rubric rules | coverage policy version |
| candidate | `routing` | **mode**, heuristic output, provisional coverage, assembled observability, detectability | SMART policy version |
| candidate | `manifest` | mode budget, assembled evidence, routing decision | manifest-builder version |
| candidate | `vision` | the canonical evidence manifest actually sent | provider + model + prompt version + material inference params |
| candidate | `coverage` | assembled detectability/observability, vision per-category assessability, applicable rubric rules | coverage policy version |
| candidate | `combine` | heuristic output, optional vision output, coverage output | combination/decision-policy version |

**Mode is an input to `routing`, and only to `routing`.** Routing's output *is* the
decision to call, and that decision is mode-dependent by definition — `OFF` never calls,
`DEEP` always does. Omitting mode from routing's fingerprint would let a card screened in
`OFF` poison the cache for the same card in `DEEP`: identical fingerprint, identical policy
version, so the stored `call_vision = false` would satisfy the `DEEP` lookup and the deep
review would silently never call the provider. Mode is therefore part of routing's
fingerprint.

**Mode is not part of `combine`'s fingerprint.** Combine consumes the heuristic assessment,
whatever vision assessment exists, and the coverage evaluation. Mode determined *whether* a
vision assessment exists; it is not itself an input. Two runs presenting combine with
identical inputs must reuse the same result regardless of the mode that produced them.
Mode is recorded on the `routing_decision` and on the `review`.

### Coverage is evaluated twice

`coverage_provisional` runs on CV evidence alone, before routing, and exists to gate spend:
a card whose photographs cannot support an assessment at all should not buy a `SMART`
vision call. `coverage` runs after vision and is **authoritative**, because the vision layer
can report a category `not_assessable` that CV suitability alone judged fine. Only the
authoritative evaluation feeds combine and the verdict. Both use the same policy artifact
and version; they differ only in whether vision evidence is available to them.

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

### Stage failure policy

A stage that cannot complete never produces a verdict by default. Specifically:

- **A permanently failed vision call** (provider down, budget exhausted, repeated malformed
  responses) leaves `vision_result_id` null and the review proceeds on CV evidence alone,
  exactly as `OFF` would. The failure is recorded as a `stage_attempt` and surfaced in
  `limitations`. It never produces `REJECT`, and it cannot yield `PASS` unless coverage is
  `SUFFICIENT` without it.
- **`load_active_rubric()` raising `RubricError`** aborts the run. There is no verdict
  without a rubric, and guessing at one is worse than declining. The CLI reports the rubric
  problem; no `review` row is written.
- **A failed image-tier stage** marks that image unusable for the purposes that stage
  serves, and the candidate continues on its remaining images. One bad photograph out of
  six must not fail the card.

No failure path may produce `REJECT`. This follows directly from the governing asymmetry.

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
verdict, `psa10_rank_score`, `rankable`, `estimated_psa_grade`, `psa10_candidate`,
`review_confidence`, and foreign keys to the exact `stage_result` rows used — including the
combine result and **both** coverage results (provisional and authoritative), and a nullable `vision_result_id` which is how `OFF` mode is represented rather
than as a missing stage. Append-only.

**`candidate_outcome`** — disposition status, date, notes. **No purchase flag and no
price.** Build plan §26's calibration record asks for images, review, predicted grade and
actual grade; it does not ask for cost. Storing a purchase price in the same schema as
returned grades puts ROI analysis one join away, which rule 14 forbids in this repository.
The asking price on `candidate` is listing metadata that the grading path never reads; the
constraint is enforced mechanically by a test asserting the evidence manifest and the review
output carry no price-derived field, not by prose alone.

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
variance), global exposure and clipping, file integrity. Emits a usability verdict that can mark an
image **unusable** — a 200×150 thumbnail supports no grading measurement — saving the cost
of attempting geometry. Marking an image unusable never contributes toward a `REJECT`
verdict; it reduces coverage, which routes toward `REVIEW` or `INSUFFICIENT_IMAGES`.

### 7.2 `geometry`

Boundary detection → perspective correction → normalized card image → normalized face
crops → **border/art segmentation**. Establishes **one normalized card coordinate system**
that every later stage, defect location and future model output refers to.

Border segmentation belongs here rather than downstream: both `observability` (is this
corner white, and therefore unable to show whitening?) and `cv_measurements` (is there a
reliable border reference to measure centering against?) need it, and neither should own a
result the other depends on. Geometry produces the segmentation once; both consume it as an
upstream value.

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

Detectability is a physical property of the photograph and the card's own design — what
*could* be seen here, independent of any rubric. `CORNERS_COLORED_001` and
`EDGES_COLORED_001` are cited as the provenance of why it is worth measuring, not as its
contract: **a white corner cannot show whitening.** Because detectability is physics rather
than policy, rubric version is deliberately *not* part of the image-tier producer
signatures — a rubric change must not invalidate stored pixel measurements. "No whitening observed" with HIGH detectability is meaningful evidence;
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

`CENTERING_NO_OVERMEASURE_001` — whose own text is *"if the borders look even, treat the
centering as passing"* — is a tolerance judgment that belongs to the heuristic layer; what
binds **here** is its corollary, that a reported measurement must carry the precision its
method actually supports and no more. `CENTERING_BORDERLESS_001` binds directly — never
force a border ratio onto a borderless design.
`CENTERING_PRODUCT_LENIENCY_001` does **not** apply here; whether 54/46 passes on Prizm and
fails on Bowman Chrome is the heuristic layer's decision.

**Corners and edges** produce four corner crops and four edge strips per image, plus
high-contrast anomaly **candidates** — explicitly not defects. (Per image, not per face:
face resolution happens downstream in §8, and each image is treated as depicting one face
whose identity is not yet known.)

**Crop ownership is split by stage and by path.** `geometry` owns normalization output at
`crops/<image_hash>/face/`; `cv_measurements` owns analysis crops at
`crops/<image_hash>/{corners,edges,surface}/`. Each is invalidated by its own stage's cache
and never by the other's.

**Surface** produces deterministic enhanced views (CLAHE, sharpened, grayscale,
edge-highlight) **alongside the preserved original**, with reproducible parameters.

Every anomaly candidate records the **enhancement level that surfaced it**, and whether it
is visible in the unenhanced original.

### The enhancement rule

"Never manufacture a defect" is not enforceable — no test can assert it. The enforceable
form is:

> **An anomaly visible only under enhancement may become a *candidate*, but can never
> independently establish an `observed` defect.**

Confirming such a candidate requires corroboration by one of exactly two routes, both
mechanically checkable:

1. The anomaly is visible in the preserved unenhanced original.
2. A vision finding reports it and **cites the original artifact** as the evidence — the
   cited artifact reference must resolve to an unenhanced image, which the combine stage
   verifies rather than trusts.

Agreement across several enhancement paths is explicitly **not** a third route. Independent
enhancements of the same pixels are not independent evidence — they can raise confidence
*within* `suspected`, but can never move a finding to `observed`. This is what makes I3
testable: construct a candidate present only in enhanced views, assert it cannot reach
`observed` under any combination of enhancements.

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

**Unknown context must be passed as `None`, never as `[]`.** Subsystem B distinguishes the
two deliberately: `None` means unconstrained, while an explicit empty list means *known to
be empty* — a real constraint that would drop every scoped rule and produce exactly the
silent narrowing this section exists to prevent. The resolver returns `None` for unknown,
and a test asserts it.

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

### The finding-state vocabulary

**Both** the heuristic layer and the vision layer emit findings in one shared vocabulary.
Defining it here, upstream of both, is what lets combine, the coverage policy and the
invariants be written once rather than per-producer — and is what makes `OFF` mode
well-defined, since in `OFF` the heuristic is the only producer of findings.

| State | Meaning |
|---|---|
| `observed` | present, and confidently a real feature of the card |
| `suspected` | possibly present; could be glare, design, or artifact |
| `not_observed` | looked for, evidence adequate, not present |
| `not_assessable` | evidence insufficient to look |

Every finding carries state, category, defect type, normalized location, evidence artifact
references, confidence, severity where assessable, PSA-10 relevance, and an explanation.
Findings additionally record their **producer** (`heuristic` or `vision`), so the two are
never conflated in storage or in output.

A heuristic finding may reach `observed` only for defect types that a measurement can
establish outright — a centering ratio outside tolerance with an adequate method, a corner
whose geometry is measurably rounded. Interpretive defect types (chipping versus glare,
print lines, dimples) can reach at most `suspected` from CV alone; promoting them requires
the vision layer. This is what stops `OFF` mode from manufacturing confident defects out of
high-contrast pixels.

---

## 10. Heuristic layer

Consumes assembled evidence against the rubric scoped by
`load_active_rubric().for_card(card_types, sets)` — where subsystem B pays off: a chrome
card receives `SURFACE_SHINY_001`, a paper card does not.

Product-specific grading rules apply here. `CENTERING_PSA10_STANDARD_002` supplies PSA's own
tolerance — *approximately* 55/45 front and 75/25 back, explicitly **not a hard arithmetic
cutoff**, so 56/44 is not an automatic failure.

Outputs per-category assessments with confidence, findings in the §9 vocabulary, and its
own PSA-10 view — stored separately from anything the vision provider produces. Its findings
are subject to the promotion limit stated in §9: measurement-establishable defect types may
reach `observed`; interpretive ones may not.

---

## 11. Vision judgment layer

### Provider abstraction

`VisionProvider.assess(evidence_manifest) -> Assessment`. Anthropic is the first
implementation. The provider owns model, prompt version and inference parameters; the
pipeline owns the evidence.

### Search aggressively, conclude conservatively

The brief is adversarial (build plan §17): *find every visible reason this card might not
receive a 10*. **But the evidence standard is conservative** — an adversarial prompt must
never promote an artifact into a confirmed defect.

Vision findings use the shared finding-state vocabulary defined in §9, with
`producer = vision`. A suspected print line stays suspected. Under a recall priority, a
wrongly confirmed defect is the expensive error.

**Per-category assessability is a required part of the response**, not an optional remark:
for each of centering, corners, edges and surface the provider states whether it could
assess that category at all. This feeds the authoritative coverage evaluation (§13), which
is why a provider saying "I could not judge the surface" cannot be lost.

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

Mode is part of the `routing` stage's fingerprint (§4), which is what keeps an `OFF` run
from satisfying a later `DEEP` lookup.

**The provisional coverage gate.** `SMART` does not call when `coverage_provisional`
returns `INADEQUATE`: a card whose photographs cannot support an assessment at all has
nothing for the provider to resolve, and the verdict is already pinned to
`INSUFFICIENT_IMAGES`. `DEEP` still calls — the owner asked for maximum evidence explicitly,
and a provider occasionally reads a face that CV could not.

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

Manifest construction is its own cached stage (`manifest` in §4), not an unnamed step: it is
deterministic, it is what the `vision` stage fingerprints, and it must be reproducible
independently of whether a call was ultimately made.

`DEEP` means *maximum useful evidence*, not mechanically every artifact. A deterministic
manifest selects useful originals, normalized faces, relevant corner/edge/surface crops,
enhanced views that add information, measurements, detectability, uncertainty, masks and the
applicable rubric — eliminating duplicates and non-informative artifacts. Both modes select against a declared **artifact budget** — v1: `SMART` up to 8 artifacts,
`DEEP` up to 20 — with a fixed priority ordering so selection is deterministic rather than
whatever fits. Everything is preserved locally regardless of what was sent. **The manifest
actually sent is what the vision cache fingerprints.**

---

## 13. EvidenceCoveragePolicy

Invariant I2 says a card must not `PASS` when a required category is "materially
unassessable". That phrase is not testable. This policy makes it so.

**A versioned, declarative artifact.** Its version participates in both coverage stages'
producer signature, so tightening it is a traceable invalidation rather than a silent
behaviour change. Every threshold below is a declared value of policy **v1**, changeable
only by a version bump.

### Declared scales

Detectability and suitability share one ordered scale:

```
NONE < LOW < MODERATE < HIGH
```

**Minimum to count as assessed: `MODERATE`.** This single threshold governs corner, edge
and surface assessability alike.

### Categories decompose into defect types

The central correction: **detectability is per defect type, not per region.** A white corner
cannot show whitening, but it shows rounding and fraying perfectly well. Collapsing a corner
to one scalar would make an entire class of cards permanently unassessable.

| Category | Defect types assessed |
|---|---|
| `centering` | border ratio measurement |
| `corners` | whitening, rounding/softness, fraying |
| `edges` | whitening, chipping, roughness |
| `surface` | scratches, print lines, dimples, stains, gloss break |

### Structural versus circumstantial undetectability

A defect type can be unassessable for two very different reasons, and conflating them is
what breaks the policy:

- **Circumstantial** — *this photograph* cannot support it: glare, blur, occlusion, low
  resolution, a missing face. A better photograph would resolve it. This is a **coverage
  failure**, and it drives `recommended_additional_photos`.
- **Structural** — *no photograph of this card* could support it, because of the card's own
  design: whitening on a white border (`CORNERS_COLORED_001`, `EDGES_COLORED_001`), a border
  ratio on a borderless design (`CENTERING_BORDERLESS_001`). A better photograph changes
  nothing.

**Structurally undetectable defect types are declared not-required for that card** and do
not block `SUFFICIENT`. They are always reported in `limitations`, so the owner knows
exactly what the photographs could never have shown (non-negotiable rule 3). Circumstantial
undetectability always counts against coverage.

This is the honest reading of the asymmetry. Demanding evidence that no photograph could
ever supply would make `PASS` unreachable for most white-bordered cards — the majority of
the modern base-card population — which is a false-rejection machine, exactly what the
governing asymmetry forbids. Demanding evidence a *better photograph* would supply is
correct and is retained in full.

### When a category counts as assessed

A category is **assessed** when, for every one of its required defect types — that is, every
defect type not declared structurally undetectable for this card — on every required face:

1. assembled detectability reaches at least `MODERATE`, **and**
2. the vision layer, where one ran, did not report that category `not_assessable`.

Condition 2 is why coverage is evaluated a second time after vision. CV suitability
measures whether the pixels *could* carry the evidence; the vision layer reports whether
anything could actually be concluded from them. Either one saying no means the category is
not assessed.

Centering additionally requires a resolved measurement: either a border ratio with declared
precision, or a design declared borderless (structural, therefore not required). The
rubric's `CENTERING_BORDERLESS_COMPARISON_001` route — comparing against the same card
already graded PSA 10 — requires an external reference corpus. That corpus is **out of scope
for this subsystem** (§6 forbids the core from touching the network, and §5 declares no such
table), so borderless centering is treated as structural and reported as a limitation.

### Required faces

The policy declares which faces are required: **front and back**.
`SURFACE_TECHNICAL_DEFECT_001` records that a crease or paper loss on the back is
grade-limiting, so an unresolved back is a circumstantial coverage failure — not an
omission, and never a defect.

### The three outcomes

| Outcome | Condition (v1) | Consequence |
|---|---|---|
| `SUFFICIENT` | all four categories assessed on all required faces | `PASS` permissible |
| `PARTIAL` | at least two categories assessed, but not all | `PASS` forbidden; verdict floor `REVIEW`; still rankable, score carries its limitations |
| `INADEQUATE` | fewer than two categories assessed, **or** no image resolves to a usable front | not rankable; `psa10_rank_score` is null |

**This is the `REVIEW` / `INSUFFICIENT_IMAGES` boundary**, and with the counts above it is a
declared threshold rather than a judgment call: `PARTIAL` means *we learned something but
not enough to pass the card*; `INADEQUATE` means *we could not evaluate*. Unknown card
context (§8 of this spec) makes product-conditional defect types circumstantially
unassessable and therefore biases toward `PARTIAL`.

Detectability is not a proxy for absence. A corner whose whitening detectability is `LOW`
because of glare is **not assessed for whitening**, regardless of whether whitening was
found — "we could not see it" never becomes "it is not there."

Making this an artifact rather than scattered conditionals is what lets I2 be tested
directly: construct a detectability map plus a set of vision assessability flags, run the
policy, assert the outcome.

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
`psa10_rank_score` is an integer **0-100** ranking heuristic — explicitly not a
probability — and is `null` whenever `rankable` is false.

The four states are **mutually exclusive and evaluated in strict order — first match
wins.** Stating them as independent conditions would leave a card with both an observed
crease and `PARTIAL` coverage matching two rows at once.

| # | Verdict | Condition |
|---|---|---|
| 1 | `REJECT` | at least one confidently `observed` PSA-10 disqualifier satisfying I1 |
| 2 | `INSUFFICIENT_IMAGES` | coverage is `INADEQUATE` |
| 3 | `REVIEW` | coverage is `PARTIAL`, or any unresolved ambiguity, suspicion or recorded contradiction |
| 4 | `PASS` | coverage is `SUFFICIENT`, no observed disqualifier, no unresolved ambiguity |

**Why `REJECT` outranks `INADEQUATE` coverage.** A confidently observed disqualifier is
knowledge, not absence of it. If the photographs are poor overall but one of them plainly
shows a crease through the middle of the card, the honest verdict is `REJECT` — the card is
not a PSA 10 candidate, and returning `INSUFFICIENT_IMAGES` would waste the owner's
inspection. This does not weaken the recall priority: rule 1 fires only when I1 is
satisfied, which already demands adequate evidence *for that specific finding* and no
material unresolved contradiction. Poor images make I1 harder to satisfy, not easier.

**Why `INADEQUATE` outranks `REVIEW`.** Both send the card onward, so nothing is lost by
recall; `INSUFFICIENT_IMAGES` is simply the more informative of the two, and preserving it
tells the owner to seek better photographs rather than to inspect.

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
contradiction undermining it.

A **material unresolved contradiction** is defined concretely, so this is testable rather
than prose: for the same defect type at the same normalized location, another finding
reports `not_observed` with detectability at least `MODERATE`, or the heuristic and vision
layers disagree on the state and neither cites evidence the other lacks. Contradictions are
recorded (§11), never silently resolved; an unresolved one blocks rule 1 of §14 and the card
falls through to `REVIEW`.

Evidence that is `suspected`, `not_assessable`, conflicting or low-confidence must never
independently produce `REJECT`. **Poor photographs are never evidence of card damage.**

**I2 — Unassessable never passes.** `PASS` requires the **authoritative** coverage
evaluation — the one that ran after vision — to return `SUFFICIENT`. The provisional
evaluation gates spend only and can never license a `PASS`. "We could not see a problem"
must never become "the card is clean."

**I3 — Enhancement alone never confirms.** An anomaly visible only under enhancement may be
a `suspected` candidate but can never independently reach `observed`.

---

## 16. Output

```
candidate / listing identity
verdict:              PASS | REVIEW | REJECT | INSUFFICIENT_IMAGES   (authoritative)
psa10_candidate:      yes | no | uncertain | unknown        (derived from verdict)
psa10_rank_score:     0-100 | null
rankable:             true | false
estimated_psa_grade:  "10" | "9-10" | "9" | "8-9" | "<=8" | null
review_confidence:    how far the evidence supports the verdict — separate from grade
coverage:             SUFFICIENT | PARTIAL | INADEQUATE, per-category and
                      per-defect-type detail, each marked circumstantial or structural
centering / corners / edges / surface   — assessment, confidence, detectability
image_quality         — per-image preflight, per-region observability, masks
roles_and_context     — resolved values with confidence and provenance
defects_found         — producer, state, category, defect type, normalized location,
                        evidence refs, enhancement level, confidence
limitations           — what could not be assessed and why, structural vs circumstantial
recommended_additional_photos
                      — specific shots that would resolve each *circumstantial*
                        limitation; empty when every limitation is structural
cv_assessment         — stored independently
vision_assessment     — stored independently, incl. independent gem view; null in OFF
reasoning             — concise, evidence-anchored
versions              — preflight, geometry, observability, CV, resolver, assembly,
                        rubric, scorer, routing policy, manifest builder, coverage
                        policy, combination policy, canonicalization scheme,
                        provider/model, prompt
```

**`review_confidence` and `estimated_psa_grade` are separate fields and must remain so**
(build plan §19): a confident "this is a 9" and a hesitant "this might be a 9" are different
claims, and collapsing them hides exactly what the owner needs to decide where to look.

**`recommended_additional_photos` is the highest-recall action available on a partially
assessable card.** `IMAGE_LIMITATIONS_REQUEST_PHOTOS_001` exists precisely for this: asking
for a better photograph beats guessing at a grade. It is derived from the coverage
evaluation's circumstantial failures — structural limitations are never turned into a photo
request, because no photograph would resolve them.

`verdict` is the authoritative field name throughout, in the schema and in this document.

Every output field has a producing stage: `combine` owns `verdict`, `psa10_candidate`,
`psa10_rank_score`, `rankable`, `estimated_psa_grade`, `review_confidence` and `reasoning`;
`coverage` owns `coverage`, `limitations` and `recommended_additional_photos`; the remaining
blocks are the corresponding stages' stored outputs surfaced unchanged.

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
validated-success-only caching, mode's participation in the routing fingerprint, role and
context resolution (including `None` versus `[]`), evidence assembly, the coverage policy
across both structural and circumstantial undetectability, routing policy, the manifest
builder's determinism and budgets, scoring, and the verdict function over its full
cross-product. Table-driven, including I1, I2 and I3.

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
10. A white-bordered card with otherwise good photographs can reach `SUFFICIENT` coverage
    and `PASS`, with corner whitening reported as a *structural* limitation — the
    regression that would otherwise make `PASS` unreachable for most modern base cards.
11. A card screened in `OFF` and then in `DEEP` issues a vision call on the second run;
    the `OFF` routing result never satisfies the `DEEP` lookup.
12. The verdict function is total and unambiguous: every combination of coverage outcome,
    observed-disqualifier presence and ambiguity maps to exactly one verdict, asserted
    table-driven over the full cross-product.
13. A vision response marking a category `not_assessable` prevents `SUFFICIENT` coverage
    even when CV suitability alone would have allowed it.
14. The synthetic image generator produces cards with known centering, border colours,
    borderless designs and controlled damage, and its own tests pass independently of the
    CV engine that consumes it.
15. No price-derived field appears in the evidence manifest, the vision prompt, or the
    review output, asserted by test.
16. Test suite green; no test calls the API.
