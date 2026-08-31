# CLAUDE.md — Card Reviewer Working Agreement

This file defines the engineering and product rules for the Card Reviewer
repository.

These rules are requirements, not suggestions.

The Superpowers skills are a **requirement for producing this application**,
not an optional aid. Invoke them when applicable; do not substitute judgement
for the required workflow.

---

# Repository purpose

Card Reviewer contains two subsystems joined through `knowledge/`.

## Subsystem A — Card Review Engine

Location:

`src/card_reviewer/`

Purpose:

Candidate card images and metadata
→ deterministic image analysis
→ grading evidence
→ optional vision-model interpretation
→ rough PSA assessment
→ PSA-10 candidate ranking
→ PASS / REVIEW / REJECT / INSUFFICIENT_IMAGES

Subsystem A is currently under implementation.

Its purpose is screening, not final human replacement.

The primary question is:

> Does this particular raw card appear to have a realistic chance of grading
> PSA 10 based on the available photographs?

Candidates may originate from:

- Flippah
- manual submissions
- future adapters

Candidate discovery itself is outside the grading engine.

## Subsystem B — Video Learning Pipeline

Location:

`src/card_reviewer/knowledge/`

Purpose:

Authorized grading/training material
→ extracted evidence
→ classified claims
→ versioned grading knowledge/rules

Subsystem B is already implemented.

It is existing grading-knowledge infrastructure and must not be casually
rewritten to accommodate Subsystem A.

The two subsystems interact through the versioned knowledge/rubric layer.

---

# Governing documents

For Subsystem A, the approved Card Review Engine design spec and implementation
plan are the governing architecture and execution blueprint.

Current documents live under:

`docs/superpowers/specs/`

and:

`docs/superpowers/plans/`

Use the approved Card Review Engine spec as the highest project-specific design
authority.

Subsystem B has its own approved design/specification documents in the same
directories.

Do not substitute older build-plan prose for a newer approved spec or design
decision.

---

# Superpowers workflow is mandatory

## Before writing code

**`superpowers:using-git-worktrees`** — branch first.

Never start implementation directly on `main`.

Use a dedicated worktree/branch for implementation work.

**`superpowers:test-driven-development`** — strict RED → GREEN.

For every behavior:

1. Write the test first.
2. Run it.
3. Verify it fails for the intended reason.
4. Implement the minimum necessary code.
5. Run the test again.
6. Run affected contract/integration tests.
7. Run the complete suite before considering the task complete.

A test that was never observed failing does not prove the behavior it claims to
guard.

A wrong-reason RED does not count.

Examples:

- import failure while testing grading logic → does not count
- broken fixture while testing cache behavior → does not count
- assertion fails because intended behavior is missing → counts

---

# Before merging

**`superpowers:requesting-code-review`** — dispatch an independent reviewer
subagent.

**Self-review does not count.**

The reviewer must inspect the actual implementation and tests, not merely the
specification or implementation plan.

Do not merge to `main` with unresolved review findings.

---

# When review comes back

**`superpowers:receiving-code-review`** — verify every finding before changing
the code.

Do not automatically agree with reviewers.

For every finding:

1. reproduce or inspect the claimed problem;
2. determine whether it is valid;
3. push back with evidence when it is wrong;
4. implement only verified findings.

Review is evidence gathering, not an instruction to perform agreement.

---

# Re-review after fixes

Review fixes require an independent scoped re-review.

Do not assume:

    finding
    → fix
    → automatically correct

A fix may repair the producer while breaking a consumer.

After fixing review findings:

1. add or strengthen the regression test;
2. observe RED where practical;
3. implement;
4. run targeted tests;
5. run producer → consumer tests;
6. mutation-test the new guard;
7. run the full suite;
8. request independent re-review.

---

# Governing design hierarchy

When implementation details disagree, use this order:

1. approved design spec
2. approved Design Decisions and governing invariants
3. interface contracts / acceptance criteria
4. executable tests
5. implementation-plan pseudocode

Implementation-plan Python is illustrative.

Do not preserve broken pseudocode merely because it appears in the plan.

If executable behavior proves plan pseudocode wrong while the governing design
is clear:

- follow the governing design;
- document the deviation;
- continue.

If the governing design itself becomes ambiguous or contradictory:

**STOP and ask.**

Do not invent a new product policy during implementation.

---

# One task at a time

Follow the approved implementation plan in dependency order.

Do not bounce between unrelated tasks.

A task is not complete until:

- its RED test was observed failing for the expected reason;
- implementation makes it pass;
- affected producer/consumer contract tests pass;
- persisted models round-trip where applicable;
- every new guard added by the diff has been mutation-tested;
- the complete suite passes;
- the task has its own commit.

Do not batch many implementation tasks into one giant commit.

Prefer one deliverable per commit so regressions remain reviewable and
bisectable.

---

# Producer → consumer contract tests are mandatory

This project has repeatedly exposed failures of the form:

    producer changed
        ↓
    producer test passed
        ↓
    consumer still expected old representation
        ↓
    end-to-end behavior broke

Independent unit tests are not sufficient once both sides exist.

Whenever one component produces data consumed by another, test:

    real producer output
        ↓
    real serialization boundary
        ↓
    real deserialization
        ↓
    real consumer

Contract tests should detect:

- tuple/list drift
- incompatible dictionary-key shapes
- missing fields
- incorrect aliases
- enum serialization mismatches
- Pydantic silently dropping fields
- nullable-field assumptions
- missing reason codes
- stale signatures
- invalid artifact references
- values that serialize but cannot deserialize
- producer changes not reflected in consumers

Do not rely solely on hand-built fixtures once the real producer exists.

---

# Mutation-test every new guard

Before requesting review, mutate every guard added by the current diff.

Verify that the intended test fails.

Do not accept merely:

> one test failed

Check **which test** killed the mutation.

Prefer plausible mutations.

Examples:

    threshold 0.80 → 0.75

    HIGH → MODERATE

    BINDING → ADVISORY

    detectability default NONE → HIGH

    remove an I1 condition

    remove an I2 condition

    remove an I3 provenance condition

    omit one fingerprint input

    remove provider cache lookup

A guard that survives mutation is not adequately protected.

---

# Full-suite discipline

Run the complete suite after each implementation task.

Subsystem B's existing tests are part of the baseline and must remain green.

Do not modify Subsystem B simply to make Card Review Engine tests easier.

If Subsystem A exposes a genuine incompatibility with Subsystem B:

**STOP and surface it explicitly.**

Do not silently rewrite the grading knowledge system.

---

# Non-negotiable product rules

1. Never use card value when judging condition.

2. Never automatically assume PSA 10. Search for visible reasons a card may not
   gem.

3. Never hide image limitations or claim certainty photographs cannot support.

4. OpenCV measurements are evidence, not the grader.

5. Claude/VisionProvider assessment is interpretation, not objective
   measurement.

6. Distinguish confidently observed defects from suspected defects.

7. Preserve original images, evidence provenance, model/analyzer versions, and
   rubric version.

8. Do not blindly learn every claim from training material. Preserve and use
   `evidence_type`.

9. Never circumvent DRM or access training material the user is not authorized
   to view.

10. No pricing, EV, buying, selling, or flip-economics logic belongs in the
    condition-grading core.

11. Never delete historical grading rules merely because they are superseded.
    Change their status/version according to Subsystem B's model.

12. Historical predictions are append-only. Never rewrite a past prediction
    after a later model version or PSA outcome becomes known.

---

# Condition grading is separate from economics

The condition-grading engine must not use:

- asking price
- market value
- EV
- expected profit
- purchase price
- resale value
- flip opportunity

to alter condition assessment.

Identical images must receive identical condition analysis regardless of the
card's value.

Economic analysis belongs outside this repository's grading core unless the
approved architecture explicitly changes.

---

# OpenCV responsibility

OpenCV is the deterministic measurement and observation layer.

It may produce:

- card boundaries
- perspective normalization
- centering measurements
- image-quality measurements
- detectability/observability
- corner crops
- edge crops
- surface views
- anomaly candidates
- normalized coordinates
- confidence/uncertainty

It does not issue the final PSA verdict.

OpenCV emits measurements and candidates.

Do not turn a CV anomaly into a confirmed defect merely because it exists.

---

# VisionProvider / Claude responsibility

Claude is a visual judgment layer.

It may interpret ambiguity such as:

- chipping vs glare
- soft corner vs lighting
- print line vs card design
- scratch vs refractor pattern
- dimple/indentation
- foil artifact
- whether evidence is actually sufficient

Claude does not replace objective measurements OpenCV can calculate reliably.

When Claude and deterministic evidence disagree, preserve that disagreement.

Do not silently overwrite one with the other.

---

# Search aggressively, conclude conservatively

The vision layer should search adversarially for reasons a card might fail PSA
10.

But conclusions must remain evidence-conservative.

Use distinctions such as:

- observed
- suspected
- not observed
- not assessable

A suspicious pattern is not automatically a confirmed defect.

---

# Governing Card Review Engine invariants

These are architectural constraints.

They must remain mechanically testable.

## I1 — Missing or uncertain evidence cannot manufacture a REJECT

REJECT requires evidence satisfying the approved I1 policy.

The following cannot independently cause rejection:

- suspicion
- inadequate detectability
- absent evidence
- unresolved contradiction
- unsupported enhancement artifacts
- provider failure
- missing regions/images

An observed finding that fails I1 routes toward REVIEW rather than REJECT.

Missing evidence removes evidence.

It never creates evidence.

## I2 — Missing required evidence cannot manufacture a PASS

Not seeing a defect does not prove a clean card when the relevant category or
region could not actually be assessed.

Required unassessable grading evidence prevents PASS according to the approved
EvidenceCoveragePolicy.

Poor photographs are not proof of damage.

They are also not proof of cleanliness.

## I3 — Enhancement-only evidence cannot independently establish a confirmed defect

Enhanced views such as:

- CLAHE
- sharpening
- grayscale
- edge highlighting

may surface anomaly candidates.

A feature visible only in enhanced evidence cannot independently establish the
evidence needed for a confirmed PSA-10 disqualifier.

Artifact provenance must survive all the way to the logic enforcing I3.

Never fabricate provenance.

---

# Detectability is part of the evidence

These are not equivalent:

> No whitening was observed.

and:

> Whitening was highly detectable here, and none was observed.

Every relevant observation must preserve detectability/observability when
applicable.

For example, a white corner may have:

    whitening_detectability = LOW

Therefore:

    no whitening observed

is weak evidence rather than proof of a clean corner.

---

# Structural and circumstantial limitations are different

Do not convert metadata problems into photography problems.

## Structural / metadata-resolvable

Examples:

- unknown card product type
- unknown set when a rule requires it
- identity clarification required
- card design inherently prevents a particular observation

These should not automatically request a better photograph.

## Circumstantial / image-resolvable

Examples:

- glare
- blur
- occlusion
- low resolution
- missing view

These may justify requesting more or better images.

Preserve the distinction throughout evidence coverage and reporting.

---

# Front-only policy

A usable front-only listing may remain:

    PARTIAL

and may remain rankable.

Missing the back prevents PASS.

Missing the back does not erase a confidently established front-side
disqualifier.

Therefore:

    usable front + no back + no I1 disqualifier
        → PARTIAL / REVIEW / rankable

    usable front + no back + I1-satisfying front defect
        → REJECT

    unusable front
        → INSUFFICIENT_IMAGES / unrankable

Coverage and condition are separate concepts.

---

# Ranking score semantics

`psa10_rank_score` is a heuristic ranking score.

It is not a calibrated probability.

Do not label it as:

- probability
- percentage chance
- PSA-10 likelihood percentage

unless future calibration genuinely supports that interpretation.

Its primary purpose is ordering candidates for human inspection.

If evidence cannot support ranking:

    psa10_rank_score = null

Do not manufacture a neutral-looking number for an unrankable card.

---

# Estimated grade is separate from rank score

`estimated_psa_grade` must not be mechanically derived from rank-score
thresholds.

The score is not calibrated.

A coarse estimated grade and the rank score may legitimately disagree.

---

# Review confidence means evidence confidence

`review_confidence` means confidence in the completeness/quality of the review.

It is not PSA-10 probability.

It may depend on:

- coverage
- detectability
- image quality
- agreement/disagreement
- ambiguity
- missing faces

A front-only candidate may be rankable while still carrying LOW review
confidence.

---

# Preserve raw producer findings

OpenCV/heuristic findings and VisionProvider findings must remain independently
recoverable.

Fusion may be used for:

- scoring
- contradiction handling
- verdict logic

but it must not destroy the source findings.

This is required for later calibration against actual PSA results.

---

# Do not double-penalize corroboration

If OpenCV and Claude identify the same physical defect, do not automatically
penalize the card twice.

Correlate findings using:

- defect/category
- normalized location/region
- overlapping evidence

Keep both source findings for provenance/calibration while using a fused
assessment where appropriate for final scoring.

---

# Rule authority and visual certainty are different concepts

Do not collapse:

> Does the defect exist?

with:

> If the defect exists, how important is the governing grading rule?

Rule authority comes from Subsystem B.

Visual certainty comes from evidence.

Unknown authority must not default to BINDING.

If relevance/authority is unresolved, fail safely toward REVIEW rather than
manufacturing a disqualifier.

---

# Card context normalization

Do not feed arbitrary listing strings directly into Subsystem B's exact-match
scoping.

Normalize context through the approved normalizer.

The currently known card-type vocabulary comes from live Subsystem B and
currently includes:

- `chrome`
- `refractor`
- `foil`

Do not use fuzzy guessing.

Unknown remains unknown.

Preserve:

- raw value
- normalized value
- source
- confidence where applicable

Unknown card context must not silently narrow the rubric.

A product-scoped rule that cannot be evaluated due to missing context remains
UNEVALUABLE.

---

# Cache architecture

SQLite is the authoritative structured state/history store.

Large images and image-derived artifacts belong in content-addressed filesystem
storage.

Do not store large images as SQLite blobs unless the approved design changes.

Stage results are append-only.

Do not overwrite historical outputs because:

- CV changed
- rubric changed
- heuristic changed
- Claude model changed
- routing policy changed
- combination policy changed
- actual PSA result later became known

---

# Input fingerprint and producer signature are different

These concepts must remain separate.

## Input fingerprint

Represents canonical values actually consumed by a stage.

Evidence floats may use declared semantic quantization.

Examples:

- centering
- normalized coordinates
- confidence values

## Producer signature

Represents exact implementation/configuration capable of changing output.

Examples:

- analyzer version
- model version
- weights
- provider
- prompt
- inference parameters
- policy versions

Producer configuration must not be rounded using measurement precision.

Use deterministic exact producer/config canonicalization.

Reject unsupported or non-finite values.

---

# Fingerprint exactly what the stage consumes

A cache identity must represent the data actually consumed.

Do not invalidate an expensive downstream result merely because an upstream
implementation version changed when the downstream-visible evidence remained
identical.

Likewise, never omit a material consumed input from a fingerprint.

Changing a value capable of changing output must invalidate that stage.

Changing an unrelated value must not.

---

# Cache only validated successful outputs

Only validated successful stage outputs may satisfy future cache lookups.

The following must never become reusable successful cache rows:

- exceptions
- provider timeouts
- malformed provider responses
- schema-validation failures
- partial writes
- failed attempts

Failed attempts may be recorded separately.

They are not successful stage results.

---

# Anthropic / external provider rules

Automated tests and CI must **never make a real Anthropic API call**.

Use `VisionProvider` mocks or saved fixtures for automated tests.

A real provider smoke test must be explicit and manually invoked.

SMART and DEEP must consult the stage cache **before** making an external API
request.

A successful provider result should be safely persisted before later cheap
stages so a crash does not unnecessarily rebill the same evidence.

If SMART determines vision is required and no provider is configured:

do not silently behave like OFF.

Surface the situation according to the approved recall-safe behavior.

---

# Vision must receive actual image evidence

When external vision analysis runs, the provider must receive the actual
selected image data, not merely hashes, paths, or descriptions.

Artifact IDs must deterministically map to the exact image blocks sent.

The provider-visible payload should include the approved useful evidence such
as:

- useful originals
- normalized front/back views
- relevant corner crops
- relevant edge crops
- relevant surface/anomaly views
- OpenCV measurements
- measurement uncertainty
- detectability
- image limitations
- anomaly candidates
- applicable rubric rules

Do not blindly send every derived image simply because it exists.

---

# Serialization boundaries must fail loudly

Do not silently stringify unsupported cache/fingerprint values.

Examples requiring deliberate representation:

- sets
- arbitrary objects
- live NumPy arrays
- Paths
- invalid dictionary keys
- NaN
- Infinity

Do not hide invalid structures with `default=str`.

Cache identity is correctness-critical.

---

# Persist artifact references, not live image arrays

Serialized stage results must remain persistence-safe.

Derived large objects such as:

- normalized card images
- border masks
- glare masks
- occlusion masks

belong in ArtifactStore/content-addressed filesystem storage.

Persist stable references plus scalar metadata.

---

# Human outcome calibration

Preserve enough historical information to compare:

    OpenCV / heuristic assessment

vs

    Claude assessment

vs

    combined assessment

vs

    actual PSA outcome

Do not rewrite historical predictions after an actual PSA result is recorded.

Outcome data is future calibration ground truth.

---

# Multiple grading outcomes must remain representable

Do not permanently assume a physical card is graded only once.

The persistence model should remain capable of representing:

- purchased but never submitted
- submitted
- returned
- resubmitted
- cracked and resubmitted
- future grader support if required

Historical review data must remain intact.

---

# Autonomous phase execution

Implementation phases do not require a user checkpoint merely because a phase
ended.

Claude may continue automatically from one phase to the next when all required
engineering gates are satisfied.

At the end of each phase:

1. run the complete test suite;
2. confirm Subsystem B remains green;
3. mutation-test every new guard introduced by the phase;
4. verify the intended test killed each mutation;
5. run affected producer → consumer contract tests;
6. run persistence round-trip tests where applicable;
7. invoke `superpowers:requesting-code-review`;
8. verify every finding using `superpowers:receiving-code-review`;
9. fix verified findings;
10. run the complete relevant verification again;
11. request independent scoped re-review of the fixes;
12. proceed to the next phase only when no unresolved findings remain.

These are engineering checkpoints, not user checkpoints.

Do not stop merely to report normal phase completion.

Keep phase reports in commits / PR history so the implementation remains
auditable.

# When to stop and ask the user

Stop implementation and ask the user only when a genuine decision is required.

Examples:

- the approved design/spec is ambiguous or contradictory;
- multiple reasonable behaviors exist and choosing among them changes product
  behavior;
- a change would weaken or alter an approved invariant;
- Subsystem A exposes a genuine incompatibility with Subsystem B;
- implementing a reviewer finding would require changing an approved product
  decision;
- an external credential, paid action, destructive action, or authorization
  decision requires the user;
- executable evidence shows that a governing assumption itself is wrong.

Do NOT stop for:

- ordinary implementation bugs;
- failing tests with a clear intended behavior;
- producer/consumer mismatches;
- serialization bugs;
- incorrect implementation-plan pseudocode when the governing design is clear;
- review findings that can be verified and corrected without changing product
  policy;
- routine refactoring required to satisfy the approved design.

Handle those through TDD, contract testing, mutation testing, review, and
re-review, then continue.

# Final implementation gate

After the final implementation phase:

1. run the entire test suite;
2. run all required end-to-end tests;
3. verify Subsystem B remains unchanged unless an approved change required it;
4. complete mutation verification for final-phase guards;
5. perform an independent full implementation review using
   `superpowers:requesting-code-review`;
6. verify findings with `superpowers:receiving-code-review`;
7. fix verified findings;
8. run an independent re-review of the fix diff;
9. ensure no unresolved review findings remain;
10. produce a concise final readiness report for the user.

The final report should include:

- phases/tasks completed;
- final test count;
- Subsystem B test count;
- implementation-plan deviations;
- independent review result;
- re-review result;
- unresolved issues, if any;
- whether the PR is ready to merge.

Do not require the user to manually review implementation code unless a
governing product/design decision remains unresolved.

---

# Before requesting code review

Before invoking `superpowers:requesting-code-review`:

- full suite green;
- no unexplained warnings;
- new guards mutation-tested;
- intended tests verified as mutation killers;
- actual diff inspected;
- producer → consumer contract tests run;
- persistence round trips run where applicable.

Do not knowingly send incomplete work for review.

---

# Definition of DONE

A task or phase is not DONE while required work remains unresolved.

DONE means:

- intended behavior implemented;
- RED → GREEN demonstrated;
- contracts connected;
- persistence validated where applicable;
- new guards mutation-tested;
- complete suite green;
- independent review completed when required;
- review fixes independently re-reviewed;
- no unresolved findings remain.

Do not use DONE as a progress label.

Use it only when the work is actually complete.

---

# Repository commands

Install dependencies:

    uv sync

Run the complete test suite:

    uv run pytest

Inspect the knowledge CLI:

    uv run card-knowledge --help

Use additional Card Reviewer CLI commands once their implementation tasks have
actually landed. Do not document or depend on commands that do not yet exist.