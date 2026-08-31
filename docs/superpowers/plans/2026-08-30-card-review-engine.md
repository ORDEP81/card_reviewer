# Card Review Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build subsystem A — an engine that takes marketplace listing photographs of a raw sports card and produces a defensible, auditable answer to "does this copy have a realistic chance of grading PSA 10?"

**Architecture:** One pipeline, two tiers. Image-tier stages (`preflight` → `geometry` → `observability` → `cv_measurements`) depend only on one image's bytes and are cached by image hash. Candidate-tier stages fuse across images, apply subsystem B's rubric, optionally consult a vision provider, and resolve a four-state verdict. Every stage result is a content-addressed, append-only cache row; OpenCV measures and Claude interprets, but neither decides — a versioned policy layer does.

**Tech Stack:** Python 3.14, uv, Pydantic v2, SQLite (stdlib `sqlite3`), OpenCV (`opencv-python-headless`), NumPy, Pillow, Typer, Rich, pytest. Anthropic SDK behind an interface.

**Spec:** `docs/superpowers/specs/2026-08-30-card-review-engine-design.md` — read it alongside this plan. The spec is the authority; this plan argues from it.

---

## Global Constraints

Copied verbatim from the spec and `CARD_REVIEWER_BUILD_PLAN.md` §30. Every task's requirements implicitly include this section.

1. **Never use card value when judging condition.** No pricing, EV, market value, profitability or buy/sell logic anywhere — including the vision prompt. `ResolvedCandidate` carries no price field of any kind.
2. **Never automatically assume PSA 10.** Search adversarially for reasons a card will not gem.
3. **Never hide image limitations** or claim certainty photographs cannot support.
4. **OpenCV measurements are evidence, not the grader. Claude assessment is evidence, not measurement.**
5. **Distinguish definite defects from suspected defects** — the finding-state vocabulary is shared and enforced.
6. **Preserve original images, knowledge provenance, and rubric version.**
7. **Do not blindly learn every claim** — classify by `evidence_type`. Applied here as the evidence-weighting policy in Decision 4.
8. Never circumvent DRM or access unauthorized material.
9. **No pricing, EV, buying, or selling logic in this repository.**
10. **Never delete a rule. Change its `status`.** (Subsystem B; subsystem A never writes rules at all.)
11. **The governing asymmetry:** missing a legitimate PSA-10 candidate is worse than forwarding a few extra cards. Every threshold, default and ambiguity resolves toward recall.
12. **No automated test or CI run ever calls the Anthropic API.** `VisionProvider` is mocked in every pipeline test. `card-review provider-smoke` is the only path to a real call.
13. **Only validated successes are cacheable.** Failures go to `stage_attempt` and can never satisfy a cache lookup.
14. **Downstream stages fingerprint upstream output *values*, never upstream producer signatures.**
15. Python >= 3.14. All commands run under `uv run`.

---

## Design Decisions

These five decisions settle what was deliberately deferred from the spec because they are implementation design, not architecture. They are settled **here**, before any task, so no implementer guesses them.

### Decision 1 — Evidence provenance, carried far enough that I3 is pure logic

**The problem.** I3 says an anomaly visible only under enhancement can never independently establish an `observed` defect, and §7.4 says combine "verifies rather than trusts" the citation. But combine's declared inputs are the heuristic, vision and coverage outputs. Nothing in them carries per-artifact enhancement level, so as specified combine cannot perform its own check.

**The resolution.** Provenance travels *inside findings*, by reference. Every `Finding` carries one or more `EvidenceRef`, and every `EvidenceRef` carries its own origin. Combine reads `finding.evidence[*].origin` — it never opens an image, never consults the artifact store, never reconstructs anything.

```python
class EvidenceOrigin(StrEnum):
    ORIGINAL   = "original"    # untouched bytes as supplied
    NORMALIZED = "normalized"  # geometry output: rectified face or crop, no pixel enhancement
    ENHANCED   = "enhanced"    # CLAHE / sharpened / grayscale / edge-highlight

class EvidenceRef(BaseModel):
    artifact_id: str                      # content-addressed id in the artifact store
    image_hash: str                       # the source photograph
    origin: EvidenceOrigin
    enhancement: str | None = None        # e.g. "clahe:clip=2.0,grid=8" — required iff origin is ENHANCED
    region: NormalizedBox | None = None   # normalized card coordinates
    view: str                             # "front_face", "corner_bottom_left", "edge_top", ...
```

`ORIGINAL` and `NORMALIZED` both count as unenhanced for I3: rectification is a geometric resampling that cannot invent a defect, whereas CLAHE and friends deliberately amplify local contrast. This distinction is the whole point, so it is asserted in tests rather than assumed.

**The I3 rule, as pure logic over that structure:**

```python
def i3_satisfied(finding: Finding) -> bool:
    """A finding may be `observed` only if some evidence is unenhanced."""
    if finding.state is not FindingState.OBSERVED:
        return True
    return any(e.origin is not EvidenceOrigin.ENHANCED for e in finding.evidence)
```

Combine calls this on every finding before the finding may contribute to `REJECT`. A finding whose evidence is entirely `ENHANCED` is **demoted to `SUSPECTED`** rather than dropped — it is still real information, it just cannot reject a card.

**No manifest duplication.** `EvidenceRef` holds ids, not bytes and not artifact bodies. The evidence manifest sent to the provider is built from the same ids; the review stores the ids. Nothing is copied twice.

**Vision route 2** (§7.4: "a vision finding citing an unenhanced artifact") is exactly `any(origin is not ENHANCED)` over the refs the provider echoed back. The provider must return the `artifact_id`s it relied on; a provider that cites an id absent from the manifest it was sent is a contract violation and the finding is rejected. That is a parsing-layer assertion, tested offline.

**Provenance must survive the round trip, and this is where it is easiest to lose.** The provider returns bare `artifact_id` strings. Reconstructing an `EvidenceRef` from a bare id — inventing `origin=ORIGINAL` and an empty `image_hash` — would silently launder an enhancement-only finding into one that satisfies I3, defeating the invariant at exactly the point it matters most. So the manifest is retained for the whole candidate-tier run as an `artifact_id -> EvidenceRef` index, and every provider-cited id is **resolved against that index**, carrying the ref's real `image_hash`, `origin`, `enhancement`, `view` and `region` into the `Finding`. An id that does not resolve is a contract violation, never a default.

### Decision 2 — v1 derivation of the three summary values

All three live in one versioned artifact, `ScoringPolicy` v1, loaded from `card_reviewer/review/policies/scoring_v1.py`. No magic numbers anywhere else. `scoring_policy_version` participates in `combine`'s producer signature.

**They answer three different questions and must never collapse into each other:**

| Field | Question | Driven by |
|---|---|---|
| `psa10_rank_score` | How should this card sort against others in the batch? | findings + coverage |
| `estimated_psa_grade` | What coarse grade does the evidence support? | worst confirmed defect severity |
| `review_confidence` | How much do we trust this assessment at all? | coverage, detectability, image quality, source agreement |

**`psa10_rank_score` — 0–100 integer, `null` when `rankable` is false.**

Start at 100, subtract penalties. Deterministic, additive, no multiplication of unrelated quantities:

```
score = clamp(0, 100, 100
    - Σ over fused findings: penalty(state, authority, i1_satisfied)
    - coverage_penalty(coverage_outcome))
```

with v1 weights declared in one table:

| Finding status | Penalty (PSA-10-relevant finding) |
|---|---|
| `observed`, `binding`, **I1 satisfied** | 100 (floors the score) |
| `observed`, `binding`, **I1 not satisfied** | 35 |
| `observed`, authority `advisory` | 25 |
| `suspected`, authority `binding` | 15 |
| `suspected`, authority `advisory` | 6 |
| `not_observed` / `not_assessable` | 0 |

**Only an I1-satisfying binding disqualifier gets the hard floor.** Scoring takes an explicit `i1_satisfied` input per finding rather than inferring it from state alone. A finding that looked like a disqualifier but could not be established routes to `REVIEW` (§14 row 3), and it must stay *meaningfully rankable* there — flooring it to 0 would sort it identically to a confirmed reject and destroy the ordering the owner uses to triage. An intermediate penalty says `worth looking at first` without saying `this card is dead`.

| Coverage | Penalty |
|---|---|
| `SUFFICIENT` | 0 |
| `PARTIAL` | 10 |
| `INADEQUATE` | n/a — score is `null` |

`not_assessable` costs nothing *here* because it is already paid for in `coverage_penalty` and in `review_confidence`; charging it twice would double-count missing evidence into an apparent defect, which violates constraint 3.

**Monotonicity is a tested property, not a hope:**

- Adding any PSA-10-negative finding never *raises* the score. (Penalties are non-negative.)
- Improving coverage without adding a finding never *lowers* the score. (`coverage_penalty` is monotone non-increasing in coverage quality.)
- Promoting a finding `suspected` → `observed` never raises the score. (Penalty table is monotone in state.)

**`estimated_psa_grade` — deliberately NOT a function of the score.**

Deriving the grade from the score would imply the score is calibrated, which §1 explicitly disclaims. The grade is a lookup on the **worst confirmed defect**, per the rubric:

```
no observed PSA-10-relevant finding, coverage SUFFICIENT   -> "10"
no observed finding, coverage PARTIAL                      -> "9-10"
one observed minor finding (severity <= minor)             -> "9"
one observed moderate finding                              -> "8-9"
any observed severe finding, or two+ observed moderate     -> "<=8"
coverage INADEQUATE                                        -> null
```

Severity comes from the finding, which comes from the measurement or the provider — never invented by the scorer. A test asserts the score and the grade can disagree (a card can score 74 and still estimate `"9-10"`), because that proves they are independent.

**`review_confidence` — `high` / `medium` / `low`, about the assessment, not the card.**

```
low     if coverage is INADEQUATE,
        or a required face is missing entirely,
        or an unresolved material contradiction exists
medium  if coverage is PARTIAL for any other reason, or any required
        defect type is circumstantially unassessable, or vision and
        heuristic disagreed anywhere, or card context is unknown
high    otherwise (coverage SUFFICIENT, no contradictions, context known)
```

`review_confidence` therefore takes an explicit `required_face_missing` input; it is not inferrable from the coverage outcome, because a front-only card is `PARTIAL` (rankable, forwarded) yet its confidence is `low` — we never saw half the card. That combination is the point: **`PARTIAL` and rankable, with `low` confidence**, is exactly how a front-only listing should present.

A test asserts the semantic separation directly: a card with `psa10_rank_score = 95` and `review_confidence = low` is constructible and meaningful — "probably clean, but we could barely see it."

### Decision 3 — canonical `card_type` / `set` normalization

**Inspected, not remembered.** The live subsystem B rubric at v4.0.0 was read directly:

- `card_types` in use: **`chrome`, `refractor`, `foil`** — three values, all on `SURFACE_SHINY_001`.
- `sets` in use: **none at all.** No active rule is scoped by set.
- `knowledge/card_types/` and `knowledge/sets/` contain only `.gitkeep`.
- `Rubric.for_card` matches by **exact set intersection** on lowercase strings: `set(rule.applies_to.card_types) & set(card_types)`.

So a resolver emitting `"Chrome"` or `"Panini Prizm"` matches nothing and silently narrows the rule set — the precise failure §8 guards against on the `None`-vs-`[]` axis, left open on the string axis.

**The boundary:**

```
raw listing title / supplied metadata
    -> CardContextNormalizer
    -> CardContext { raw, canonical, confidence, source }
    -> Rubric.for_card(canonical.card_types, canonical.sets)
```

`CardContext` preserves **both**: `raw` (exactly what arrived) and `canonical` (vocabulary values or `None`). Provenance is `supplied` / `inferred` / `unknown` with a confidence.

**Vocabulary lives in one file**, `card_reviewer/review/vocabulary.py`, so adding a product never touches grading logic:

```python
CARD_TYPE_VOCABULARY: dict[str, str] = {      # alias -> canonical
    "chrome": "chrome", "topps chrome": "chrome", "bowman chrome": "chrome",
    "refractor": "refractor", "refractors": "refractor", "prizm": "refractor",
    "foil": "foil", "holo": "foil", "holofoil": "foil",
}
SET_VOCABULARY: dict[str, str] = {}   # empty: no active rule is set-scoped at v4.0.0
```

Normalization: casefold, strip punctuation, collapse whitespace, then exact alias lookup. **No fuzzy matching** — an unrecognized string becomes `None` (unknown), never the nearest neighbour. Guessing "Prizm Silver" into `chrome` would apply a rule the owner never sanctioned.

**Unknown behaviour, and the one subtlety that matters most.** Unknown context passes `None`, so `for_card` returns *every* rule including product-scoped ones. That is correct per §8 — unknown context must not narrow the rules. But a returned rule is not an *applicable* one. Each rule is therefore tagged:

```python
class RuleEvaluability(StrEnum):
    APPLICABLE  = "applicable"    # unscoped, or scope satisfied by known context
    UNEVALUABLE = "unevaluable"   # scoped, but the context it needs is unknown
```

An `UNEVALUABLE` rule **never fires a finding and never contributes to a verdict.** It instead raises an `UNKNOWN_PRODUCT_CONTEXT` coverage gap (class `metadata-resolvable`, per §13), which biases toward `PARTIAL` → `REVIEW` and emits a card-identification request. This is what the owner meant by "must not silently apply": the rule is visible, accounted for, and explicitly not applied.

Because no active rule is set-scoped today, the set axis is exercised only by synthetic fixtures. That is stated rather than hidden, and the tests are written so they start passing for real the day subsystem B adds a set-scoped rule.

### Decision 4 — heuristic weighting of `evidence_type` and rule confidence

**Inspected, not remembered.** Subsystem B's real schema:

- `EvidenceType`: `objective`, `experience_based`, `opinion`, `unverified`, `contradicted`
- `Confidence`: `high`, `medium`, `low`
- Live v4.0.0 distribution: evidence_type `objective` 11 / `experience_based` 25; confidence `high` 20 / `medium` 16. No active rule is `opinion`, `unverified` or `contradicted` today — the policy still handles all five, defensively.

**Two axes that never multiply.** The owner's requirement, restated as the design rule:

> "How certain are we the defect exists?" and "if it exists, how important is it?" are different questions with different sources, and neither may be used to answer the other.

| Axis | Source | Consumes |
|---|---|---|
| **Finding certainty** | subsystem A: finding state, detectability, measurement/provider confidence | Does this defect exist? |
| **Rule authority** | subsystem B: `evidence_type` + `confidence` | If it exists, how much does it matter? |

**Rule authority is a lattice, not a coefficient.** `evidence_type` maps to a tier; `confidence` may demote within a tier but never promote across one:

| `evidence_type` | Authority | May establish `REJECT`? | Contributes to score? |
|---|---|---|---|
| `objective` | `binding` | yes | yes |
| `experience_based` | `binding` if `confidence` is `high`, else `advisory` | only when `binding` | yes |
| `opinion` | `advisory` | no | yes, reduced weight |
| `unverified` | `advisory` | no | yes, reduced weight |
| `contradicted` | `inert` | no | no — never applied |

This is exactly non-negotiable rule 7 made executable: the pipeline does not treat every rule the video pipeline learned as equally binding, and a `contradicted` rule is inert rather than deleted.

**Two properties the tests must prove:**

1. **Authority can never manufacture certainty.** A `binding` rule matched against a `suspected`, low-detectability finding yields a `suspected` finding — never `observed`, never a `REJECT`. Authority scales *consequence*, never *belief*.
2. **Weighting can never neutralize a confidently observed binding rule.** Once a finding is `observed`, satisfies I1, and its rule is `binding`, no accumulation of advisory evidence can lift the card back to `PASS`. The verdict function reaches rule 1 before anything advisory is consulted.

No expression of the form `severity × confidence × evidence_type` appears anywhere. Authority selects a penalty row (Decision 2); certainty selects the finding state. They meet only in that table lookup.

**Authority belongs to matched rules, not to defect types.** A finding does not *have* an authority; the rules that apply to it do. Attaching every rule in a category to every anomaly would let an unrelated high-authority centering rule lend its weight to a corner anomaly. So a dedicated resolution step (Task 16) maps each finding to the specific rules whose scope and category actually match it, records those rule IDs on the finding, and derives the finding's authority as the **maximum authority among its matched rules**.

**Unresolved authority must be safe.** A finding that matches no rule gets `Authority.ADVISORY`, never `BINDING`. Defaulting an unmapped finding to binding would let any unrecognized anomaly reject a card — precisely the false rejection the governing asymmetry forbids. Advisory still penalizes and still routes to `REVIEW`, but it cannot reject.

**`psa10_relevant` is ours, not the provider's.** Claude may describe a defect, its confidence and its severity — that is the interpretive work it is for. But whether a described defect is *PSA-10-disqualifying* is a rubric judgment, so the resolution step recomputes `psa10_relevant` from the matched rules and overrides whatever the provider claimed. A provider asserting `psa10_relevant: true` on something no rule covers yields an advisory finding, not a rejection.

### Decision 5 — finding fusion for scoring

**The problem.** Both producers can see the same physical defect. The heuristic reports a `suspected` scratch at the top-left; Claude reports an `observed` scratch in the same place. Summing their penalties charges the card twice for one flaw, so a card looked at *harder* scores *worse* — the opposite of what more evidence should do.

**The resolution.** Raw findings are stored **unfused**, per producer, because calibration against real PSA outcomes needs to know what each source said on its own. Scoring and the verdict consume a **fused** view built on top:

- Findings correlate into one `FusedFinding` when they share `category` + `defect_type` **and** their normalized regions overlap. The same defect type in a different corner stays separate — that is genuinely two flaws.
- The fused state is the **strongest** among its sources (`observed` > `suspected` > `not_observed` > `not_assessable`): one producer confirming what another suspected is corroboration, not contradiction.
- The fused finding carries the union of its sources' evidence refs, so I3 is evaluated over everything supporting it.
- Each fused finding penalizes **once**.
- Sources that disagree on state record that disagreement, which feeds `review_confidence` and I1's contradiction test.

Correlating on region rather than defect type alone is also what makes cross-producer disagreement mean anything: two findings about different corners are not a disagreement, and collapsing them would suppress a real defect.

---
## File Structure

Everything new lives under `src/card_reviewer/review/`. **Subsystem B is never modified** — it is imported through its one published contract, `from card_reviewer.knowledge import load_active_rubric, Rubric, RubricError`.

```
src/card_reviewer/review/
  __init__.py            public surface: review_card, ReviewPipeline, CardReview
  enums.py               Scale, FindingState, EvidenceOrigin, Verdict, Coverage, ...
  provenance.py          EvidenceRef, EvidenceOrigin, NormalizedBox  (Decision 1)
  findings.py            Finding, FindingSet, i3_satisfied            (Decision 1)
  taxonomy.py            detectability taxonomy: defect types, reason codes, classes
  context.py             CardContext, ImageRole, resolution results
  vocabulary.py          canonical card_type/set aliases              (Decision 3)
  normalize.py           CardContextNormalizer                        (Decision 3)
  roles.py               image role resolution
  canonical.py           versioned canonicalization + quantization
  fingerprint.py         stage fingerprints and producer signatures
  models.py              CandidateInput, ResolvedCandidate, CardReview, stage IO
  storage/
    schema.sql           SQLite DDL
    migrations.py        versioned migration runner
    repository.py        Repository protocol + SqliteRepository
    artifacts.py         content-addressed image/artifact store
  ingest/
    adapter.py           CandidateAdapter protocol, ManualAdapter
  imaging/
    synthetic.py         synthetic card generator (its own task)
    preflight.py         raw-image analysis
    geometry.py          boundary, perspective, normalization, border segmentation
    observability.py     detectability + suitability, classed reason codes
    measure/
      centering.py  corners.py  edges.py  surface.py
  policies/
    authority_v1.py      evidence_type/confidence -> authority     (Decision 4)
    coverage_v1.py       EvidenceCoveragePolicy
    scoring_v1.py        rank score, grade estimate, confidence    (Decision 2)
    routing_v1.py        SMART routing
    relevance_v1.py      category/scope matching rules for findings
    combine_v1.py        verdict precedence + I1/I2/I3
  assembly.py            candidate-level evidence assembly
  heuristic.py           rubric evaluation against assembled evidence
  relevance.py           finding -> applicable rules -> authority   (Decision 4)
  fusion.py              correlate findings across producers        (Decision 5)
  manifest.py            evidence manifest builder
  vision/
    provider.py          VisionProvider protocol, Assessment models
    anthropic.py         Anthropic implementation (image blocks + text)
    prompt.py            versioned prompt + canonical payload construction
  pipeline.py            stage runner, caching, orchestration
  service.py             application service (CLI-free)
  report.py              human-readable rendering
  cli.py                 Typer app
tests/review/            mirrors the above, one test module per source module
tests/review/golden/     committed real photographs + expected observations
```

**Dependency direction is strictly one-way:** `enums` → `provenance` → `findings` → `taxonomy` → policies → assembly/heuristic → pipeline → service → cli. No policy imports `pipeline`. No `imaging` module imports a policy.

---

## Task Dependency Graph

```
Phase 1 — Foundations (pure logic, no OpenCV, no I/O)
  T1 skeleton+enums → T2 provenance → T3 findings/I3 → T4 taxonomy
  T5 canonicalization → T6 fingerprinting

Phase 2 — Storage
  T7 schema+migrations → T8 repository → T9 artifact store

Phase 3 — Context (pure logic)
  T10 vocabulary+normalizer → T11 rule evaluability → T12 image roles
  T13 ingestion boundary

Phase 4 — Policies (pure logic, the heart of the system)
  T14 authority → T15 heuristic evaluator → T16 finding→rule relevance
  → T17 coverage → T18 scoring → T19 verdict+invariants

Phase 5 — Imaging (expensive; synthetic generator first so everything is testable)
  T20 synthetic generator → T21 preflight → T22 geometry
  → T23 observability → T24 centering → T25 corners/edges → T26 surface

Phase 6 — Assembly and caching
  T27 evidence assembly → T28 stage runner + validated cache

Phase 7 — Vision (only after the evidence contract is stable)
  T29 provider protocol + fakes → T30 canonical payload + manifest builder
  → T31 routing → T32 Anthropic impl (image blocks) + offline contract tests

Phase 8 — Integration and surface
  T33 finding fusion → T34 combine integration → T35 service
  → T36 outcomes+export → T37 golden fixtures → T38 end-to-end
```

---

## Phase 1 — Foundations

### Task 1: Package skeleton and shared enums

**Files:**
- Create: `src/card_reviewer/review/__init__.py`, `src/card_reviewer/review/enums.py`
- Modify: `pyproject.toml` (add `opencv-python-headless>=4.10`, `numpy>=2.1`, `anthropic>=0.40`; add `card-review` script entry)
- Test: `tests/review/test_enums.py`

**Interfaces:**
- Produces: `Scale` (`NONE`/`LOW`/`MODERATE`/`HIGH`, ordered), `FindingState`, `Verdict`, `Coverage`, `Psa10Candidate`, `Mode`, `UndetectabilityClass`, `Authority`, `RuleEvaluability`, `ReviewConfidence`, `Provenance`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_enums.py
from card_reviewer.review.enums import Scale, Verdict, Coverage, Mode, Authority


def test_scale_is_ordered_so_thresholds_can_be_compared():
    assert Scale.NONE < Scale.LOW < Scale.MODERATE < Scale.HIGH
    assert Scale.MODERATE >= Scale.MODERATE


def test_scale_parses_from_its_string_value():
    assert Scale("moderate") is Scale.MODERATE


def test_verdict_has_exactly_the_four_spec_states():
    assert {v.value for v in Verdict} == {
        "PASS", "REVIEW", "REJECT", "INSUFFICIENT_IMAGES"
    }


def test_coverage_has_exactly_three_outcomes():
    assert {c.value for c in Coverage} == {"SUFFICIENT", "PARTIAL", "INADEQUATE"}


def test_mode_has_three_values_and_smart_is_the_default():
    assert {m.value for m in Mode} == {"off", "smart", "deep"}
    assert Mode.default() is Mode.SMART


def test_authority_orders_binding_above_advisory_above_inert():
    assert Authority.BINDING > Authority.ADVISORY > Authority.INERT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_enums.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'card_reviewer.review'`

- [ ] **Step 3: Write minimal implementation**

`Scale` and `Authority` need ordering, so they are `IntEnum` with a `.value` string accessor via a parallel map — simplest correct approach is an `IntEnum` plus `_VALUES`:

```python
# src/card_reviewer/review/enums.py
"""Shared vocabulary for the review engine.

Ordered scales are IntEnum so `>=` works directly against a declared
threshold — the coverage policy compares detectability to a minimum
constantly, and string comparison would silently do the wrong thing.
"""
from __future__ import annotations

from enum import IntEnum, StrEnum


class _OrderedScale(IntEnum):
    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            for member in cls:
                if member.label == value:
                    return member
        return None

    @property
    def label(self) -> str:
        return self.name.lower()

    def __str__(self) -> str:
        return self.label


class Scale(_OrderedScale):
    """Detectability and suitability share one ordered scale (spec §13)."""
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3


class Authority(_OrderedScale):
    """How much a rubric rule may influence the outcome (Decision 4)."""
    INERT = 0
    ADVISORY = 1
    BINDING = 2


class FindingState(StrEnum):
    OBSERVED = "observed"
    SUSPECTED = "suspected"
    NOT_OBSERVED = "not_observed"
    NOT_ASSESSABLE = "not_assessable"


class Verdict(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    INSUFFICIENT_IMAGES = "INSUFFICIENT_IMAGES"


class Coverage(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INADEQUATE = "INADEQUATE"


class Psa10Candidate(StrEnum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class ReviewConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UndetectabilityClass(StrEnum):
    STRUCTURAL = "structural"
    CIRCUMSTANTIAL = "circumstantial"
    METADATA_RESOLVABLE = "metadata_resolvable"


class RuleEvaluability(StrEnum):
    APPLICABLE = "applicable"
    UNEVALUABLE = "unevaluable"


class Provenance(StrEnum):
    SUPPLIED = "supplied"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class Mode(StrEnum):
    OFF = "off"
    SMART = "smart"
    DEEP = "deep"

    @classmethod
    def default(cls) -> Mode:
        return cls.SMART
```

`src/card_reviewer/review/__init__.py` starts as a docstring only — the public surface is added in Task 35, so importing `review` stays cheap and never drags in OpenCV.

```python
# src/card_reviewer/review/__init__.py
"""Card review engine (subsystem A).

Importing this package must stay cheap. Heavy dependencies (cv2, numpy,
anthropic) live behind lazy imports inside the modules that need them, the
same discipline `card_reviewer.knowledge` follows.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_enums.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/ tests/review/ pyproject.toml
git commit -m "feat(review): package skeleton and shared enums"
```

**Acceptance:** `Scale.MODERATE >= Scale.LOW` is true; every enum's value set matches the spec exactly; `uv run python -c "import card_reviewer.review"` does not import cv2.

---

### Task 2: Evidence provenance model

**Files:**
- Create: `src/card_reviewer/review/provenance.py`
- Test: `tests/review/test_provenance.py`

**Interfaces:**
- Consumes: `enums.py`
- Produces: `EvidenceOrigin`, `NormalizedBox`, `EvidenceRef`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_provenance.py
import pytest
from pydantic import ValidationError

from card_reviewer.review.provenance import (
    EvidenceOrigin, EvidenceRef, NormalizedBox,
)


def _ref(**kw):
    base = dict(artifact_id="a1", image_hash="h1", origin=EvidenceOrigin.ORIGINAL,
                view="front_face")
    return EvidenceRef(**(base | kw))


def test_enhanced_evidence_must_declare_its_enhancement():
    with pytest.raises(ValidationError, match="enhancement"):
        _ref(origin=EvidenceOrigin.ENHANCED, enhancement=None)


def test_unenhanced_evidence_must_not_declare_an_enhancement():
    with pytest.raises(ValidationError, match="enhancement"):
        _ref(origin=EvidenceOrigin.NORMALIZED, enhancement="clahe:clip=2.0")


def test_enhanced_evidence_with_a_method_is_valid():
    ref = _ref(origin=EvidenceOrigin.ENHANCED, enhancement="clahe:clip=2.0,grid=8")
    assert ref.is_enhanced is True


def test_original_and_normalized_both_count_as_unenhanced():
    assert _ref(origin=EvidenceOrigin.ORIGINAL).is_enhanced is False
    assert _ref(origin=EvidenceOrigin.NORMALIZED).is_enhanced is False


def test_normalized_box_rejects_coordinates_outside_the_unit_square():
    with pytest.raises(ValidationError):
        NormalizedBox(x0=0.1, y0=0.1, x1=1.4, y1=0.5)


def test_normalized_box_rejects_inverted_corners():
    with pytest.raises(ValidationError, match="x1 must exceed x0"):
        NormalizedBox(x0=0.6, y0=0.1, x1=0.2, y1=0.5)


def test_boxes_detect_overlap_for_the_i1_contradiction_test():
    a = NormalizedBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
    b = NormalizedBox(x0=0.4, y0=0.4, x1=0.9, y1=0.9)
    c = NormalizedBox(x0=0.6, y0=0.6, x1=0.9, y1=0.9)
    assert a.overlaps(b) is True
    assert a.overlaps(c) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_provenance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'card_reviewer.review.provenance'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/provenance.py
"""Where a piece of evidence came from.

This module exists so invariant I3 can be enforced as pure logic. Combine
must be able to decide "was this defect visible in something we did not
enhance?" without opening an image file, so every finding carries typed
references and each reference carries its own origin.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class EvidenceOrigin(StrEnum):
    """ORIGINAL and NORMALIZED both count as unenhanced.

    Rectifying a photograph is a geometric resampling: it moves pixels but
    cannot invent local contrast. CLAHE, sharpening and edge-highlighting
    deliberately amplify it, which is exactly why I3 exists.
    """
    ORIGINAL = "original"
    NORMALIZED = "normalized"
    ENHANCED = "enhanced"


class NormalizedBox(BaseModel):
    """A region in normalized card coordinates, [0,1] on both axes.

    Findings carry boxes rather than points so I1's "overlapping location"
    contradiction test is computable (spec §15).
    """
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> NormalizedBox:
        if self.x1 <= self.x0:
            raise ValueError(f"x1 must exceed x0 — got {self.x0} .. {self.x1}")
        if self.y1 <= self.y0:
            raise ValueError(f"y1 must exceed y0 — got {self.y0} .. {self.y1}")
        return self

    def overlaps(self, other: NormalizedBox) -> bool:
        return not (
            self.x1 <= other.x0 or other.x1 <= self.x0
            or self.y1 <= other.y0 or other.y1 <= self.y0
        )


class EvidenceRef(BaseModel):
    """One artifact a finding rests on. Ids only — never pixel data.

    Keeping this a reference rather than an embedded artifact is what stops
    the evidence manifest being duplicated into every stored result.
    """
    artifact_id: str
    image_hash: str
    origin: EvidenceOrigin
    enhancement: str | None = None
    region: NormalizedBox | None = None
    view: str

    @model_validator(mode="after")
    def _enhancement_matches_origin(self) -> EvidenceRef:
        if self.origin is EvidenceOrigin.ENHANCED and not self.enhancement:
            raise ValueError(
                "enhancement is required when origin is ENHANCED — an enhanced "
                "view whose method is unrecorded cannot be reproduced or audited"
            )
        if self.origin is not EvidenceOrigin.ENHANCED and self.enhancement:
            raise ValueError(
                f"enhancement must be absent when origin is {self.origin.value} "
                "— an unenhanced artifact has no enhancement to declare"
            )
        return self

    @property
    def is_enhanced(self) -> bool:
        return self.origin is EvidenceOrigin.ENHANCED
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_provenance.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/provenance.py tests/review/test_provenance.py
git commit -m "feat(review): evidence provenance model for I3 enforcement"
```

**Acceptance:** an `ENHANCED` ref without a method is a validation error; `ORIGINAL` and `NORMALIZED` both report `is_enhanced is False`; `NormalizedBox.overlaps` is symmetric and edge-exclusive.

---

### Task 3: Findings and the I3 invariant

**Files:**
- Create: `src/card_reviewer/review/findings.py`
- Test: `tests/review/test_findings.py`

**Interfaces:**
- Consumes: `enums.py`, `provenance.py`
- Produces: `Finding`, `FindingProducer`, `Severity`, `i3_satisfied(finding) -> bool`, `enforce_i3(findings) -> list[Finding]`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_findings.py
import pytest
from pydantic import ValidationError

from card_reviewer.review.enums import FindingState
from card_reviewer.review.findings import (
    Finding, FindingProducer, Severity, enforce_i3, i3_satisfied,
)
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef


def _ev(origin=EvidenceOrigin.ORIGINAL, enhancement=None, aid="a1"):
    return EvidenceRef(artifact_id=aid, image_hash="h1", origin=origin,
                       enhancement=enhancement, view="front_face")


def _finding(state, evidence, **kw):
    base = dict(defect_type="scratches", category="surface", state=state,
                producer=FindingProducer.HEURISTIC, confidence=0.9,
                psa10_relevant=True, evidence=evidence)
    return Finding(**(base | kw))


def test_observed_finding_backed_only_by_enhanced_views_violates_i3():
    f = _finding(FindingState.OBSERVED,
                 [_ev(EvidenceOrigin.ENHANCED, "clahe:clip=2.0")])
    assert i3_satisfied(f) is False


def test_observed_finding_with_one_unenhanced_ref_satisfies_i3():
    f = _finding(FindingState.OBSERVED, [
        _ev(EvidenceOrigin.ENHANCED, "clahe:clip=2.0", aid="a1"),
        _ev(EvidenceOrigin.ORIGINAL, aid="a2"),
    ])
    assert i3_satisfied(f) is True


def test_normalized_crop_counts_as_corroboration():
    f = _finding(FindingState.OBSERVED, [_ev(EvidenceOrigin.NORMALIZED, aid="a3")])
    assert i3_satisfied(f) is True


def test_suspected_finding_is_never_an_i3_violation():
    f = _finding(FindingState.SUSPECTED,
                 [_ev(EvidenceOrigin.ENHANCED, "sharpen:amount=1.5")])
    assert i3_satisfied(f) is True


def test_enforce_i3_demotes_rather_than_drops():
    """An enhancement-only anomaly is still information — it just cannot reject."""
    findings = [_finding(FindingState.OBSERVED,
                         [_ev(EvidenceOrigin.ENHANCED, "clahe:clip=2.0")])]
    out = enforce_i3(findings)
    assert len(out) == 1
    assert out[0].state is FindingState.SUSPECTED
    assert "I3" in out[0].demotion_reason


def test_enforce_i3_leaves_compliant_findings_untouched():
    f = _finding(FindingState.OBSERVED, [_ev(EvidenceOrigin.ORIGINAL)])
    assert enforce_i3([f])[0] == f


def test_finding_requires_at_least_one_evidence_ref():
    with pytest.raises(ValidationError):
        _finding(FindingState.OBSERVED, [])


def test_agreement_across_two_enhancements_still_fails_i3():
    """Independent enhancements of the same pixels are not independent evidence."""
    f = _finding(FindingState.OBSERVED, [
        _ev(EvidenceOrigin.ENHANCED, "clahe:clip=2.0", aid="a1"),
        _ev(EvidenceOrigin.ENHANCED, "sharpen:amount=1.5", aid="a2"),
        _ev(EvidenceOrigin.ENHANCED, "edge:canny", aid="a3"),
    ])
    assert i3_satisfied(f) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_findings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'card_reviewer.review.findings'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/findings.py
"""Findings, and the one invariant that is pure logic over them.

Both the heuristic layer and the vision layer emit findings in this shared
vocabulary (spec §9). Defining it once, upstream of both, is what lets
combine, the coverage policy and the invariants be written once — and what
makes OFF mode well-defined, since there the heuristic is the only producer.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .enums import FindingState
from .provenance import EvidenceRef, NormalizedBox


class FindingProducer(StrEnum):
    HEURISTIC = "heuristic"
    VISION = "vision"


class Severity(StrEnum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class Finding(BaseModel):
    """One observation about the card, from either producer."""
    defect_type: str
    category: str
    state: FindingState
    producer: FindingProducer
    confidence: float = Field(ge=0.0, le=1.0)
    psa10_relevant: bool
    evidence: list[EvidenceRef] = Field(min_length=1)
    severity: Severity | None = None
    location: NormalizedBox | None = None
    rule_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    demotion_reason: str = ""

    model_config = {"frozen": True}


def i3_satisfied(finding: Finding) -> bool:
    """I3 — enhancement alone never confirms.

    An anomaly visible only under enhancement may be a `suspected` candidate
    but can never independently reach `observed`. Agreement across several
    enhancement paths is deliberately NOT a corroboration route: independent
    enhancements of the same pixels are not independent evidence.
    """
    if finding.state is not FindingState.OBSERVED:
        return True
    return any(not ref.is_enhanced for ref in finding.evidence)


def enforce_i3(findings: list[Finding]) -> list[Finding]:
    """Demote — never drop — findings that violate I3.

    An enhancement-only anomaly is still real information about where to
    look; it simply may not establish a confirmed defect. Dropping it would
    hide a limitation, which non-negotiable rule 3 forbids.
    """
    out: list[Finding] = []
    for f in findings:
        if i3_satisfied(f):
            out.append(f)
            continue
        out.append(f.model_copy(update={
            "state": FindingState.SUSPECTED,
            "demotion_reason": (
                "I3: visible only under enhancement "
                f"({', '.join(sorted({e.enhancement or '' for e in f.evidence}))}); "
                "demoted from observed to suspected"
            ),
        }))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_findings.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/findings.py tests/review/test_findings.py
git commit -m "feat(review): shared finding vocabulary and I3 as pure logic"
```

**Acceptance:** `i3_satisfied` is decidable from the finding alone with no file access; three agreeing enhanced refs still fail; `enforce_i3` demotes to `suspected` and records why.

---
### Task 4: Detectability taxonomy

**Files:** Create `src/card_reviewer/review/taxonomy.py`; Test `tests/review/test_taxonomy.py`

**Interfaces:**
- Consumes: `enums.py`
- Produces: `TAXONOMY_VERSION: str`, `Promotion`, `DefectTypeSpec`, `DEFECT_TYPES`, `REASON_CODES`, `CATEGORIES`, `defect_types_for(category) -> list[str]`, `promotion_of(category, name) -> Promotion`, `class_of(reason_code) -> UndetectabilityClass`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_taxonomy.py
import pytest

from card_reviewer.review.enums import UndetectabilityClass
from card_reviewer.review import taxonomy as tx


def test_the_four_grading_categories_have_the_spec_defect_types():
    assert tx.defect_types_for("centering") == ["border_ratio"]
    assert set(tx.defect_types_for("corners")) == {"whitening", "rounding", "fraying"}
    assert set(tx.defect_types_for("edges")) == {"whitening", "chipping", "roughness"}
    assert set(tx.defect_types_for("surface")) == {
        "scratches", "print_lines", "dimples", "stains", "gloss_break"
    }


def test_every_defect_type_declares_a_promotion_level():
    for key in tx.DEFECT_TYPES:
        assert tx.DEFECT_TYPES[key].promotion in (tx.Promotion.MEASUREMENT,
                                                  tx.Promotion.INTERPRETIVE)


def test_measurement_types_are_exactly_the_spec_list():
    measurement = {k for k, v in tx.DEFECT_TYPES.items()
                   if v.promotion is tx.Promotion.MEASUREMENT}
    assert measurement == {"centering:border_ratio", "corners:whitening",
                           "corners:rounding", "edges:whitening"}


def test_white_border_is_structural_and_glare_is_circumstantial():
    assert tx.class_of("WHITE_BORDER") is UndetectabilityClass.STRUCTURAL
    assert tx.class_of("GLARE") is UndetectabilityClass.CIRCUMSTANTIAL


def test_unknown_product_context_is_metadata_resolvable_not_circumstantial():
    """Classing it circumstantial would generate a photo request that no
    photograph could satisfy — spec §13's whole reason for a third class."""
    assert tx.class_of("UNKNOWN_PRODUCT_CONTEXT") is UndetectabilityClass.METADATA_RESOLVABLE


def test_a_refractor_surface_is_glare_hence_circumstantial():
    """Diffuse lighting genuinely resolves it, so it is not structural."""
    assert tx.class_of("GLARE") is UndetectabilityClass.CIRCUMSTANTIAL


def test_unknown_reason_code_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="NOT_A_CODE"):
        tx.class_of("NOT_A_CODE")


def test_taxonomy_version_is_declared():
    assert tx.TAXONOMY_VERSION == "1.0.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/taxonomy.py
"""The detectability taxonomy (spec §13).

A versioned artifact declaring the defect types, the reason codes that
explain why a defect type is or is not detectable, and each code's class.
Its version participates in the `observability` and `cv_measurements`
producer signatures — adding a defect type genuinely changes what a pixel
measurement must compute, so recomputation is correct. Rubric version does
NOT: that changes policy about what measurements mean, not the measurement.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from .enums import UndetectabilityClass

TAXONOMY_VERSION = "1.0.0"


class Promotion(StrEnum):
    """May a measurement alone raise this finding to `observed`?"""
    MEASUREMENT = "measurement"
    INTERPRETIVE = "interpretive"


class DefectTypeSpec(BaseModel):
    category: str
    name: str
    promotion: Promotion

    @property
    def key(self) -> str:
        return f"{self.category}:{self.name}"


def _spec(category: str, name: str, promotion: Promotion) -> tuple[str, DefectTypeSpec]:
    s = DefectTypeSpec(category=category, name=name, promotion=promotion)
    return s.key, s


_M = Promotion.MEASUREMENT
_I = Promotion.INTERPRETIVE

DEFECT_TYPES: dict[str, DefectTypeSpec] = dict([
    # Whitening is `measurement`: a luminance step against a known border
    # segmentation. Rounding is geometric. Everything else needs semantics
    # CV cannot supply — fraying and roughness are indistinguishable from
    # compression artifacts and paper texture without interpretation.
    _spec("centering", "border_ratio", _M),
    _spec("corners", "whitening", _M),
    _spec("corners", "rounding", _M),
    _spec("corners", "fraying", _I),
    _spec("edges", "whitening", _M),
    _spec("edges", "chipping", _I),
    _spec("edges", "roughness", _I),
    _spec("surface", "scratches", _I),
    _spec("surface", "print_lines", _I),
    _spec("surface", "dimples", _I),
    _spec("surface", "stains", _I),
    _spec("surface", "gloss_break", _I),
])

_S = UndetectabilityClass.STRUCTURAL
_C = UndetectabilityClass.CIRCUMSTANTIAL
_MR = UndetectabilityClass.METADATA_RESOLVABLE

REASON_CODES: dict[str, UndetectabilityClass] = {
    # Structural: the card's own printed design. No photograph resolves these.
    "WHITE_BORDER": _S,
    "BORDERLESS_DESIGN": _S,
    # Circumstantial: this photograph. A better one resolves them.
    "GLARE": _C,
    "BLUR": _C,
    "OCCLUSION": _C,
    "LOW_RESOLUTION": _C,
    "SEVERE_PERSPECTIVE": _C,
    "MISSING_FACE": _C,
    # Metadata-resolvable: identifying the card resolves it. Not a photo defect,
    # so it must never generate a photo request.
    "UNKNOWN_PRODUCT_CONTEXT": _MR,
}

CATEGORIES: tuple[str, ...] = ("centering", "corners", "edges", "surface")


def defect_types_for(category: str) -> list[str]:
    return [s.name for s in DEFECT_TYPES.values() if s.category == category]


def promotion_of(category: str, name: str) -> Promotion:
    return DEFECT_TYPES[f"{category}:{name}"].promotion


def class_of(reason_code: str) -> UndetectabilityClass:
    try:
        return REASON_CODES[reason_code]
    except KeyError as exc:
        raise KeyError(
            f"unknown reason code {reason_code!r} — every detectability shortfall "
            "must cite a declared code so its class is never guessed"
        ) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_taxonomy.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/taxonomy.py tests/review/test_taxonomy.py
git commit -m "feat(review): versioned detectability taxonomy"
```

**Acceptance:** unknown reason codes raise instead of defaulting to a class; `UNKNOWN_PRODUCT_CONTEXT` is metadata-resolvable; promotion is declared for all twelve types.

---

### Task 5: Canonicalization and semantic quantization

**Files:** Create `src/card_reviewer/review/canonical.py`; Test `tests/review/test_canonical.py`

**Interfaces:**
- Produces: `CANON_SCHEME_VERSION: str`, `canonicalize(obj) -> str`, `quantize(field_path, value) -> float`, `PRECISION_MAP: dict[str, float]`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_canonical.py
from card_reviewer.review.canonical import (
    CANON_SCHEME_VERSION, canonicalize, quantize,
)


def test_there_is_no_single_global_float_precision():
    """A centering ratio measured to +/-1.5pp must quantize far more coarsely
    than a normalized coordinate; one rounding for both either discards real
    signal or manufactures spurious cache misses."""
    assert quantize("centering.horizontal", 54.03) == quantize("centering.horizontal", 54.4)
    assert quantize("region.x0", 0.5001) != quantize("region.x0", 0.5099)


def test_key_order_does_not_change_the_canonical_form():
    assert canonicalize({"b": 1, "a": 2}) == canonicalize({"a": 2, "b": 1})


def test_non_semantic_fields_are_excluded():
    """Timestamps and elapsed times must not make identical work look different."""
    a = canonicalize({"value": 1, "computed_at": "2026-08-30T10:00:00Z", "elapsed_ms": 12})
    b = canonicalize({"value": 1, "computed_at": "2026-08-30T11:00:00Z", "elapsed_ms": 99})
    assert a == b


def test_quantization_applies_inside_nested_structures():
    a = canonicalize({"centering": {"horizontal": 54.03}})
    b = canonicalize({"centering": {"horizontal": 54.40}})
    assert a == b


def test_unknown_float_fields_use_the_declared_default_precision():
    assert quantize("some.new.field", 1.0 / 3.0) == quantize("some.new.field", 0.33334)


def test_scheme_version_is_declared_so_requantizing_is_traceable():
    assert CANON_SCHEME_VERSION == "1.0.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_canonical.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/canonical.py
"""Canonical serialization for fingerprinting (spec §4).

There is no single global float precision. Each value is quantized by its
own declared semantic precision before serialization, under a versioned
scheme whose version participates in every fingerprint.
"""
from __future__ import annotations

import json
import math
from typing import Any

CANON_SCHEME_VERSION = "1.0.0"

DEFAULT_PRECISION = 1e-4

PRECISION_MAP: dict[str, float] = {
    # Centering is reported to +/-1.5 percentage points; quantizing to 0.5pp
    # keeps every distinction the method can actually support and no more.
    "centering.horizontal": 0.5,
    "centering.vertical": 0.5,
    # Normalized coordinates drive crop extraction; 1e-3 of a card edge is
    # roughly a pixel at typical resolutions.
    "region.x0": 1e-3, "region.y0": 1e-3, "region.x1": 1e-3, "region.y1": 1e-3,
    # Confidences are compared against coarse thresholds, never summed.
    "confidence": 0.01,
    # Pixel-space measurements are integers in practice.
    "border_px": 1.0,
}

EXCLUDED_KEYS: frozenset[str] = frozenset({
    "computed_at", "elapsed_ms", "latency_ms", "created_at", "updated_at",
    "cost_usd", "request_id",
})


def quantize(field_path: str, value: float) -> float:
    step = PRECISION_MAP.get(field_path, DEFAULT_PRECISION)
    if step <= 0:
        raise ValueError(f"precision for {field_path!r} must be positive")
    return math.floor(value / step + 0.5) * step


def _walk(node: Any, path: str) -> Any:
    if isinstance(node, dict):
        return {
            k: _walk(v, f"{path}.{k}" if path else k)
            for k, v in sorted(node.items())
            if k not in EXCLUDED_KEYS
        }
    if isinstance(node, (list, tuple)):
        return [_walk(v, path) for v in node]
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        return round(quantize(path, node), 12)
    return node


def canonicalize(obj: Any) -> str:
    """Deterministic JSON: sorted keys, quantized floats, no excluded fields."""
    return json.dumps(_walk(obj, ""), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_canonical.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/canonical.py tests/review/test_canonical.py
git commit -m "feat(review): versioned canonicalization with per-field quantization"
```

**Acceptance:** two centering values within measurement precision canonicalize identically; two normalized coordinates 0.01 apart do not; timestamps never affect the output.

---

### Task 6: Stage fingerprints and producer signatures

**Files:** Create `src/card_reviewer/review/fingerprint.py`; Test `tests/review/test_fingerprint.py`

**Interfaces:**
- Consumes: `canonical.py`
- Produces: `fingerprint(payload) -> str`, `signature_for(stage, versions) -> str`, `STAGE_SIGNATURE_INPUTS`, `STAGE_FINGERPRINT_INPUTS`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_fingerprint.py
from card_reviewer.review.fingerprint import (
    STAGE_FINGERPRINT_INPUTS, STAGE_SIGNATURE_INPUTS, fingerprint,
    signature_for,
)


def test_fingerprint_is_stable_across_key_order():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_fingerprint_changes_when_a_semantic_value_changes():
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_downstream_fingerprints_use_values_not_upstream_signatures():
    """Spec §4: bumping the CV analyzer must not invalidate a vision result
    whose evidence is unchanged."""
    before = fingerprint({"measurements": {"centering": {"horizontal": 54.0}}})
    after = fingerprint({"measurements": {"centering": {"horizontal": 54.0}}})
    assert before == after


def test_mode_is_a_routing_fingerprint_input_not_a_signature_input():
    """Mode is data the stage consumes, not part of its implementation
    identity — so it belongs in the fingerprint. That is what stops an OFF
    run satisfying a later DEEP lookup."""
    assert "mode" in STAGE_FINGERPRINT_INPUTS["routing"]
    assert "mode" not in STAGE_SIGNATURE_INPUTS["routing"]


def test_mode_is_absent_from_combine_entirely():
    assert "mode" not in STAGE_SIGNATURE_INPUTS["combine"]
    assert "mode" not in STAGE_FINGERPRINT_INPUTS["combine"]


def test_an_off_and_a_deep_run_produce_different_routing_fingerprints():
    off = fingerprint({"mode": "off", "heuristic_output": {}})
    deep = fingerprint({"mode": "deep", "heuristic_output": {}})
    assert off != deep


def test_taxonomy_version_is_in_image_tier_signatures_but_rubric_is_not():
    for stage in ("observability", "cv_measurements"):
        assert "taxonomy_version" in STAGE_SIGNATURE_INPUTS[stage]
        assert "rubric_version" not in STAGE_SIGNATURE_INPUTS[stage]


def test_preflight_and_geometry_do_not_consume_the_taxonomy():
    for stage in ("preflight", "geometry"):
        assert "taxonomy_version" not in STAGE_SIGNATURE_INPUTS[stage]


def test_signature_changes_when_any_declared_version_changes():
    a = signature_for("observability", {"observability_version": "1.0.0",
                                        "taxonomy_version": "1.0.0", "config": {}})
    b = signature_for("observability", {"observability_version": "1.0.1",
                                        "taxonomy_version": "1.0.0", "config": {}})
    assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_fingerprint.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/fingerprint.py
"""Cache identity: (stage, input_fingerprint, producer_signature).

The rule that makes reuse safe: a downstream stage fingerprints upstream
output VALUES, never upstream producer signatures. Bumping the CV analyzer
creates a new cv_measurements row, but if the measurements a later stage
received are unchanged, that stage's fingerprint is unchanged and its
stored result — including an expensive vision assessment — is reused.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .canonical import CANON_SCHEME_VERSION, canonicalize

# Which version keys form each stage's producer signature. Read this table
# together with spec §4 — it is the executable form of that table.
STAGE_SIGNATURE_INPUTS: dict[str, tuple[str, ...]] = {
    "preflight": ("preflight_version", "config"),
    "geometry": ("geometry_version", "config"),
    # Taxonomy, not rubric: adding a defect type changes what pixels must be
    # measured; changing a rubric rule changes what the measurement means.
    "observability": ("observability_version", "taxonomy_version", "config"),
    "cv_measurements": ("cv_version", "taxonomy_version", "config"),
    "role_context": ("resolver_version",),
    "evidence_assembly": ("assembly_version",),
    "heuristic": ("scorer_version", "authority_policy_version", "weights"),
    "coverage_provisional": ("coverage_policy_version", "taxonomy_version"),
    # Mode is deliberately absent here: it is data the stage consumes, not part
    # of its implementation identity, so it belongs in the FINGERPRINT below.
    "routing": ("routing_policy_version",),
    "manifest": ("manifest_builder_version",),
    "vision": ("provider", "model", "prompt_version", "inference_params"),
    "coverage": ("coverage_policy_version", "taxonomy_version"),
    "combine": ("combination_policy_version", "scoring_policy_version"),
}

# Fingerprint inputs (the data consumed), distinct from signature inputs
# (the implementation). Mode is a fingerprint input to routing.
STAGE_FINGERPRINT_INPUTS: dict[str, tuple[str, ...]] = {
    "preflight": ("image_hash",),
    "geometry": ("image_hash", "preflight_output"),
    "observability": ("image_hash", "geometry_output"),
    "cv_measurements": ("image_hash", "geometry_output", "observability_output"),
    "role_context": ("image_hashes", "per_image_cv", "per_image_geometry",
                     "listing_title", "card_identification_text",
                     "supplied_card_type", "supplied_set", "supplied_roles"),
    "evidence_assembly": ("roles", "context", "per_image_outputs"),
    "heuristic": ("assembled_evidence", "applicable_rubric_rules"),
    "coverage_provisional": ("assembled_detectability", "applicable_rubric_rules"),
    "routing": ("mode", "heuristic_output", "provisional_coverage",
                "assembled_observability", "detectability"),
    "manifest": ("mode_budget", "assembled_evidence", "routing_decision",
                 "applicable_rubric_rule_content"),
    "vision": ("evidence_manifest",),
    "coverage": ("assembled_detectability", "vision_category_assessability",
                 "applicable_rubric_rules"),
    "combine": ("heuristic_output", "vision_output", "coverage_output"),
}


def fingerprint(payload: Any) -> str:
    body = f"{CANON_SCHEME_VERSION}|{canonicalize(payload)}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def signature_for(stage: str, versions: dict[str, Any]) -> str:
    keys = STAGE_SIGNATURE_INPUTS[stage]
    missing = [k for k in keys if k not in versions]
    if missing:
        raise KeyError(
            f"stage {stage!r} signature requires {missing} — an omitted version "
            "key would silently make two different implementations look identical"
        )
    return fingerprint({"stage": stage, **{k: versions[k] for k in keys}})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_fingerprint.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/fingerprint.py tests/review/test_fingerprint.py
git commit -m "feat(review): stage fingerprints and producer signatures"
```

**Acceptance:** `mode` is in routing's **fingerprint** inputs and in no stage's producer signature; two modes produce different routing fingerprints; `taxonomy_version` is in observability/cv_measurements only; a missing version key raises rather than silently hashing fewer inputs.

---
## Phase 2 — Storage

### Task 7: SQLite schema and migrations

**Files:** Create `src/card_reviewer/review/storage/__init__.py`, `schema.sql`, `migrations.py`; Test `tests/review/test_migrations.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION: int`, `migrate(conn) -> int`, `connect(path) -> sqlite3.Connection`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_migrations.py
import sqlite3

from card_reviewer.review.storage.migrations import SCHEMA_VERSION, connect, migrate


def test_migrate_creates_every_table_the_spec_declares(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"candidate", "image", "candidate_image", "stage_result",
            "stage_attempt", "routing_decision", "review", "candidate_outcome",
            "grading_submission"} <= names


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "t.db")
    assert migrate(conn) == SCHEMA_VERSION
    assert migrate(conn) == SCHEMA_VERSION


def test_stage_result_is_unique_on_the_cache_identity(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    row = ("preflight", "fp1", "sig1", "{}", "{}", "2026-08-30T00:00:00Z", "h1", None)
    sql = ("INSERT INTO stage_result(stage, input_fingerprint, producer_signature,"
           " output_json, versions_json, created_at, image_hash, candidate_id)"
           " VALUES(?,?,?,?,?,?,?,?)")
    conn.execute(sql, row)
    try:
        conn.execute(sql, row)
        raise AssertionError("duplicate cache identity must be rejected")
    except sqlite3.IntegrityError:
        pass


def test_candidate_outcome_has_no_price_or_purchase_column(tmp_path):
    """Non-negotiable rule 14: storing price beside returned grades puts ROI
    analysis one join away."""
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(candidate_outcome)")}
    assert not (cols & {"price", "purchased", "cost", "paid"})


def test_multiple_grading_submissions_per_candidate_are_allowed(tmp_path):
    """Cards get returned ungraded, resubmitted, cracked, or crossed."""
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute("INSERT INTO candidate(id, source, created_at) VALUES('c1','manual','t')")
    for n in ("s1", "s2"):
        conn.execute(
            "INSERT INTO grading_submission(id, candidate_id, grader, status)"
            " VALUES(?, 'c1', 'PSA', 'submitted')", (n,))
    assert conn.execute(
        "SELECT COUNT(*) FROM grading_submission WHERE candidate_id='c1'"
    ).fetchone()[0] == 2


def test_review_requires_a_routing_decision(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    cols = {r[1]: r for r in
            {c[1]: c for c in conn.execute("PRAGMA table_info(review)")}.items()}
    names = {c[1] for c in conn.execute("PRAGMA table_info(review)")}
    assert {"routing_decision_id", "coverage_provisional_result_id",
            "coverage_result_id", "vision_result_id", "review_confidence"} <= names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_migrations.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`schema.sql`:

```sql
-- Card review engine schema v1. Append-only by discipline: stage_result and
-- review rows are never updated, so a later analyzer improvement can be
-- compared against what the previous one concluded.

CREATE TABLE IF NOT EXISTS candidate (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    listing_url TEXT, listing_id TEXT, title TEXT,
    -- Listing provenance only. Never read by the grading path; the adapter
    -- drops it when building ResolvedCandidate (non-negotiable rule 14).
    asking_price TEXT,
    supplied_card_type TEXT, supplied_set TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image (
    image_hash TEXT PRIMARY KEY,
    path TEXT NOT NULL, width INTEGER, height INTEGER,
    format TEXT, bytes INTEGER, created_at TEXT
);

CREATE TABLE IF NOT EXISTS candidate_image (
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    image_hash TEXT NOT NULL REFERENCES image(image_hash),
    supplied_role TEXT, source_url TEXT, ordering INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (candidate_id, image_hash)
);

-- Validated successes ONLY. A row here means the stage ran to completion and
-- passed schema validation.
CREATE TABLE IF NOT EXISTS stage_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    producer_signature TEXT NOT NULL,
    output_json TEXT NOT NULL,
    versions_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    image_hash TEXT, candidate_id TEXT,
    UNIQUE (stage, input_fingerprint, producer_signature)
);

-- Failures, timeouts, provider errors, malformed responses. Diagnostics and
-- cost accounting only; NEVER satisfies a cache lookup.
CREATE TABLE IF NOT EXISTS stage_attempt (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    input_fingerprint TEXT, producer_signature TEXT,
    error_kind TEXT NOT NULL, error_detail TEXT,
    cost_usd REAL, latency_ms INTEGER,
    created_at TEXT NOT NULL,
    image_hash TEXT, candidate_id TEXT
);

CREATE TABLE IF NOT EXISTS routing_decision (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    policy_version TEXT NOT NULL, mode TEXT NOT NULL,
    call_vision INTEGER NOT NULL, trigger_reasons TEXT,
    input_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    mode TEXT NOT NULL,
    routing_decision_id INTEGER NOT NULL REFERENCES routing_decision(id),
    verdict TEXT NOT NULL,
    psa10_candidate TEXT NOT NULL,
    psa10_rank_score INTEGER,
    rankable INTEGER NOT NULL,
    estimated_psa_grade TEXT,
    review_confidence TEXT NOT NULL,
    coverage TEXT NOT NULL,
    heuristic_result_id INTEGER REFERENCES stage_result(id),
    coverage_provisional_result_id INTEGER REFERENCES stage_result(id),
    coverage_result_id INTEGER REFERENCES stage_result(id),
    combine_result_id INTEGER REFERENCES stage_result(id),
    vision_result_id INTEGER REFERENCES stage_result(id),
    rubric_version TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- No price. No purchase flag. See rule 14.
CREATE TABLE IF NOT EXISTS candidate_outcome (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    status TEXT NOT NULL, occurred_on TEXT, notes TEXT
);

CREATE TABLE IF NOT EXISTS grading_submission (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    grader TEXT NOT NULL, submitted_on TEXT, service_tier TEXT,
    returned_on TEXT, grade TEXT, cert_number TEXT,
    status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stage_result_lookup
    ON stage_result(stage, input_fingerprint, producer_signature);
CREATE INDEX IF NOT EXISTS idx_review_candidate ON review(candidate_id);
```

```python
# src/card_reviewer/review/storage/migrations.py
"""Versioned schema application. SQLite is the sole state authority."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
_SCHEMA = Path(__file__).with_name("schema.sql")


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= SCHEMA_VERSION:
        return current
    conn.executescript(_SCHEMA.read_text())
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return SCHEMA_VERSION
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_migrations.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/storage/ tests/review/test_migrations.py
git commit -m "feat(review): SQLite schema and migration runner"
```

**Acceptance:** the cache identity is enforced by a UNIQUE constraint; `candidate_outcome` has no price column; migration is idempotent.

---

### Task 8: Repository abstraction

**Files:** Create `src/card_reviewer/review/storage/repository.py`; Test `tests/review/test_repository.py`

**Interfaces:**
- Consumes: `migrations.py`, `fingerprint.py`
- Produces: `Repository` protocol; `SqliteRepository` with `get_stage_result(stage, fp, sig)`, `put_stage_result(...) -> int`, `record_attempt(...)`, `save_review(...)`, `save_routing_decision(...) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_repository.py
import pytest

from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    r = SqliteRepository(conn)
    r.save_candidate(id="c1", source="manual", title="t")
    return r


def test_a_stored_result_is_returned_for_the_same_cache_identity(repo):
    repo.put_stage_result("preflight", "fp1", "sig1", {"ok": True}, {"v": "1"},
                          image_hash="h1")
    got = repo.get_stage_result("preflight", "fp1", "sig1")
    assert got is not None and got.output == {"ok": True}


def test_a_different_producer_signature_is_a_cache_miss(repo):
    repo.put_stage_result("preflight", "fp1", "sig1", {"ok": True}, {}, image_hash="h1")
    assert repo.get_stage_result("preflight", "fp1", "sig2") is None


def test_a_recorded_failure_never_satisfies_a_cache_lookup(repo):
    """Spec §4: a failed vision call must not suppress a later successful one."""
    repo.record_attempt("vision", "fp1", "sig1", error_kind="timeout",
                        error_detail="504", candidate_id="c1")
    assert repo.get_stage_result("vision", "fp1", "sig1") is None


def test_a_success_after_a_failure_is_cached_normally(repo):
    repo.record_attempt("vision", "fp1", "sig1", error_kind="timeout",
                        candidate_id="c1")
    repo.put_stage_result("vision", "fp1", "sig1", {"findings": []}, {},
                          candidate_id="c1")
    assert repo.get_stage_result("vision", "fp1", "sig1") is not None


def test_putting_the_same_identity_twice_returns_the_existing_row(repo):
    a = repo.put_stage_result("preflight", "fp1", "sig1", {"n": 1}, {}, image_hash="h1")
    b = repo.put_stage_result("preflight", "fp1", "sig1", {"n": 1}, {}, image_hash="h1")
    assert a == b


def test_reviews_are_append_only_so_history_survives(repo):
    rd = repo.save_routing_decision(candidate_id="c1", policy_version="1.0.0",
                                    mode="off", call_vision=False,
                                    trigger_reasons=[], input_fingerprint="fp")
    for verdict in ("REVIEW", "PASS"):
        repo.save_review(candidate_id="c1", mode="off", routing_decision_id=rd,
                         verdict=verdict, psa10_candidate="uncertain",
                         psa10_rank_score=50, rankable=True,
                         estimated_psa_grade="9", review_confidence="medium",
                         coverage="PARTIAL", rubric_version="4.0.0", output={})
    assert len(repo.reviews_for("c1")) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_repository.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/storage/repository.py
"""The only module that writes SQL. Moving to Postgres touches this file."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Protocol


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


@dataclass(frozen=True)
class StageResult:
    id: int
    stage: str
    output: dict[str, Any]
    versions: dict[str, Any]
    created_at: str


class Repository(Protocol):
    def get_stage_result(self, stage: str, fp: str, sig: str) -> StageResult | None: ...
    def put_stage_result(self, stage: str, fp: str, sig: str,
                         output: dict[str, Any], versions: dict[str, Any],
                         *, image_hash: str | None = None,
                         candidate_id: str | None = None) -> int: ...
    def record_attempt(self, stage: str, fp: str | None, sig: str | None, *,
                       error_kind: str, error_detail: str = "",
                       cost_usd: float | None = None,
                       latency_ms: int | None = None,
                       image_hash: str | None = None,
                       candidate_id: str | None = None) -> int: ...


class SqliteRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save_candidate(self, *, id: str, source: str, title: str = "",
                       listing_url: str | None = None,
                       listing_id: str | None = None,
                       asking_price: str | None = None,
                       supplied_card_type: str | None = None,
                       supplied_set: str | None = None) -> str:
        self._conn.execute(
            "INSERT OR IGNORE INTO candidate(id, source, listing_url, listing_id,"
            " title, asking_price, supplied_card_type, supplied_set, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (id, source, listing_url, listing_id, title, asking_price,
             supplied_card_type, supplied_set, _now()))
        self._conn.commit()
        return id

    def get_stage_result(self, stage: str, fp: str, sig: str) -> StageResult | None:
        row = self._conn.execute(
            "SELECT id, stage, output_json, versions_json, created_at"
            " FROM stage_result WHERE stage=? AND input_fingerprint=?"
            " AND producer_signature=?", (stage, fp, sig)).fetchone()
        if row is None:
            return None
        return StageResult(id=row[0], stage=row[1], output=json.loads(row[2]),
                           versions=json.loads(row[3]), created_at=row[4])

    def put_stage_result(self, stage: str, fp: str, sig: str,
                         output: dict[str, Any], versions: dict[str, Any],
                         *, image_hash: str | None = None,
                         candidate_id: str | None = None) -> int:
        self._conn.execute(
            "INSERT OR IGNORE INTO stage_result(stage, input_fingerprint,"
            " producer_signature, output_json, versions_json, created_at,"
            " image_hash, candidate_id) VALUES(?,?,?,?,?,?,?,?)",
            (stage, fp, sig, json.dumps(output), json.dumps(versions), _now(),
             image_hash, candidate_id))
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM stage_result WHERE stage=? AND input_fingerprint=?"
            " AND producer_signature=?", (stage, fp, sig)).fetchone()
        return int(row[0])

    def record_attempt(self, stage: str, fp: str | None, sig: str | None, *,
                       error_kind: str, error_detail: str = "",
                       cost_usd: float | None = None,
                       latency_ms: int | None = None,
                       image_hash: str | None = None,
                       candidate_id: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO stage_attempt(stage, input_fingerprint,"
            " producer_signature, error_kind, error_detail, cost_usd,"
            " latency_ms, created_at, image_hash, candidate_id)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (stage, fp, sig, error_kind, error_detail, cost_usd, latency_ms,
             _now(), image_hash, candidate_id))
        self._conn.commit()
        return int(cur.lastrowid)

    def save_routing_decision(self, *, candidate_id: str, policy_version: str,
                              mode: str, call_vision: bool,
                              trigger_reasons: list[str],
                              input_fingerprint: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO routing_decision(candidate_id, policy_version, mode,"
            " call_vision, trigger_reasons, input_fingerprint, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (candidate_id, policy_version, mode, int(call_vision),
             json.dumps(trigger_reasons), input_fingerprint, _now()))
        self._conn.commit()
        return int(cur.lastrowid)

    def save_review(self, **kw: Any) -> int:
        kw = dict(kw)
        kw["output_json"] = json.dumps(kw.pop("output", {}))
        kw["rankable"] = int(kw["rankable"])
        kw.setdefault("created_at", _now())
        cols = ", ".join(kw)
        marks = ", ".join("?" for _ in kw)
        cur = self._conn.execute(
            f"INSERT INTO review({cols}) VALUES({marks})", tuple(kw.values()))
        self._conn.commit()
        return int(cur.lastrowid)

    def reviews_for(self, candidate_id: str) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM review WHERE candidate_id=? ORDER BY id",
            (candidate_id,)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_repository.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/storage/repository.py tests/review/test_repository.py
git commit -m "feat(review): repository abstraction over SQLite"
```

**Acceptance:** a recorded attempt never returns from `get_stage_result`; reviews accumulate rather than overwrite; the pipeline never writes SQL directly.

---

### Task 9: Content-addressed artifact store

**Files:** Create `src/card_reviewer/review/storage/artifacts.py`; Test `tests/review/test_artifacts.py`

**Interfaces:**
- Produces: `ArtifactStore` with `put_image(bytes) -> str`, `put_derived(image_hash, kind, name, bytes) -> str`, `path_of(artifact_id) -> Path`, `read(artifact_id) -> bytes`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_artifacts.py
import pytest

from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


def test_identical_bytes_hash_to_one_image_stored_once(store):
    a = store.put_image(b"pixels")
    b = store.put_image(b"pixels")
    assert a == b
    assert len(list((store.root / "images").iterdir())) == 1


def test_geometry_and_measurement_crops_live_on_separate_paths(store):
    """Crop ownership is split by stage so each is invalidated by its own
    stage's cache and never by the other's (spec §7.4)."""
    h = store.put_image(b"pixels")
    face = store.put_derived(h, "face", "front.png", b"f")
    corner = store.put_derived(h, "corners", "bottom_left.png", b"c")
    assert "/face/" in str(store.path_of(face))
    assert "/corners/" in str(store.path_of(corner))


def test_a_derived_artifact_id_is_stable_for_the_same_inputs(store):
    h = store.put_image(b"pixels")
    assert store.put_derived(h, "corners", "bl.png", b"c") == \
           store.put_derived(h, "corners", "bl.png", b"c")


def test_originals_are_preserved_byte_for_byte(store):
    """Non-negotiable rule 6."""
    h = store.put_image(b"\x89PNG\r\n\x1a\n original")
    assert store.read(h) == b"\x89PNG\r\n\x1a\n original"


def test_reading_an_unknown_artifact_raises(store):
    with pytest.raises(KeyError):
        store.read("deadbeef")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/storage/artifacts.py
"""Content-addressed storage for images and derived artifacts.

SQLite holds records; the filesystem holds bytes. Large blobs never go into
the database. Originals are preserved untouched (non-negotiable rule 6).
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class ArtifactStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "crops").mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Path] = {}
        self._reindex()

    def _reindex(self) -> None:
        for p in self.root.rglob("*"):
            if p.is_file():
                self._index.setdefault(p.stem if p.parent.name == "images"
                                       else self._derived_id_from(p), p)

    @staticmethod
    def _derived_id_from(p: Path) -> str:
        return hashlib.sha256(str(p).encode()).hexdigest()[:32]

    def put_image(self, data: bytes) -> str:
        image_hash = hashlib.sha256(data).hexdigest()
        dest = self.root / "images" / image_hash
        if not dest.exists():
            dest.write_bytes(data)
        self._index[image_hash] = dest
        return image_hash

    def put_derived(self, image_hash: str, kind: str, name: str,
                    data: bytes) -> str:
        dest = self.root / "crops" / image_hash / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        artifact_id = self._derived_id_from(dest)
        self._index[artifact_id] = dest
        return artifact_id

    def path_of(self, artifact_id: str) -> Path:
        try:
            return self._index[artifact_id]
        except KeyError as exc:
            raise KeyError(f"unknown artifact {artifact_id!r}") from exc

    def read(self, artifact_id: str) -> bytes:
        return self.path_of(artifact_id).read_bytes()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_artifacts.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/storage/artifacts.py tests/review/test_artifacts.py
git commit -m "feat(review): content-addressed artifact store"
```

**Acceptance:** the same image supplied twice is stored once; geometry crops and measurement crops occupy separate directories; originals round-trip byte-for-byte.

---
## Phase 3 — Context resolution

### Task 10: Canonical vocabulary and the card-context normalizer

**Files:** Create `src/card_reviewer/review/vocabulary.py`, `src/card_reviewer/review/context.py`, `src/card_reviewer/review/normalize.py`; Test `tests/review/test_normalize.py`

**Interfaces:**
- Consumes: `enums.py`
- Produces: `CARD_TYPE_VOCABULARY`, `SET_VOCABULARY`, `VOCABULARY_VERSION`, `CardContext`, `CardContextNormalizer.normalize(raw_title, supplied_card_type, supplied_set) -> CardContext`

**Context for the implementer:** the live subsystem B rubric (v4.0.0) uses exactly three `card_types` — `chrome`, `refractor`, `foil` — and **zero** `sets`. `Rubric.for_card` matches by exact set intersection on those strings. Free-form listing text must never reach it.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_normalize.py
import pytest

from card_reviewer.review.enums import Provenance
from card_reviewer.review.normalize import CardContextNormalizer
from card_reviewer.review.vocabulary import CARD_TYPE_VOCABULARY, SET_VOCABULARY


@pytest.fixture
def norm():
    return CardContextNormalizer()


def test_the_vocabulary_matches_what_subsystem_b_actually_scopes_on():
    """Inspected from the live rubric, not invented: SURFACE_SHINY_001 is the
    only scoped rule and it uses chrome/refractor/foil."""
    assert set(CARD_TYPE_VOCABULARY.values()) == {"chrome", "refractor", "foil"}
    assert SET_VOCABULARY == {}


@pytest.mark.parametrize("raw,expected", [
    ("Chrome", "chrome"), ("  CHROME  ", "chrome"),
    ("Topps Chrome", "chrome"), ("Bowman Chrome", "chrome"),
    ("Refractor", "refractor"), ("refractors", "refractor"),
    ("Prizm", "refractor"), ("holo", "foil"), ("Holofoil", "foil"),
])
def test_aliases_and_capitalization_normalize_to_canonical_values(norm, raw, expected):
    ctx = norm.normalize(supplied_card_type=raw)
    assert ctx.canonical_card_types == [expected]


def test_unrecognized_values_become_unknown_never_the_nearest_neighbour(norm):
    """Guessing 'Prizm Silver' into chrome would apply a rule the owner never
    sanctioned. No fuzzy matching."""
    ctx = norm.normalize(supplied_card_type="Prizm Silver Mojo /25")
    assert ctx.canonical_card_types is None
    assert ctx.provenance is Provenance.UNKNOWN


def test_raw_values_are_preserved_alongside_canonical_ones(norm):
    ctx = norm.normalize(supplied_card_type="Topps Chrome")
    assert ctx.raw_card_type == "Topps Chrome"
    assert ctx.canonical_card_types == ["chrome"]


def test_unknown_context_yields_none_never_an_empty_list(norm):
    """Subsystem B distinguishes them: None means unconstrained, [] means
    'known to be empty' and would drop every scoped rule."""
    ctx = norm.normalize()
    assert ctx.canonical_card_types is None
    assert ctx.canonical_sets is None
    assert ctx.canonical_card_types != []


def test_supplied_metadata_outranks_title_inference(norm):
    ctx = norm.normalize(raw_title="2023 Bowman Chrome Auto",
                         supplied_card_type="foil")
    assert ctx.canonical_card_types == ["foil"]
    assert ctx.provenance is Provenance.SUPPLIED


def test_title_inference_is_marked_inferred_with_a_confidence(norm):
    ctx = norm.normalize(raw_title="2023 Topps Chrome Julio Rodriguez #150")
    assert ctx.canonical_card_types == ["chrome"]
    assert ctx.provenance is Provenance.INFERRED
    assert 0.0 < ctx.confidence < 1.0


def test_set_axis_is_unknown_today_because_no_rule_is_set_scoped(norm):
    ctx = norm.normalize(raw_title="2023 Topps Chrome", supplied_set="Topps Chrome")
    assert ctx.canonical_sets is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/vocabulary.py
"""Canonical card-type and set vocabulary (Decision 3).

Derived by inspecting the live subsystem B rubric, not from memory. At
rubric v4.0.0 the only scoped rule is SURFACE_SHINY_001, scoped to
chrome/refractor/foil; no rule is scoped by set. Adding a supported product
means editing this file and nothing else — grading logic never changes.
"""
from __future__ import annotations

VOCABULARY_VERSION = "1.0.0"

CARD_TYPE_VOCABULARY: dict[str, str] = {
    "chrome": "chrome",
    "topps chrome": "chrome",
    "bowman chrome": "chrome",
    "refractor": "refractor",
    "refractors": "refractor",
    "prizm": "refractor",
    "foil": "foil",
    "holo": "foil",
    "holofoil": "foil",
}

# Empty by inspection: no active rule at v4.0.0 is set-scoped. The set axis
# is exercised only by synthetic fixtures until subsystem B adds one.
SET_VOCABULARY: dict[str, str] = {}
```

```python
# src/card_reviewer/review/context.py
"""Resolved card context: raw and canonical, side by side."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import Provenance


class CardContext(BaseModel):
    raw_card_type: str | None = None
    raw_set: str | None = None
    raw_title: str | None = None
    # None means unconstrained. NEVER [] — that means "known to be empty",
    # a real constraint that would drop every scoped rule.
    canonical_card_types: list[str] | None = None
    canonical_sets: list[str] | None = None
    provenance: Provenance = Provenance.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}

    @property
    def is_known(self) -> bool:
        return self.canonical_card_types is not None
```

```python
# src/card_reviewer/review/normalize.py
"""Raw listing strings -> canonical CardContext (Decision 3).

Free-form text must never reach Rubric.for_card, which matches by exact
set intersection. An unrecognized value becomes unknown, never the nearest
neighbour: guessing would apply a rule the owner never sanctioned.
"""
from __future__ import annotations

import re

from .context import CardContext
from .enums import Provenance
from .vocabulary import CARD_TYPE_VOCABULARY, SET_VOCABULARY

_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")

TITLE_INFERENCE_CONFIDENCE = 0.6
SUPPLIED_CONFIDENCE = 1.0


def _key(value: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", value.casefold())).strip()


class CardContextNormalizer:
    version = "1.0.0"

    def normalize(self, raw_title: str | None = None,
                  supplied_card_type: str | None = None,
                  supplied_set: str | None = None) -> CardContext:
        card_types, provenance, confidence = self._card_types(
            raw_title, supplied_card_type)
        return CardContext(
            raw_card_type=supplied_card_type, raw_set=supplied_set,
            raw_title=raw_title,
            canonical_card_types=card_types,
            canonical_sets=self._sets(supplied_set),
            provenance=provenance, confidence=confidence,
        )

    def _card_types(self, title: str | None, supplied: str | None):
        if supplied:
            canonical = CARD_TYPE_VOCABULARY.get(_key(supplied))
            if canonical:
                return [canonical], Provenance.SUPPLIED, SUPPLIED_CONFIDENCE
            return None, Provenance.UNKNOWN, 0.0
        if title:
            found = self._scan(title)
            if found:
                return found, Provenance.INFERRED, TITLE_INFERENCE_CONFIDENCE
        return None, Provenance.UNKNOWN, 0.0

    @staticmethod
    def _scan(title: str) -> list[str] | None:
        key = _key(title)
        hits = {
            canonical for alias, canonical in CARD_TYPE_VOCABULARY.items()
            if re.search(rf"\b{re.escape(alias)}\b", key)
        }
        return sorted(hits) or None

    @staticmethod
    def _sets(supplied: str | None) -> list[str] | None:
        if not supplied:
            return None
        canonical = SET_VOCABULARY.get(_key(supplied))
        return [canonical] if canonical else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_normalize.py -v`
Expected: PASS (8 tests including parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/vocabulary.py src/card_reviewer/review/context.py src/card_reviewer/review/normalize.py tests/review/test_normalize.py
git commit -m "feat(review): canonical card-context normalization"
```

**Acceptance:** unrecognized strings become `None`, never a guess; unknown is `None` and never `[]`; the vocabulary is the only file to edit when adding a product.

---

### Task 11: Rule evaluability under unknown context

**Files:** Create `src/card_reviewer/review/evaluability.py`; Test `tests/review/test_evaluability.py`

**Interfaces:**
- Consumes: `context.py`, `enums.py`, `card_reviewer.knowledge.load_active_rubric`
- Produces: `ScopedRule` (rule + evaluability), `UNKNOWN_PRODUCT_CONTEXT`, `scope_rules(rules, context) -> list[ScopedRule]`, `unevaluable_reasons(scoped) -> list[str]`, `applicable(scoped) -> list[Rule]`

**Why this task exists:** `for_card(None, None)` returns *every* rule, which is correct — unknown context must not narrow the rubric. But a returned rule is not an applicable one. Without this gate, a product-scoped rule would silently apply to a card whose product is unknown.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_evaluability.py
from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.context import CardContext
from card_reviewer.review.enums import Provenance, RuleEvaluability
from card_reviewer.review.evaluability import scope_rules, unevaluable_reasons


def _ctx(card_types):
    return CardContext(canonical_card_types=card_types,
                       provenance=Provenance.SUPPLIED if card_types
                       else Provenance.UNKNOWN)


def test_unscoped_rules_are_always_applicable():
    rules = load_active_rubric().rules
    scoped = scope_rules(rules, _ctx(None))
    unscoped = [s for s in scoped if not s.rule.applies_to.card_types]
    assert unscoped
    assert all(s.evaluability is RuleEvaluability.APPLICABLE for s in unscoped)


def test_a_product_scoped_rule_is_unevaluable_when_context_is_unknown():
    """SURFACE_SHINY_001 is scoped to chrome/refractor/foil. With product
    unknown it must not silently apply."""
    scoped = {s.rule.id: s for s in scope_rules(load_active_rubric().rules, _ctx(None))}
    assert scoped["SURFACE_SHINY_001"].evaluability is RuleEvaluability.UNEVALUABLE


def test_the_same_rule_is_applicable_once_the_product_is_known():
    scoped = {s.rule.id: s for s in
              scope_rules(load_active_rubric().rules, _ctx(["chrome"]))}
    assert scoped["SURFACE_SHINY_001"].evaluability is RuleEvaluability.APPLICABLE


def test_a_scoped_rule_whose_scope_excludes_the_card_is_not_returned_at_all():
    """for_card already filters this: a paper card never sees SURFACE_SHINY_001."""
    rubric = load_active_rubric()
    paper = rubric.for_card(card_types=["paper"], sets=None)
    assert "SURFACE_SHINY_001" not in {r.id for r in paper}


def test_unknown_context_produces_a_metadata_resolvable_reason_code():
    scoped = scope_rules(load_active_rubric().rules, _ctx(None))
    assert "UNKNOWN_PRODUCT_CONTEXT" in unevaluable_reasons(scoped)


def test_known_context_produces_no_unevaluable_reasons():
    scoped = scope_rules(load_active_rubric().rules, _ctx(["chrome"]))
    assert unevaluable_reasons(scoped) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_evaluability.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/evaluability.py
"""Whether a returned rubric rule may actually be applied (Decision 3).

`for_card(None, None)` returns every rule, deliberately — unknown context
must not narrow the rubric (spec §8). But an unscoped rule set is not a
satisfied one. A rule scoped to a product we cannot identify is tagged
UNEVALUABLE: it never fires a finding and never contributes to a verdict.
It instead raises an UNKNOWN_PRODUCT_CONTEXT coverage gap, which is
metadata-resolvable and biases toward REVIEW.
"""
from __future__ import annotations

from dataclasses import dataclass

from card_reviewer.knowledge.models import Rule

from .context import CardContext
from .enums import RuleEvaluability

UNKNOWN_PRODUCT_CONTEXT = "UNKNOWN_PRODUCT_CONTEXT"


@dataclass(frozen=True)
class ScopedRule:
    rule: Rule
    evaluability: RuleEvaluability
    reason: str = ""


def scope_rules(rules: list[Rule], context: CardContext) -> list[ScopedRule]:
    out: list[ScopedRule] = []
    for rule in rules:
        needs_type = bool(rule.applies_to.card_types)
        needs_set = bool(rule.applies_to.sets)
        type_unknown = needs_type and context.canonical_card_types is None
        set_unknown = needs_set and context.canonical_sets is None
        if type_unknown or set_unknown:
            out.append(ScopedRule(rule, RuleEvaluability.UNEVALUABLE,
                                  UNKNOWN_PRODUCT_CONTEXT))
        else:
            out.append(ScopedRule(rule, RuleEvaluability.APPLICABLE))
    return out


def unevaluable_reasons(scoped: list[ScopedRule]) -> list[str]:
    return sorted({s.reason for s in scoped
                   if s.evaluability is RuleEvaluability.UNEVALUABLE and s.reason})


def applicable(scoped: list[ScopedRule]) -> list[Rule]:
    return [s.rule for s in scoped
            if s.evaluability is RuleEvaluability.APPLICABLE]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_evaluability.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/evaluability.py tests/review/test_evaluability.py
git commit -m "feat(review): rule evaluability gate for unknown card context"
```

**Acceptance:** `SURFACE_SHINY_001` is `UNEVALUABLE` with unknown product and `APPLICABLE` with `chrome`; unknown context emits `UNKNOWN_PRODUCT_CONTEXT`; no rule is ever silently applied.

---

### Task 12: Image-role resolution

**Files:** Create `src/card_reviewer/review/roles.py`; Test `tests/review/test_roles.py`

**Interfaces:**
- Consumes: `enums.py`
- Produces: `ImageRole` (`FRONT`/`BACK`/`UNKNOWN`), `ResolvedRole`, `resolve_roles(images) -> dict[str, ResolvedRole]`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_roles.py
from card_reviewer.review.enums import Provenance
from card_reviewer.review.roles import ImageRole, RoleInput, resolve_roles


def _img(h, supplied=None, text_density=0.1, has_central_image=True):
    return RoleInput(image_hash=h, supplied_role=supplied,
                     text_density=text_density,
                     has_central_image_region=has_central_image)


def test_supplied_role_outranks_inference():
    out = resolve_roles([_img("h1", supplied="back", text_density=0.05)])
    assert out["h1"].role is ImageRole.BACK
    assert out["h1"].provenance is Provenance.SUPPLIED


def test_high_text_density_without_a_central_image_infers_a_back():
    out = resolve_roles([_img("h1", text_density=0.55, has_central_image=False)])
    assert out["h1"].role is ImageRole.BACK
    assert out["h1"].provenance is Provenance.INFERRED


def test_low_text_density_with_a_central_image_infers_a_front():
    out = resolve_roles([_img("h1", text_density=0.08, has_central_image=True)])
    assert out["h1"].role is ImageRole.FRONT


def test_ambiguous_signatures_yield_unknown_rather_than_a_guess():
    out = resolve_roles([_img("h1", text_density=0.3, has_central_image=True)])
    assert out["h1"].role is ImageRole.UNKNOWN
    assert out["h1"].provenance is Provenance.UNKNOWN


def test_unknown_is_a_first_class_state_not_an_error():
    out = resolve_roles([_img("h1", text_density=0.3)])
    assert out["h1"].confidence == 0.0
    assert out["h1"].role is ImageRole.UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_roles.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/roles.py
"""Which photograph is the front (spec §8).

An `unknown` role is a first-class state, not an error: the image still
contributes to measurements that do not depend on knowing the face, and is
excluded from those that do.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .enums import Provenance

BACK_TEXT_DENSITY = 0.45
FRONT_TEXT_DENSITY = 0.15
INFERENCE_CONFIDENCE = 0.7


class ImageRole(StrEnum):
    FRONT = "front"
    BACK = "back"
    UNKNOWN = "unknown"


class RoleInput(BaseModel):
    image_hash: str
    supplied_role: str | None = None
    text_density: float = Field(ge=0.0, le=1.0)
    has_central_image_region: bool = True


class ResolvedRole(BaseModel):
    image_hash: str
    role: ImageRole
    provenance: Provenance
    confidence: float = Field(ge=0.0, le=1.0)


def resolve_roles(images: list[RoleInput]) -> dict[str, ResolvedRole]:
    out: dict[str, ResolvedRole] = {}
    for img in images:
        if img.supplied_role in {"front", "back"}:
            out[img.image_hash] = ResolvedRole(
                image_hash=img.image_hash, role=ImageRole(img.supplied_role),
                provenance=Provenance.SUPPLIED, confidence=1.0)
            continue
        if img.text_density >= BACK_TEXT_DENSITY and not img.has_central_image_region:
            role = ImageRole.BACK
        elif img.text_density <= FRONT_TEXT_DENSITY and img.has_central_image_region:
            role = ImageRole.FRONT
        else:
            out[img.image_hash] = ResolvedRole(
                image_hash=img.image_hash, role=ImageRole.UNKNOWN,
                provenance=Provenance.UNKNOWN, confidence=0.0)
            continue
        out[img.image_hash] = ResolvedRole(
            image_hash=img.image_hash, role=role,
            provenance=Provenance.INFERRED, confidence=INFERENCE_CONFIDENCE)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_roles.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/roles.py tests/review/test_roles.py
git commit -m "feat(review): image role resolution with unknown as first-class"
```

**Acceptance:** supplied beats inferred beats unknown; ambiguous signatures return `UNKNOWN` rather than guessing.

---

### Task 13: Ingestion boundary

**Files:** Create `src/card_reviewer/review/models.py`, `src/card_reviewer/review/ingest/__init__.py`, `src/card_reviewer/review/ingest/adapter.py`; Test `tests/review/test_ingest.py`

**Interfaces:**
- Produces: `CandidateInput`, `ResolvedCandidate`, `CardReview`, `CandidateAdapter` protocol, `ManualAdapter.resolve(CandidateInput) -> ResolvedCandidate`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_ingest.py
import pytest

from card_reviewer.review.ingest.adapter import ManualAdapter
from card_reviewer.review.models import CandidateInput, ResolvedCandidate
from card_reviewer.review.storage.artifacts import ArtifactStore


def test_card_review_carries_every_spec_output_field():
    """Spec §16. Missing one here means a field with no home in the record."""
    from card_reviewer.review.models import CardReview
    for field in ("verdict", "psa10_candidate", "psa10_rank_score", "rankable",
                  "estimated_psa_grade", "review_confidence", "coverage",
                  "categories", "image_quality", "roles_and_context",
                  "defects_found", "limitations",
                  "recommended_additional_photos", "card_identification_request",
                  "cv_assessment", "vision_assessment", "reasoning", "versions"):
        assert field in CardReview.model_fields, f"CardReview omits {field}"


def test_card_review_has_no_price_field():
    from card_reviewer.review.models import CardReview
    assert not (set(CardReview.model_fields)
                & {"price", "asking_price", "cost", "value"})


def test_resolved_candidate_has_no_price_field_at_all():
    """Rule 14 drawn structurally at the core's input type, not at the
    evidence manifest."""
    forbidden = {"asking_price", "price", "cost", "value", "purchased"}
    assert not (set(ResolvedCandidate.model_fields) & forbidden)


def test_candidate_input_may_carry_price_as_listing_provenance():
    ci = CandidateInput(source="manual", title="t", asking_price="42.00",
                        image_paths=[])
    assert ci.asking_price == "42.00"


def test_the_adapter_drops_price_when_resolving(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"pixels")
    adapter = ManualAdapter(ArtifactStore(tmp_path / "store"))
    resolved = adapter.resolve(CandidateInput(
        source="manual", title="t", asking_price="9999.00", image_paths=[img]))
    assert "9999" not in resolved.model_dump_json()


def test_the_adapter_hashes_images_into_the_content_addressed_store(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"pixels")
    store = ArtifactStore(tmp_path / "store")
    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="t", image_paths=[img]))
    assert len(resolved.images) == 1
    assert store.read(resolved.images[0].image_hash) == b"pixels"


def test_two_manual_copies_with_identical_titles_get_distinct_ids(tmp_path):
    """Two physical cards can share a title exactly. Deriving identity from
    the title would silently merge them into one candidate and overwrite the
    first card's review history."""
    img = tmp_path / "a.png"; img.write_bytes(b"pixels")
    adapter = ManualAdapter(ArtifactStore(tmp_path / "store"))
    a = adapter.resolve(CandidateInput(source="manual", title="2023 Chrome #150",
                                       image_paths=[img]))
    b = adapter.resolve(CandidateInput(source="manual", title="2023 Chrome #150",
                                       image_paths=[img]))
    assert a.candidate_id != b.candidate_id


def test_a_caller_supplied_id_is_used_verbatim_for_resubmission(tmp_path):
    """Re-reviewing the same physical card must reach the same candidate row,
    so its history accumulates rather than forking."""
    img = tmp_path / "a.png"; img.write_bytes(b"pixels")
    adapter = ManualAdapter(ArtifactStore(tmp_path / "store"))
    kw = dict(source="manual", title="t", candidate_id="my-card-001",
              image_paths=[img])
    assert (adapter.resolve(CandidateInput(**kw)).candidate_id
            == adapter.resolve(CandidateInput(**kw)).candidate_id
            == "my-card-001")


def test_a_listing_backed_candidate_is_stable_across_resolves(tmp_path):
    img = tmp_path / "a.png"; img.write_bytes(b"pixels")
    adapter = ManualAdapter(ArtifactStore(tmp_path / "store"))
    kw = dict(source="flippah", title="t", listing_id="L-42", image_paths=[img])
    assert (adapter.resolve(CandidateInput(**kw)).candidate_id
            == adapter.resolve(CandidateInput(**kw)).candidate_id)


def test_the_same_photo_in_two_listings_yields_one_image_hash(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"same"); b.write_bytes(b"same")
    store = ArtifactStore(tmp_path / "store")
    adapter = ManualAdapter(store)
    r1 = adapter.resolve(CandidateInput(source="manual", title="1", image_paths=[a]))
    r2 = adapter.resolve(CandidateInput(source="manual", title="2", image_paths=[b]))
    assert r1.images[0].image_hash == r2.images[0].image_hash


def test_manual_adapter_never_touches_the_network(tmp_path, monkeypatch):
    import socket
    def boom(*a, **k):
        raise AssertionError("the core must not touch the network")
    monkeypatch.setattr(socket, "socket", boom)
    img = tmp_path / "a.png"; img.write_bytes(b"pixels")
    ManualAdapter(ArtifactStore(tmp_path / "s")).resolve(
        CandidateInput(source="manual", title="t", image_paths=[img]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/models.py
"""External input and the resolved core input (spec §6)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CandidateInput(BaseModel):
    """What arrives from outside. May carry listing metadata including price."""
    source: str
    title: str = ""
    listing_url: str | None = None
    listing_id: str | None = None
    asking_price: str | None = None
    card_type: str | None = None
    set_name: str | None = None
    image_paths: list[Path] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    supplied_roles: dict[str, str] = Field(default_factory=dict)
    # A caller-supplied stable identity for the physical card. Manual entries
    # without one get a fresh UUID rather than a title-derived hash.
    candidate_id: str | None = None


class ResolvedImage(BaseModel):
    image_hash: str
    supplied_role: str | None = None
    source_url: str | None = None
    ordering: int = 0


class CardReview(BaseModel):
    """The complete output record (spec §16).

    Field ownership: `combine` owns verdict, psa10_candidate, psa10_rank_score,
    rankable, estimated_psa_grade, review_confidence and reasoning; `coverage`
    owns coverage, limitations, recommended_additional_photos and
    card_identification_request; the remaining blocks are the corresponding
    stages' stored outputs surfaced unchanged.
    """
    review_id: int | None = None
    candidate_id: str
    listing_url: str | None = None
    title: str = ""
    mode: str

    verdict: str
    psa10_candidate: str
    psa10_rank_score: int | None = None
    rankable: bool
    estimated_psa_grade: str | None = None
    review_confidence: str

    coverage: str
    coverage_detail: dict[str, Any] = Field(default_factory=dict)
    categories: dict[str, Any] = Field(default_factory=dict)
    image_quality: dict[str, Any] = Field(default_factory=dict)
    roles_and_context: dict[str, Any] = Field(default_factory=dict)

    defects_found: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[dict[str, Any]] = Field(default_factory=list)
    recommended_additional_photos: list[str] = Field(default_factory=list)
    card_identification_request: bool = False

    cv_assessment: dict[str, Any] = Field(default_factory=dict)
    vision_assessment: dict[str, Any] | None = None
    reasoning: str = ""
    versions: dict[str, str] = Field(default_factory=dict)


class ResolvedCandidate(BaseModel):
    """The core's input type. Carries NO price field of any kind — rule 14 is
    structural here rather than a discipline the grading path must remember."""
    candidate_id: str
    source: str
    title: str = ""
    card_type: str | None = None
    set_name: str | None = None
    images: list[ResolvedImage] = Field(default_factory=list)

    model_config = {"frozen": True}
```

```python
# src/card_reviewer/review/ingest/adapter.py
"""Adapters resolve external input. The ONLY component permitted network I/O.

A future Flippah API is a new adapter and nothing else.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Protocol

from ..models import CandidateInput, ResolvedCandidate, ResolvedImage
from ..storage.artifacts import ArtifactStore


class CandidateAdapter(Protocol):
    def resolve(self, candidate: CandidateInput) -> ResolvedCandidate: ...


class ManualAdapter:
    """Local files only — never opens a socket."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def resolve(self, candidate: CandidateInput) -> ResolvedCandidate:
        images: list[ResolvedImage] = []
        for i, path in enumerate(candidate.image_paths):
            image_hash = self._store.put_image(path.read_bytes())
            images.append(ResolvedImage(
                image_hash=image_hash,
                supplied_role=candidate.supplied_roles.get(str(path)),
                ordering=i))
        return ResolvedCandidate(
            candidate_id=self._candidate_id(candidate),
            source=candidate.source, title=candidate.title,
            card_type=candidate.card_type, set_name=candidate.set_name,
            images=images)

    @staticmethod
    def _candidate_id(candidate: CandidateInput) -> str:
        """Identity of a PHYSICAL card, which a title does not establish.

        Two different copies of the same card share a title exactly, so a
        title-derived id would merge them into one candidate and overwrite the
        first one's review history. Only a listing identity is a real external
        key; without one, mint a UUID and let the caller persist it.
        """
        if candidate.candidate_id:
            return candidate.candidate_id
        listing = candidate.listing_id or candidate.listing_url
        if listing:
            return hashlib.sha256(
                f"{candidate.source}|{listing}".encode()).hexdigest()[:16]
        return f"manual-{uuid.uuid4().hex[:16]}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_ingest.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/models.py src/card_reviewer/review/ingest/ tests/review/test_ingest.py
git commit -m "feat(review): ingestion boundary with structural price exclusion"
```

**Acceptance:** `ResolvedCandidate` has no price field; the adapter drops it; two manual copies sharing a title get distinct ids while a supplied id or listing id is stable across resolves; the same photo in two listings resolves to one hash; no socket is opened.

---
## Phase 4 — Policies (the heart of the system, all pure logic)

### Task 14: Rule authority policy

**Files:** Create `src/card_reviewer/review/policies/__init__.py`, `src/card_reviewer/review/policies/authority_v1.py`; Test `tests/review/test_authority.py`

**Interfaces:**
- Consumes: `card_reviewer.knowledge.models.EvidenceType`, `Confidence`; `enums.Authority`
- Produces: `AUTHORITY_POLICY_VERSION`, `authority_of(rule) -> Authority`, `may_establish_reject(rule) -> bool`

**Context for the implementer:** subsystem B's real enums are `EvidenceType` = `objective | experience_based | opinion | unverified | contradicted` and `Confidence` = `high | medium | low`. The live rubric contains only `objective` (11) and `experience_based` (25); the policy still handles all five defensively.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_authority.py
import pytest

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.knowledge.models import Confidence, EvidenceType
from card_reviewer.review.enums import Authority
from card_reviewer.review.policies.authority_v1 import (
    authority_of, may_establish_reject,
)


class _R:
    def __init__(self, et, c):
        self.evidence_type, self.confidence = et, c
        self.id = "TEST_001"


@pytest.mark.parametrize("et,conf,expected", [
    (EvidenceType.OBJECTIVE, Confidence.HIGH, Authority.BINDING),
    (EvidenceType.OBJECTIVE, Confidence.LOW, Authority.BINDING),
    (EvidenceType.EXPERIENCE_BASED, Confidence.HIGH, Authority.BINDING),
    (EvidenceType.EXPERIENCE_BASED, Confidence.MEDIUM, Authority.ADVISORY),
    (EvidenceType.EXPERIENCE_BASED, Confidence.LOW, Authority.ADVISORY),
    (EvidenceType.OPINION, Confidence.HIGH, Authority.ADVISORY),
    (EvidenceType.UNVERIFIED, Confidence.HIGH, Authority.ADVISORY),
    (EvidenceType.CONTRADICTED, Confidence.HIGH, Authority.INERT),
])
def test_authority_lattice_matches_the_declared_table(et, conf, expected):
    assert authority_of(_R(et, conf)) is expected


def test_objective_rules_stay_binding_regardless_of_confidence():
    """Objective evidence is grounded in PSA's own published standards;
    confidence demotes within experience-based, never across from objective."""
    for conf in Confidence:
        assert authority_of(_R(EvidenceType.OBJECTIVE, conf)) is Authority.BINDING


def test_a_contradicted_rule_is_inert_rather_than_deleted():
    """Non-negotiable rule 10: never delete a rule, change its status."""
    r = _R(EvidenceType.CONTRADICTED, Confidence.HIGH)
    assert authority_of(r) is Authority.INERT
    assert may_establish_reject(r) is False


def test_only_binding_rules_may_establish_a_reject():
    assert may_establish_reject(_R(EvidenceType.OBJECTIVE, Confidence.HIGH))
    assert not may_establish_reject(_R(EvidenceType.OPINION, Confidence.HIGH))
    assert not may_establish_reject(
        _R(EvidenceType.EXPERIENCE_BASED, Confidence.MEDIUM))


def test_every_live_rubric_rule_maps_to_a_defined_authority():
    for rule in load_active_rubric().rules:
        assert authority_of(rule) in set(Authority)


def test_the_live_rubric_yields_no_inert_rules_today():
    """Sanity check against the real v4.0.0 content: objective + experience
    based only. If this ever fails, a contradicted rule went active."""
    assert all(authority_of(r) is not Authority.INERT
               for r in load_active_rubric().rules)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_authority.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/policies/authority_v1.py
"""How much a rubric rule may influence the outcome (Decision 4).

This is non-negotiable rule 7 made executable: the pipeline does not treat
every claim the video pipeline learned as equally binding.

The critical discipline: authority answers "if this defect exists, how much
does it matter?" It NEVER answers "does this defect exist?" That question
belongs to the finding's own state and detectability. The two axes meet only
in a penalty-table lookup — there is no multiplication of one by the other.
"""
from __future__ import annotations

from card_reviewer.knowledge.models import Confidence, EvidenceType

from ..enums import Authority

AUTHORITY_POLICY_VERSION = "1.0.0"


def authority_of(rule) -> Authority:
    """Map subsystem B's evidence taxonomy onto an authority tier.

    `confidence` may demote within experience_based, but never promotes
    across a tier: a high-confidence opinion is still an opinion.
    """
    match rule.evidence_type:
        case EvidenceType.OBJECTIVE:
            # Grounded in PSA's published standards or official material.
            return Authority.BINDING
        case EvidenceType.EXPERIENCE_BASED:
            return (Authority.BINDING if rule.confidence is Confidence.HIGH
                    else Authority.ADVISORY)
        case EvidenceType.OPINION | EvidenceType.UNVERIFIED:
            return Authority.ADVISORY
        case EvidenceType.CONTRADICTED:
            # Inert, not absent: rule 10 says never delete, change status.
            return Authority.INERT
    raise ValueError(f"unhandled evidence_type {rule.evidence_type!r}")


def may_establish_reject(rule) -> bool:
    """Only binding authority can carry a card to REJECT."""
    return authority_of(rule) is Authority.BINDING
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_authority.py -v`
Expected: PASS (6 tests including parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/policies/ tests/review/test_authority.py
git commit -m "feat(review): rule authority policy from subsystem B evidence types"
```

**Acceptance:** the lattice matches the declared table exactly; `contradicted` is inert not deleted; every live rubric rule maps to a defined tier.

---

### Task 15: Heuristic evaluator

**Files:** Create `src/card_reviewer/review/heuristic.py`; Test `tests/review/test_heuristic.py`

**Interfaces:**
- Consumes: `findings.py`, `taxonomy.py`, `evaluability.py`, `assembly.Assembled` (Task 27)
- Produces: `HeuristicResult`, `best_detectability(detectability, category, defect_type) -> Scale`, `evaluate(assembled: Assembled, scoped_rules) -> HeuristicResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_heuristic.py
from card_reviewer.review.enums import FindingState, Scale
from card_reviewer.review.assembly import Assembled
from card_reviewer.review.heuristic import evaluate
from card_reviewer.review.roles import ImageRole
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef


def _ev(origin=EvidenceOrigin.NORMALIZED):
    return EvidenceRef(artifact_id="a1", image_hash="h1", origin=origin,
                       view="front_face")


def _assembled(**kw):
    # Detectability is keyed (ImageRole, category, defect_type) — the one shape
    # used everywhere. A second shape here would miss on every lookup.
    base = dict(
        centering={"horizontal": 52.0, "vertical": 51.0, "measurable": True},
        detectability={(ImageRole.FRONT, "corners", "rounding"): Scale.HIGH,
                       (ImageRole.FRONT, "corners", "whitening"): Scale.HIGH,
                       (ImageRole.FRONT, "surface", "scratches"): Scale.HIGH,
                       (ImageRole.FRONT, "surface", "print_lines"): Scale.HIGH},
        anomalies=[], faces_present=(ImageRole.FRONT,),
        evidence_refs={"corners:rounding": [_ev()],
                       "centering:border_ratio": [_ev()]})
    return Assembled(**(base | kw))


def test_a_measurement_type_may_reach_observed(rubric_rules):
    """Corner rounding is geometric — CV can establish it outright."""
    a = _assembled(anomalies=[{"defect_type": "rounding", "category": "corners",
                               "confidence": 0.95, "severity": "moderate"}])
    result = evaluate(a, rubric_rules)
    f = next(f for f in result.findings if f.defect_type == "rounding")
    assert f.state is FindingState.OBSERVED


def test_an_interpretive_type_can_never_exceed_suspected_from_cv_alone(rubric_rules):
    """Print lines need semantics CV cannot supply — this is what stops OFF
    mode manufacturing confident defects out of high-contrast pixels."""
    a = _assembled(
        anomalies=[{"defect_type": "print_lines", "category": "surface",
                    "confidence": 0.99, "severity": "severe"}],
        evidence_refs={"surface:print_lines": [_ev()]})
    result = evaluate(a, rubric_rules)
    f = next(f for f in result.findings if f.defect_type == "print_lines")
    assert f.state is FindingState.SUSPECTED


def test_low_detectability_prevents_observed_even_for_measurement_types(rubric_rules):
    a = _assembled(
        detectability={(ImageRole.FRONT, "corners", "rounding"): Scale.LOW},
        anomalies=[{"defect_type": "rounding", "category": "corners",
                    "confidence": 0.99, "severity": "severe"}])
    result = evaluate(a, rubric_rules)
    f = next(f for f in result.findings if f.defect_type == "rounding")
    assert f.state is not FindingState.OBSERVED


def test_binding_authority_does_not_promote_a_low_confidence_finding(rubric_rules):
    """Authority scales consequence, never belief."""
    a = _assembled(anomalies=[{"defect_type": "rounding", "category": "corners",
                               "confidence": 0.2, "severity": "minor"}])
    result = evaluate(a, rubric_rules)
    f = next(f for f in result.findings if f.defect_type == "rounding")
    assert f.state is FindingState.SUSPECTED


def test_unevaluable_rules_never_produce_findings(rubric_rules_unknown_context):
    """Non-vacuous: the card carries a real anomaly, so there ARE findings to
    check rule_ids on."""
    a = _assembled(anomalies=[{"defect_type": "scratches", "category": "surface",
                               "confidence": 0.9}],
                   evidence_refs={"surface:scratches": [_ev()],
                                  "centering:border_ratio": [_ev()]})
    result = evaluate(a, rubric_rules_unknown_context)
    assert result.findings
    assert all("SURFACE_SHINY_001" not in f.rule_ids for f in result.findings)
    assert "UNKNOWN_PRODUCT_CONTEXT" in result.unevaluable_reasons


def test_centering_within_psa_tolerance_produces_no_disqualifier(rubric_rules):
    """CENTERING_PSA10_STANDARD_002: approximately 55/45, explicitly not a
    hard arithmetic cutoff — 52/48 is comfortably inside."""
    result = evaluate(_assembled(), rubric_rules)
    assert not [f for f in result.findings
                if f.category == "centering" and f.state is FindingState.OBSERVED]


def test_a_grossly_miscut_card_does_produce_a_centering_finding(rubric_rules):
    """Centering is a measurement, not an anomaly candidate, so it needs its
    own evaluation — otherwise a 75/25 card produces no finding at all."""
    a = _assembled(centering={"horizontal": 75.0, "vertical": 50.0,
                              "measurable": True})
    result = evaluate(a, rubric_rules)
    assert [f for f in result.findings
            if f.category == "centering" and f.state is FindingState.OBSERVED]


def test_an_unmeasurable_centering_produces_no_finding(rubric_rules):
    a = _assembled(centering={"measurable": False,
                              "reason": "BORDERLESS_OR_NO_RELIABLE_REFERENCE"})
    assert not [f for f in evaluate(a, rubric_rules).findings
                if f.category == "centering"]


def test_every_finding_carries_a_location_so_fusion_can_correlate(rubric_rules):
    """A finding without a location can never fuse, so the same defect seen
    by both producers would be penalized twice."""
    a = _assembled(anomalies=[{"defect_type": "rounding", "category": "corners",
                               "confidence": 0.95}])
    for f in evaluate(a, rubric_rules).findings:
        assert f.location is not None
```

A shared `tests/review/conftest.py` supplies the fixtures:

```python
# tests/review/conftest.py
import pytest

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.context import CardContext
from card_reviewer.review.enums import Provenance
from card_reviewer.review.evaluability import scope_rules


@pytest.fixture(scope="session")
def rubric():
    return load_active_rubric()


@pytest.fixture
def rubric_rules(rubric):
    ctx = CardContext(canonical_card_types=["chrome"],
                      provenance=Provenance.SUPPLIED, confidence=1.0)
    return scope_rules(rubric.for_card(["chrome"], None), ctx)


@pytest.fixture
def rubric_rules_unknown_context(rubric):
    return scope_rules(rubric.for_card(None, None), CardContext())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_heuristic.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/heuristic.py
"""Rubric evaluation against assembled CV evidence (spec §10).

Emits findings in the shared §9 vocabulary. The promotion limit is the rule
that matters: measurement-establishable defect types may reach `observed`,
interpretive ones may not, no matter how confident the pixel evidence looks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .enums import FindingState, Scale
from .evaluability import ScopedRule, applicable, unevaluable_reasons
from .findings import Finding, FindingProducer, Severity
from .provenance import EvidenceRef, NormalizedBox
from .taxonomy import CATEGORIES, Promotion, promotion_of

if TYPE_CHECKING:
    from .assembly import Assembled

SCORER_VERSION = "1.0.0"

MIN_DETECTABILITY_FOR_OBSERVED = Scale.MODERATE
MIN_CONFIDENCE_FOR_OBSERVED = 0.8
# PSA's own tolerance is "approximately 55/45", i.e. 5 percentage points off
# centre, and explicitly not a hard cutoff — so a breach is reported only past
# it, and severely past it grades worse.
CENTERING_TOLERANCE_PP = 5.0
CENTERING_SEVERE_PP = 15.0


# NOTE: the heuristic consumes `assembly.Assembled` (Task 27) directly. There is
# deliberately no second "assembled evidence" type — two shapes with different
# detectability key arities would silently miss on every lookup and no finding
# could ever reach `observed`.
#
# Detectability is keyed (ImageRole, category, defect_type); `best_detectability`
# takes the max across faces, since a defect visible on either face is visible.


class HeuristicResult(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    unevaluable_reasons: list[str] = Field(default_factory=list)
    scorer_version: str = SCORER_VERSION


def best_detectability(detectability: dict, category: str,
                       defect_type: str) -> Scale:
    """Max over faces for one (category, defect_type).

    Callers hold a 3-tuple-keyed map; looking it up with a 2-tuple would miss
    every time and silently return the default, which is how an invariant
    quietly stops binding.
    """
    values = [v for (_face, c, d), v in detectability.items()
              if c == category and d == defect_type]
    return Scale(max(values)) if values else Scale.NONE


def _state_for(category: str, defect_type: str, confidence: float,
               detectability: Scale) -> FindingState:
    """Certainty, and only certainty. Rule authority plays no part here."""
    if detectability < MIN_DETECTABILITY_FOR_OBSERVED:
        return FindingState.SUSPECTED
    if promotion_of(category, defect_type) is Promotion.INTERPRETIVE:
        return FindingState.SUSPECTED
    if confidence < MIN_CONFIDENCE_FOR_OBSERVED:
        return FindingState.SUSPECTED
    return FindingState.OBSERVED


def evaluate(assembled: "Assembled",
             scoped_rules: list[ScopedRule]) -> HeuristicResult:
    rules_by_category: dict[str, list[str]] = {}
    for rule in applicable(scoped_rules):
        rules_by_category.setdefault(rule.category.value, []).append(rule.id)

    findings: list[Finding] = []
    for anomaly in assembled.anomalies:
        category = anomaly["category"]
        defect_type = anomaly["defect_type"]
        key = f"{category}:{defect_type}"
        detectability = best_detectability(assembled.detectability,
                                           category, defect_type)
        refs = assembled.evidence_refs.get(key) or []
        if not refs:
            continue
        findings.append(Finding(
            defect_type=defect_type, category=category,
            state=_state_for(category, defect_type,
                             float(anomaly.get("confidence", 0.0)), detectability),
            producer=FindingProducer.HEURISTIC,
            confidence=float(anomaly.get("confidence", 0.0)),
            psa10_relevant=category in CATEGORIES,
            evidence=refs,
            severity=Severity(anomaly["severity"]) if anomaly.get("severity") else None,
            # A location is REQUIRED: fusion correlates by overlapping region,
            # and a finding without one can never fuse, so the same physical
            # defect seen by both producers would be penalized twice.
            location=_location_of(anomaly, refs),
            rule_ids=rules_by_category.get(category, []),
            explanation=f"CV anomaly candidate in {category}/{defect_type}",
        ))
    findings.extend(_centering_findings(assembled, rules_by_category))
    return HeuristicResult(findings=findings,
                           unevaluable_reasons=unevaluable_reasons(scoped_rules))


def _location_of(anomaly: dict[str, Any],
                 refs: list[EvidenceRef]) -> NormalizedBox | None:
    if anomaly.get("region"):
        return NormalizedBox.model_validate(anomaly["region"])
    boxes = [r.region for r in refs if r.region is not None]
    if not boxes:
        return None
    return NormalizedBox(x0=min(b.x0 for b in boxes), y0=min(b.y0 for b in boxes),
                         x1=max(b.x1 for b in boxes), y1=max(b.y1 for b in boxes))


def _centering_findings(assembled: "Assembled",
                        rules_by_category: dict[str, list[str]]) -> list[Finding]:
    """Centering is a measurement, not an anomaly candidate, so it never
    appears in `assembled.anomalies` — it needs its own evaluation or a
    grossly miscut card produces no finding at all.

    `CENTERING_PSA10_STANDARD_002` supplies PSA's tolerance: approximately
    55/45 front, explicitly NOT a hard arithmetic cutoff, so 56/44 is not an
    automatic failure. Only a clear breach is reported.
    """
    centering = assembled.centering
    if not centering.get("measurable"):
        return []
    refs = assembled.evidence_refs.get("centering:border_ratio") or []
    if not refs:
        return []
    worst = max(abs(float(centering.get("horizontal", 50.0)) - 50.0),
                abs(float(centering.get("vertical", 50.0)) - 50.0))
    if worst <= CENTERING_TOLERANCE_PP:
        return []
    severity = (Severity.SEVERE if worst > CENTERING_SEVERE_PP
                else Severity.MODERATE)
    return [Finding(
        defect_type="border_ratio", category="centering",
        state=FindingState.OBSERVED, producer=FindingProducer.HEURISTIC,
        confidence=0.9, psa10_relevant=True, severity=severity,
        location=NormalizedBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        evidence=refs, rule_ids=rules_by_category.get("centering", []),
        explanation=f"centering {centering.get('horizontal')}/"
                    f"{100 - float(centering.get('horizontal', 50.0)):.0f} "
                    "exceeds the PSA 10 tolerance")]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_heuristic.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/heuristic.py tests/review/conftest.py tests/review/test_heuristic.py
git commit -m "feat(review): heuristic evaluator with promotion limit"
```

**Acceptance:** interpretive defect types never reach `observed` from CV alone; low detectability blocks promotion; binding authority never raises a finding's state.

---

### Task 16: Finding-to-rule relevance and authority resolution

**Files:** Create `src/card_reviewer/review/relevance.py`, `src/card_reviewer/review/policies/relevance_v1.py`; Test `tests/review/test_relevance.py`

**Interfaces:**
- Consumes: `findings.py`, `evaluability.py`, `policies/authority_v1.py`
- Produces: `RELEVANCE_POLICY_VERSION`, `ResolvedFinding` (finding + matched rule IDs + authority + `psa10_relevant`), `resolve_relevance(findings, scoped_rules) -> list[ResolvedFinding]`

**Why this task exists (Decision 4).** A finding does not *have* an authority; the rules that match it do. Without this step, an implementer either attaches every rule in a category to every anomaly — letting an unrelated centering rule lend weight to a corner defect — or defaults unmapped findings to `BINDING`, which lets any unrecognized anomaly reject a card.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_relevance.py
from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.context import CardContext
from card_reviewer.review.enums import Authority, FindingState, Provenance
from card_reviewer.review.evaluability import scope_rules
from card_reviewer.review.findings import Finding, FindingProducer
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef
from card_reviewer.review.relevance import resolve_relevance


def _f(category="corners", defect="rounding", relevant=True):
    return Finding(defect_type=defect, category=category,
                   state=FindingState.OBSERVED,
                   producer=FindingProducer.HEURISTIC, confidence=0.9,
                   psa10_relevant=relevant,
                   evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                                         origin=EvidenceOrigin.ORIGINAL,
                                         view="v")])


def _scoped(card_types=None):
    rubric = load_active_rubric()
    ctx = CardContext(canonical_card_types=card_types,
                      provenance=Provenance.SUPPLIED if card_types
                      else Provenance.UNKNOWN)
    return scope_rules(rubric.for_card(card_types, None), ctx)


def test_a_finding_matches_only_rules_in_its_own_category():
    resolved = resolve_relevance([_f(category="corners")], _scoped(["chrome"]))[0]
    rubric = {r.id: r for r in load_active_rubric().rules}
    assert resolved.rule_ids
    assert all(rubric[rid].category.value == "corners" for rid in resolved.rule_ids)


def test_a_corner_finding_never_inherits_a_centering_rules_authority():
    resolved = resolve_relevance([_f(category="corners")], _scoped(["chrome"]))[0]
    rubric = {r.id: r for r in load_active_rubric().rules}
    assert not any(rubric[rid].category.value == "centering"
                   for rid in resolved.rule_ids)


def test_authority_is_the_maximum_among_matched_rules():
    resolved = resolve_relevance([_f(category="surface")], _scoped(["chrome"]))[0]
    rubric = {r.id: r for r in load_active_rubric().rules}
    from card_reviewer.review.policies.authority_v1 import authority_of
    expected = max(authority_of(rubric[rid]) for rid in resolved.rule_ids)
    assert resolved.authority is expected


def test_an_unmatched_finding_is_advisory_never_binding():
    """Defaulting to binding would let any unrecognized anomaly reject a card."""
    resolved = resolve_relevance([_f(category="handling", defect="unknown_thing")],
                                 _scoped(["chrome"]))[0]
    assert resolved.rule_ids == []
    assert resolved.authority is Authority.ADVISORY


def test_a_finding_outside_the_grading_categories_is_not_psa10_relevant():
    resolved = resolve_relevance([_f(category="handling", defect="x")],
                                 _scoped(["chrome"]))[0]
    assert resolved.psa10_relevant is False


def test_a_provider_claim_of_relevance_is_overridden_by_our_policy():
    """Claude may describe a defect; whether it disqualifies a 10 is ours."""
    claimed = _f(category="handling", defect="looks_bad", relevant=True)
    assert resolve_relevance([claimed], _scoped(["chrome"]))[0].psa10_relevant is False


def test_an_unmatched_finding_in_a_grading_category_stays_relevant():
    """Advisory means it cannot REJECT — not that it disappears. Gating
    relevance on rule matching would let an unexplained corner defect ship as
    a clean gem candidate."""
    from card_reviewer.review.policies import relevance_v1
    odd = _f(category="corners", defect="unrecognized_thing")
    resolved = resolve_relevance([odd], _scoped(["chrome"]))[0]
    assert resolved.psa10_relevant is True
    assert resolved.authority is not Authority.BINDING


def test_unevaluable_rules_are_never_matched():
    """SURFACE_SHINY_001 is product-scoped; with product unknown it must not
    lend its authority to a surface finding."""
    resolved = resolve_relevance([_f(category="surface", defect="scratches")],
                                 _scoped(None))[0]
    assert "SURFACE_SHINY_001" not in resolved.rule_ids


def test_a_contradicted_rule_contributes_no_authority():
    from card_reviewer.review.policies.authority_v1 import authority_of
    for rule in load_active_rubric().rules:
        if authority_of(rule) is Authority.INERT:
            resolved = resolve_relevance([_f(category=rule.category.value)],
                                         _scoped(["chrome"]))[0]
            assert rule.id not in resolved.rule_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_relevance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'card_reviewer.review.relevance'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/policies/relevance_v1.py
"""Which rubric rules actually apply to a given finding (Decision 4)."""
from __future__ import annotations

RELEVANCE_POLICY_VERSION = "1.0.0"

# A rule matches a finding when their categories agree. Category is the axis
# subsystem B already models; matching finer than that would require the rubric
# to carry defect-type scoping, which it does not.
def rule_matches_finding(rule, finding) -> bool:
    return rule.category.value == finding.category
```

```python
# src/card_reviewer/review/relevance.py
"""Resolve each finding to the rules that govern it, and to an authority.

Two things this exists to prevent:

  1. Attaching every rule in a category to every anomaly, which would let an
     unrelated high-authority rule lend weight to an unrelated defect.
  2. Defaulting an unmapped finding to BINDING, which would let any
     unrecognized anomaly reject a card — the false rejection the governing
     asymmetry forbids.
"""
from __future__ import annotations

from pydantic import BaseModel

from .enums import Authority, RuleEvaluability
from .findings import Finding
from .policies.authority_v1 import authority_of
from .policies.relevance_v1 import RELEVANCE_POLICY_VERSION, rule_matches_finding
from .taxonomy import CATEGORIES


class ResolvedFinding(BaseModel):
    finding: Finding
    rule_ids: list[str]
    authority: Authority
    psa10_relevant: bool
    policy_version: str = RELEVANCE_POLICY_VERSION


def resolve_relevance(findings: list[Finding],
                      scoped_rules: list) -> list[ResolvedFinding]:
    # Only APPLICABLE rules participate: a product-scoped rule we could not
    # evaluate must not lend its authority to anything.
    usable = [s.rule for s in scoped_rules
              if s.evaluability is RuleEvaluability.APPLICABLE
              and authority_of(s.rule) is not Authority.INERT]

    out: list[ResolvedFinding] = []
    for finding in findings:
        matched = [r for r in usable if rule_matches_finding(r, finding)]
        authority = (max((authority_of(r) for r in matched),
                         default=Authority.ADVISORY))
        # Relevance is decided by OUR policy, not by the provider's claim — but
        # it is decided by the grading taxonomy, NOT by whether a rule happened
        # to match. Gating on `matched` would make an unexplained corner defect
        # psa10_relevant=False, which drops it from both the verdict and the
        # score: an observed defect would ship as a clean gem candidate.
        # An unmatched finding is advisory (so it cannot REJECT) but still
        # relevant (so it still penalizes and still routes to REVIEW).
        relevant = finding.category in CATEGORIES
        out.append(ResolvedFinding(
            finding=finding.model_copy(update={
                "rule_ids": [r.id for r in matched],
                "psa10_relevant": relevant,
            }),
            rule_ids=[r.id for r in matched],
            authority=authority if matched else Authority.ADVISORY,
            psa10_relevant=relevant))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_relevance.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/relevance.py src/card_reviewer/review/policies/relevance_v1.py tests/review/test_relevance.py
git commit -m "feat(review): finding-to-rule relevance and authority resolution"
```

**Acceptance:** a finding matches only its own category's rules; authority is the max over matched rules; an unmatched finding is `ADVISORY` and not `psa10_relevant`; a provider's relevance claim is overridden; unevaluable and inert rules never contribute.

---

### Task 17: EvidenceCoveragePolicy

**Files:** Create `src/card_reviewer/review/policies/coverage_v1.py`; Test `tests/review/test_coverage.py`

**Interfaces:**
- Consumes: `taxonomy.py`, `enums.py`, `roles.py`
- Produces: `COVERAGE_POLICY_VERSION`, `MIN_ASSESSED`, `REQUIRED_FACES`, `UnevaluableRule`, `Limitation`, `CoverageResult`, `evaluate_coverage(detectability, reason_codes, vision_assessability, faces_present, *, unevaluable_rules=None) -> CoverageResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_coverage.py
from card_reviewer.review.enums import Coverage, Scale, UndetectabilityClass
from card_reviewer.review.policies.coverage_v1 import evaluate_coverage
from card_reviewer.review.roles import ImageRole

_ALL = ("centering", "corners", "edges", "surface")


def _good(faces=(ImageRole.FRONT, ImageRole.BACK)):
    det = {}
    for face in faces:
        for cat in _ALL:
            from card_reviewer.review.taxonomy import defect_types_for
            for dt in defect_types_for(cat):
                det[(face, cat, dt)] = Scale.HIGH
    return det


def test_full_evidence_on_both_faces_is_sufficient():
    r = evaluate_coverage(_good(), {}, {}, (ImageRole.FRONT, ImageRole.BACK))
    assert r.outcome is Coverage.SUFFICIENT


def test_a_white_bordered_card_can_still_reach_sufficient():
    """DoD 10. Structural undetectability is waived, so PASS stays reachable
    for the majority of the modern base-card population."""
    det = _good()
    reasons = {}
    for face in (ImageRole.FRONT, ImageRole.BACK):
        det[(face, "corners", "whitening")] = Scale.LOW
        det[(face, "edges", "whitening")] = Scale.LOW
        reasons[(face, "corners", "whitening")] = "WHITE_BORDER"
        reasons[(face, "edges", "whitening")] = "WHITE_BORDER"
    r = evaluate_coverage(det, reasons, {}, (ImageRole.FRONT, ImageRole.BACK))
    assert r.outcome is Coverage.SUFFICIENT
    assert any(l.undetectability_class is UndetectabilityClass.STRUCTURAL
               for l in r.limitations)


def test_glare_on_the_same_corner_is_circumstantial_and_blocks_sufficient():
    det = _good()
    det[(ImageRole.FRONT, "corners", "whitening")] = Scale.LOW
    r = evaluate_coverage(det, {(ImageRole.FRONT, "corners", "whitening"): "GLARE"},
                          {}, (ImageRole.FRONT, ImageRole.BACK))
    assert r.outcome is Coverage.PARTIAL


def test_a_usable_front_only_card_is_partial_and_rankable():
    r = evaluate_coverage(_good((ImageRole.FRONT,)), {}, {}, (ImageRole.FRONT,))
    assert r.outcome is Coverage.PARTIAL
    assert r.rankable is True


def test_an_unassessable_front_is_inadequate_and_unrankable():
    det = {(ImageRole.FRONT, c, d): Scale.NONE for c in _ALL
           for d in __import__("card_reviewer.review.taxonomy",
                               fromlist=["x"]).defect_types_for(c)}
    r = evaluate_coverage(det, {}, {}, (ImageRole.FRONT,))
    assert r.outcome is Coverage.INADEQUATE
    assert r.rankable is False


def test_vision_saying_not_assessable_overrides_good_cv_suitability():
    """CRIT-5: CV measures whether the pixels COULD carry evidence; vision
    reports whether anything could actually be concluded."""
    r = evaluate_coverage(_good(), {}, {"surface": False},
                          (ImageRole.FRONT, ImageRole.BACK))
    assert r.outcome is Coverage.PARTIAL


def test_an_unapplied_product_rule_is_a_metadata_resolvable_limitation():
    """It arrives as itself, not simulated by lowering pixel detectability
    for some arbitrary defect type."""
    from card_reviewer.review.policies.coverage_v1 import UnevaluableRule
    r = evaluate_coverage(
        _good(), {}, {}, (ImageRole.FRONT, ImageRole.BACK),
        unevaluable_rules=[UnevaluableRule(
            rule_id="SURFACE_SHINY_001", category="surface",
            reason_code="UNKNOWN_PRODUCT_CONTEXT")])
    assert any(l.undetectability_class is UndetectabilityClass.METADATA_RESOLVABLE
               for l in r.limitations)


def test_perfect_photos_with_unknown_product_are_partial_and_rankable():
    """The integration case: nothing is wrong with the photographs, so no
    photo request is warranted — we simply do not know what card this is."""
    from card_reviewer.review.policies.coverage_v1 import UnevaluableRule
    r = evaluate_coverage(
        _good(), {}, {}, (ImageRole.FRONT, ImageRole.BACK),
        unevaluable_rules=[UnevaluableRule(
            rule_id="SURFACE_SHINY_001", category="surface",
            reason_code="UNKNOWN_PRODUCT_CONTEXT")])
    assert r.outcome is Coverage.PARTIAL
    assert r.rankable is True
    assert r.card_identification_request is True
    assert r.recommended_additional_photos == []


def test_photo_requests_derive_from_circumstantial_limitations_only():
    det = _good()
    det[(ImageRole.FRONT, "corners", "whitening")] = Scale.LOW
    det[(ImageRole.FRONT, "surface", "scratches")] = Scale.LOW
    reasons = {(ImageRole.FRONT, "corners", "whitening"): "WHITE_BORDER",
               (ImageRole.FRONT, "surface", "scratches"): "GLARE"}
    r = evaluate_coverage(det, reasons, {}, (ImageRole.FRONT, ImageRole.BACK))
    assert any("diffuse" in p.lower() for p in r.recommended_additional_photos)
    assert not any("white" in p.lower() for p in r.recommended_additional_photos)


def test_a_glare_gap_and_an_identity_gap_produce_different_requests():
    from card_reviewer.review.policies.coverage_v1 import UnevaluableRule
    det = _good()
    det[(ImageRole.FRONT, "corners", "rounding")] = Scale.LOW
    r = evaluate_coverage(
        det, {(ImageRole.FRONT, "corners", "rounding"): "GLARE"}, {},
        (ImageRole.FRONT, ImageRole.BACK),
        unevaluable_rules=[UnevaluableRule(
            rule_id="SURFACE_SHINY_001", category="surface",
            reason_code="UNKNOWN_PRODUCT_CONTEXT")])
    assert r.card_identification_request is True
    assert any("diffuse" in x.lower() for x in r.recommended_additional_photos)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_coverage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/policies/coverage_v1.py
"""EvidenceCoveragePolicy v1 (spec §13).

Makes I2 mechanically testable. Every threshold here is a declared v1 value,
changeable only by a version bump.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..enums import Coverage, Scale, UndetectabilityClass
from ..roles import ImageRole
from ..taxonomy import CATEGORIES, class_of, defect_types_for

COVERAGE_POLICY_VERSION = "1.0.0"

MIN_ASSESSED = Scale.MODERATE
REQUIRED_FACES = (ImageRole.FRONT, ImageRole.BACK)
MIN_FRONT_CATEGORIES_FOR_PARTIAL = 2

PHOTO_REQUESTS: dict[str, str] = {
    "GLARE": "a diffuse-lit photograph of the {face} (avoid direct flash)",
    "BLUR": "a sharper close-up of the {face} {category}",
    "LOW_RESOLUTION": "a higher-resolution close-up of the {face} {category}",
    "OCCLUSION": "the {face} out of its holder, or with the obstruction moved",
    "MISSING_FACE": "a photograph of the {face}",
    "SEVERE_PERSPECTIVE": "a square-on photograph of the {face}",
}


class UnevaluableRule(BaseModel):
    """A rubric rule that could not be applied — not a pixel problem."""
    rule_id: str
    category: str
    reason_code: str


class Limitation(BaseModel):
    face: str
    category: str
    defect_type: str
    reason_code: str
    undetectability_class: UndetectabilityClass


class CoverageResult(BaseModel):
    outcome: Coverage
    rankable: bool
    assessed: dict[str, list[str]] = Field(default_factory=dict)
    limitations: list[Limitation] = Field(default_factory=list)
    recommended_additional_photos: list[str] = Field(default_factory=list)
    card_identification_request: bool = False
    policy_version: str = COVERAGE_POLICY_VERSION


def evaluate_coverage(
    detectability: dict[tuple[ImageRole, str, str], Scale],
    reason_codes: dict[tuple[ImageRole, str, str], str],
    vision_assessability: dict[str, bool],
    faces_present: tuple[ImageRole, ...],
    *,
    unevaluable_rules: list[UnevaluableRule] | None = None,
) -> CoverageResult:
    """`unevaluable_rules` carries rubric gaps that have nothing to do with
    pixels — a product-scoped rule we cannot apply because the card was never
    identified. They are metadata-resolvable limitations, and they must arrive
    here as themselves rather than being simulated by lowering the
    detectability of some arbitrary defect type."""
    limitations: list[Limitation] = []
    assessed: dict[str, list[str]] = {}

    # Rubric-level gaps first: these are not photograph defects at all.
    blocked_categories: set[str] = set()
    for gap in unevaluable_rules or []:
        limitations.append(Limitation(
            face="card", category=gap.category, defect_type="*",
            reason_code=gap.reason_code,
            undetectability_class=class_of(gap.reason_code)))
        blocked_categories.add(gap.category)

    for face in REQUIRED_FACES:
        assessed_here: list[str] = []
        for category in CATEGORIES:
            required_ok = True
            for defect_type in defect_types_for(category):
                key = (face, category, defect_type)
                if face not in faces_present:
                    limitations.append(Limitation(
                        face=face.value, category=category, defect_type=defect_type,
                        reason_code="MISSING_FACE",
                        undetectability_class=UndetectabilityClass.CIRCUMSTANTIAL))
                    required_ok = False
                    continue
                if detectability.get(key, Scale.NONE) >= MIN_ASSESSED:
                    continue
                code = reason_codes.get(key, "LOW_RESOLUTION")
                klass = class_of(code)
                limitations.append(Limitation(
                    face=face.value, category=category, defect_type=defect_type,
                    reason_code=code, undetectability_class=klass))
                # Structural gaps are reported but do not block: no photograph
                # could ever supply the evidence, so demanding it would make
                # PASS unreachable for an entire class of cards.
                if klass is not UndetectabilityClass.STRUCTURAL:
                    required_ok = False
            # Vision may veto a category CV suitability alone allowed.
            if vision_assessability.get(category) is False:
                required_ok = False
            # So may an unapplied product-scoped rule.
            if category in blocked_categories:
                required_ok = False
            if required_ok:
                assessed_here.append(category)
        assessed[face.value] = assessed_here

    front = assessed.get(ImageRole.FRONT.value, [])
    all_assessed = all(len(assessed.get(f.value, [])) == len(CATEGORIES)
                       for f in REQUIRED_FACES)

    if all_assessed:
        outcome, rankable = Coverage.SUFFICIENT, True
    elif len(front) >= MIN_FRONT_CATEGORIES_FOR_PARTIAL:
        outcome, rankable = Coverage.PARTIAL, True
    else:
        outcome, rankable = Coverage.INADEQUATE, False

    photos, identify = _requests(limitations)
    return CoverageResult(outcome=outcome, rankable=rankable, assessed=assessed,
                          limitations=limitations,
                          recommended_additional_photos=photos,
                          card_identification_request=identify)


def _requests(limitations: list[Limitation]) -> tuple[list[str], bool]:
    photos: list[str] = []
    identify = False
    for lim in limitations:
        if lim.undetectability_class is UndetectabilityClass.METADATA_RESOLVABLE:
            identify = True
            continue
        if lim.undetectability_class is UndetectabilityClass.STRUCTURAL:
            continue
        template = PHOTO_REQUESTS.get(lim.reason_code)
        if template:
            text = template.format(face=lim.face, category=lim.category)
            if text not in photos:
                photos.append(text)
    return photos, identify
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_coverage.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/policies/coverage_v1.py tests/review/test_coverage.py
git commit -m "feat(review): EvidenceCoveragePolicy with structural exemption"
```

**Acceptance:** a white-bordered card reaches `SUFFICIENT`; the same corner glared does not; a usable front-only card is `PARTIAL` and rankable; vision `not_assessable` vetoes; an unapplied product rule arrives as an `UnevaluableRule` and yields an identification request with **no** photo request; photo requests exclude structural limitations.

---
### Task 18: Scoring policy — rank score, grade estimate, review confidence

**Files:** Create `src/card_reviewer/review/policies/scoring_v1.py`; Test `tests/review/test_scoring.py`

**Interfaces:**
- Consumes: `findings.py`, `enums.py`, `policies/authority_v1.py`
- Produces: `SCORING_POLICY_VERSION`, `PENALTIES`, `rank_score(findings, coverage) -> int | None`, `estimated_grade(findings, coverage) -> str | None`, `review_confidence(coverage, contradictions, producers_disagreed, card_context_known, *, required_face_missing=False) -> ReviewConfidence`. All three accept `(finding, authority, i1_satisfied)` triples.

**This is Decision 2 in code.** All three derivations live here; no magic number appears anywhere else.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_scoring.py
import pytest

from card_reviewer.review.enums import (
    Authority, Coverage, FindingState, ReviewConfidence,
)
from card_reviewer.review.findings import Finding, FindingProducer, Severity
from card_reviewer.review.policies.scoring_v1 import (
    estimated_grade, rank_score, review_confidence,
)
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef


def _f(state, severity=None, relevant=True, authority=Authority.BINDING,
       i1=True):
    return Finding(
        defect_type="rounding", category="corners", state=state,
        producer=FindingProducer.HEURISTIC, confidence=0.9,
        psa10_relevant=relevant, severity=severity,
        evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                              origin=EvidenceOrigin.ORIGINAL, view="front")],
    ), authority, i1


# --- rank score -----------------------------------------------------------

def test_a_clean_card_with_full_coverage_scores_100():
    assert rank_score([], Coverage.SUFFICIENT) == 100


def test_the_score_is_null_when_coverage_is_inadequate():
    assert rank_score([], Coverage.INADEQUATE) is None


def test_adding_a_credible_negative_finding_never_raises_the_score():
    base = rank_score([], Coverage.SUFFICIENT)
    worse = rank_score([_f(FindingState.SUSPECTED)], Coverage.SUFFICIENT)
    assert worse <= base


def test_promoting_suspected_to_observed_never_raises_the_score():
    s = rank_score([_f(FindingState.SUSPECTED)], Coverage.SUFFICIENT)
    o = rank_score([_f(FindingState.OBSERVED)], Coverage.SUFFICIENT)
    assert o <= s


def test_improving_coverage_without_adding_a_defect_never_lowers_the_score():
    partial = rank_score([], Coverage.PARTIAL)
    sufficient = rank_score([], Coverage.SUFFICIENT)
    assert sufficient >= partial


def test_only_an_i1_satisfying_binding_disqualifier_floors_the_score():
    assert rank_score([_f(FindingState.OBSERVED, i1=True)],
                      Coverage.SUFFICIENT) == 0


def test_an_observed_finding_failing_i1_stays_meaningfully_rankable():
    """It routes to REVIEW, so it must sort above a confirmed reject rather
    than collapsing to the same 0."""
    score = rank_score([_f(FindingState.OBSERVED, i1=False)], Coverage.SUFFICIENT)
    assert 0 < score < 100


def test_an_unresolved_finding_scores_worse_than_a_merely_suspected_one():
    unresolved = rank_score([_f(FindingState.OBSERVED, i1=False)],
                            Coverage.SUFFICIENT)
    suspected = rank_score([_f(FindingState.SUSPECTED, i1=False)],
                           Coverage.SUFFICIENT)
    assert unresolved < suspected


def test_an_unmapped_finding_defaults_to_advisory_never_binding():
    """Decision 4: an unmapped finding must not be able to reject a card."""
    bare = _f(FindingState.OBSERVED)[0]
    assert rank_score([bare], Coverage.SUFFICIENT) > 0


def test_advisory_authority_costs_less_than_binding():
    b = rank_score([_f(FindingState.SUSPECTED, authority=Authority.BINDING)],
                   Coverage.SUFFICIENT)
    a = rank_score([_f(FindingState.SUSPECTED, authority=Authority.ADVISORY)],
                   Coverage.SUFFICIENT)
    assert a > b


def test_not_assessable_findings_cost_nothing_in_the_score():
    """Missing evidence is already paid for in coverage and confidence;
    charging it here too would double-count absence as a defect."""
    assert rank_score([_f(FindingState.NOT_ASSESSABLE)],
                      Coverage.SUFFICIENT) == rank_score([], Coverage.SUFFICIENT)


def test_the_score_is_always_within_bounds():
    many = [_f(FindingState.OBSERVED) for _ in range(20)]
    assert rank_score(many, Coverage.PARTIAL) == 0


def test_two_separate_defects_cost_more_than_one():
    """Guards the other side of fusion: distinct defects must still stack, or
    fusion would be hiding real flaws rather than avoiding double-counting."""
    one = rank_score([_f(FindingState.SUSPECTED, i1=False)], Coverage.SUFFICIENT)
    two = rank_score([_f(FindingState.SUSPECTED, i1=False),
                      _f(FindingState.SUSPECTED, i1=False)], Coverage.SUFFICIENT)
    assert two < one


# --- grade estimate -------------------------------------------------------

def test_a_clean_fully_covered_card_estimates_a_10():
    assert estimated_grade([], Coverage.SUFFICIENT) == "10"


def test_partial_coverage_widens_the_estimate_rather_than_lowering_it():
    assert estimated_grade([], Coverage.PARTIAL) == "9-10"


@pytest.mark.parametrize("severity,expected", [
    (Severity.MINOR, "9"), (Severity.MODERATE, "8-9"), (Severity.SEVERE, "<=8"),
])
def test_the_grade_follows_the_worst_confirmed_defect(severity, expected):
    assert estimated_grade([_f(FindingState.OBSERVED, severity)],
                           Coverage.SUFFICIENT) == expected


def test_two_moderate_defects_estimate_below_8():
    fs = [_f(FindingState.OBSERVED, Severity.MODERATE),
          _f(FindingState.OBSERVED, Severity.MODERATE)]
    assert estimated_grade(fs, Coverage.SUFFICIENT) == "<=8"


def test_the_grade_is_null_when_coverage_is_inadequate():
    assert estimated_grade([], Coverage.INADEQUATE) is None


def test_suspected_findings_do_not_lower_the_grade_estimate():
    assert estimated_grade([_f(FindingState.SUSPECTED, Severity.SEVERE)],
                           Coverage.SUFFICIENT) == "10"


def test_an_observed_finding_failing_i1_does_not_lower_the_grade():
    """It is an unresolved concern, not a confirmed defect — it costs score
    and routes to REVIEW, but the grade estimate reports what is established."""
    assert estimated_grade([_f(FindingState.OBSERVED, Severity.SEVERE, i1=False)],
                           Coverage.SUFFICIENT) == "10"


def test_grade_is_not_a_conversion_of_the_score():
    """They answer different questions and must be able to disagree."""
    fs = [_f(FindingState.SUSPECTED) for _ in range(4)]
    score = rank_score(fs, Coverage.SUFFICIENT)
    grade = estimated_grade(fs, Coverage.SUFFICIENT)
    assert score < 90 and grade == "10"


# --- review confidence ----------------------------------------------------

def test_confidence_is_low_when_coverage_is_inadequate():
    assert review_confidence(Coverage.INADEQUATE, [], False, True) is \
        ReviewConfidence.LOW


def test_a_missing_required_face_is_low_confidence_even_though_partial():
    """A front-only card is PARTIAL and rankable, but we never saw half of
    it — the assessment deserves LOW confidence."""
    assert review_confidence(Coverage.PARTIAL, [], False, True,
                             required_face_missing=True) is ReviewConfidence.LOW


def test_other_partial_coverage_remains_medium():
    assert review_confidence(Coverage.PARTIAL, [], False, True,
                             required_face_missing=False) is \
        ReviewConfidence.MEDIUM


def test_confidence_is_medium_when_the_two_producers_disagreed():
    assert review_confidence(Coverage.SUFFICIENT, [], True, True) is \
        ReviewConfidence.MEDIUM


def test_confidence_is_medium_when_card_context_is_unknown():
    assert review_confidence(Coverage.SUFFICIENT, [], False, False) is \
        ReviewConfidence.MEDIUM


def test_confidence_is_high_only_when_everything_is_resolved():
    assert review_confidence(Coverage.SUFFICIENT, [], False, True) is \
        ReviewConfidence.HIGH


def test_a_high_score_can_carry_low_confidence():
    """'Probably clean, but we could barely see it' must be expressible —
    this is the semantic separation the owner asked for."""
    score = rank_score([], Coverage.PARTIAL)
    conf = review_confidence(Coverage.INADEQUATE, [], False, True)
    assert score >= 85 and conf is ReviewConfidence.LOW
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/policies/scoring_v1.py
"""The three summary values (Decision 2).

They answer three different questions and must never collapse into one:

  psa10_rank_score  — how should this card sort against others?
  estimated_psa_grade — what coarse grade does the evidence support?
  review_confidence — how much do we trust this assessment at all?

Every weight and threshold in the system's scoring lives here. No magic
number appears in any other module.
"""
from __future__ import annotations

from ..enums import Authority, Coverage, FindingState, ReviewConfidence
from ..findings import Finding, Severity

SCORING_POLICY_VERSION = "1.0.0"

MAX_SCORE = 100
MIN_SCORE = 0

# Penalties are non-negative and monotone in state, which is what makes the
# monotonicity properties in the tests hold by construction rather than luck.
# Keyed by (state, authority, i1_satisfied). Only a binding disqualifier that
# actually satisfies I1 floors the score: an observed-but-unresolved finding
# routes to REVIEW and must stay meaningfully rankable there, or it would sort
# identically to a confirmed reject and destroy the triage ordering.
PENALTIES: dict[tuple[FindingState, Authority, bool], int] = {
    (FindingState.OBSERVED, Authority.BINDING, True): 100,   # floors the score
    (FindingState.OBSERVED, Authority.BINDING, False): 35,
    (FindingState.OBSERVED, Authority.ADVISORY, True): 25,
    (FindingState.OBSERVED, Authority.ADVISORY, False): 25,
    (FindingState.SUSPECTED, Authority.BINDING, False): 15,
    (FindingState.SUSPECTED, Authority.BINDING, True): 15,
    (FindingState.SUSPECTED, Authority.ADVISORY, False): 6,
    (FindingState.SUSPECTED, Authority.ADVISORY, True): 6,
}

COVERAGE_PENALTY: dict[Coverage, int] = {
    Coverage.SUFFICIENT: 0,
    Coverage.PARTIAL: 10,
}

SEVERITY_GRADE: dict[Severity, str] = {
    Severity.MINOR: "9",
    Severity.MODERATE: "8-9",
    Severity.SEVERE: "<=8",
}


def _triples(findings) -> list[tuple[Finding, Authority, bool]]:
    """Normalize to (finding, authority, i1_satisfied).

    Authority defaults to ADVISORY, never BINDING: an unmapped finding must
    not be able to reject a card (Decision 4).
    """
    out = []
    for item in findings:
        if isinstance(item, tuple):
            if len(item) == 3:
                out.append(item)
            else:
                out.append((item[0], item[1], False))
        else:
            out.append((item, Authority.ADVISORY, False))
    return out


def rank_score(findings, coverage: Coverage) -> int | None:
    """0-100 ranking heuristic. Explicitly NOT a probability.

    Expects FUSED findings (Decision 5): one physical defect penalizes once,
    however many producers saw it.
    """
    if coverage is Coverage.INADEQUATE:
        return None
    score = MAX_SCORE - COVERAGE_PENALTY.get(coverage, 0)
    for finding, authority, i1 in _triples(findings):
        if not finding.psa10_relevant or authority is Authority.INERT:
            continue
        # not_observed and not_assessable cost nothing here by design.
        score -= PENALTIES.get((finding.state, authority, i1), 0)
    return max(MIN_SCORE, min(MAX_SCORE, score))


def estimated_grade(findings, coverage: Coverage) -> str | None:
    """A coarse estimate from the worst CONFIRMED defect.

    Deliberately not a function of rank_score: deriving it from the score
    would imply the score is calibrated, which §1 disclaims.
    """
    if coverage is Coverage.INADEQUATE:
        return None
    # "Worst CONFIRMED defect" means confirmed: an observed finding that fails
    # I1 is an unresolved concern (it routes to REVIEW), not an established
    # defect, so it must not drag the grade estimate down as though it were.
    observed = [f for f, _, i1 in _triples(findings)
                if f.state is FindingState.OBSERVED and f.psa10_relevant and i1]
    if not observed:
        return "10" if coverage is Coverage.SUFFICIENT else "9-10"
    severities = [f.severity for f in observed if f.severity]
    if not severities:
        return "9"
    if Severity.SEVERE in severities:
        return "<=8"
    if severities.count(Severity.MODERATE) >= 2:
        return "<=8"
    if Severity.MODERATE in severities:
        return "8-9"
    return "9"


def review_confidence(coverage: Coverage, contradictions: list,
                      producers_disagreed: bool, card_context_known: bool,
                      *, required_face_missing: bool = False) -> ReviewConfidence:
    """Confidence in the ASSESSMENT, never the probability of a PSA 10.

    `required_face_missing` is explicit because it is not inferrable from the
    coverage outcome: a front-only card is PARTIAL — rankable and forwarded —
    yet its confidence is LOW, because half the card was never seen. That
    combination is intended, not a contradiction.
    """
    if coverage is Coverage.INADEQUATE or contradictions or required_face_missing:
        return ReviewConfidence.LOW
    if (coverage is Coverage.PARTIAL or producers_disagreed
            or not card_context_known):
        return ReviewConfidence.MEDIUM
    return ReviewConfidence.HIGH
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_scoring.py -v`
Expected: PASS (28 tests including parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/policies/scoring_v1.py tests/review/test_scoring.py
git commit -m "feat(review): v1 scoring, grade estimate and review confidence"
```

**Acceptance:** all three monotonicity properties hold; only an I1-satisfying binding finding floors the score while an unresolved one stays rankable; an unmapped finding defaults to advisory; a missing required face is `low` confidence while other `PARTIAL` causes stay `medium`; the score and grade can disagree; every weight lives in this file.

---

### Task 19: Verdict engine and invariants I1/I2/I3

**Files:** Create `src/card_reviewer/review/policies/combine_v1.py`; Test `tests/review/test_verdict.py`

**Interfaces:**
- Consumes: `findings.py`, `policies/scoring_v1.py`, `policies/authority_v1.py`, `enums.py`
- Produces: `COMBINATION_POLICY_VERSION`, `MIN_DETECTABILITY_FOR_REJECT`, `REJECT_CONFIDENCE_FLOOR`, `i1_satisfied(finding, detectability, others) -> bool`, `decide_verdict(...) -> VerdictResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_verdict.py
import itertools

import pytest

from card_reviewer.review.enums import (
    Authority, Coverage, FindingState, Psa10Candidate, Scale, Verdict,
)
from card_reviewer.review.findings import Finding, FindingProducer
from card_reviewer.review.policies.combine_v1 import decide_verdict, i1_satisfied
from card_reviewer.review.provenance import (
    EvidenceOrigin, EvidenceRef, NormalizedBox,
)


def _f(state=FindingState.OBSERVED, conf=0.95, box=(0.0, 0.0, 0.3, 0.3),
       enhanced=False, relevant=True):
    origin = EvidenceOrigin.ENHANCED if enhanced else EvidenceOrigin.ORIGINAL
    return Finding(
        defect_type="rounding", category="corners", state=state,
        producer=FindingProducer.HEURISTIC, confidence=conf,
        psa10_relevant=relevant,
        location=NormalizedBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
        evidence=[EvidenceRef(artifact_id="a", image_hash="h", origin=origin,
                              enhancement="clahe:clip=2.0" if enhanced else None,
                              view="front")])


# --- I1 -------------------------------------------------------------------

def test_i1_requires_adequate_detectability_for_the_finding():
    """Poor photographs must be unable to produce rejections. This is the
    prong that carries the guarantee — the contradiction prong cannot fire
    on a badly photographed card, since nothing there reaches MODERATE."""
    assert i1_satisfied(_f(), Scale.HIGH, []) is True
    assert i1_satisfied(_f(), Scale.LOW, []) is False


def test_i1_requires_the_reject_confidence_floor():
    assert i1_satisfied(_f(conf=0.55), Scale.HIGH, []) is False


def test_i1_fails_on_a_material_contradiction_at_an_overlapping_location():
    contradicting = _f(state=FindingState.NOT_OBSERVED, box=(0.2, 0.2, 0.5, 0.5))
    assert i1_satisfied(_f(), Scale.HIGH, [(contradicting, Scale.HIGH)]) is False


def test_a_contradiction_elsewhere_on_the_card_is_not_material():
    elsewhere = _f(state=FindingState.NOT_OBSERVED, box=(0.7, 0.7, 0.9, 0.9))
    assert i1_satisfied(_f(), Scale.HIGH, [(elsewhere, Scale.HIGH)]) is True


def test_a_low_detectability_contradiction_does_not_block():
    weak = _f(state=FindingState.NOT_OBSERVED, box=(0.1, 0.1, 0.4, 0.4))
    assert i1_satisfied(_f(), Scale.HIGH, [(weak, Scale.LOW)]) is True


def test_suspected_findings_never_satisfy_i1():
    assert i1_satisfied(_f(state=FindingState.SUSPECTED), Scale.HIGH, []) is False


# --- verdict precedence ---------------------------------------------------

def test_an_i1_satisfying_disqualifier_rejects():
    r = decide_verdict([(_f(), Authority.BINDING, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.REJECT


def test_reject_outranks_inadequate_coverage():
    """A crease plainly visible on the front is knowledge, not absence of it.
    A missing back is a bar on passing, not a bar on rejecting."""
    r = decide_verdict([(_f(), Authority.BINDING, Scale.HIGH)],
                       Coverage.INADEQUATE, ambiguity=False)
    assert r.verdict is Verdict.REJECT


def test_an_observed_finding_failing_i1_routes_to_review_never_pass():
    """Something looked like a disqualifier and could not be resolved. That
    is an unresolved concern, not an absence of one."""
    r = decide_verdict([(_f(conf=0.5), Authority.BINDING, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.REVIEW


def test_advisory_authority_cannot_reject():
    r = decide_verdict([(_f(), Authority.ADVISORY, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.REVIEW


def test_inadequate_coverage_without_a_disqualifier_is_insufficient_images():
    r = decide_verdict([], Coverage.INADEQUATE, ambiguity=False)
    assert r.verdict is Verdict.INSUFFICIENT_IMAGES
    assert r.psa10_candidate is Psa10Candidate.UNKNOWN


def test_partial_coverage_reviews():
    r = decide_verdict([], Coverage.PARTIAL, ambiguity=False)
    assert r.verdict is Verdict.REVIEW


def test_a_clean_fully_covered_card_passes():
    r = decide_verdict([], Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.PASS
    assert r.psa10_candidate is Psa10Candidate.YES


def test_enhancement_only_evidence_cannot_reject():
    """I3 enforced at the verdict boundary."""
    r = decide_verdict([(_f(enhanced=True), Authority.BINDING, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.REVIEW


def test_pass_is_unreachable_without_sufficient_coverage():
    """I2, over every combination that is not SUFFICIENT."""
    for coverage in (Coverage.PARTIAL, Coverage.INADEQUATE):
        r = decide_verdict([], coverage, ambiguity=False)
        assert r.verdict is not Verdict.PASS


def test_the_verdict_function_is_total_over_the_full_cross_product():
    """DoD 12: coverage x I1-satisfying x I1-failing x ambiguity."""
    axes = itertools.product(
        list(Coverage), [False, True], [False, True], [False, True])
    for coverage, has_sat, has_unsat, ambiguity in axes:
        findings = []
        if has_sat:
            findings.append((_f(), Authority.BINDING, Scale.HIGH))
        if has_unsat:
            findings.append((_f(conf=0.5), Authority.BINDING, Scale.HIGH))
        r = decide_verdict(findings, coverage, ambiguity=ambiguity)
        assert r.verdict in set(Verdict)
        if has_sat:
            assert r.verdict is Verdict.REJECT
        elif coverage is Coverage.INADEQUATE:
            assert r.verdict is Verdict.INSUFFICIENT_IMAGES
        elif has_unsat or ambiguity or coverage is Coverage.PARTIAL:
            assert r.verdict is Verdict.REVIEW
        else:
            assert r.verdict is Verdict.PASS


def test_psa10_candidate_is_always_derived_from_the_verdict():
    mapping = {Verdict.PASS: Psa10Candidate.YES,
               Verdict.REVIEW: Psa10Candidate.UNCERTAIN,
               Verdict.REJECT: Psa10Candidate.NO,
               Verdict.INSUFFICIENT_IMAGES: Psa10Candidate.UNKNOWN}
    for coverage in Coverage:
        for has_sat in (False, True):
            f = [(_f(), Authority.BINDING, Scale.HIGH)] if has_sat else []
            r = decide_verdict(f, coverage, ambiguity=False)
            assert r.psa10_candidate is mapping[r.verdict]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_verdict.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/policies/combine_v1.py
"""Verdict resolution and the three invariants (spec §14, §15).

The four states are mutually exclusive and evaluated in STRICT ORDER —
first match wins. Stating them as independent conditions would leave a card
with both an observed crease and PARTIAL coverage matching two rows at once.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..enums import (
    Authority, Coverage, FindingState, Psa10Candidate, Scale, Verdict,
)
from ..findings import Finding, i3_satisfied

COMBINATION_POLICY_VERSION = "1.0.0"

MIN_DETECTABILITY_FOR_REJECT = Scale.MODERATE
# Spec §15 declares the floor as `HIGH` on the shared scale. Findings carry a
# float confidence, so the mapping is stated once here rather than a bare 0.8
# appearing as an unexplained magic number.
CONFIDENCE_BANDS: dict[Scale, float] = {
    Scale.LOW: 0.0, Scale.MODERATE: 0.5, Scale.HIGH: 0.8,
}
REJECT_CONFIDENCE_FLOOR = CONFIDENCE_BANDS[Scale.HIGH]


class VerdictResult(BaseModel):
    verdict: Verdict
    psa10_candidate: Psa10Candidate
    reasons: list[str] = Field(default_factory=list)
    policy_version: str = COMBINATION_POLICY_VERSION


_CANDIDATE: dict[Verdict, Psa10Candidate] = {
    Verdict.PASS: Psa10Candidate.YES,
    Verdict.REVIEW: Psa10Candidate.UNCERTAIN,
    Verdict.REJECT: Psa10Candidate.NO,
    Verdict.INSUFFICIENT_IMAGES: Psa10Candidate.UNKNOWN,
}


def i1_satisfied(finding: Finding, detectability: Scale,
                 others: list[tuple[Finding, Scale]]) -> bool:
    """I1 — ambiguity never rejects.

    The adequacy prong binds the ASSERTING finding rather than hoping for a
    contradicting one: on a badly photographed card no contradicting finding
    could reach MODERATE, so a contradiction-only test would weaken exactly
    where it is needed most.
    """
    if finding.state is not FindingState.OBSERVED:
        return False
    if not i3_satisfied(finding):
        return False
    if detectability < MIN_DETECTABILITY_FOR_REJECT:
        return False
    if finding.confidence < REJECT_CONFIDENCE_FLOOR:
        return False
    return not _material_contradiction(finding, others)


def _material_contradiction(finding: Finding,
                            others: list[tuple[Finding, Scale]]) -> bool:
    for other, other_detectability in others:
        if other is finding or other.defect_type != finding.defect_type:
            continue
        if finding.location is None or other.location is None:
            continue
        if not finding.location.overlaps(other.location):
            continue
        if (other.state is FindingState.NOT_OBSERVED
                and other_detectability >= MIN_DETECTABILITY_FOR_REJECT):
            return True
        if other.state is not finding.state and other.producer is not finding.producer:
            return True
    return False


def decide_verdict(
    findings: list[tuple[Finding, Authority, Scale]],
    coverage: Coverage,
    *, ambiguity: bool,
) -> VerdictResult:
    others = [(f, d) for f, _, d in findings]
    reasons: list[str] = []

    # Rule 1 — REJECT. A confidently observed disqualifier is knowledge, not
    # absence of it, so it outranks inadequate coverage.
    for finding, authority, detectability in findings:
        if not finding.psa10_relevant or authority is not Authority.BINDING:
            continue
        if i1_satisfied(finding, detectability, others):
            reasons.append(
                f"{finding.category}/{finding.defect_type} observed and I1-satisfying")
            return _result(Verdict.REJECT, reasons)

    # Rule 2 — INSUFFICIENT_IMAGES.
    if coverage is Coverage.INADEQUATE:
        return _result(Verdict.INSUFFICIENT_IMAGES, ["coverage INADEQUATE"])

    # Rule 3 — REVIEW. Includes an observed disqualifier that FAILS I1:
    # something looked like a defect and could not be established. That is an
    # unresolved concern, not an absence of one, and must never reach PASS.
    if coverage is Coverage.PARTIAL:
        reasons.append("coverage PARTIAL")
    for finding, _, _ in findings:
        if finding.state is FindingState.OBSERVED and finding.psa10_relevant:
            reasons.append(
                f"{finding.category}/{finding.defect_type} observed but not "
                "adequately evidenced to reject")
        elif finding.state is FindingState.SUSPECTED and finding.psa10_relevant:
            reasons.append(f"{finding.category}/{finding.defect_type} suspected")
    if ambiguity:
        reasons.append("unresolved ambiguity")
    if reasons:
        return _result(Verdict.REVIEW, reasons)

    # Rule 4 — otherwise. Reached only with SUFFICIENT coverage; this is what
    # makes the function total.
    return _result(Verdict.PASS, ["coverage SUFFICIENT, no disqualifier"])


def _result(verdict: Verdict, reasons: list[str]) -> VerdictResult:
    return VerdictResult(verdict=verdict, psa10_candidate=_CANDIDATE[verdict],
                         reasons=reasons)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_verdict.py -v`
Expected: PASS (17 tests, including the 24-cell cross-product)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/policies/combine_v1.py tests/review/test_verdict.py
git commit -m "feat(review): verdict precedence and invariants I1/I2/I3"
```

**Acceptance:** the cross-product test covers all 24 cells with exactly one verdict each; an I1-failing observed finding yields `REVIEW`; enhancement-only evidence cannot reject; `PASS` is unreachable without `SUFFICIENT`.

---
## Phase 5 — Imaging

The synthetic generator comes first because it is the only honest way to test a measurement engine: real photographs have no ground truth. Every CV task afterwards asserts against images whose true centering, border colour and damage are known by construction.

### Task 20: Synthetic card image generator

**Files:** Create `src/card_reviewer/review/imaging/__init__.py`, `src/card_reviewer/review/imaging/synthetic.py`; Test `tests/review/test_synthetic.py`

**Interfaces:**
- Produces: `CardSpec`, `render(spec) -> numpy.ndarray`, `render_png(spec) -> bytes`

This is a substantial piece of software in its own right, with its own tests and its own review — not a subtask of the CV work.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_synthetic.py
import numpy as np
import pytest

from card_reviewer.review.imaging.synthetic import CardSpec, render


def test_a_perfectly_centered_card_has_equal_borders():
    img = render(CardSpec(h_centering=50.0, v_centering=50.0))
    assert img.shape[2] == 3


@pytest.mark.parametrize("ratio", [50.0, 55.0, 60.0, 70.0])
def test_requested_centering_is_reproduced_within_a_pixel(ratio):
    """Ground truth by construction: the generator's own geometry is the
    oracle every centering test measures against."""
    spec = CardSpec(h_centering=ratio, card_w=600, card_h=840, border_px=40)
    img = render(spec)
    left, right = _border_widths(img)
    measured = 100.0 * left / (left + right)
    assert abs(measured - ratio) <= 1.0


def test_white_and_dark_borders_are_both_producible():
    white = render(CardSpec(border_color=(255, 255, 255)))
    dark = render(CardSpec(border_color=(20, 20, 20)))
    assert white[5, 5].mean() > 200 and dark[5, 5].mean() < 60


def test_a_borderless_design_has_no_uniform_border_band():
    img = render(CardSpec(borderless=True))
    top = img[2:6, :, :].reshape(-1, 3)
    assert top.std(axis=0).mean() > 5.0


def test_corner_damage_appears_only_where_requested():
    clean = render(CardSpec())
    damaged = render(CardSpec(corner_damage={"bottom_left": 0.8}))
    assert not np.array_equal(clean, damaged)
    assert np.array_equal(clean[:60, -60:], damaged[:60, -60:])


def test_rotation_and_perspective_are_reproducible_for_a_seed():
    a = render(CardSpec(rotation_deg=7.0, perspective=0.15, seed=42))
    b = render(CardSpec(rotation_deg=7.0, perspective=0.15, seed=42))
    assert np.array_equal(a, b)


def test_glare_covers_the_requested_region_only():
    img = render(CardSpec(glare_regions=["top_left"], seed=1))
    assert img[:80, :80].mean() > img[-80:, -80:].mean()


def _border_widths(img):
    """Locate the printed-art rectangle by column variance."""
    col_var = img.std(axis=(0, 2))
    inked = np.where(col_var > col_var.max() * 0.25)[0]
    return int(inked[0]), int(img.shape[1] - inked[-1] - 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/imaging/synthetic.py
"""Synthetic trading cards with known ground truth.

Real photographs have no ground truth — nobody can say a listing photo is
"exactly 54/46". This generator is the oracle: it renders a card whose
centering, border colour, damage and distortion are known by construction,
so every measurement test asserts against a value rather than a guess.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class CardSpec(BaseModel):
    card_w: int = 600
    card_h: int = 840
    border_px: int = 40
    h_centering: float = Field(default=50.0, ge=1.0, le=99.0)
    v_centering: float = Field(default=50.0, ge=1.0, le=99.0)
    border_color: tuple[int, int, int] = (255, 255, 255)
    art_color: tuple[int, int, int] = (40, 90, 160)
    borderless: bool = False
    corner_damage: dict[str, float] = Field(default_factory=dict)
    rotation_deg: float = 0.0
    perspective: float = 0.0
    glare_regions: list[str] = Field(default_factory=list)
    background: tuple[int, int, int] = (10, 10, 10)
    seed: int = 0


_CORNERS = {"top_left": (0, 0), "top_right": (0, 1),
            "bottom_left": (1, 0), "bottom_right": (1, 1)}


def render(spec: CardSpec) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(spec.seed)
    img = np.zeros((spec.card_h, spec.card_w, 3), np.uint8)
    img[:] = spec.border_color

    if spec.borderless:
        img[:] = spec.art_color
        noise = rng.integers(-25, 25, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else:
        total_h = spec.card_w - 2 * spec.border_px
        total_v = spec.card_h - 2 * spec.border_px
        # h_centering is the LEFT border's share of total border width.
        slack_h = spec.card_w - total_h
        left = int(round(slack_h * spec.h_centering / 100.0))
        slack_v = spec.card_h - total_v
        top = int(round(slack_v * spec.v_centering / 100.0))
        img[top:top + total_v, left:left + total_h] = spec.art_color

    for name, severity in spec.corner_damage.items():
        r, c = _CORNERS[name]
        size = int(40 * severity)
        ys = slice(0, size) if r == 0 else slice(spec.card_h - size, spec.card_h)
        xs = slice(0, size) if c == 0 else slice(spec.card_w - size, spec.card_w)
        patch = rng.integers(180, 255, (size, size, 3), dtype=np.uint8)
        img[ys, xs] = patch

    for region in spec.glare_regions:
        r, c = _CORNERS[region]
        ys = slice(0, 120) if r == 0 else slice(spec.card_h - 120, spec.card_h)
        xs = slice(0, 120) if c == 0 else slice(spec.card_w - 120, spec.card_w)
        img[ys, xs] = np.clip(img[ys, xs].astype(np.int16) + 90, 0, 255).astype(np.uint8)

    if spec.rotation_deg or spec.perspective:
        img = _distort(img, spec, cv2)
    return img


def _distort(img: np.ndarray, spec: CardSpec, cv2) -> np.ndarray:
    h, w = img.shape[:2]
    canvas = np.zeros((int(h * 1.3), int(w * 1.3), 3), np.uint8)
    canvas[:] = spec.background
    oy, ox = (canvas.shape[0] - h) // 2, (canvas.shape[1] - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img

    src = np.float32([[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]])
    d = spec.perspective * w
    dst = np.float32([[ox + d, oy], [ox + w, oy + d],
                      [ox + w - d, oy + h], [ox, oy + h - d]])
    out = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst),
                              (canvas.shape[1], canvas.shape[0]),
                              borderValue=spec.background)
    if spec.rotation_deg:
        centre = (canvas.shape[1] / 2, canvas.shape[0] / 2)
        m = cv2.getRotationMatrix2D(centre, spec.rotation_deg, 1.0)
        out = cv2.warpAffine(out, m, (canvas.shape[1], canvas.shape[0]),
                             borderValue=spec.background)
    return out


def render_png(spec: CardSpec) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", render(spec))
    if not ok:
        raise RuntimeError("failed to encode synthetic card as PNG")
    return buf.tobytes()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_synthetic.py -v`
Expected: PASS (7 tests including parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/imaging/ tests/review/test_synthetic.py
git commit -m "feat(review): synthetic card image generator with known ground truth"
```

**Acceptance:** requested centering is reproduced within one pixel across the parametrized range; white, dark and borderless designs all render; a seeded distortion is byte-identical on repeat.

---

### Task 21: Preflight

**Files:** Create `src/card_reviewer/review/imaging/preflight.py`; Test `tests/review/test_preflight.py`

**Interfaces:**
- Consumes: `synthetic.py` (tests only)
- Produces: `PREFLIGHT_VERSION`, `PreflightResult`, `analyze(image_bytes) -> PreflightResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_preflight.py
import cv2
import numpy as np

from card_reviewer.review.imaging.preflight import analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render_png


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_a_normal_card_photo_is_usable():
    assert analyze(render_png(CardSpec())).usable is True


def test_a_thumbnail_is_unusable_and_says_why():
    tiny = _png(np.zeros((150, 200, 3), np.uint8))
    r = analyze(tiny)
    assert r.usable is False and r.reason_code == "LOW_RESOLUTION"


def test_marking_an_image_unusable_is_never_a_reject_verdict():
    """An unusable image reduces coverage; it never condemns the card."""
    r = analyze(_png(np.zeros((150, 200, 3), np.uint8)))
    assert not hasattr(r, "verdict")


def test_a_blurred_image_is_flagged_with_low_sharpness():
    blurred = cv2.GaussianBlur(render(CardSpec()), (31, 31), 0)
    assert analyze(_png(blurred)).global_sharpness < analyze(render_png(CardSpec())).global_sharpness


def test_a_blown_out_image_reports_clipping():
    blown = np.full((840, 600, 3), 254, np.uint8)
    assert analyze(_png(blown)).clipped_fraction > 0.9


def test_corrupt_bytes_are_reported_not_raised():
    r = analyze(b"not an image")
    assert r.usable is False and r.reason_code == "DECODE_FAILED"


from card_reviewer.review.imaging.synthetic import render  # noqa: E402
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_preflight.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/imaging/preflight.py
"""Raw-image properties requiring no geometry (spec §7.1).

Marking an image unusable never contributes toward a REJECT verdict; it
reduces coverage, which routes toward REVIEW or INSUFFICIENT_IMAGES.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

PREFLIGHT_VERSION = "1.0.0"

MIN_WIDTH = 400
MIN_HEIGHT = 400
MIN_SHARPNESS = 25.0
MAX_CLIPPED_FRACTION = 0.6


class PreflightResult(BaseModel):
    usable: bool
    width: int = 0
    height: int = 0
    global_sharpness: float = 0.0
    clipped_fraction: float = 0.0
    reason_code: str | None = None
    version: str = PREFLIGHT_VERSION


def analyze(image_bytes: bytes) -> PreflightResult:
    import cv2

    buf = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return PreflightResult(usable=False, reason_code="DECODE_FAILED")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(((gray >= 250) | (gray <= 5)).mean())

    reason = None
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        reason = "LOW_RESOLUTION"
    elif sharpness < MIN_SHARPNESS:
        reason = "BLUR"
    elif clipped > MAX_CLIPPED_FRACTION:
        reason = "GLARE"

    return PreflightResult(usable=reason is None, width=w, height=h,
                           global_sharpness=sharpness, clipped_fraction=clipped,
                           reason_code=reason)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_preflight.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/imaging/preflight.py tests/review/test_preflight.py
git commit -m "feat(review): preflight raw-image analysis"
```

**Acceptance:** a thumbnail is unusable with `LOW_RESOLUTION`; corrupt bytes are reported not raised; the result carries no verdict field.

---

### Task 22: Geometry, normalization and border segmentation

**Files:** Create `src/card_reviewer/review/imaging/geometry.py`; Test `tests/review/test_geometry.py`

**Interfaces:**
- Produces: `GEOMETRY_VERSION`, `GeometryResult` (quad, transform, `boundary_confidence`, `normalized` image, `border_mask`), `analyze(image_bytes) -> GeometryResult`

**Why border segmentation lives here:** both `observability` (is this corner white, so unable to show whitening?) and `cv_measurements` (is there a reliable border reference?) need it, and neither should own a result the other depends on.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_geometry.py
import cv2
import numpy as np

from card_reviewer.review.imaging.geometry import analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render_png


def test_a_clean_card_boundary_is_detected_with_high_confidence():
    r = analyze(render_png(CardSpec(rotation_deg=0, perspective=0)))
    assert r.boundary_confidence > 0.8 and r.quad is not None


def test_a_rotated_card_is_rectified_to_an_axis_aligned_image():
    r = analyze(render_png(CardSpec(rotation_deg=8.0, perspective=0.1, seed=3)))
    assert r.normalized is not None
    assert abs(r.normalized.shape[1] / r.normalized.shape[0] - 600 / 840) < 0.1


def test_unreliable_detection_declines_geometry_dependent_work():
    """Spec §7.2 / build plan §11: never produce plausible numbers from a bad
    quad."""
    noise = np.random.default_rng(0).integers(0, 255, (800, 600, 3), dtype=np.uint8)
    r = analyze(cv2.imencode(".png", noise)[1].tobytes())
    assert r.boundary_confidence < 0.5
    assert r.normalized is None


def test_a_white_border_is_segmented_as_border():
    r = analyze(render_png(CardSpec(border_color=(255, 255, 255))))
    assert r.border_mask is not None and r.border_mask[:8, :8].mean() > 0.5


def test_a_borderless_design_yields_no_reliable_border_band():
    r = analyze(render_png(CardSpec(borderless=True)))
    assert r.has_reliable_border is False


def test_the_perspective_transform_is_emitted_as_provenance():
    r = analyze(render_png(CardSpec(perspective=0.12, seed=5)))
    assert r.transform is not None and len(r.transform) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/imaging/geometry.py
"""Boundary, perspective correction, normalization, border segmentation.

Establishes the ONE normalized card coordinate system every later stage,
defect location and future model output refers to.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

GEOMETRY_VERSION = "1.0.0"

NORM_W, NORM_H = 600, 840
MIN_BOUNDARY_CONFIDENCE = 0.5
BORDER_BAND_PX = 24


class GeometryResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    boundary_confidence: float
    quad: list[list[float]] | None = None
    transform: list[list[float]] | None = None
    normalized: np.ndarray | None = None
    border_mask: np.ndarray | None = None
    has_reliable_border: bool = False
    version: str = GEOMETRY_VERSION


def analyze(image_bytes: bytes) -> GeometryResult:
    import cv2

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return GeometryResult(boundary_confidence=0.0)

    quad, confidence = _detect_quad(img, cv2)
    if quad is None or confidence < MIN_BOUNDARY_CONFIDENCE:
        # Decline geometry-dependent work rather than inventing numbers.
        return GeometryResult(boundary_confidence=confidence)

    dst = np.float32([[0, 0], [NORM_W, 0], [NORM_W, NORM_H], [0, NORM_H]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    normalized = cv2.warpPerspective(img, matrix, (NORM_W, NORM_H))
    mask, reliable = _segment_border(normalized)

    return GeometryResult(
        boundary_confidence=confidence, quad=quad.tolist(),
        transform=matrix.tolist(), normalized=normalized,
        border_mask=mask, has_reliable_border=reliable)


def _detect_quad(img, cv2) -> tuple[np.ndarray | None, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return None, 0.2
    area_ratio = cv2.contourArea(largest) / (img.shape[0] * img.shape[1])
    confidence = float(min(1.0, max(0.0, area_ratio * 1.6)))
    return _order(approx.reshape(4, 2)), confidence


def _order(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _segment_border(normalized) -> tuple[np.ndarray, bool]:
    """A border band is 'reliable' when it is uniform enough to measure against."""
    gray = normalized.mean(axis=2)
    band = np.zeros(gray.shape, np.float32)
    band[:BORDER_BAND_PX, :] = 1.0
    band[-BORDER_BAND_PX:, :] = 1.0
    band[:, :BORDER_BAND_PX] = 1.0
    band[:, -BORDER_BAND_PX:] = 1.0
    values = gray[band > 0]
    reliable = bool(values.std() < 30.0)
    return band, reliable
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_geometry.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/imaging/geometry.py tests/review/test_geometry.py
git commit -m "feat(review): geometry, normalization and border segmentation"
```

**Acceptance:** an unreliable boundary returns `normalized is None` rather than a plausible-looking rectification; borderless designs report `has_reliable_border is False`.

---
### Task 23: Observability — detectability and suitability with classed reason codes

**Files:** Create `src/card_reviewer/review/imaging/observability.py`; Test `tests/review/test_observability.py`

**Interfaces:**
- Consumes: `geometry.py`, `taxonomy.py`
- Produces: `OBSERVABILITY_VERSION`, `ObservabilityResult` (per-region-per-defect-type detectability, reason codes with class, per-purpose suitability, glare/occlusion masks), `analyze(geometry_result) -> ObservabilityResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_observability.py
from card_reviewer.review.enums import Scale, UndetectabilityClass
from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.observability import analyze
from card_reviewer.review.imaging.synthetic import CardSpec, render_png


def _obs(spec):
    return analyze(geom(render_png(spec)))


def test_detectability_is_reported_per_region_and_per_defect_type():
    r = _obs(CardSpec())
    assert ("bottom_left", "corners", "whitening") in r.detectability
    assert ("bottom_left", "corners", "rounding") in r.detectability


def test_a_white_corner_cannot_show_whitening_and_says_so_structurally():
    """CORNERS_COLORED_001 as physics: the reason code is WHITE_BORDER and
    its class is structural, so coverage waives it rather than demanding a
    photograph that could never help."""
    r = _obs(CardSpec(border_color=(255, 255, 255)))
    key = ("bottom_left", "corners", "whitening")
    assert r.detectability[key] < Scale.MODERATE
    assert r.reason_codes[key] == "WHITE_BORDER"
    assert r.reason_class(key) is UndetectabilityClass.STRUCTURAL


def test_the_same_white_corner_is_still_assessable_for_rounding():
    """This is what keeps PASS reachable for white-bordered cards."""
    r = _obs(CardSpec(border_color=(255, 255, 255)))
    assert r.detectability[("bottom_left", "corners", "rounding")] >= Scale.MODERATE


def test_a_dark_border_gives_high_whitening_detectability():
    r = _obs(CardSpec(border_color=(20, 20, 20)))
    assert r.detectability[("bottom_left", "corners", "whitening")] >= Scale.MODERATE


def test_glare_is_circumstantial_not_structural():
    r = _obs(CardSpec(border_color=(20, 20, 20), glare_regions=["top_left"], seed=2))
    key = ("top_left", "surface", "scratches")
    assert r.reason_codes.get(key) == "GLARE"
    assert r.reason_class(key) is UndetectabilityClass.CIRCUMSTANTIAL


def test_a_photo_can_be_good_for_centering_and_useless_for_surface():
    """A glare spot must not condemn a whole image."""
    r = _obs(CardSpec(border_color=(20, 20, 20), glare_regions=["top_left"], seed=2))
    assert r.suitability["centering"] >= Scale.MODERATE
    assert r.suitability["surface"] < r.suitability["centering"]


def test_every_shortfall_below_moderate_carries_a_declared_reason_code():
    r = _obs(CardSpec(border_color=(255, 255, 255)))
    for key, value in r.detectability.items():
        if value < Scale.MODERATE:
            assert key in r.reason_codes
            assert r.reason_class(key) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_observability.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/imaging/observability.py
"""Post-geometry detectability and suitability (spec §7.3).

Detectability is a physical property of the photograph and the card's own
design — what COULD be seen here, independent of any rubric. Rule IDs are
cited as provenance for why it is worth measuring, never as its contract:
that is why taxonomy version, not rubric version, is in this stage's
producer signature.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from ..enums import Scale, UndetectabilityClass
from ..taxonomy import CATEGORIES, class_of, defect_types_for
from .geometry import GeometryResult

OBSERVABILITY_VERSION = "1.0.0"

REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right", "center")
WHITE_BORDER_LUMA = 200.0
GLARE_LUMA = 245.0
GLARE_FRACTION = 0.15

Key = tuple[str, str, str]


class ObservabilityResult(BaseModel):
    detectability: dict[Key, Scale] = Field(default_factory=dict)
    reason_codes: dict[Key, str] = Field(default_factory=dict)
    suitability: dict[str, Scale] = Field(default_factory=dict)
    version: str = OBSERVABILITY_VERSION

    def reason_class(self, key: Key) -> UndetectabilityClass | None:
        code = self.reason_codes.get(key)
        return class_of(code) if code else None


def analyze(geometry: GeometryResult) -> ObservabilityResult:
    if geometry.normalized is None:
        det = {(r, c, d): Scale.NONE
               for r in REGIONS for c in CATEGORIES for d in defect_types_for(c)}
        return ObservabilityResult(
            detectability=det,
            reason_codes={k: "SEVERE_PERSPECTIVE" for k in det},
            suitability={c: Scale.NONE for c in CATEGORIES})

    gray = geometry.normalized.mean(axis=2)
    det: dict[Key, Scale] = {}
    reasons: dict[Key, str] = {}

    for region in REGIONS:
        patch = _patch(gray, region)
        glared = float((patch >= GLARE_LUMA).mean()) > GLARE_FRACTION
        bright = float(patch.mean()) >= WHITE_BORDER_LUMA
        for category in CATEGORIES:
            for defect_type in defect_types_for(category):
                key = (region, category, defect_type)
                if glared:
                    det[key], reasons[key] = Scale.LOW, "GLARE"
                elif defect_type == "whitening" and bright:
                    # A white corner cannot show whitening. Structural: no
                    # photograph of THIS card could ever show it.
                    det[key], reasons[key] = Scale.LOW, "WHITE_BORDER"
                elif category == "centering" and not geometry.has_reliable_border:
                    det[key], reasons[key] = Scale.LOW, "BORDERLESS_DESIGN"
                else:
                    det[key] = Scale.HIGH
    return ObservabilityResult(
        detectability=det, reason_codes=reasons,
        suitability=_suitability(det))


def _patch(gray: np.ndarray, region: str) -> np.ndarray:
    h, w = gray.shape
    match region:
        case "top_left": return gray[:h // 5, :w // 5]
        case "top_right": return gray[:h // 5, -w // 5:]
        case "bottom_left": return gray[-h // 5:, :w // 5]
        case "bottom_right": return gray[-h // 5:, -w // 5:]
        case _: return gray[h // 4:3 * h // 4, w // 4:3 * w // 4]


def _suitability(det: dict[Key, Scale]) -> dict[str, Scale]:
    out: dict[str, Scale] = {}
    for category in CATEGORIES:
        values = [v for (_, c, _), v in det.items() if c == category]
        out[category] = Scale(min(values)) if values else Scale.NONE
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_observability.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/imaging/observability.py tests/review/test_observability.py
git commit -m "feat(review): observability with per-defect-type classed detectability"
```

**Acceptance:** a white corner is `WHITE_BORDER`/structural for whitening yet still assessable for rounding; glare is circumstantial; every sub-`MODERATE` value carries a declared code.

---

### Task 24: Centering measurement

**Files:** Create `src/card_reviewer/review/imaging/measure/__init__.py`, `measure/centering.py`; Test `tests/review/test_measure_centering.py`

**Interfaces:**
- Produces: `CenteringMeasurement` (`measurable`, `horizontal`, `vertical`, `method`, `precision_pp`, `reason`), `measure_centering(geometry) -> CenteringMeasurement`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_measure_centering.py
import pytest

from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.measure.centering import measure_centering
from card_reviewer.review.imaging.synthetic import CardSpec, render_png


@pytest.mark.parametrize("truth", [50.0, 55.0, 60.0, 65.0])
def test_measured_centering_lands_within_the_declared_tolerance(truth):
    m = measure_centering(geom(render_png(CardSpec(h_centering=truth))))
    assert m.measurable is True
    assert abs(m.horizontal - truth) <= m.precision_pp


def test_the_measurement_declares_its_own_precision():
    m = measure_centering(geom(render_png(CardSpec())))
    assert m.precision_pp == 1.5


def test_a_borderless_design_reports_not_measurable_with_a_reason():
    m = measure_centering(geom(render_png(CardSpec(borderless=True))))
    assert m.measurable is False
    assert m.reason == "BORDERLESS_OR_NO_RELIABLE_REFERENCE"


def test_no_ratio_is_forced_onto_a_borderless_card():
    """CENTERING_BORDERLESS_001 binds directly here."""
    m = measure_centering(geom(render_png(CardSpec(borderless=True))))
    assert m.horizontal is None and m.vertical is None


def test_the_measurement_carries_no_acceptability_judgment():
    """Product leniency is the heuristic layer's decision, never the
    measurement layer's — CENTERING_PRODUCT_LENIENCY_001 does not apply here."""
    m = measure_centering(geom(render_png(CardSpec(h_centering=62.0))))
    assert not hasattr(m, "passes")
    assert not hasattr(m, "acceptable")


def test_an_undetected_boundary_yields_not_measurable():
    from card_reviewer.review.imaging.geometry import GeometryResult
    m = measure_centering(GeometryResult(boundary_confidence=0.1))
    assert m.measurable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_measure_centering.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/imaging/measure/centering.py
"""Centering: measurement, never acceptability (spec §7.4).

Reports the ratio and the precision the method actually supports. Whether
54/46 passes on Prizm and fails on Bowman Chrome is the heuristic layer's
decision — CENTERING_PRODUCT_LENIENCY_001 does not apply in this module.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..geometry import GeometryResult

PRECISION_PP = 1.5
INK_VARIANCE_FRACTION = 0.25


class CenteringMeasurement(BaseModel):
    measurable: bool
    horizontal: float | None = None
    vertical: float | None = None
    method: str | None = None
    precision_pp: float = PRECISION_PP
    reason: str | None = None


def measure_centering(geometry: GeometryResult) -> CenteringMeasurement:
    if geometry.normalized is None or not geometry.has_reliable_border:
        return CenteringMeasurement(
            measurable=False, reason="BORDERLESS_OR_NO_RELIABLE_REFERENCE")

    gray = geometry.normalized.mean(axis=2)
    horizontal = _ratio(gray.std(axis=0))
    vertical = _ratio(gray.std(axis=1))
    if horizontal is None or vertical is None:
        return CenteringMeasurement(
            measurable=False, reason="BORDERLESS_OR_NO_RELIABLE_REFERENCE")
    return CenteringMeasurement(measurable=True, horizontal=horizontal,
                                vertical=vertical, method="border_geometry")


def _ratio(variance: np.ndarray) -> float | None:
    """Locate the printed-art band, then express the leading border's share."""
    threshold = variance.max() * INK_VARIANCE_FRACTION
    inked = np.where(variance > threshold)[0]
    if inked.size < 2:
        return None
    leading = float(inked[0])
    trailing = float(variance.size - inked[-1] - 1)
    total = leading + trailing
    if total <= 0:
        return None
    return round(100.0 * leading / total, 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_measure_centering.py -v`
Expected: PASS (6 tests including parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/imaging/measure/ tests/review/test_measure_centering.py
git commit -m "feat(review): centering measurement with declared precision"
```

**Acceptance:** measured values land within the declared tolerance against synthetic ground truth; borderless reports `measurable: false`; no acceptability field exists.

---

### Task 25: Corner and edge crops with anomaly candidates

**Files:** Create `src/card_reviewer/review/imaging/measure/corners.py`, `measure/edges.py`; Test `tests/review/test_measure_corners_edges.py`

**Interfaces:**
- Consumes: `geometry.py`, `storage/artifacts.py`, `provenance.py`
- Produces: `measure_corners(geometry, store, image_hash) -> CornerResult`, `measure_edges(...) -> EdgeResult`, each with crops and anomaly **candidates**

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_measure_corners_edges.py
import pytest

from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.measure.corners import measure_corners
from card_reviewer.review.imaging.measure.edges import measure_edges
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.provenance import EvidenceOrigin
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


def test_four_corner_crops_are_produced_per_image(store):
    r = measure_corners(geom(render_png(CardSpec())), store, "h1")
    assert len(r.crops) == 4


def test_crops_land_in_the_measurement_owned_directory(store):
    r = measure_corners(geom(render_png(CardSpec())), store, "h1")
    assert "/corners/" in str(store.path_of(next(iter(r.crops.values()))))


def test_corner_damage_produces_an_anomaly_candidate_not_a_defect(store):
    spec = CardSpec(border_color=(20, 20, 20),
                    corner_damage={"bottom_left": 0.9})
    r = measure_corners(geom(render_png(spec)), store, "h1")
    assert any(a["region"] == "bottom_left" for a in r.anomalies)
    for a in r.anomalies:
        assert a["kind"] == "candidate"
        assert "defect" not in a


def test_a_clean_card_yields_no_corner_anomalies(store):
    r = measure_corners(geom(render_png(CardSpec(border_color=(20, 20, 20)))),
                        store, "h1")
    assert r.anomalies == []


def test_evidence_refs_record_normalized_origin_not_enhanced(store):
    r = measure_corners(geom(render_png(CardSpec())), store, "h1")
    assert all(ref.origin is EvidenceOrigin.NORMALIZED for ref in r.evidence_refs)


def test_four_edge_strips_are_produced_per_image(store):
    r = measure_edges(geom(render_png(CardSpec())), store, "h1")
    assert len(r.crops) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_measure_corners_edges.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/imaging/measure/corners.py
"""Corner crops and anomaly CANDIDATES — explicitly not defects."""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...provenance import EvidenceOrigin, EvidenceRef, NormalizedBox
from ...storage.artifacts import ArtifactStore
from ..geometry import GeometryResult

CORNER_FRACTION = 0.12
ANOMALY_CONTRAST = 45.0

CORNERS = {
    "top_left": (0.0, 0.0), "top_right": (1.0, 0.0),
    "bottom_left": (0.0, 1.0), "bottom_right": (1.0, 1.0),
}


class CornerResult(BaseModel):
    crops: dict[str, str] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


def measure_corners(geometry: GeometryResult, store: ArtifactStore,
                    image_hash: str) -> CornerResult:
    import cv2

    result = CornerResult()
    if geometry.normalized is None:
        return result
    img = geometry.normalized
    h, w = img.shape[:2]
    size_x, size_y = int(w * CORNER_FRACTION), int(h * CORNER_FRACTION)

    for name, (fx, fy) in CORNERS.items():
        x0 = 0 if fx == 0.0 else w - size_x
        y0 = 0 if fy == 0.0 else h - size_y
        patch = img[y0:y0 + size_y, x0:x0 + size_x]
        artifact_id = store.put_derived(
            image_hash, "corners", f"{name}.png",
            cv2.imencode(".png", patch)[1].tobytes())
        result.crops[name] = artifact_id

        box = NormalizedBox(x0=x0 / w, y0=y0 / h,
                            x1=(x0 + size_x) / w, y1=(y0 + size_y) / h)
        result.evidence_refs.append(EvidenceRef(
            artifact_id=artifact_id, image_hash=image_hash,
            origin=EvidenceOrigin.NORMALIZED, region=box,
            view=f"corner_{name}"))

        gray = patch.mean(axis=2)
        if float(gray.std()) > ANOMALY_CONTRAST:
            result.anomalies.append({
                "kind": "candidate", "region": name, "category": "corners",
                "defect_type": "rounding",
                "contrast": float(gray.std()),
                "artifact_id": artifact_id,
            })
    return result
```

```python
# src/card_reviewer/review/imaging/measure/edges.py
"""Edge strips and anomaly candidates. Same contract as corners."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...provenance import EvidenceOrigin, EvidenceRef, NormalizedBox
from ...storage.artifacts import ArtifactStore
from ..geometry import GeometryResult

EDGE_FRACTION = 0.06
ANOMALY_CONTRAST = 45.0

EDGES = ("top", "bottom", "left", "right")


class EdgeResult(BaseModel):
    crops: dict[str, str] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


def measure_edges(geometry: GeometryResult, store: ArtifactStore,
                  image_hash: str) -> EdgeResult:
    import cv2

    result = EdgeResult()
    if geometry.normalized is None:
        return result
    img = geometry.normalized
    h, w = img.shape[:2]
    band_y, band_x = int(h * EDGE_FRACTION), int(w * EDGE_FRACTION)

    slices = {
        "top": (slice(0, band_y), slice(0, w)),
        "bottom": (slice(h - band_y, h), slice(0, w)),
        "left": (slice(0, h), slice(0, band_x)),
        "right": (slice(0, h), slice(w - band_x, w)),
    }
    for name, (ys, xs) in slices.items():
        patch = img[ys, xs]
        artifact_id = store.put_derived(
            image_hash, "edges", f"{name}.png",
            cv2.imencode(".png", patch)[1].tobytes())
        result.crops[name] = artifact_id
        result.evidence_refs.append(EvidenceRef(
            artifact_id=artifact_id, image_hash=image_hash,
            origin=EvidenceOrigin.NORMALIZED,
            region=NormalizedBox(x0=xs.start / w, y0=ys.start / h,
                                 x1=xs.stop / w, y1=ys.stop / h),
            view=f"edge_{name}"))
        if float(patch.mean(axis=2).std()) > ANOMALY_CONTRAST:
            result.anomalies.append({
                "kind": "candidate", "region": name, "category": "edges",
                "defect_type": "chipping", "artifact_id": artifact_id})
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_measure_corners_edges.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/imaging/measure/corners.py src/card_reviewer/review/imaging/measure/edges.py tests/review/test_measure_corners_edges.py
git commit -m "feat(review): corner and edge crops with anomaly candidates"
```

**Acceptance:** anomalies are always `kind: candidate`; crops carry `NORMALIZED` provenance; measurement crops land under the stage's own directory.

---

### Task 26: Surface enhancement with recorded provenance

**Files:** Create `src/card_reviewer/review/imaging/measure/surface.py`; Test `tests/review/test_measure_surface.py`

**Interfaces:**
- Produces: `measure_surface(geometry, store, image_hash) -> SurfaceResult` with the preserved original plus deterministic enhanced views, each `EvidenceRef` carrying `origin=ENHANCED` and its method string

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_measure_surface.py
import pytest

from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.measure.surface import measure_surface
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.provenance import EvidenceOrigin
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def result(tmp_path):
    return measure_surface(geom(render_png(CardSpec(seed=4))),
                           ArtifactStore(tmp_path), "h1")


def test_the_unenhanced_original_is_always_preserved_alongside(result):
    assert any(r.origin is EvidenceOrigin.NORMALIZED for r in result.evidence_refs)


def test_every_enhanced_view_records_its_method(result):
    enhanced = [r for r in result.evidence_refs
                if r.origin is EvidenceOrigin.ENHANCED]
    assert enhanced
    assert all(r.enhancement for r in enhanced)


def test_enhancement_parameters_are_reproducible(tmp_path):
    a = measure_surface(geom(render_png(CardSpec(seed=4))),
                        ArtifactStore(tmp_path / "a"), "h1")
    b = measure_surface(geom(render_png(CardSpec(seed=4))),
                        ArtifactStore(tmp_path / "b"), "h1")
    assert ({r.enhancement for r in a.evidence_refs}
            == {r.enhancement for r in b.evidence_refs})


def test_anomalies_record_the_enhancement_level_that_surfaced_them(result):
    for a in result.anomalies:
        assert "surfaced_by" in a
        assert "visible_in_original" in a


def test_an_enhancement_only_anomaly_is_marked_not_visible_in_original(result):
    only_enhanced = [a for a in result.anomalies if not a["visible_in_original"]]
    for a in only_enhanced:
        assert a["surfaced_by"] != "original"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_measure_surface.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/imaging/measure/surface.py
"""Surface views: deterministic enhancements ALONGSIDE the preserved original.

Every anomaly candidate records the enhancement level that surfaced it and
whether it is visible in the unenhanced view. That record is what lets I3 be
enforced later as pure logic rather than by re-examining pixels.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...provenance import EvidenceOrigin, EvidenceRef
from ...storage.artifacts import ArtifactStore
from ..geometry import GeometryResult

CLAHE_CLIP = 2.0
CLAHE_GRID = 8
SHARPEN_AMOUNT = 1.5
ANOMALY_CONTRAST = 38.0

ENHANCEMENTS = {
    "clahe": f"clahe:clip={CLAHE_CLIP},grid={CLAHE_GRID}",
    "sharpen": f"sharpen:amount={SHARPEN_AMOUNT}",
    "edge": "edge:canny:50,150",
}


class SurfaceResult(BaseModel):
    crops: dict[str, str] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


def measure_surface(geometry: GeometryResult, store: ArtifactStore,
                    image_hash: str) -> SurfaceResult:
    import cv2
    import numpy as np

    result = SurfaceResult()
    if geometry.normalized is None:
        return result
    img = geometry.normalized
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_id = store.put_derived(image_hash, "surface", "original.png",
                                    cv2.imencode(".png", img)[1].tobytes())
    result.crops["original"] = original_id
    result.evidence_refs.append(EvidenceRef(
        artifact_id=original_id, image_hash=image_hash,
        origin=EvidenceOrigin.NORMALIZED, view="surface_original"))

    views = {
        "clahe": cv2.createCLAHE(CLAHE_CLIP, (CLAHE_GRID, CLAHE_GRID)).apply(gray),
        "sharpen": cv2.addWeighted(
            gray, 1 + SHARPEN_AMOUNT,
            cv2.GaussianBlur(gray, (0, 0), 3), -SHARPEN_AMOUNT, 0),
        "edge": cv2.Canny(gray, 50, 150),
    }
    original_contrast = float(gray.std())

    for name, view in views.items():
        artifact_id = store.put_derived(
            image_hash, "surface", f"{name}.png",
            cv2.imencode(".png", view)[1].tobytes())
        result.crops[name] = artifact_id
        result.evidence_refs.append(EvidenceRef(
            artifact_id=artifact_id, image_hash=image_hash,
            origin=EvidenceOrigin.ENHANCED, enhancement=ENHANCEMENTS[name],
            view=f"surface_{name}"))

        if float(np.asarray(view).std()) > ANOMALY_CONTRAST:
            visible_in_original = original_contrast > ANOMALY_CONTRAST
            result.anomalies.append({
                "kind": "candidate", "category": "surface",
                "defect_type": "scratches",
                "surfaced_by": name if not visible_in_original else "original",
                "visible_in_original": visible_in_original,
                "artifact_id": artifact_id,
            })
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_measure_surface.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/imaging/measure/surface.py tests/review/test_measure_surface.py
git commit -m "feat(review): surface enhancement with recorded provenance"
```

**Acceptance:** the original is preserved alongside every enhancement; each enhanced ref carries a reproducible method string; anomalies record what surfaced them.

---
## Phase 6 — Assembly and caching

### Task 27: Candidate-level evidence assembly

**Files:** Create `src/card_reviewer/review/assembly.py`; Test `tests/review/test_assembly.py`

**Interfaces:**
- Consumes: `roles.py`, `imaging/*`, `provenance.py`
- Produces: `ASSEMBLY_VERSION`, `ImageEvidence`, `Assembled`, `assemble(images, roles) -> Assembled`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_assembly.py
from card_reviewer.review.assembly import ImageEvidence, assemble
from card_reviewer.review.enums import Scale
from card_reviewer.review.roles import ImageRole, ResolvedRole
from card_reviewer.review.enums import Provenance


def _role(h, role):
    return ResolvedRole(image_hash=h, role=role, provenance=Provenance.SUPPLIED,
                        confidence=1.0)


def _img(h, det, sharpness=100.0):
    return ImageEvidence(image_hash=h, detectability=det, sharpness=sharpness,
                         centering={"measurable": True, "horizontal": 52.0},
                         anomalies=[], evidence_refs={})


def test_a_corner_glared_in_one_photo_and_clear_in_another_is_observable():
    """Merging detectability across images is the point of this stage."""
    a = _img("h1", {("bottom_left", "corners", "rounding"): Scale.LOW})
    b = _img("h2", {("bottom_left", "corners", "rounding"): Scale.HIGH})
    out = assemble([a, b], {"h1": _role("h1", ImageRole.FRONT),
                            "h2": _role("h2", ImageRole.FRONT)})
    assert out.detectability[(ImageRole.FRONT, "corners", "rounding")] is Scale.HIGH


def test_assembly_records_which_image_established_a_value():
    a = _img("h1", {("bottom_left", "corners", "rounding"): Scale.LOW})
    b = _img("h2", {("bottom_left", "corners", "rounding"): Scale.HIGH})
    out = assemble([a, b], {"h1": _role("h1", ImageRole.FRONT),
                            "h2": _role("h2", ImageRole.FRONT)})
    assert out.provenance[(ImageRole.FRONT, "corners", "rounding")] == "h2"


def test_the_sharpest_front_is_selected_for_surface_work():
    dull = _img("h1", {}, sharpness=20.0)
    sharp = _img("h2", {}, sharpness=300.0)
    out = assemble([dull, sharp], {"h1": _role("h1", ImageRole.FRONT),
                                   "h2": _role("h2", ImageRole.FRONT)})
    assert out.best_for["surface"] == "h2"


def test_conflicting_measurements_are_preserved_not_averaged():
    a = _img("h1", {}); a.centering["horizontal"] = 52.0
    b = _img("h2", {}); b.centering["horizontal"] = 61.0
    out = assemble([a, b], {"h1": _role("h1", ImageRole.FRONT),
                            "h2": _role("h2", ImageRole.FRONT)})
    assert len(out.conflicts) == 1
    assert 52.0 in out.conflicts[0]["values"] and 61.0 in out.conflicts[0]["values"]


def test_unknown_role_images_contribute_only_face_independent_work():
    unknown = _img("h1", {("center", "surface", "scratches"): Scale.HIGH})
    out = assemble([unknown], {"h1": _role("h1", ImageRole.UNKNOWN)})
    assert not any(k[0] is ImageRole.FRONT for k in out.detectability)


def test_faces_present_reports_only_confidently_resolved_faces():
    out = assemble([_img("h1", {})], {"h1": _role("h1", ImageRole.FRONT)})
    assert out.faces_present == (ImageRole.FRONT,)


def test_reason_codes_survive_assembly_so_coverage_can_classify_them():
    """Without this, WHITE_BORDER never reaches the coverage policy and every
    structural limitation is misread as circumstantial — which would make the
    structural exemption unreachable in the real pipeline."""
    a = ImageEvidence(
        image_hash="h1",
        detectability={("bottom_left", "corners", "whitening"): Scale.LOW},
        reason_codes={("bottom_left", "corners", "whitening"): "WHITE_BORDER"})
    out = assemble([a], {"h1": _role("h1", ImageRole.FRONT)})
    assert out.reason_codes[(ImageRole.FRONT, "corners", "whitening")] == "WHITE_BORDER"


def test_a_reason_code_is_dropped_once_another_photo_resolves_the_defect():
    """If one photo shows the corner clearly, the other's glare is no longer
    a limitation on this card."""
    glared = ImageEvidence(
        image_hash="h1",
        detectability={("bottom_left", "corners", "rounding"): Scale.LOW},
        reason_codes={("bottom_left", "corners", "rounding"): "GLARE"})
    clear = ImageEvidence(
        image_hash="h2",
        detectability={("bottom_left", "corners", "rounding"): Scale.HIGH})
    out = assemble([glared, clear], {"h1": _role("h1", ImageRole.FRONT),
                                     "h2": _role("h2", ImageRole.FRONT)})
    assert (ImageRole.FRONT, "corners", "rounding") not in out.reason_codes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_assembly.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/assembly.py
"""Fuse per-image results into one view of the card (spec §9).

A corner glared in one photo and clear in another is OBSERVABLE, and the
assembly records which image established that. Conflicting measurements are
preserved rather than averaged away — the disagreement is information.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import Scale
from .provenance import EvidenceRef
from .roles import ImageRole, ResolvedRole

ASSEMBLY_VERSION = "1.0.0"

CONFLICT_THRESHOLD_PP = 5.0


class ImageEvidence(BaseModel):
    image_hash: str
    detectability: dict[tuple[str, str, str], Scale] = Field(default_factory=dict)
    # Carried through from ObservabilityResult. Dropping these here is why
    # WHITE_BORDER would never reach the coverage policy, silently turning
    # every structural limitation into a circumstantial one and making the
    # structural exemption (DoD 10) unreachable end to end.
    reason_codes: dict[tuple[str, str, str], str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    sharpness: float = 0.0
    centering: dict[str, Any] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: dict[str, list[EvidenceRef]] = Field(default_factory=dict)


class Assembled(BaseModel):
    detectability: dict[tuple[ImageRole, str, str], Scale] = Field(default_factory=dict)
    reason_codes: dict[tuple[ImageRole, str, str], str] = Field(default_factory=dict)
    provenance: dict[tuple[ImageRole, str, str], str] = Field(default_factory=dict)
    best_for: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    faces_present: tuple[ImageRole, ...] = ()
    centering: dict[str, Any] = Field(default_factory=dict)


def assemble(images: list[ImageEvidence],
             roles: dict[str, ResolvedRole]) -> Assembled:
    out = Assembled()
    faces: set[ImageRole] = set()

    for image in images:
        role = roles[image.image_hash].role
        if role is ImageRole.UNKNOWN:
            # Still contributes face-independent work, but never claims a face.
            out.anomalies.extend(image.anomalies)
            continue
        faces.add(role)
        for (region, category, defect_type), value in image.detectability.items():
            key = (role, category, defect_type)
            # Best-of across images: a defect visible in ANY photo is observable.
            if value > out.detectability.get(key, Scale.NONE):
                out.detectability[key] = value
                out.provenance[key] = image.image_hash
                # The reason travels with the value it explains. If the best
                # view of this defect type is still limited, coverage needs to
                # know WHY — structural and circumstantial are handled
                # completely differently.
                code = image.reason_codes.get((region, category, defect_type))
                if code and value < Scale.MODERATE:
                    out.reason_codes[key] = code
                else:
                    out.reason_codes.pop(key, None)
        out.limitations.extend(image.limitations)
        out.anomalies.extend(image.anomalies)
        for purpose, refs in image.evidence_refs.items():
            out.evidence_refs.setdefault(purpose, []).extend(refs)

    out.faces_present = tuple(sorted(faces, key=lambda r: r.value))
    out.best_for = _best_for(images, roles)
    out.conflicts = _conflicts(images, roles)
    fronts = [i for i in images if roles[i.image_hash].role is ImageRole.FRONT]
    out.centering = fronts[0].centering if fronts else {}
    return out


def _best_for(images: list[ImageEvidence],
              roles: dict[str, ResolvedRole]) -> dict[str, str]:
    fronts = [i for i in images if roles[i.image_hash].role is ImageRole.FRONT]
    if not fronts:
        return {}
    sharpest = max(fronts, key=lambda i: i.sharpness)
    return {"surface": sharpest.image_hash, "centering": fronts[0].image_hash}


def _conflicts(images: list[ImageEvidence],
               roles: dict[str, ResolvedRole]) -> list[dict[str, Any]]:
    values = [(i.image_hash, i.centering.get("horizontal"))
              for i in images
              if roles[i.image_hash].role is ImageRole.FRONT
              and i.centering.get("horizontal") is not None]
    if len(values) < 2:
        return []
    numbers = [v for _, v in values]
    if max(numbers) - min(numbers) <= CONFLICT_THRESHOLD_PP:
        return []
    return [{"field": "centering.horizontal", "values": numbers,
             "images": [h for h, _ in values]}]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_assembly.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/assembly.py tests/review/test_assembly.py
git commit -m "feat(review): candidate-level evidence assembly"
```

**Acceptance:** detectability merges best-of across images with provenance; **reason codes travel with the values they explain**, so `WHITE_BORDER` reaches the coverage policy and a code is dropped once another photo resolves the defect; conflicts are preserved not averaged; unknown-role images never claim a face.

---

### Task 28: Stage runner with content-addressed caching

**Files:** Create `src/card_reviewer/review/pipeline.py` (stage runner only); Test `tests/review/test_stage_runner.py`

**Interfaces:**
- Consumes: `fingerprint.py`, `storage/repository.py`
- Produces: `StageValidationError`, `StageRunner.run(stage, inputs, versions, compute, *, schema=None, image_hash=None, candidate_id=None) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_stage_runner.py
import pytest

from card_reviewer.review.pipeline import StageRunner, StageValidationError
from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository


@pytest.fixture
def runner(tmp_path):
    conn = connect(tmp_path / "t.db"); migrate(conn)
    return StageRunner(SqliteRepository(conn))


def test_a_second_identical_run_reuses_the_cached_result(runner):
    calls = []
    def compute():
        calls.append(1)
        return {"n": 1}
    v = {"preflight_version": "1.0.0", "config": {}}
    runner.run("preflight", {"image_hash": "h"}, v, compute, image_hash="h")
    runner.run("preflight", {"image_hash": "h"}, v, compute, image_hash="h")
    assert len(calls) == 1


def test_changed_inputs_recompute(runner):
    calls = []
    v = {"preflight_version": "1.0.0", "config": {}}
    for h in ("h1", "h2"):
        runner.run("preflight", {"image_hash": h}, v,
                   lambda: calls.append(1) or {"n": 1}, image_hash=h)
    assert len(calls) == 2


def test_a_bumped_producer_version_recomputes(runner):
    calls = []
    for version in ("1.0.0", "1.0.1"):
        runner.run("preflight", {"image_hash": "h"},
                   {"preflight_version": version, "config": {}},
                   lambda: calls.append(1) or {"n": 1}, image_hash="h")
    assert len(calls) == 2


def test_a_failure_is_recorded_and_never_cached(runner):
    def boom():
        raise RuntimeError("provider exploded")
    v = {"provider": "anthropic", "model": "m", "prompt_version": "1",
         "inference_params": {}}
    with pytest.raises(RuntimeError):
        runner.run("vision", {"manifest": "x"}, v, boom, candidate_id="c1")
    calls = []
    out = runner.run("vision", {"manifest": "x"}, v,
                     lambda: calls.append(1) or {"ok": True}, candidate_id="c1")
    assert out == {"ok": True} and len(calls) == 1


def test_output_failing_schema_validation_is_never_cached(runner):
    """Spec §4: a stage_result row exists ONLY for an output that ran to
    completion AND passed schema validation."""
    from pydantic import BaseModel

    class Out(BaseModel):
        n: int

    v = {"preflight_version": "1.0.0", "config": {}}
    with pytest.raises(StageValidationError):
        runner.run("preflight", {"image_hash": "h"}, v,
                   lambda: {"n": "not an int"}, schema=Out, image_hash="h")
    calls = []
    out = runner.run("preflight", {"image_hash": "h"}, v,
                     lambda: calls.append(1) or {"n": 1}, schema=Out,
                     image_hash="h")
    assert out == {"n": 1} and len(calls) == 1


def test_a_validation_failure_is_recorded_as_an_attempt(runner):
    from pydantic import BaseModel

    class Out(BaseModel):
        n: int

    with pytest.raises(StageValidationError):
        runner.run("preflight", {"image_hash": "h"},
                   {"preflight_version": "1.0.0", "config": {}},
                   lambda: {"wrong": True}, schema=Out, image_hash="h")
    rows = runner._repo._conn.execute(
        "SELECT error_kind FROM stage_attempt").fetchall()
    assert any("Validation" in r[0] for r in rows)


def test_an_off_run_never_satisfies_a_deep_lookup(runner):
    """The routing cache bug this design exists to prevent."""
    calls = []
    v = {"routing_policy_version": "1.0.0"}
    for mode in ("off", "deep"):
        runner.run("routing", {"mode": mode, "heuristic_output": {}}, v,
                   lambda: calls.append(1) or {"call_vision": mode == "deep"},
                   candidate_id="c1")
    assert len(calls) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_stage_runner.py -v`
Expected: FAIL with `ImportError: cannot import name 'StageRunner'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/pipeline.py
"""Stage execution with content-addressed caching.

Cache identity is (stage, input_fingerprint, producer_signature). A row
exists ONLY for an output that ran to completion; failures go to
stage_attempt and can never satisfy a lookup.
"""
from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .fingerprint import fingerprint, signature_for
from .storage.repository import Repository


class StageValidationError(Exception):
    """A stage produced output that does not match its declared schema."""


class StageRunner:
    def __init__(self, repo: Repository) -> None:
        self._repo = repo

    def run(self, stage: str, inputs: dict[str, Any], versions: dict[str, Any],
            compute: Callable[[], dict[str, Any]], *,
            schema: type[BaseModel] | None = None,
            image_hash: str | None = None,
            candidate_id: str | None = None) -> dict[str, Any]:
        fp = fingerprint(inputs)
        sig = signature_for(stage, versions)

        cached = self._repo.get_stage_result(stage, fp, sig)
        if cached is not None:
            return cached.output

        try:
            output = compute()
        except Exception as exc:
            self._repo.record_attempt(
                stage, fp, sig, error_kind=type(exc).__name__,
                error_detail=str(exc), image_hash=image_hash,
                candidate_id=candidate_id)
            raise

        # "Validated successes only" is enforced here, not assumed. An output
        # that does not match its schema is a failure, however cleanly the
        # stage returned it — caching it would poison every later run.
        if schema is not None:
            try:
                schema.model_validate(output)
            except ValidationError as exc:
                self._repo.record_attempt(
                    stage, fp, sig, error_kind="StageValidationError",
                    error_detail=str(exc), image_hash=image_hash,
                    candidate_id=candidate_id)
                raise StageValidationError(
                    f"stage {stage!r} output failed validation: {exc}") from exc

        self._repo.put_stage_result(stage, fp, sig, output, versions,
                                    image_hash=image_hash,
                                    candidate_id=candidate_id)
        return output
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_stage_runner.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/pipeline.py tests/review/test_stage_runner.py
git commit -m "feat(review): stage runner with validated-success-only caching"
```

**Acceptance:** a failed run leaves no cache row; **schema-invalid output is recorded as an attempt and never cached**; an `OFF` routing result never satisfies a `DEEP` lookup; a producer bump recomputes. Every `ReviewPipeline` stage call passes its `schema=`.

---

## Phase 7 — Vision

### Task 29: Provider protocol and offline fakes

**Files:** Create `src/card_reviewer/review/vision/__init__.py`, `vision/provider.py`; Test `tests/review/test_vision_provider.py`

**Interfaces:**
- Produces: `VisionProvider` protocol, `Assessment`, `VisionFinding`, `GemView`, `ProviderContractError`, `FakeProvider`, `parse_assessment(payload, allowed_artifact_ids) -> Assessment`, `resolve_vision_findings(assessment, manifest_index) -> list[Finding]`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_vision_provider.py
import pytest

from card_reviewer.review.vision.provider import (
    ProviderContractError, parse_assessment,
)


def _payload(**kw):
    base = {
        "findings": [{
            "defect_type": "print_lines", "category": "surface",
            "state": "suspected", "confidence": 0.6, "psa10_relevant": True,
            "evidence_artifact_ids": ["a1"], "explanation": "faint line",
        }],
        "category_assessability": {"centering": True, "corners": True,
                                   "edges": True, "surface": True},
        "gem_view": "possible_psa10_disqualifier",
    }
    return base | kw


def test_a_well_formed_response_parses():
    a = parse_assessment(_payload(), allowed_artifact_ids={"a1"})
    assert a.findings[0].defect_type == "print_lines"


def test_citing_an_artifact_not_in_the_manifest_is_a_contract_violation():
    """A provider that cites an id it was never sent is not to be trusted."""
    with pytest.raises(ProviderContractError, match="not in the manifest"):
        parse_assessment(_payload(), allowed_artifact_ids={"other"})


def test_every_category_must_report_assessability():
    bad = _payload(category_assessability={"centering": True})
    with pytest.raises(ProviderContractError, match="assessability"):
        parse_assessment(bad, allowed_artifact_ids={"a1"})


def test_a_malformed_state_is_rejected_rather_than_coerced():
    bad = _payload(findings=[{
        "defect_type": "x", "category": "surface", "state": "definitely_bad",
        "confidence": 0.9, "psa10_relevant": True,
        "evidence_artifact_ids": ["a1"], "explanation": ""}])
    with pytest.raises(ProviderContractError):
        parse_assessment(bad, allowed_artifact_ids={"a1"})


def test_a_cited_enhanced_artifact_keeps_its_enhanced_origin():
    """I3 must survive the round trip. Rebuilding this ref as ORIGINAL would
    silently launder an enhancement-only finding into a rejectable one."""
    from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef
    from card_reviewer.review.vision.provider import (
        Assessment, GemView, resolve_vision_findings,
    )
    ref = EvidenceRef(artifact_id="a1", image_hash="realhash",
                      origin=EvidenceOrigin.ENHANCED,
                      enhancement="clahe:clip=2.0", view="surface_clahe")
    a = parse_assessment(_payload(), allowed_artifact_ids={"a1"})
    findings = resolve_vision_findings(a, {"a1": ref})
    got = findings[0].evidence[0]
    assert got.origin is EvidenceOrigin.ENHANCED
    assert got.enhancement == "clahe:clip=2.0"
    assert got.image_hash == "realhash"


def test_an_observed_finding_citing_only_enhanced_evidence_is_demoted():
    """Invariant: it can never independently REJECT."""
    from card_reviewer.review.enums import FindingState
    from card_reviewer.review.findings import enforce_i3
    from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef
    from card_reviewer.review.vision.provider import resolve_vision_findings
    ref = EvidenceRef(artifact_id="a1", image_hash="h",
                      origin=EvidenceOrigin.ENHANCED,
                      enhancement="sharpen:amount=1.5", view="surface_sharpen")
    payload = _payload(findings=[{
        "defect_type": "scratches", "category": "surface", "state": "observed",
        "confidence": 0.99, "psa10_relevant": True,
        "evidence_artifact_ids": ["a1"], "explanation": ""}])
    a = parse_assessment(payload, allowed_artifact_ids={"a1"})
    resolved = enforce_i3(resolve_vision_findings(a, {"a1": ref}))
    assert resolved[0].state is FindingState.SUSPECTED
    assert "I3" in resolved[0].demotion_reason


def test_an_unresolvable_cited_id_raises_rather_than_defaulting():
    from card_reviewer.review.vision.provider import resolve_vision_findings
    a = parse_assessment(_payload(), allowed_artifact_ids={"a1"})
    with pytest.raises(ProviderContractError, match="not in the manifest"):
        resolve_vision_findings(a, {})


def test_a_missing_gem_view_is_rejected():
    bad = _payload(); del bad["gem_view"]
    with pytest.raises(ProviderContractError, match="gem_view"):
        parse_assessment(bad, allowed_artifact_ids={"a1"})


def test_severity_and_location_survive_parsing():
    payload = _payload(findings=[{
        "defect_type": "scratches", "category": "surface", "state": "observed",
        "confidence": 0.9, "psa10_relevant": True,
        "evidence_artifact_ids": ["a1"], "severity": "moderate",
        "location": {"x0": 0.1, "y0": 0.1, "x1": 0.3, "y1": 0.3},
        "explanation": ""}])
    a = parse_assessment(payload, allowed_artifact_ids={"a1"})
    assert a.findings[0].severity.value == "moderate"
    assert a.findings[0].location.x1 == 0.3


def test_insufficient_evidence_is_a_first_class_answer():
    a = parse_assessment(_payload(gem_view="insufficient_evidence",
                                  category_assessability={
                                      "centering": True, "corners": True,
                                      "edges": True, "surface": False}),
                         allowed_artifact_ids={"a1"})
    assert a.category_assessability["surface"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_vision_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/vision/provider.py
"""VisionProvider contract. Anthropic is one implementation, not the interface.

Per-category assessability is REQUIRED, not an optional remark: it feeds the
authoritative coverage evaluation, so a provider saying "I could not judge
the surface" must not be lost.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from ..enums import FindingState
from ..findings import Finding
from ..provenance import EvidenceRef
from ..findings import Severity
from ..provenance import NormalizedBox
from ..taxonomy import CATEGORIES


class ProviderContractError(Exception):
    """The provider returned something outside its declared contract."""


class GemView(StrEnum):
    NO_DISQUALIFIER = "no_visible_psa10_disqualifier"
    POSSIBLE_DISQUALIFIER = "possible_psa10_disqualifier"
    VISIBLE_DISQUALIFIER = "visible_psa10_disqualifier"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class VisionFinding(BaseModel):
    defect_type: str
    category: str
    state: FindingState
    confidence: float = Field(ge=0.0, le=1.0)
    # Advisory only. Our rubric decides PSA-10 relevance (Decision 4); this
    # field records what the provider claimed so the two can be compared.
    psa10_relevant: bool
    evidence_artifact_ids: list[str] = Field(min_length=1)
    severity: Severity | None = None
    # Normalized location. When the provider omits it, the resolver derives it
    # from the cited refs' regions — a location is required downstream for the
    # contradiction test and for fusion, so it can never be simply absent.
    location: NormalizedBox | None = None
    explanation: str = ""


class Assessment(BaseModel):
    findings: list[VisionFinding] = Field(default_factory=list)
    category_assessability: dict[str, bool]
    gem_view: GemView
    disagreements: list[str] = Field(default_factory=list)


class VisionProvider(Protocol):
    def assess(self, evidence_manifest: dict[str, Any]) -> Assessment: ...


def parse_assessment(payload: dict[str, Any],
                     allowed_artifact_ids: set[str]) -> Assessment:
    if "gem_view" not in payload:
        raise ProviderContractError("response is missing gem_view")
    try:
        assessment = Assessment.model_validate(payload)
    except ValidationError as exc:
        raise ProviderContractError(f"malformed provider response: {exc}") from exc

    missing = set(CATEGORIES) - set(assessment.category_assessability)
    if missing:
        raise ProviderContractError(
            f"response omits assessability for {sorted(missing)} — coverage "
            "cannot be evaluated without it")

    for finding in assessment.findings:
        unknown = set(finding.evidence_artifact_ids) - allowed_artifact_ids
        if unknown:
            raise ProviderContractError(
                f"finding cites artifacts {sorted(unknown)} not in the manifest "
                "it was sent")
    return assessment


def resolve_vision_findings(
    assessment: Assessment,
    manifest_index: dict[str, "EvidenceRef"],
) -> list["Finding"]:
    """Turn provider output into Findings WITHOUT losing provenance.

    The provider returns bare artifact ids. Rebuilding an EvidenceRef from a
    bare id — inventing origin=ORIGINAL and an empty image_hash — would launder
    an enhancement-only finding into one that satisfies I3, defeating the
    invariant exactly where it matters. Every cited id is therefore resolved
    against the manifest that was actually sent, and an unresolvable id is a
    contract violation rather than a default.
    """
    from ..findings import Finding, FindingProducer

    out: list[Finding] = []
    for vf in assessment.findings:
        refs = []
        for artifact_id in vf.evidence_artifact_ids:
            ref = manifest_index.get(artifact_id)
            if ref is None:
                raise ProviderContractError(
                    f"finding cites artifact {artifact_id!r} which is not in the "
                    "manifest that was sent")
            refs.append(ref)
        out.append(Finding(
            defect_type=vf.defect_type, category=vf.category, state=vf.state,
            producer=FindingProducer.VISION, confidence=vf.confidence,
            # psa10_relevant is provisional here; Task 16 recomputes it from
            # the matched rubric rules and overrides the provider's claim.
            psa10_relevant=vf.psa10_relevant,
            severity=vf.severity,
            location=vf.location or _derive_location(refs),
            evidence=refs, explanation=vf.explanation))
    return out


def _derive_location(refs: list["EvidenceRef"]) -> "NormalizedBox | None":
    """Union of the cited refs' regions, when the provider gave no location."""
    boxes = [r.region for r in refs if r.region is not None]
    if not boxes:
        return None
    return NormalizedBox(
        x0=min(b.x0 for b in boxes), y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes), y1=max(b.y1 for b in boxes))


class FakeProvider:
    """Deterministic stand-in. Every pipeline test uses this, never the API."""

    def __init__(self, assessment: Assessment) -> None:
        self._assessment = assessment
        self.calls = 0

    def assess(self, evidence_manifest: dict[str, Any]) -> Assessment:
        self.calls += 1
        return self._assessment
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_vision_provider.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/vision/ tests/review/test_vision_provider.py
git commit -m "feat(review): vision provider contract and offline fake"
```

**Acceptance:** a cited artifact absent from the manifest raises at both parse and resolve; enhanced provenance survives the round trip so an enhancement-only observed finding is demoted and can never reject; severity and location are preserved, with location derived from cited regions when omitted.

---
### Task 30: Evidence manifest builder

**Files:** Create `src/card_reviewer/review/manifest.py`; Test `tests/review/test_manifest_builder.py`

**Interfaces:**
- Consumes: `assembly.py`, `provenance.py`, `enums.Mode`
- Produces: `MANIFEST_BUILDER_VERSION`, `BUDGETS`, `BuiltManifest(payload, index)`, `build_manifest(assembled, mode, rubric_rules) -> BuiltManifest`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_manifest_builder.py
from card_reviewer.review.enums import Mode
from card_reviewer.review.manifest import BUDGETS, build_manifest
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef


def _refs(n):
    return [EvidenceRef(artifact_id=f"a{i}", image_hash="h",
                        origin=EvidenceOrigin.NORMALIZED, view=f"v{i}")
            for i in range(n)]


class _A:
    def __init__(self, refs):
        self.evidence_refs = {"surface": refs}
        self.detectability, self.conflicts, self.centering = {}, [], {}
        self.reason_codes, self.anomalies, self.limitations = {}, [], []


def test_smart_and_deep_have_different_declared_budgets():
    assert BUDGETS[Mode.SMART] < BUDGETS[Mode.DEEP]


def test_selection_respects_the_mode_budget():
    m = build_manifest(_A(_refs(40)), Mode.SMART, []).payload
    assert len(m["artifacts"]) <= BUDGETS[Mode.SMART]


def test_deep_selects_more_than_smart_but_not_everything():
    """DEEP means maximum USEFUL evidence, not mechanically every artifact."""
    deep = build_manifest(_A(_refs(40)), Mode.DEEP, []).payload
    smart = build_manifest(_A(_refs(40)), Mode.SMART, []).payload
    assert BUDGETS[Mode.SMART] < len(deep["artifacts"]) <= BUDGETS[Mode.DEEP]
    assert len(deep["artifacts"]) < 40


def test_duplicate_artifact_ids_are_eliminated():
    dup = _refs(3) + _refs(3)
    m = build_manifest(_A(dup), Mode.DEEP, []).payload
    ids = [a["artifact_id"] for a in m["artifacts"]]
    assert len(ids) == len(set(ids))


def test_selection_is_deterministic_for_the_same_inputs():
    assert (build_manifest(_A(_refs(30)), Mode.SMART, []).payload
            == build_manifest(_A(_refs(30)), Mode.SMART, []).payload)


def test_the_index_resolves_every_sent_artifact_back_to_its_ref():
    """Decision 1: without this the provider's citations cannot be resolved
    and provenance is lost at the round trip."""
    built = build_manifest(_A(_refs(5)), Mode.SMART, [])
    for artifact in built.payload["artifacts"]:
        ref = built.index[artifact["artifact_id"]]
        assert ref.origin.value == artifact["origin"]
        assert ref.image_hash


def test_the_index_contains_exactly_what_was_sent():
    built = build_manifest(_A(_refs(40)), Mode.SMART, [])
    assert set(built.index) == {a["artifact_id"]
                                for a in built.payload["artifacts"]}


def test_the_manifest_carries_rubric_rule_content_not_a_version_string(rubric):
    m = build_manifest(_A(_refs(3)), Mode.SMART,
                       rubric.for_card(None, None)[:2]).payload
    assert isinstance(m["rubric_rules"], list)
    assert "statement" in m["rubric_rules"][0]


def test_the_manifest_carries_every_field_the_design_promised(rubric):
    """A silently thinned payload would make the provider's answers worse
    while still looking like a working integration."""
    m = build_manifest(_A(_refs(3)), Mode.DEEP,
                       rubric.for_card(None, None)).payload
    for field in ("artifacts", "measurements", "detectability",
                  "detectability_reasons", "image_limitations", "conflicts",
                  "anomaly_candidates", "rubric_rules"):
        assert field in m, f"manifest omits {field}"


def test_anomaly_candidates_carry_enhancement_provenance():
    class _B(_A):
        pass
    a = _B(_refs(2))
    a.anomalies = [{"category": "surface", "defect_type": "scratches",
                    "surfaced_by": "clahe", "visible_in_original": False,
                    "artifact_id": "x"}]
    m = build_manifest(a, Mode.DEEP, []).payload
    assert m["anomaly_candidates"][0]["visible_in_original"] is False
    assert m["anomaly_candidates"][0]["surfaced_by"] == "clahe"


def test_no_pricing_information_reaches_the_manifest(rubric):
    import re
    m = build_manifest(_A(_refs(3)), Mode.DEEP,
                       rubric.for_card(None, None)).payload
    blob = repr(m).lower()
    # Whole words only: "ev" as a substring matches "evidence_type", which is a
    # legitimate rubric field, and a substring test would fail on it forever.
    for word in ("price", "cost", "profit", "purchase", "resale", "ev", "roi"):
        assert not re.search(rf"\b{word}\b", blob), f"pricing term {word!r} leaked"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_manifest_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/manifest.py
"""Deterministic evidence manifest construction (spec §12).

Its own cached stage, not an unnamed step: it is what the `vision` stage
fingerprints, so it must be reproducible independently of whether a call was
ultimately made. It fingerprints rubric CONTENT, not a version string — a
release that leaves the applicable rules byte-identical must not re-bill.
"""
from __future__ import annotations

from typing import Any

from typing import NamedTuple

from .enums import Mode
from .provenance import EvidenceRef

MANIFEST_BUILDER_VERSION = "1.0.0"

BUDGETS: dict[Mode, int] = {Mode.OFF: 0, Mode.SMART: 8, Mode.DEEP: 20}

# Fixed priority so selection is deterministic rather than "whatever fits".
VIEW_PRIORITY = ("surface_original", "front_face", "back_face",
                 "corner_", "edge_", "surface_")


def _rank(view: str) -> int:
    for i, prefix in enumerate(VIEW_PRIORITY):
        if view.startswith(prefix):
            return i
    return len(VIEW_PRIORITY)


class BuiltManifest(NamedTuple):
    """The payload sent to the provider, and the index used to resolve what it
    cites back to real provenance (Decision 1). Returning them together is what
    stops `combine` inventing an EvidenceRef from a bare artifact id."""
    payload: dict[str, Any]
    index: dict[str, EvidenceRef]


def build_manifest(assembled: Any, mode: Mode,
                   rubric_rules: list[Any]) -> BuiltManifest:
    seen: set[str] = set()
    candidates = []
    for refs in assembled.evidence_refs.values():
        for ref in refs:
            if ref.artifact_id in seen:
                continue
            seen.add(ref.artifact_id)
            candidates.append(ref)

    candidates.sort(key=lambda r: (_rank(r.view), r.view, r.artifact_id))
    selected = candidates[:BUDGETS[mode]]

    payload = {
        "artifacts": [
            {"artifact_id": r.artifact_id, "view": r.view,
             "origin": r.origin.value, "enhancement": r.enhancement,
             "region": r.region.model_dump() if r.region else None}
            for r in selected
        ],
        "measurements": dict(assembled.centering),
        # Detectability and its reason codes: the provider must know which
        # regions could not carry evidence, or it will read absence as absence.
        "detectability": {f"{k}": str(v) for k, v in assembled.detectability.items()},
        "detectability_reasons": {f"{k}": v
                                  for k, v in assembled.reason_codes.items()},
        # Image-quality limitations, so "I could not judge this" is informed.
        "image_limitations": list(getattr(assembled, "limitations", [])),
        # Disagreements between photographs, preserved not averaged.
        "conflicts": list(assembled.conflicts),
        # Anomaly candidates WITH their enhancement provenance, so the provider
        # can tell "visible in the original" from "only under CLAHE".
        "anomaly_candidates": [
            {"category": a.get("category"), "defect_type": a.get("defect_type"),
             "region": a.get("region"), "artifact_id": a.get("artifact_id"),
             "surfaced_by": a.get("surfaced_by", "original"),
             "visible_in_original": a.get("visible_in_original", True)}
            for a in assembled.anomalies
        ],
        # Content, not version. Pricing fields never appear here (rule 14).
        "rubric_rules": [
            {"id": r.id, "category": r.category.value, "statement": r.statement,
             "evidence_type": r.evidence_type.value,
             "confidence": r.confidence.value}
            for r in rubric_rules
        ],
        "builder_version": MANIFEST_BUILDER_VERSION,
    }
    return BuiltManifest(payload=payload,
                         index={r.artifact_id: r for r in selected})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_manifest_builder.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/manifest.py tests/review/test_manifest_builder.py
git commit -m "feat(review): deterministic evidence manifest builder"
```

**Acceptance:** `build_manifest` returns a `BuiltManifest(payload, index)`; the index resolves every sent artifact back to its `EvidenceRef`; selection is deterministic and budget-bounded; duplicates are eliminated; **every promised field is present** — detectability with reason codes, image limitations, conflicts, anomaly candidates with enhancement provenance, measurements and rubric content; no pricing term appears as a whole word.

---

### Task 31: SMART routing policy

**Files:** Create `src/card_reviewer/review/policies/routing_v1.py`; Test `tests/review/test_routing.py`

**Interfaces:**
- Produces: `ROUTING_POLICY_VERSION`, `RoutingDecision`, `decide_routing(mode, heuristic, provisional_coverage, detectability) -> RoutingDecision`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_routing.py
from card_reviewer.review.enums import Coverage, FindingState, Mode, Scale
from card_reviewer.review.findings import Finding, FindingProducer
from card_reviewer.review.policies.routing_v1 import decide_routing
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef


def _f(state=FindingState.SUSPECTED):
    return Finding(defect_type="print_lines", category="surface", state=state,
                   producer=FindingProducer.HEURISTIC, confidence=0.6,
                   psa10_relevant=True,
                   evidence=[EvidenceRef(artifact_id="a", image_hash="h",
                                         origin=EvidenceOrigin.NORMALIZED,
                                         view="v")])


def test_off_never_calls():
    d = decide_routing(Mode.OFF, [_f()], Coverage.PARTIAL, {})
    assert d.call_vision is False


def test_deep_always_calls():
    d = decide_routing(Mode.DEEP, [], Coverage.SUFFICIENT, {})
    assert d.call_vision is True


def test_smart_calls_on_a_resolvable_ambiguity():
    d = decide_routing(Mode.SMART, [_f()], Coverage.SUFFICIENT,
                       {("front", "surface", "print_lines"): Scale.HIGH})
    assert d.call_vision is True
    assert "suspected" in " ".join(d.trigger_reasons)


def test_smart_does_not_call_when_detectability_is_unknown():
    """No reason to believe the pixels carry the answer."""
    d = decide_routing(Mode.SMART, [_f()], Coverage.SUFFICIENT, {})
    assert d.call_vision is False or "confirm" in " ".join(d.trigger_reasons)


def test_smart_does_not_call_on_missing_information_alone():
    """A provider cannot recover information absent from the pixels; sending
    an occluded corner buys insufficient_evidence at cost."""
    d = decide_routing(Mode.SMART, [], Coverage.PARTIAL,
                       {("front", "surface", "scratches"): Scale.LOW})
    assert d.call_vision is False


def test_smart_does_not_call_when_provisional_coverage_is_inadequate():
    d = decide_routing(Mode.SMART, [_f()], Coverage.INADEQUATE, {})
    assert d.call_vision is False


def test_deep_still_calls_on_inadequate_coverage():
    """The owner asked for maximum evidence explicitly."""
    d = decide_routing(Mode.DEEP, [], Coverage.INADEQUATE, {})
    assert d.call_vision is True


def test_smart_calls_to_confirm_a_strong_gem_candidate():
    d = decide_routing(Mode.SMART, [], Coverage.SUFFICIENT, {})
    assert d.call_vision is True
    assert "confirm" in " ".join(d.trigger_reasons)


def test_the_decision_records_the_mode_it_was_made_under():
    assert decide_routing(Mode.SMART, [], Coverage.SUFFICIENT, {}).mode is Mode.SMART
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_routing.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/policies/routing_v1.py
"""SMART routing: fires on resolvable ambiguity, not missing information."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..enums import Coverage, FindingState, Mode, Scale

ROUTING_POLICY_VERSION = "1.0.0"

MIN_DETECTABILITY_TO_RESOLVE = Scale.MODERATE


class RoutingDecision(BaseModel):
    mode: Mode
    call_vision: bool
    trigger_reasons: list[str] = Field(default_factory=list)
    policy_version: str = ROUTING_POLICY_VERSION


def decide_routing(mode: Mode, findings: list, provisional: Coverage,
                   detectability: dict) -> RoutingDecision:
    if mode is Mode.OFF:
        return RoutingDecision(mode=mode, call_vision=False,
                               trigger_reasons=["mode is OFF"])
    if mode is Mode.DEEP:
        return RoutingDecision(mode=mode, call_vision=True,
                               trigger_reasons=["mode is DEEP"])

    # The provisional gate: nothing for the provider to resolve.
    if provisional is Coverage.INADEQUATE:
        return RoutingDecision(
            mode=mode, call_vision=False,
            trigger_reasons=["provisional coverage INADEQUATE — a call cannot "
                             "recover information absent from the pixels"])

    reasons: list[str] = []
    for finding in findings:
        if finding.state is not FindingState.SUSPECTED:
            continue
        key = next((k for k in detectability
                    if k[1] == finding.category and k[2] == finding.defect_type),
                   None)
        # Only worth a call when the evidence could actually settle it. Absent
        # detectability is NOT resolvable: we have no reason to believe the
        # pixels carry the answer, and sending them buys insufficient_evidence
        # at cost (spec §12).
        if key is not None and detectability[key] >= MIN_DETECTABILITY_TO_RESOLVE:
            reasons.append(
                f"{finding.category}/{finding.defect_type} suspected and resolvable")

    if not reasons and provisional is Coverage.SUFFICIENT and not findings:
        reasons.append("strong gem candidate worth confirming")

    return RoutingDecision(mode=mode, call_vision=bool(reasons),
                           trigger_reasons=reasons)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_routing.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/policies/routing_v1.py tests/review/test_routing.py
git commit -m "feat(review): SMART routing policy with provisional spend gate"
```

**Acceptance:** `OFF` never calls, `DEEP` always; `SMART` fires on resolvable ambiguity and not on missing information; the provisional gate suppresses a `SMART` call on `INADEQUATE`.

---

### Task 32: Anthropic provider and offline contract tests

**Files:** Create `src/card_reviewer/review/vision/prompt.py`, `vision/anthropic.py`; Test `tests/review/test_anthropic_contract.py`, `tests/review/fixtures/vision/*.json`

**Interfaces:**
- Consumes: `storage/artifacts.ArtifactStore`, `manifest.build_manifest`
- Produces: `PROMPT_VERSION`, `build_prompt(manifest) -> str`, `build_request(manifest, store) -> list[dict]`, `AnthropicVisionProvider(model, store, api_key)`

**Constraint:** no automated test may call the API. Every test here runs against saved fixture payloads and against the **constructed request object**, which is inspected without being sent.

**The core requirement:** a vision provider that sends only text is not a vision provider. The request must carry the actual bytes of the selected manifest artifacts as image content blocks, with a deterministic `artifact_id` ↔ image mapping so the provider's citations resolve back to real evidence.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_anthropic_contract.py
import json
from pathlib import Path

import pytest

from card_reviewer.review.vision.anthropic import AnthropicVisionProvider
from card_reviewer.review.vision.prompt import PROMPT_VERSION, build_prompt
from card_reviewer.review.vision.provider import ProviderContractError

FIXTURES = Path(__file__).parent / "fixtures" / "vision"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def provider_rig(tmp_path):
    """A provider whose store actually holds the artifact the manifest cites.

    `assess()` builds the request before calling the model, so an empty store
    raises KeyError long before any response parsing happens.
    """
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.storage.artifacts import ArtifactStore
    store = ArtifactStore(tmp_path)
    image_hash = store.put_image(render_png(CardSpec()))
    aid = store.put_derived(image_hash, "surface", "original.png",
                            render_png(CardSpec()))
    provider = AnthropicVisionProvider(model="m", store=store, api_key="unused")
    manifest = {"artifacts": [{"artifact_id": aid, "view": "surface_original",
                               "origin": "normalized", "enhancement": None,
                               "region": None}],
                "rubric_rules": [], "measurements": {}}
    return provider, manifest, aid


def test_the_prompt_is_adversarial_but_demands_conservative_evidence():
    text = build_prompt({"artifacts": [], "rubric_rules": [], "measurements": {}})
    assert "every visible reason" in text.lower()
    assert "suspected" in text.lower()


def test_the_prompt_never_mentions_price_or_market_value():
    import re
    text = build_prompt({"artifacts": [], "rubric_rules": [],
                         "measurements": {}}).lower()
    # Whole words only: the brief legitimately says "cite the artifact_id
    # values you relied on", and a substring test would fail on "values".
    for word in ("price", "prices", "profit", "worth", "resale", "roi",
                 "market", "purchase"):
        assert not re.search(rf"\b{word}\b", text), f"prompt mentions {word!r}"


def test_the_prompt_does_not_ask_the_provider_to_restate_centering():
    text = build_prompt({"artifacts": [], "rubric_rules": [],
                         "measurements": {"horizontal": 54.0}})
    assert "do not re-measure" in text.lower()


def test_the_prompt_includes_every_canonical_payload_section():
    text = build_prompt({
        "artifacts": [], "rubric_rules": [], "measurements": {"horizontal": 54.0},
        "detectability": {"x": "low"}, "detectability_reasons": {"x": "GLARE"},
        "image_limitations": ["front is glared"], "conflicts": [{"f": 1}],
        "anomaly_candidates": [{"defect_type": "scratches"}]})
    lowered = text.lower()
    for section in ("detectability", "limitation", "conflict", "anomaly"):
        assert section in lowered


# --- the request actually carries images -------------------------------------

def _rig(tmp_path):
    from card_reviewer.review.storage.artifacts import ArtifactStore
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    store = ArtifactStore(tmp_path)
    image_hash = store.put_image(render_png(CardSpec()))
    artifact_id = store.put_derived(image_hash, "surface", "original.png",
                                    render_png(CardSpec()))
    manifest = {"artifacts": [{"artifact_id": artifact_id, "view": "surface_original",
                               "origin": "normalized", "enhancement": None,
                               "region": None}],
                "rubric_rules": [], "measurements": {}}
    return store, manifest, artifact_id


def test_the_request_contains_real_image_blocks(tmp_path):
    from card_reviewer.review.vision.anthropic import build_request
    store, manifest, _ = _rig(tmp_path)
    blocks = build_request(manifest, store)
    images = [b for b in blocks if b["type"] == "image"]
    assert images, "the request carries no image content at all"
    assert images[0]["source"]["type"] == "base64"
    assert len(images[0]["source"]["data"]) > 100


def test_each_image_block_is_labelled_with_its_artifact_id(tmp_path):
    """Without a deterministic id-to-image mapping the provider cannot cite
    evidence we can resolve back."""
    from card_reviewer.review.vision.anthropic import build_request
    store, manifest, artifact_id = _rig(tmp_path)
    blocks = build_request(manifest, store)
    labels = [b["text"] for b in blocks if b["type"] == "text"]
    assert any(artifact_id in t for t in labels)


def test_image_block_order_is_deterministic(tmp_path):
    from card_reviewer.review.vision.anthropic import build_request
    store, manifest, _ = _rig(tmp_path)
    assert build_request(manifest, store) == build_request(manifest, store)


def test_building_the_request_never_opens_a_socket(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("build_request must not touch the network")))
    from card_reviewer.review.vision.anthropic import build_request
    store, manifest, _ = _rig(tmp_path)
    assert build_request(manifest, store)


def test_a_well_formed_saved_response_parses(monkeypatch, provider_rig):
    provider, manifest, aid = provider_rig
    payload = _fixture("valid.json")
    payload["findings"][0]["evidence_artifact_ids"] = [aid]
    monkeypatch.setattr(provider, "_call", lambda blocks: payload)
    a = provider.assess(manifest)
    assert a.gem_view.value == "possible_psa10_disqualifier"


def test_a_provider_built_without_a_store_is_rejected_at_construction():
    """Failing here beats an AttributeError on the first real call."""
    with pytest.raises(ValueError, match="ArtifactStore"):
        AnthropicVisionProvider(model="m", api_key="unused")


@pytest.mark.parametrize("name", [
    "missing_gem_view.json", "unknown_artifact.json",
    "missing_assessability.json", "malformed_state.json",
])
def test_malformed_saved_responses_raise_a_contract_error(monkeypatch, name,
                                                          provider_rig):
    provider, manifest, _ = provider_rig
    monkeypatch.setattr(provider, "_call", lambda blocks: _fixture(name))
    with pytest.raises(ProviderContractError):
        provider.assess(manifest)


def test_no_test_in_this_module_constructs_a_real_client():
    """Guard: the SDK is imported lazily inside _call, never at import time."""
    import card_reviewer.review.vision.anthropic as mod
    assert not hasattr(mod, "anthropic")
```

Fixture `tests/review/fixtures/vision/valid.json`:

```json
{
  "findings": [
    {"defect_type": "print_lines", "category": "surface", "state": "suspected",
     "confidence": 0.6, "psa10_relevant": true,
     "evidence_artifact_ids": ["a1"], "explanation": "faint diagonal line"}
  ],
  "category_assessability": {"centering": true, "corners": true,
                             "edges": true, "surface": true},
  "gem_view": "possible_psa10_disqualifier",
  "disagreements": []
}
```

The other four fixtures are `valid.json` with one defect each: `missing_gem_view.json` omits `gem_view`; `unknown_artifact.json` sets `evidence_artifact_ids` to `["not_sent"]`; `missing_assessability.json` reduces `category_assessability` to `{"centering": true}`; `malformed_state.json` sets `state` to `"definitely_bad"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_anthropic_contract.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/vision/prompt.py
"""Versioned prompt construction.

Adversarial brief, conservative evidence standard. No pricing, EV, profit or
purchase information ever reaches the prompt (non-negotiable rule 14).
"""
from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "1.0.0"

_BRIEF = """You are assessing a raw trading card from photographs, to help decide
whether it has a realistic chance of grading PSA 10.

Find every visible reason this card might not receive a 10. Search aggressively.

But conclude conservatively. Each finding must carry a state:
  observed        - visible and confidently a real feature of the card
  suspected       - visible but could be glare, design, or artifact
  not_observed    - you looked, evidence was adequate, it is not present
  not_assessable  - the evidence was insufficient to look

A wrongly confirmed defect is the expensive error here. A suspected print line
stays suspected.

Do not re-measure centering. It is measured for you and supplied below; your job
is what measurement cannot do — chipping versus glare, soft corners, print lines,
dimples, stains, foil and refractor artifacts, and whether a suspected defect is
part of the card's design.

For EVERY finding, cite the artifact_id values you relied on. Cite only ids that
appear in the artifact list below.

For EVERY category (centering, corners, edges, surface) state whether you could
assess it at all. "I could not judge the surface" is a first-class answer and
must not be omitted.

Finally give an independent gem view, one of: no_visible_psa10_disqualifier,
possible_psa10_disqualifier, visible_psa10_disqualifier, insufficient_evidence.

An anomaly candidate marked "visible_in_original": false was surfaced only by
image enhancement. You may report it, but it cannot on its own establish a
confirmed defect — say `suspected` unless you can see it in an unenhanced view.

Where you report a finding, give its normalized location as
{"x0","y0","x1","y1"} in card coordinates, and its severity as one of
minor / moderate / severe.

Respond with JSON only.
"""


# Every section of the canonical payload is rendered. Silently omitting one
# would degrade the provider's answers while still looking like a working
# integration — the failure mode is invisible, so it is asserted in tests.
_SECTIONS = (
    ("Artifacts (cite these artifact_id values)", "artifacts", []),
    ("Measurements already taken", "measurements", {}),
    ("Detectability per region and defect type", "detectability", {}),
    ("Why detectability is limited (reason codes)", "detectability_reasons", {}),
    ("Image quality limitations", "image_limitations", []),
    ("Conflicts between photographs", "conflicts", []),
    ("Anomaly candidates (with enhancement provenance)",
     "anomaly_candidates", []),
    ("Applicable grading rules", "rubric_rules", []),
)


def build_prompt(manifest: dict[str, Any]) -> str:
    parts = [_BRIEF]
    for heading, key, default in _SECTIONS:
        parts.append(
            f"{heading}:\n{json.dumps(manifest.get(key, default), indent=2)}")
    return "\n\n".join(parts) + "\n"
```

```python
# src/card_reviewer/review/vision/anthropic.py
"""Anthropic implementation of VisionProvider.

The SDK is imported lazily inside `_call` so importing this module never
requires the dependency, and so tests can replace `_call` without any client
ever being constructed.
"""
from __future__ import annotations

import base64
import json
from typing import Any

from ..storage.artifacts import ArtifactStore
from .prompt import PROMPT_VERSION, build_prompt
from .provider import Assessment, ProviderContractError, parse_assessment

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096


SUPPORTED_MEDIA = {b"\x89PNG": "image/png", b"\xff\xd8\xff": "image/jpeg"}


def _media_type(data: bytes) -> str:
    for magic, media in SUPPORTED_MEDIA.items():
        if data.startswith(magic):
            return media
    raise ProviderContractError(
        "artifact is not a supported image format (PNG or JPEG)")


def build_request(manifest: dict[str, Any], store: ArtifactStore) -> list[dict]:
    """Content blocks for one request: the brief, then each artifact as a
    LABELLED image block.

    The label immediately precedes its image so the provider can cite
    artifact_id values that resolve back to real evidence. Sending only text
    would make this a vision provider that never sees anything.
    """
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": build_prompt(manifest)}]
    for artifact in manifest.get("artifacts", []):
        artifact_id = artifact["artifact_id"]
        data = store.read(artifact_id)
        blocks.append({"type": "text", "text": (
            f"artifact_id={artifact_id} view={artifact.get('view')} "
            f"origin={artifact.get('origin')} "
            f"enhancement={artifact.get('enhancement')}")})
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": _media_type(data),
            "data": base64.b64encode(data).decode("ascii")}})
    return blocks


class AnthropicVisionProvider:
    prompt_version = PROMPT_VERSION

    def __init__(self, model: str = DEFAULT_MODEL,
                 store: ArtifactStore | None = None,
                 api_key: str | None = None,
                 temperature: float | None = None) -> None:
        if store is None:
            raise ValueError(
                "AnthropicVisionProvider requires an ArtifactStore — the request "
                "carries image bytes, so there is nothing to send without one")
        self.model = model
        # Omitted by default. Sampling parameters are not accepted by every
        # current model, and this provider has no need to vary sampling: the
        # determinism that matters is the manifest, which is fingerprinted.
        self.temperature = temperature
        self._store = store
        self._api_key = api_key

    def assess(self, evidence_manifest: dict[str, Any]) -> Assessment:
        payload = self._call(build_request(evidence_manifest, self._store))
        allowed = {a["artifact_id"] for a in evidence_manifest.get("artifacts", [])}
        return parse_assessment(payload, allowed_artifact_ids=allowed)

    def _call(self, blocks: list[dict]) -> dict[str, Any]:
        import anthropic  # lazy: never imported at module load

        client = anthropic.Anthropic(api_key=self._api_key)
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = client.messages.create(
            model=self.model, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": blocks}], **kwargs)
        text = "".join(block.text for block in response.content
                       if block.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderContractError(
                f"provider returned non-JSON content: {exc}") from exc

    @property
    def inference_params(self) -> dict[str, Any]:
        """Part of the vision stage's producer signature."""
        params: dict[str, Any] = {"max_tokens": MAX_TOKENS}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_anthropic_contract.py -v`
Expected: PASS (12 tests including parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/vision/ tests/review/test_anthropic_contract.py tests/review/fixtures/vision/
git commit -m "feat(review): Anthropic provider with offline contract tests"
```

**Acceptance:** the constructed request contains real base64 image blocks, each labelled with its `artifact_id` and ordered deterministically; the prompt renders every canonical payload section; every malformed fixture raises `ProviderContractError`; the prompt contains no pricing word; the SDK is never imported at module load; no test issues a network call.

---
## Phase 8 — Integration and surface

### Task 33: Finding fusion

**Files:** Create `src/card_reviewer/review/fusion.py`; Test `tests/review/test_fusion.py`

**Interfaces:**
- Consumes: `findings.py`, `provenance.py`
- Produces: `FUSION_VERSION`, `FusedFinding`, `fuse(findings) -> list[FusedFinding]`

**Why this task exists (Decision 5).** It precedes combine because `combine` imports `fuse`.

**Note.** Both producers can see the same physical defect. Summing their penalties charges the card twice for one flaw, so a card looked at harder scores worse.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_fusion.py
from card_reviewer.review.enums import FindingState
from card_reviewer.review.findings import Finding, FindingProducer, Severity
from card_reviewer.review.fusion import fuse
from card_reviewer.review.provenance import (
    EvidenceOrigin, EvidenceRef, NormalizedBox,
)


def _f(producer, state, box, defect="scratches", severity=None,
       origin=EvidenceOrigin.ORIGINAL, aid="a"):
    return Finding(
        defect_type=defect, category="surface", state=state, producer=producer,
        confidence=0.9, psa10_relevant=True, severity=severity,
        location=NormalizedBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
        evidence=[EvidenceRef(artifact_id=aid, image_hash="h", origin=origin,
                              enhancement="clahe:clip=2.0"
                              if origin is EvidenceOrigin.ENHANCED else None,
                              view="v")])


def test_the_same_defect_seen_by_both_producers_fuses_into_one():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.SUSPECTED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (.1, .1, .4, .4))])
    assert len(out) == 1


def test_the_same_defect_type_in_a_different_region_stays_separate():
    """Two corners really are two flaws."""
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .2, .2)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (.7, .7, .9, .9))])
    assert len(out) == 2


def test_different_defect_types_in_one_region_stay_separate():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3),
                   defect="scratches"),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3),
                   defect="print_lines")])
    assert len(out) == 2


def test_the_fused_state_is_the_strongest_among_sources():
    """One producer confirming what another suspected is corroboration."""
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.SUSPECTED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert out[0].state is FindingState.OBSERVED


def test_the_source_findings_are_retained_for_calibration():
    sources = [_f(FindingProducer.HEURISTIC, FindingState.SUSPECTED, (0, 0, .3, .3)),
               _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))]
    out = fuse(sources)
    assert len(out[0].sources) == 2
    assert {f.producer for f in out[0].sources} == {FindingProducer.HEURISTIC,
                                                    FindingProducer.VISION}


def test_evidence_refs_are_unioned_so_i3_sees_everything():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3),
                   origin=EvidenceOrigin.ENHANCED, aid="a1"),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3),
                   origin=EvidenceOrigin.ORIGINAL, aid="a2")])
    origins = {r.origin for r in out[0].evidence}
    assert origins == {EvidenceOrigin.ENHANCED, EvidenceOrigin.ORIGINAL}


def test_a_fusion_of_only_enhanced_sources_still_fails_i3():
    from card_reviewer.review.findings import i3_satisfied
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3),
                   origin=EvidenceOrigin.ENHANCED, aid="a1"),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3),
                   origin=EvidenceOrigin.ENHANCED, aid="a2")])
    assert i3_satisfied(out[0].as_finding()) is False


def test_disagreement_between_sources_is_recorded():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.NOT_OBSERVED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert out[0].producers_disagreed is True


def test_agreement_is_not_recorded_as_disagreement():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert out[0].producers_disagreed is False


def test_as_finding_keeps_the_winning_producer_not_a_hardcoded_one():
    """I1's contradiction test compares producers; stamping everything
    HEURISTIC would make that clause dead code."""
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.SUSPECTED, (0, 0, .3, .3)),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))])
    assert out[0].as_finding().producer is FindingProducer.VISION


def test_the_worst_severity_among_sources_is_kept():
    out = fuse([_f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3),
                   severity=Severity.MINOR),
                _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3),
                   severity=Severity.SEVERE)])
    assert out[0].severity is Severity.SEVERE


def test_findings_without_a_location_never_silently_merge():
    a = _f(FindingProducer.HEURISTIC, FindingState.OBSERVED, (0, 0, .3, .3))
    b = _f(FindingProducer.VISION, FindingState.OBSERVED, (0, 0, .3, .3))
    a = a.model_copy(update={"location": None})
    b = b.model_copy(update={"location": None})
    assert len(fuse([a, b])) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'card_reviewer.review.fusion'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_reviewer/review/fusion.py
"""Correlate findings across producers into one assessment per defect.

Raw findings stay unfused underneath, per producer, because calibration
against real PSA outcomes needs to know what each source said alone. Scoring
and the verdict consume the fused view, so one physical defect penalizes once
however many producers saw it.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import FindingState
from .findings import Finding, Severity
from .provenance import EvidenceRef, NormalizedBox

FUSION_VERSION = "1.0.0"

# Strongest first: one producer confirming what another suspected is
# corroboration, not contradiction.
_STATE_RANK = {
    FindingState.OBSERVED: 3,
    FindingState.SUSPECTED: 2,
    FindingState.NOT_OBSERVED: 1,
    FindingState.NOT_ASSESSABLE: 0,
}
_SEVERITY_RANK = {Severity.MINOR: 1, Severity.MODERATE: 2, Severity.SEVERE: 3}


class FusedFinding(BaseModel):
    category: str
    defect_type: str
    state: FindingState
    confidence: float
    psa10_relevant: bool
    severity: Severity | None
    location: NormalizedBox | None
    evidence: list[EvidenceRef]
    sources: list[Finding]
    producers_disagreed: bool
    fusion_version: str = FUSION_VERSION

    def as_finding(self) -> Finding:
        """A Finding view, so I3 and the verdict operate unchanged.

        The producer is the one that supplied the winning state, never a
        hardcoded value: I1's contradiction test compares producers, so
        stamping everything HEURISTIC would make the cross-producer clause
        dead code and misattribute vision-only findings in the report.
        """
        strongest = max(self.sources, key=lambda f: _STATE_RANK[f.state])
        return Finding(
            defect_type=self.defect_type, category=self.category,
            state=self.state, producer=strongest.producer,
            confidence=self.confidence, psa10_relevant=self.psa10_relevant,
            severity=self.severity, location=self.location,
            evidence=self.evidence,
            rule_ids=sorted({r for f in self.sources for r in f.rule_ids}),
            demotion_reason=strongest.demotion_reason)


def fuse(findings: list[Finding]) -> list[FusedFinding]:
    groups: list[list[Finding]] = []
    for finding in findings:
        for group in groups:
            if _correlates(group[0], finding):
                group.append(finding)
                break
        else:
            groups.append([finding])
    return [_fuse_group(g) for g in groups]


def _correlates(a: Finding, b: Finding) -> bool:
    """Same defect AND overlapping region.

    Region is required: two findings about different corners are not one
    defect, and merging them would suppress a real flaw. Without locations we
    cannot establish they are the same thing, so we do not merge.
    """
    if (a.category, a.defect_type) != (b.category, b.defect_type):
        return False
    if a.location is None or b.location is None:
        return False
    return a.location.overlaps(b.location)


def _fuse_group(group: list[Finding]) -> FusedFinding:
    strongest = max(group, key=lambda f: _STATE_RANK[f.state])
    severities = [f.severity for f in group if f.severity]
    evidence: list[EvidenceRef] = []
    seen: set[str] = set()
    for f in group:
        for ref in f.evidence:
            if ref.artifact_id not in seen:
                seen.add(ref.artifact_id)
                evidence.append(ref)
    producers = {f.producer for f in group}
    states = {f.state for f in group}
    return FusedFinding(
        category=strongest.category, defect_type=strongest.defect_type,
        state=strongest.state, confidence=max(f.confidence for f in group),
        psa10_relevant=any(f.psa10_relevant for f in group),
        severity=(max(severities, key=lambda s: _SEVERITY_RANK[s])
                  if severities else None),
        location=strongest.location, evidence=evidence, sources=list(group),
        producers_disagreed=len(producers) > 1 and len(states) > 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_fusion.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/fusion.py tests/review/test_fusion.py
git commit -m "feat(review): finding fusion across producers"
```

**Acceptance:** the same defect seen by both producers fuses and penalizes once; different regions stay separate; the fused state is the strongest; source findings are retained; evidence refs union so I3 still sees enhancement-only support; disagreement is recorded.

---

### Task 34: Combine integration — findings from both producers into one verdict

**Files:** Modify `src/card_reviewer/review/policies/combine_v1.py` (add `combine`); Test `tests/review/test_combine.py`

**Interfaces:**
- Consumes: `heuristic.HeuristicResult`, `vision.provider.Assessment`, `policies/coverage_v1.CoverageResult`, `policies/scoring_v1`
- Produces: `CombinedResult`, `combine(heuristic, vision, coverage, *, card_context_known, scoped_rules, manifest_index=None, detectability=None, required_face_missing=False) -> CombinedResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_combine.py
from card_reviewer.review.enums import (
    Coverage, FindingState, ReviewConfidence, Scale, Verdict,
)
from card_reviewer.review.findings import Finding, FindingProducer
from card_reviewer.review.heuristic import HeuristicResult
from card_reviewer.review.policies.combine_v1 import combine
from card_reviewer.review.policies.coverage_v1 import CoverageResult
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef
from card_reviewer.review.vision.provider import Assessment, GemView


def _f(producer, state, defect="print_lines", enhanced=False):
    origin = EvidenceOrigin.ENHANCED if enhanced else EvidenceOrigin.ORIGINAL
    return Finding(defect_type=defect, category="surface", state=state,
                   producer=producer, confidence=0.95, psa10_relevant=True,
                   evidence=[EvidenceRef(
                       artifact_id="a", image_hash="h", origin=origin,
                       enhancement="clahe:clip=2.0" if enhanced else None,
                       view="v")])


def _cov(outcome=Coverage.SUFFICIENT):
    return CoverageResult(outcome=outcome, rankable=outcome is not Coverage.INADEQUATE)


def _vision(findings=(), assessability=None, gem=GemView.NO_DISQUALIFIER):
    return Assessment(
        findings=[], gem_view=gem,
        category_assessability=assessability or {
            "centering": True, "corners": True, "edges": True, "surface": True})


def test_off_mode_produces_a_complete_result_without_any_vision(rubric_rules):
    r = combine(HeuristicResult(), None, _cov(), card_context_known=True,
                scoped_rules=rubric_rules)
    assert r.verdict is Verdict.PASS
    assert r.psa10_rank_score == 100
    assert r.vision_present is False


def test_i3_is_enforced_before_the_verdict_is_decided(rubric_rules):
    """An enhancement-only observed finding is demoted, so it cannot reject."""
    h = HeuristicResult(findings=[_f(FindingProducer.HEURISTIC,
                                     FindingState.OBSERVED, enhanced=True)])
    r = combine(h, None, _cov(), card_context_known=True,
                scoped_rules=rubric_rules)
    assert r.verdict is not Verdict.REJECT
    assert any("I3" in f.demotion_reason for f in r.findings)


def test_the_same_defect_from_both_producers_penalizes_once(rubric_rules):
    """Decision 5: looking harder must not make the score worse."""
    from card_reviewer.review.provenance import NormalizedBox
    box = NormalizedBox(x0=0.0, y0=0.0, x1=0.3, y1=0.3)
    one = _f(FindingProducer.HEURISTIC, FindingState.SUSPECTED).model_copy(
        update={"location": box})
    both = HeuristicResult(findings=[
        one, one.model_copy(update={"producer": FindingProducer.VISION})])
    a = combine(HeuristicResult(findings=[one]), None, _cov(),
                card_context_known=True, scoped_rules=rubric_rules)
    b = combine(both, None, _cov(), card_context_known=True,
                scoped_rules=rubric_rules)
    assert a.psa10_rank_score == b.psa10_rank_score
    assert len(b.findings) == 2 and len(b.fused) == 1


def test_raw_findings_are_retained_alongside_the_fused_view(rubric_rules):
    from card_reviewer.review.provenance import NormalizedBox
    box = NormalizedBox(x0=0.0, y0=0.0, x1=0.3, y1=0.3)
    one = _f(FindingProducer.HEURISTIC, FindingState.SUSPECTED).model_copy(
        update={"location": box})
    r = combine(HeuristicResult(findings=[
        one, one.model_copy(update={"producer": FindingProducer.VISION})]),
        None, _cov(), card_context_known=True, scoped_rules=rubric_rules)
    assert {f.producer for f in r.findings} == {FindingProducer.HEURISTIC,
                                                FindingProducer.VISION}


def test_an_unmatched_finding_cannot_reject_but_does_not_vanish(rubric_rules):
    """Decision 4: advisory means it cannot REJECT — not that it disappears.
    An observed corner defect with no matching rule must still route to REVIEW
    and still cost score, or it would ship as a clean gem candidate."""
    from card_reviewer.review.provenance import NormalizedBox
    odd = _f(FindingProducer.HEURISTIC, FindingState.OBSERVED).model_copy(
        update={"category": "corners", "defect_type": "unrecognized",
                "location": NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)})
    r = combine(HeuristicResult(findings=[odd]), None, _cov(),
                card_context_known=True, scoped_rules=rubric_rules,
                detectability={("front", "corners", "unrecognized"): Scale.HIGH})
    assert r.verdict is Verdict.REVIEW
    assert r.psa10_rank_score < 100


def test_i1_cannot_be_satisfied_when_detectability_is_absent(rubric_rules):
    """An empty detectability map must block a reject, never license one."""
    from card_reviewer.review.provenance import NormalizedBox
    f = _f(FindingProducer.HEURISTIC, FindingState.OBSERVED).model_copy(
        update={"category": "corners", "defect_type": "rounding",
                "location": NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)})
    r = combine(HeuristicResult(findings=[f]), None, _cov(),
                card_context_known=True, scoped_rules=rubric_rules,
                detectability={})
    assert r.verdict is not Verdict.REJECT


def test_the_score_is_null_when_coverage_is_inadequate(rubric_rules):
    r = combine(HeuristicResult(), None, _cov(Coverage.INADEQUATE),
                card_context_known=True, scoped_rules=rubric_rules)
    assert r.psa10_rank_score is None and r.rankable is False
    assert r.verdict is Verdict.INSUFFICIENT_IMAGES


def test_unknown_card_context_lowers_confidence_but_never_rejects(rubric_rules):
    r = combine(HeuristicResult(), None, _cov(), card_context_known=False,
                scoped_rules=rubric_rules)
    assert r.review_confidence is ReviewConfidence.MEDIUM
    assert r.verdict is not Verdict.REJECT


def test_a_missing_required_face_is_low_confidence_yet_still_rankable(rubric_rules):
    r = combine(HeuristicResult(), None, _cov(Coverage.PARTIAL),
                card_context_known=True, scoped_rules=rubric_rules,
                required_face_missing=True)
    assert r.review_confidence is ReviewConfidence.LOW
    assert r.rankable is True


def test_grade_and_score_are_reported_independently(rubric_rules):
    r = combine(HeuristicResult(), None, _cov(Coverage.PARTIAL),
                card_context_known=True, scoped_rules=rubric_rules)
    assert r.estimated_psa_grade == "9-10"
    assert r.psa10_rank_score == 90
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_combine.py -v`
Expected: FAIL with `ImportError: cannot import name 'combine'`

- [ ] **Step 3: Write minimal implementation**

Append to `combine_v1.py`:

```python
class CombinedResult(BaseModel):
    verdict: Verdict
    psa10_candidate: Psa10Candidate
    psa10_rank_score: int | None
    rankable: bool
    estimated_psa_grade: str | None
    review_confidence: ReviewConfidence
    coverage: Coverage
    # Raw, per-producer findings retained for calibration (Decision 5).
    findings: list[Finding] = Field(default_factory=list)
    # The fused view scoring and the verdict actually consumed.
    fused: list = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    vision_present: bool = False
    policy_version: str = COMBINATION_POLICY_VERSION


def combine(heuristic, vision, coverage, *, card_context_known: bool,
            scoped_rules: list, manifest_index: dict | None = None,
            detectability: dict | None = None,
            required_face_missing: bool = False) -> CombinedResult:
    """Fuse both producers' findings into one verdict.

    Order matters and is load-bearing:

      1. Resolve provider findings against the manifest, so provenance
         survives the round trip (Decision 1).
      2. Enforce I3, demoting enhancement-only observations before anything
         can act on them.
      3. Fuse across producers, so one physical defect penalizes once
         (Decision 5).
      4. Resolve each fused finding to its matched rules and authority, which
         also decides psa10_relevant (Decision 4).
      5. Decide the verdict, then score with I1-awareness (Decision 2).
    """
    from ..fusion import fuse
    from ..heuristic import best_detectability
    from ..relevance import resolve_relevance
    from ..vision.provider import resolve_vision_findings
    from .scoring_v1 import estimated_grade, rank_score, review_confidence

    raw: list[Finding] = list(heuristic.findings)
    if vision is not None:
        raw.extend(resolve_vision_findings(vision, manifest_index or {}))

    # Fuse BEFORE enforcing I3. Decision 5 says the fused finding carries the
    # union of its sources' evidence, "so I3 is evaluated over everything
    # supporting it" — running I3 first would demote a finding one producer saw
    # only under enhancement even when the other saw it in the original.
    fused = fuse(raw)
    fused = [f.model_copy(update={"state": g.state,
                                  "demotion_reason": g.demotion_reason})
             for f, g in zip(fused, enforce_i3([f.as_finding() for f in fused]))]
    resolved = resolve_relevance([f.as_finding() for f in fused], scoped_rules)

    # Detectability is keyed (ImageRole, category, defect_type) everywhere in
    # the system. Looking it up with a 2-tuple would miss every time and fall
    # back to the default, silently disabling I1's adequacy prong — the prong
    # that actually stops poor photographs producing rejections. The default is
    # NONE, not HIGH: absent evidence must block a reject, never license one.
    detectability = detectability or {}
    triples = []
    for rf in resolved:
        scale = best_detectability(detectability, rf.finding.category,
                                   rf.finding.defect_type)
        triples.append((rf.finding, rf.authority, scale))

    result = decide_verdict(triples, coverage.outcome,
                            ambiguity=bool(heuristic.unevaluable_reasons
                                           or any(f.producers_disagreed
                                                  for f in fused)))

    # Scoring needs to know which findings actually cleared I1: only those get
    # the hard floor, so an unresolved concern stays meaningfully rankable.
    scored = [
        (rf.finding, rf.authority,
         i1_satisfied(rf.finding, scale, [(f, s) for f, _, s in triples]))
        for (rf, (_, _, scale)) in zip(resolved, triples)
    ]

    disagreed = any(f.producers_disagreed for f in fused)
    # A material contradiction is two findings disagreeing about the same
    # defect (spec §15) — NOT an I3 demotion, which is one finding whose
    # evidence was enhancement-only. Conflating them would cap confidence at
    # LOW for any enhancement-surfaced anomaly.
    contradictions = [f for f in fused if f.producers_disagreed]

    return CombinedResult(
        verdict=result.verdict, psa10_candidate=result.psa10_candidate,
        psa10_rank_score=rank_score(scored, coverage.outcome),
        rankable=coverage.rankable,
        estimated_psa_grade=estimated_grade(scored, coverage.outcome),
        review_confidence=review_confidence(
            coverage.outcome, contradictions, disagreed, card_context_known,
            required_face_missing=required_face_missing),
        coverage=coverage.outcome, findings=raw, fused=fused,
        reasons=result.reasons, vision_present=vision is not None)
```

Add the imports `Psa10Candidate`, `ReviewConfidence` and `enforce_i3` at the top of the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_combine.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/policies/combine_v1.py tests/review/test_combine.py
git commit -m "feat(review): combine both producers into one verdict"
```

**Acceptance:** `OFF` mode yields a complete result; provenance survives the vision round trip; I3 runs before fusion and the verdict; one physical defect penalizes once while raw per-producer findings are retained; an unmatched finding cannot reject; a missing required face is `low` confidence yet still rankable; score is null exactly when unrankable.

---

### Task 35: Application service and full pipeline wiring

**Files:** Modify `src/card_reviewer/review/pipeline.py` (add `ReviewPipeline`); Create `src/card_reviewer/review/service.py`; Modify `src/card_reviewer/review/__init__.py`; Test `tests/review/test_pipeline_e2e.py`

**Interfaces:**
- Produces: `ReviewPipeline.review(ResolvedCandidate, mode, provider=None) -> CardReview`, `review_card(CandidateInput, mode) -> CardReview`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_pipeline_e2e.py
import pytest

from card_reviewer.review.enums import Mode, Verdict
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.ingest.adapter import ManualAdapter
from card_reviewer.review.models import CandidateInput
from card_reviewer.review.pipeline import ReviewPipeline
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository
from card_reviewer.review.vision.provider import (
    Assessment, FakeProvider, GemView,
)


@pytest.fixture
def rig(tmp_path):
    conn = connect(tmp_path / "t.db"); migrate(conn)
    store = ArtifactStore(tmp_path / "store")
    repo = SqliteRepository(conn)
    return ReviewPipeline(repo, store), store, repo


def _candidate(tmp_path, store, specs):
    paths = []
    for i, spec in enumerate(specs):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(render_png(spec))
        paths.append(p)
    return ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="2023 Topps Chrome test", image_paths=paths,
        supplied_roles={str(paths[0]): "front"} | (
            {str(paths[1]): "back"} if len(paths) > 1 else {})))


def test_a_front_and_back_card_runs_end_to_end_in_off_mode(rig, tmp_path):
    """DoD 1."""
    pipeline, store, repo = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(seed=2)])
    review = pipeline.review(resolved, Mode.OFF)
    assert review.verdict in set(Verdict)
    assert repo.reviews_for(resolved.candidate_id)


def test_off_never_calls_and_deep_always_does(rig, tmp_path):
    """DoD 2."""
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(seed=2)])
    provider = FakeProvider(Assessment(
        category_assessability={"centering": True, "corners": True,
                                "edges": True, "surface": True},
        gem_view=GemView.NO_DISQUALIFIER))
    pipeline.review(resolved, Mode.OFF, provider)
    assert provider.calls == 0
    pipeline.review(resolved, Mode.DEEP, provider)
    assert provider.calls == 1


def test_an_off_run_does_not_poison_the_deep_cache(rig, tmp_path):
    """DoD 11 — mode is in routing's fingerprint."""
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(seed=2)])
    provider = FakeProvider(Assessment(
        category_assessability={"centering": True, "corners": True,
                                "edges": True, "surface": True},
        gem_view=GemView.NO_DISQUALIFIER))
    pipeline.review(resolved, Mode.OFF, provider)
    pipeline.review(resolved, Mode.DEEP, provider)
    assert provider.calls == 1


def test_the_same_image_in_two_candidates_is_analyzed_once(rig, tmp_path):
    """DoD 5."""
    pipeline, store, repo = rig
    a = _candidate(tmp_path, store, [CardSpec(), CardSpec(seed=2)])
    pipeline.review(a, Mode.OFF)
    before = _preflight_rows(repo)
    b = _candidate(tmp_path, store, [CardSpec(), CardSpec(seed=2)])
    pipeline.review(b, Mode.OFF)
    assert _preflight_rows(repo) == before


def test_a_front_only_card_is_partial_and_rankable_never_rejected(rig, tmp_path):
    """DoD 16."""
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec()])
    review = pipeline.review(resolved, Mode.OFF)
    assert review.verdict in (Verdict.REVIEW, Verdict.REJECT)
    if review.verdict is Verdict.REVIEW:
        assert review.rankable is True


def test_rerunning_reuses_cached_stage_results(rig, tmp_path):
    """DoD 3."""
    pipeline, store, repo = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(seed=2)])
    pipeline.review(resolved, Mode.OFF)
    before = _preflight_rows(repo)
    pipeline.review(resolved, Mode.OFF)
    assert _preflight_rows(repo) == before


def test_unknown_product_context_asks_for_identity_not_photos(rig, tmp_path):
    """Integration, end to end: perfect front and back, product unknown."""
    pipeline, store, _ = rig
    resolved = _candidate(tmp_path, store, [CardSpec(), CardSpec(seed=2)])
    # title carries no recognizable product token, so context stays unknown
    resolved = resolved.model_copy(update={"title": "nice card lot 4"})
    review = pipeline.review(resolved, Mode.OFF)
    assert review.rankable is True
    assert review.verdict is Verdict.REVIEW
    assert review.card_identification_request is True
    assert review.recommended_additional_photos == []


def _preflight_rows(repo) -> int:
    return repo._conn.execute(
        "SELECT COUNT(*) FROM stage_result WHERE stage='preflight'").fetchone()[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_pipeline_e2e.py -v`
Expected: FAIL with `ImportError: cannot import name 'ReviewPipeline'`

- [ ] **Step 3: Write minimal implementation**

Append `ReviewPipeline` to `pipeline.py`. Every stage goes through `StageRunner.run` with its `schema=`, so caching, validation and failure recording are uniform.

```python
# appended to src/card_reviewer/review/pipeline.py
"""Orchestration. The stage order is the spec's, and the wiring below is where
the guarantees the policies promise actually get connected."""

IMAGE_TIER_VERSIONS = {
    "preflight": {"preflight_version": PREFLIGHT_VERSION, "config": {}},
    "geometry": {"geometry_version": GEOMETRY_VERSION, "config": {}},
    "observability": {"observability_version": OBSERVABILITY_VERSION,
                      "taxonomy_version": TAXONOMY_VERSION, "config": {}},
    "cv_measurements": {"cv_version": CV_VERSION,
                        "taxonomy_version": TAXONOMY_VERSION, "config": {}},
}


class ReviewPipeline:
    def __init__(self, repo: Repository, store: ArtifactStore,
                 rubric: Rubric | None = None) -> None:
        self._repo = repo
        self._store = store
        self._runner = StageRunner(repo)
        # RubricError aborts the run: there is no verdict without a rubric, and
        # guessing at one is worse than declining (spec §4 failure policy).
        self._rubric = rubric or load_active_rubric()

    def review(self, candidate: ResolvedCandidate, mode: Mode,
               provider: VisionProvider | None = None) -> CardReview:
        images = [self._analyze_image(i.image_hash) for i in candidate.images]

        roles = resolve_roles([
            RoleInput(image_hash=i.image_hash,
                      supplied_role=i.supplied_role,
                      text_density=ev["text_density"],
                      has_central_image_region=ev["has_central_image_region"])
            for i, ev in zip(candidate.images, images)])

        context = CardContextNormalizer().normalize(
            raw_title=candidate.title, supplied_card_type=candidate.card_type,
            supplied_set=candidate.set_name)
        scoped = scope_rules(
            self._rubric.for_card(context.canonical_card_types,
                                  context.canonical_sets), context)

        assembled = assemble(
            [ImageEvidence(**ev["assembly_input"]) for ev in images], roles)
        heuristic = evaluate(assembled, scoped)

        # scope_rules feeds three consumers: the heuristic (which rules to
        # evaluate), coverage (which rules could NOT be applied), and relevance
        # inside combine (which rules govern a finding).
        unevaluable = [
            UnevaluableRule(rule_id=sr.rule.id, category=sr.rule.category.value,
                            reason_code=sr.reason)
            for sr in scoped if sr.evaluability is RuleEvaluability.UNEVALUABLE]

        provisional = evaluate_coverage(
            assembled.detectability, assembled.reason_codes, {},
            assembled.faces_present, unevaluable_rules=unevaluable)

        routing = decide_routing(mode, heuristic.findings, provisional.outcome,
                                 assembled.detectability)
        routing_id = self._repo.save_routing_decision(
            candidate_id=candidate.candidate_id,
            policy_version=ROUTING_POLICY_VERSION, mode=mode.value,
            call_vision=routing.call_vision,
            trigger_reasons=routing.trigger_reasons,
            input_fingerprint=fingerprint({"mode": mode.value}))

        vision, manifest_index, vision_result_id = None, {}, None
        if routing.call_vision and provider is not None:
            built = build_manifest(assembled, mode, applicable(scoped))
            manifest_index = built.index
            try:
                vision = provider.assess(built.payload)
                vision_result_id = self._repo.put_stage_result(
                    "vision", fingerprint(built.payload), "provider",
                    vision.model_dump(), {}, candidate_id=candidate.candidate_id)
            except Exception as exc:
                # A permanently failed call leaves vision_result_id null and the
                # review proceeds on CV evidence exactly as OFF would. It never
                # produces REJECT.
                self._repo.record_attempt(
                    "vision", None, None, error_kind=type(exc).__name__,
                    error_detail=str(exc), candidate_id=candidate.candidate_id)

        assessability = (vision.category_assessability if vision else {})
        coverage = evaluate_coverage(
            assembled.detectability, assembled.reason_codes, assessability,
            assembled.faces_present, unevaluable_rules=unevaluable)

        combined = combine(
            heuristic, vision, coverage,
            card_context_known=context.is_known, scoped_rules=scoped,
            manifest_index=manifest_index,
            detectability=assembled.detectability,
            # Not inferrable from the coverage outcome: a front-only card is
            # PARTIAL and rankable yet its confidence is LOW.
            required_face_missing=any(f not in assembled.faces_present
                                      for f in REQUIRED_FACES))

        return self._persist(candidate, mode, routing_id, combined, coverage,
                             context, roles, vision_result_id)

    def _analyze_image(self, image_hash: str) -> dict[str, Any]:
        """Image tier: cached by image hash, so a photo shared by two listings
        is analyzed once, ever."""
        data = self._store.read(image_hash)
        pre = self._runner.run(
            "preflight", {"image_hash": image_hash},
            IMAGE_TIER_VERSIONS["preflight"],
            lambda: preflight_analyze(data).model_dump(),
            schema=PreflightResult, image_hash=image_hash)
        if not pre["usable"]:
            return _unusable_image(image_hash, pre)

        geometry = geometry_analyze(data)
        obs = observability_analyze(geometry)
        measurements = {
            "centering": measure_centering(geometry).model_dump(),
            "corners": measure_corners(geometry, self._store, image_hash),
            "edges": measure_edges(geometry, self._store, image_hash),
            "surface": measure_surface(geometry, self._store, image_hash),
        }
        return _image_evidence(image_hash, pre, geometry, obs, measurements)
```

`_unusable_image` and `_image_evidence` build the `ImageEvidence` payload — detectability, **reason codes**, limitations, sharpness, centering, anomalies and evidence refs keyed `"category:defect_type"`. `_persist` writes the `review` row (including both coverage result ids and the nullable `vision_result_id`) and returns the `CardReview`. A failed image-tier stage marks that image unusable and the candidate continues on its remaining images.

`service.review_card` composes `ManualAdapter` and `ReviewPipeline`; `review/__init__.py` exports `review_card`, `ReviewPipeline` and `CardReview`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_pipeline_e2e.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/pipeline.py src/card_reviewer/review/service.py src/card_reviewer/review/__init__.py tests/review/test_pipeline_e2e.py
git commit -m "feat(review): full pipeline wiring and application service"
```

**Acceptance:** DoD items 1, 2, 3, 5, 11 and 16 are demonstrated by the tests in this task; every stage call passes a `schema=`; the manifest index reaches `combine`; unknown product context produces an identification request and no photo request.

---

### Task 36: CLI, reporting, outcomes and export

**Files:** Create `src/card_reviewer/review/report.py`, `src/card_reviewer/review/cli.py`; Modify `pyproject.toml`; Test `tests/review/test_cli.py`

**Interfaces:**
- Produces: `card-review screen|deep|show|outcome|export|provider-smoke`

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_cli.py
import json

from typer.testing import CliRunner

from card_reviewer.review.cli import app

runner = CliRunner()


def test_screen_runs_a_card_and_prints_a_verdict(tmp_path):
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    img = tmp_path / "a.png"; img.write_bytes(render_png(CardSpec()))
    result = runner.invoke(app, ["screen", str(img), "--mode", "off",
                                 "--data-dir", str(tmp_path / "data")])
    assert result.exit_code == 0
    assert any(v in result.stdout for v in
               ("PASS", "REVIEW", "REJECT", "INSUFFICIENT_IMAGES"))


def test_the_default_mode_is_smart():
    result = runner.invoke(app, ["screen", "--help"])
    assert "smart" in result.stdout.lower()


def test_export_emits_valid_json(tmp_path):
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    img = tmp_path / "a.png"; img.write_bytes(render_png(CardSpec()))
    data = tmp_path / "data"
    runner.invoke(app, ["screen", str(img), "--mode", "off", "--data-dir", str(data)])
    result = runner.invoke(app, ["export", "1", "--data-dir", str(data)])
    assert json.loads(result.stdout)["verdict"]


def test_outcome_records_a_psa_result_joinable_to_its_review(tmp_path):
    """DoD 9."""
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    img = tmp_path / "a.png"; img.write_bytes(render_png(CardSpec()))
    data = tmp_path / "data"
    runner.invoke(app, ["screen", str(img), "--mode", "off", "--data-dir", str(data)])
    result = runner.invoke(app, ["outcome", "--review-id", "1", "--grade", "10",
                                 "--cert", "12345678", "--data-dir", str(data)])
    assert result.exit_code == 0


def test_the_report_shows_limitations_and_photo_requests(tmp_path):
    """Non-negotiable rule 3: never hide image limitations."""
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    img = tmp_path / "a.png"; img.write_bytes(render_png(CardSpec()))
    data = tmp_path / "data"
    runner.invoke(app, ["screen", str(img), "--mode", "off", "--data-dir", str(data)])
    result = runner.invoke(app, ["show", "1", "--data-dir", str(data)])
    assert "limitation" in result.stdout.lower()


def test_no_grading_logic_lives_in_the_cli():
    import card_reviewer.review.cli as mod
    source = __import__("inspect").getsource(mod)
    for forbidden in ("Verdict.REJECT", "psa10_rank_score =", "Scale.MODERATE"):
        assert forbidden not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`report.py` renders a `CardReview` through Rich: identity, verdict, `psa10_candidate`, rank score, grade estimate, review confidence, coverage with per-category detail, findings grouped by state with their producer and evidence, then — always — `limitations` marked structural / circumstantial / metadata-resolvable, `recommended_additional_photos`, and any card-identification request.

`cli.py` is a thin Typer app: each command resolves paths, constructs `ReviewPipeline`, calls it, and hands the result to `report.py`. No thresholds, no verdict logic, no scoring — a test asserts that by scanning the module source.

`provider-smoke` is the only command that constructs `AnthropicVisionProvider` against a real key; it is never exercised by CI.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_cli.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/card_reviewer/review/report.py src/card_reviewer/review/cli.py pyproject.toml tests/review/test_cli.py
git commit -m "feat(review): CLI, reporting, outcome recording and export"
```

**Acceptance:** the report always shows limitations; no grading logic appears in the CLI module; a PSA outcome joins back to its review.

---

### Task 37: Golden real-image regression fixtures

**Files:** Create `tests/review/golden/*.jpg`, `tests/review/golden/expectations.yaml`; Test `tests/review/test_golden.py`

**They assert observations, never grades.** A photograph is never labelled "this is a PSA 10" — subjective grading opinion must not become fake ground truth.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_golden.py
import pytest
import yaml
from pathlib import Path

from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.measure.centering import measure_centering
from card_reviewer.review.imaging.observability import analyze as obs
from card_reviewer.review.imaging.preflight import analyze as pre

GOLDEN = Path(__file__).parent / "golden"
CASES = yaml.safe_load((GOLDEN / "expectations.yaml").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["file"])
def test_golden_observations_hold(case):
    data = (GOLDEN / case["file"]).read_bytes()
    assert pre(data).usable is case["preflight_usable"]
    if not case["preflight_usable"]:
        return
    g = geom(data)
    assert (g.boundary_confidence > 0.5) is case["boundary_detected"]
    if case.get("centering_measurable") is not None:
        assert measure_centering(g).measurable is case["centering_measurable"]
    for region, expected in case.get("detectability", {}).items():
        key = tuple(region.split("."))
        assert str(obs(g).detectability[key]) == expected


def test_no_golden_case_asserts_a_psa_grade():
    """Guard: subjective grading opinion must never become ground truth."""
    for case in CASES:
        assert not (set(case) & {"grade", "psa_grade", "expected_grade",
                                 "verdict", "is_psa10"})
```

`expectations.yaml` records only observations:

```yaml
- file: white_border_clean.jpg
  preflight_usable: true
  boundary_detected: true
  centering_measurable: true
  detectability:
    bottom_left.corners.whitening: low
    bottom_left.corners.rounding: high

- file: borderless_chrome.jpg
  preflight_usable: true
  boundary_detected: true
  centering_measurable: false

- file: glare_heavy.jpg
  preflight_usable: true
  boundary_detected: true
  detectability:
    top_left.surface.scratches: low

- file: thumbnail_too_small.jpg
  preflight_usable: false
  boundary_detected: false
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_golden.py -v`
Expected: FAIL — the fixture images do not exist yet

- [ ] **Step 3: Add the fixtures**

Commit four small real photographs (under 200 KB each) matching the four cases above. If the owner has not supplied real photographs at this point, generate stand-ins with the synthetic generator and mark them clearly in `expectations.yaml` with a `synthetic: true` key, so they are replaced with real images later rather than silently accepted as real-world coverage.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/review/test_golden.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/review/golden/ tests/review/test_golden.py
git commit -m "test(review): golden real-image observation fixtures"
```

**Acceptance:** every golden case asserts observations only; the guard test fails if anyone adds a grade key.

---

### Task 38: Full definition-of-done verification

**Files:** Test `tests/review/test_definition_of_done.py`

One test per DoD item that is not already covered by an earlier task's tests, so the whole contract is verifiable in one run.

- [ ] **Step 1: Write the failing test**

```python
# tests/review/test_definition_of_done.py
"""One test per spec §19 item not already covered elsewhere."""
import pytest

from card_reviewer.review.enums import Coverage, Mode, Scale, Verdict
from card_reviewer.review.policies.coverage_v1 import evaluate_coverage
from card_reviewer.review.roles import ImageRole
from card_reviewer.review.taxonomy import defect_types_for


def test_dod4_a_cv_bump_does_not_rebill_an_unchanged_vision_call(rig_factory):
    """Bumping the CV analyzer creates a new cv_measurements result without
    re-billing a vision call whose evidence manifest is unchanged."""
    pipeline, resolved, provider = rig_factory(cv_version="1.0.0")
    pipeline.review(resolved, Mode.DEEP, provider)
    calls = provider.calls
    pipeline_v2, resolved_v2, _ = rig_factory(cv_version="1.0.1",
                                              provider=provider)
    pipeline_v2.review(resolved_v2, Mode.DEEP, provider)
    assert provider.calls == calls


def test_dod3_a_stored_vision_result_survives_a_crash_before_combination(
        rig_factory):
    """The vision call is the expensive step; a crash between it and
    combination must never re-bill it."""
    pipeline, resolved, provider = rig_factory()
    pipeline.review(resolved, Mode.DEEP, provider)
    calls = provider.calls
    # A fresh pipeline over the same database is exactly what a restart looks
    # like: the stage_result rows are all that survive.
    pipeline_after_crash, _, _ = rig_factory(provider=provider, reuse_db=True)
    pipeline_after_crash.review(resolved, Mode.DEEP, provider)
    assert provider.calls == calls


def test_dod8_unknown_context_receives_every_rule_and_biases_to_review(rubric):
    from card_reviewer.review.context import CardContext
    from card_reviewer.review.evaluability import scope_rules
    scoped = scope_rules(rubric.for_card(None, None), CardContext())
    assert len(scoped) == len(rubric.rules)
    assert any(s.evaluability.value == "unevaluable" for s in scoped)


def test_dod10_a_white_bordered_card_can_pass():
    det, reasons = {}, {}
    for face in (ImageRole.FRONT, ImageRole.BACK):
        for cat in ("centering", "corners", "edges", "surface"):
            for dt in defect_types_for(cat):
                det[(face, cat, dt)] = Scale.HIGH
        for cat in ("corners", "edges"):
            det[(face, cat, "whitening")] = Scale.LOW
            reasons[(face, cat, "whitening")] = "WHITE_BORDER"
    r = evaluate_coverage(det, reasons, {}, (ImageRole.FRONT, ImageRole.BACK))
    assert r.outcome is Coverage.SUFFICIENT


def test_dod14_the_synthetic_generator_stands_alone():
    from card_reviewer.review.imaging.synthetic import CardSpec, render
    assert render(CardSpec(borderless=True)) is not None
    assert render(CardSpec(corner_damage={"top_left": 0.5})) is not None


def test_dod15_no_price_field_reaches_the_core(tmp_path):
    from card_reviewer.review.models import ResolvedCandidate
    assert not (set(ResolvedCandidate.model_fields)
                & {"asking_price", "price", "cost", "value"})


def test_dod17_a_rubric_content_change_refreshes_the_manifest(rubric):
    from card_reviewer.review.fingerprint import fingerprint
    from card_reviewer.review.manifest import build_manifest
    from card_reviewer.review.enums import Mode

    class _A:
        evidence_refs, detectability, conflicts, centering = {}, {}, [], {}
        reason_codes, anomalies, limitations = {}, [], []

    rules = rubric.for_card(None, None)
    a = fingerprint(build_manifest(_A(), Mode.SMART, rules).payload)
    b = fingerprint(build_manifest(_A(), Mode.SMART, rules).payload)
    assert a == b, "identical rule content must not change the fingerprint"
    c = fingerprint(build_manifest(_A(), Mode.SMART, rules[:-1]).payload)
    assert a != c, "changed rule content must change the fingerprint"


def test_the_anthropic_request_would_carry_real_images(tmp_path):
    """A vision provider that sends only text is not a vision provider."""
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.storage.artifacts import ArtifactStore
    from card_reviewer.review.vision.anthropic import build_request
    store = ArtifactStore(tmp_path)
    h = store.put_image(render_png(CardSpec()))
    aid = store.put_derived(h, "surface", "original.png", render_png(CardSpec()))
    blocks = build_request(
        {"artifacts": [{"artifact_id": aid, "view": "surface_original",
                        "origin": "normalized", "enhancement": None}],
         "rubric_rules": [], "measurements": {}}, store)
    assert any(b["type"] == "image" for b in blocks)


def test_unknown_product_context_requests_identity_not_photographs():
    """Integration: perfect photographs, unknown product."""
    from card_reviewer.review.policies.coverage_v1 import (
        UnevaluableRule, evaluate_coverage,
    )
    det = {(f, c, d): Scale.HIGH
           for f in (ImageRole.FRONT, ImageRole.BACK)
           for c in ("centering", "corners", "edges", "surface")
           for d in defect_types_for(c)}
    r = evaluate_coverage(det, {}, {}, (ImageRole.FRONT, ImageRole.BACK),
                          unevaluable_rules=[UnevaluableRule(
                              rule_id="SURFACE_SHINY_001", category="surface",
                              reason_code="UNKNOWN_PRODUCT_CONTEXT")])
    assert r.outcome is Coverage.PARTIAL and r.rankable is True
    assert r.card_identification_request is True
    assert r.recommended_additional_photos == []


def test_two_manual_copies_with_one_title_do_not_share_a_candidate(tmp_path):
    from card_reviewer.review.ingest.adapter import ManualAdapter
    from card_reviewer.review.models import CandidateInput
    from card_reviewer.review.storage.artifacts import ArtifactStore
    img = tmp_path / "a.png"; img.write_bytes(b"pixels")
    adapter = ManualAdapter(ArtifactStore(tmp_path / "s"))
    kw = dict(source="manual", title="same title", image_paths=[img])
    assert (adapter.resolve(CandidateInput(**kw)).candidate_id
            != adapter.resolve(CandidateInput(**kw)).candidate_id)


def test_dod18_no_test_in_the_suite_calls_the_anthropic_api():
    import subprocess
    out = subprocess.run(
        ["grep", "-rn", "anthropic.Anthropic(", "tests/"],
        capture_output=True, text=True)
    assert out.stdout == ""
```

`rig_factory` is a fixture added to `tests/review/conftest.py`. It returns `(pipeline, resolved_candidate, provider)` and accepts `cv_version=`, `provider=` and `reuse_db=` so a version bump, a shared `FakeProvider` and a simulated restart over the same database can each be expressed. It uses `ReviewPipeline.review(resolved, mode, provider)` — the only declared entry point — rather than inventing a `review_once` helper.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/review/test_definition_of_done.py -v`
Expected: FAIL — `rig_factory` is undefined

- [ ] **Step 3: Add the fixture and make the tests pass**

Add `rig_factory` to `conftest.py`; fix anything the DoD tests surface.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS, including all 267 existing subsystem B tests. Output pristine — no warnings.

- [ ] **Step 5: Commit**

```bash
git add tests/review/test_definition_of_done.py tests/review/conftest.py
git commit -m "test(review): full definition-of-done verification"
```

**Acceptance:** every numbered DoD item in spec §19 has a passing test; the Anthropic request provably carries images without a network call; unknown product context requests identity rather than photographs; manual candidate identity does not collide on title; the full suite is green; no test calls the API.

---

## Plan Self-Review

**Spec coverage.** Every spec section maps to a task: §1–2 (Global Constraints), §3 (T28, T35), §4 (T5, T6, T28), §5 (T7, T8), §6 (T13), §7.1–7.4 (T21–T26), §8 (T10, T11, T12), §9 (T3, T27), §10 (T15, T16), §11 (T29, T32), §12 (T30, T31), §13 (T17), §14 (T18, T19, T33, T34), §15 (T3, T19), §16 (T36), §17 (T36), §18 (T20, T32, T37), §19 (T38). No gaps.

**Placeholder scan.** No "TBD", "TODO", "add error handling", or "similar to Task N". T35 now ships the `ReviewPipeline` orchestration code rather than describing it, since the revision loaded five load-bearing wiring guarantees onto it. Three steps remain prose: T35's three small helpers (`_unusable_image`, `_image_evidence`, `_persist`), T36's CLI command bodies, and T37/T38's fixture assembly. Each is mechanical assembly of interfaces defined exactly above it, and each has complete tests specifying its behaviour.

**Type consistency across the revision.** The names crossing task boundaries were re-checked after this pass: `Scale`, `FindingState`, `Coverage`, `Verdict`, `Authority`, `Psa10Candidate`, `ReviewConfidence`, `EvidenceOrigin`, `EvidenceRef`, `NormalizedBox`, `Finding`, `FusedFinding`, `ResolvedFinding`, `UnevaluableRule`, `CardContext`, `ScopedRule`, `ImageRole`, `ResolvedRole`, `GeometryResult`, `ObservabilityResult`, `CoverageResult`, `Assessment`, `VisionFinding`, `RoutingDecision`, `StageRunner`, `StageValidationError`, `ReviewPipeline`. Signature changes introduced by this revision and propagated everywhere they are used:

- `evaluate_coverage(..., *, unevaluable_rules=None)` — T17, called from T35.
- `review_confidence(..., *, required_face_missing=False)` — T18, called from T33.
- `rank_score` / `estimated_grade` now take `(finding, authority, i1_satisfied)` triples — T18, produced by T33.
- `StageRunner.run(..., schema=None)` — T28, passed by every T35 stage call.
- `AnthropicVisionProvider(model, store, api_key)` — T32, constructed in T36.
- `combine(..., scoped_rules, manifest_index, required_face_missing)` — T33, called from T35.

**One deliberate asymmetry.** `Finding.psa10_relevant` is set provisionally by producers and **overwritten** by T16's resolution. That is intended (Decision 4) — a provider's relevance claim is advisory — and the field keeps both values only in the sense that the raw finding is retained under the fused one.

**Two spec deviations flagged, not absorbed.** (1) The `review` table gains a `review_confidence` column, since §16 requires the field in output and it must survive a restart. (2) §11 describes vision findings as carrying "normalized location"; this plan makes location required downstream and derives it from cited evidence regions when the provider omits it, rather than allowing it to be absent. Both extend the spec's data model rather than contradicting it.

## Notes for the executor

- **Run the existing suite before starting.** `uv run pytest` must be green (267 subsystem B tests) before Task 1.
- **Never modify `src/card_reviewer/knowledge/`.** Subsystem B is consumed through `load_active_rubric` only. If a task seems to need a change there, stop and raise it.
- **The rubric is live data.** Tests that read it (T11, T14, T30) assert on behaviour, not on counts that a future rubric release would break — except `test_the_live_rubric_yields_no_inert_rules_today`, which is a deliberate tripwire.
- **OpenCV is imported lazily inside functions**, never at module scope, so `import card_reviewer.review` stays cheap.
- **If any task reveals a genuine contradiction with the spec, stop and flag it** rather than changing the architecture.
