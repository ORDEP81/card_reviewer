# Card Reviewer Build Plan

## Mission

Build a standalone **Sports Card Review Engine** that evaluates raw sports cards from marketplace listing photographs and estimates the likely PSA grade, especially the probability that a card will receive a PSA 10.

The system will learn card-inspection knowledge from video tutorials supplied by the user, including:

- YouTube
- Skool
- Major League Profits / MLP
- Local MP4/MOV files
- Additional grading videos added later

The system must combine:

1. Claude visual reasoning
2. OpenCV image measurements
3. A persistent grading knowledge base learned from training videos
4. Set/card-specific knowledge
5. Listing image quality assessment
6. Strict uncertainty handling

The reviewer must be **independent from card prices and profitability**.

It must NEVER receive:

- PSA 10 market price
- raw market price
- asking price
- EV
- expected profit
- buy/pass recommendation

Its only job is:

> Examine the card and determine how likely it is to receive each PSA grade.

The separate Card App will use the review result to calculate economics.

---

# 1. Development Philosophy

This is NOT initially a fine-tuned machine-learning model.

"Learning" means creating and maintaining a persistent structured knowledge base from the supplied training material.

Architecture:

```text
Training Videos
       ↓
Video Learning Pipeline
       ↓
Structured Grading Knowledge
       ↓
Card Review Skill
       ↓
OpenCV + Claude Vision
       ↓
Structured Card Review
```

The knowledge base must persist between Claude sessions.

Do not rely on Claude remembering previous conversations.

---

# 2. Mac Development Environment

First verify required dependencies.

```bash
brew --version
python3 --version
claude --version
yt-dlp --version
ffmpeg -version
```

If yt-dlp and FFmpeg are missing:

```bash
brew install yt-dlp ffmpeg
```

If Claude Code is missing:

```bash
brew install --cask claude-code
```

Python dependencies should be managed in a virtual environment.

Preferred Python dependencies:

```text
opencv-python
numpy
Pillow
pydantic
pytest
rich
typer
```

Create a `requirements.txt` or `pyproject.toml`.

Do not require Homebrew OpenCV unless actually necessary. Prefer the Python `opencv-python` package.

---

# 3. Install Claude Video Analysis

Inside Claude Code:

```text
/plugin marketplace add bradautomates/claude-video
/plugin install watch@claude-video
```

Verify `/watch` is available.

The video tool supports:

```text
/watch VIDEO_OR_URL QUESTION
```

Example:

```text
/watch lesson_video.mp4 analyze the sports-card grading techniques shown
```

Public YouTube material can generally be passed directly to `/watch`.

For long videos, do NOT perform one superficial scan and treat it as complete.

First obtain the transcript and identify relevant sections.

Then perform focused visual analysis of sections containing:

- card inspection
- centering demonstrations
- surface defects
- edge defects
- corner defects
- print defects
- card handling
- lighting techniques
- PSA examples
- comparisons between PSA grades

Use focused time ranges when appropriate.

---

# 4. Authenticated Course Videos

The user may supply training material from services to which the user legitimately has access.

Never attempt to defeat DRM, bypass payment, or obtain material the user is not authorized to access.

Where yt-dlp supports authenticated playback, the user's active browser session may be used.

Example:

```bash
yt-dlp --cookies-from-browser chrome \
  -f "bv*+ba/b" \
  "COURSE_LESSON_URL" \
  -o "training/source/%(title)s.%(ext)s"
```

Other browser names may be used if supported:

```text
chrome
brave
edge
firefox
safari
```

Do NOT permanently export browser cookies into this repository.

Do NOT commit:

```text
cookies.txt
browser session data
authentication tokens
course credentials
API keys
```

Add sensitive files to `.gitignore`.

Important:

Skool pages are not guaranteed to work directly with yt-dlp.

If authenticated extraction fails, report the failure rather than attempting to bypass platform protections.

A locally obtained video file may always be supplied to the learning pipeline.

---

# 5. Project Structure

Build approximately this structure:

```text
card-reviewer/
│
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .gitignore
│
├── src/
│   └── card_reviewer/
│       ├── __init__.py
│       ├── cli.py
│       ├── review.py
│       ├── schemas.py
│       ├── image_quality.py
│       ├── image_normalizer.py
│       ├── centering.py
│       ├── corners.py
│       ├── edges.py
│       ├── surface.py
│       ├── evidence.py
│       └── grader.py
│
├── skills/
│   └── card-review/
│       └── SKILL.md
│
├── knowledge/
│   ├── ACTIVE_RUBRIC.md
│   ├── psa/
│   │   ├── centering.md
│   │   ├── corners.md
│   │   ├── edges.md
│   │   ├── surface.md
│   │   ├── print_defects.md
│   │   └── image_limitations.md
│   ├── card_types/
│   │   ├── chrome.md
│   │   ├── refractor.md
│   │   ├── paper.md
│   │   ├── autograph.md
│   │   └── thick_cards.md
│   ├── sets/
│   │   └── README.md
│   └── pending_rules/
│
├── training/
│   ├── source/
│   ├── transcripts/
│   ├── frames/
│   ├── lessons/
│   └── manifests/
│
├── examples/
│   ├── psa10/
│   ├── psa9/
│   ├── psa8/
│   └── uncertain/
│
├── tests/
│   ├── unit/
│   ├── fixtures/
│   └── golden_cases/
│
└── output/
```

---

# 6. Video Learning Pipeline

Implement a repeatable learning workflow.

Every training video should generate a lesson record.

Example:

```text
training/lessons/lesson_001.md
```

The lesson record should contain:

```text
SOURCE
TITLE
DATE PROCESSED
FILE HASH
DURATION

TOPICS
- centering
- surface
- corners
- edges

RULES TAUGHT

VISUAL EXAMPLES

DEFECTS SHOWN

PSA GRADE EXAMPLES

INSTRUCTOR OPINIONS

OBJECTIVE OBSERVATIONS

POTENTIAL CONTRADICTIONS

SOURCE TIMESTAMPS

CANDIDATE KNOWLEDGE-BASE UPDATES
```

Do not treat everything an instructor says as fact.

Classify learned information as:

```text
OBJECTIVE
EXPERIENCE-BASED
OPINION
UNVERIFIED
CONTRADICTED
```

Preserve provenance.

Example:

```yaml
rule:
  id: SURFACE_PRINT_LINE_001
  category: surface
  description: Vertical print lines may prevent gem-mint grading.
  confidence: high
  evidence_type: experience_based
  sources:
    - lesson_014
    - lesson_028
```

Do not silently overwrite existing rules.

New knowledge enters:

```text
knowledge/pending_rules/
```

before being promoted to:

```text
knowledge/ACTIVE_RUBRIC.md
```

This makes the reviewer auditable.

---

# 7. Video Processing Strategy

For each video:

## Pass 1 — Transcript

Determine:

- what the lesson covers
- relevant grading sections
- timestamps containing visual demonstrations

## Pass 2 — Focused visual analysis

Inspect frames around relevant timestamps.

Pay particular attention when the instructor:

- rotates a card under light
- zooms into a surface
- compares two cards
- points at damage
- demonstrates centering
- examines corners
- discusses print lines
- shows PSA slabs
- explains why a card received PSA 9 instead of PSA 10

## Pass 3 — Extract knowledge

Produce the structured lesson file.

## Pass 4 — Compare against existing knowledge

Identify:

```text
new rule
supporting evidence
contradiction
duplicate knowledge
set-specific rule
instructor opinion
```

## Pass 5 — Update pending knowledge

Never delete provenance.

---

# 8. Card Image Input

The review engine should accept:

```json
{
  "card": {
    "year": "2025-26",
    "manufacturer": "Topps",
    "set": "Topps Chrome Basketball",
    "player": "Cooper Flagg",
    "card_number": "251",
    "parallel": "Red Refractor",
    "serial_number": "7/10"
  },
  "images": [
    "front.jpg",
    "back.jpg"
  ]
}
```

Images may eventually originate from an eBay listing.

The card reviewer should NOT be responsible for pricing or eBay business logic.

The main application should obtain the highest-resolution listing images available and hand them to this engine.

Create adapters later if required.

---

# 9. Image Preprocessing with OpenCV

OpenCV should provide OBJECTIVE measurements and enhanced inspection views.

Claude provides judgment.

Do not make OpenCV itself decide the PSA grade.

Pipeline:

```text
Original Image
      ↓
Image Quality Check
      ↓
Card Boundary Detection
      ↓
Perspective Correction
      ↓
Normalized Card Image
      ↓
Regions of Interest
      ↓
Measurements
      ↓
Claude Visual Inspection
```

---

# 10. Image Quality Analysis

Before grading anything, calculate whether the photographs are usable.

Measure:

## Resolution

Record width and height.

## Blur

Use Laplacian variance or similar sharpness measurement.

## Glare

Estimate clipped/highlight regions.

## Occlusion

Detect or ask Claude whether:

- card is in sleeve
- card is inside top loader
- card is inside one-touch
- fingers cover portions
- stickers obscure card
- glare obscures surface

## Perspective

Estimate card rotation and keystone distortion.

## Missing views

Detect:

```text
front missing
back missing
corners not visible
edges obstructed
surface impossible to inspect
```

Return an overall:

```text
image_quality_score
```

and:

```text
review_confidence
```

Never turn bad photographs into false certainty.

---

# 11. Card Boundary Detection

Implement OpenCV card detection.

Investigate:

- grayscale
- Gaussian blur
- Canny edge detection
- contour detection
- quadrilateral approximation
- perspective transform

Goal:

Find the four outer card corners.

Generate:

```text
normalized_front.jpg
normalized_back.jpg
```

with perspective corrected.

If reliable boundary detection fails:

```text
boundary_confidence = low
```

and retain the original image.

Never invent measurements.

---

# 12. Centering Engine

Centering is one area where OpenCV measurements can be very useful.

For cards with measurable borders or design reference points, calculate:

```text
left
right
top
bottom
```

and estimated ratios:

```text
LR = 53/47
TB = 51/49
```

However, many modern cards are:

```text
borderless
asymmetrical
full bleed
intentionally off-center in design
```

Therefore implement TWO centering modes.

## Geometric centering

Used when actual borders/design landmarks can be detected.

## Visual centering

Claude evaluates the intended card design.

Eventually support:

```text
knowledge/sets/<set>/<card-or-template>.json
```

so cards from known sets can have layout templates.

Never penalize an intentionally asymmetric design.

---

# 13. Corner Analysis

Generate detailed crops:

```text
front_top_left
front_top_right
front_bottom_left
front_bottom_right

back_top_left
back_top_right
back_bottom_left
back_bottom_right
```

Claude should inspect for:

```text
whitening
softness
rounding
fraying
compression
bending
peeling
foil damage
physical deformation
```

Return individual findings for every visible corner.

---

# 14. Edge Analysis

Generate edge strips:

```text
front_left
front_right
front_top
front_bottom

back_left
back_right
back_top
back_bottom
```

Inspect for:

```text
whitening
chipping
rough factory cut
foil separation
indentations
peeling
edge wear
damage
```

OpenCV may highlight high-contrast anomalies but those findings are only evidence.

Claude must determine whether they appear to be actual damage.

---

# 15. Surface Analysis

Surface inspection is particularly difficult from eBay photographs.

Analyze for visible:

```text
scratches
print lines
roller marks
dimples
indentations
stains
spots
refractor lines
foil defects
print defects
creases
surface damage
```

Enhancement techniques may include:

```text
contrast enhancement
CLAHE
sharpened view
grayscale view
edge-highlight view
multiple crops
```

Never "enhance" an image in a way that creates fictional defects.

Preserve the original image alongside every enhanced image.

For every suspected defect return:

```text
type
location
severity
confidence
source_image
```

---

# 16. Defect Map

Use normalized coordinates.

Example:

```json
{
  "type": "possible_print_line",
  "side": "front",
  "location": {
    "x": 0.72,
    "y": 0.44
  },
  "severity": "minor",
  "confidence": 0.71
}
```

This will eventually let the Card App display markers over the listing image.

---

# 17. Claude Review Process

For each card, Claude receives:

```text
original images
normalized images
corner crops
edge crops
surface-enhanced views
OpenCV centering measurements
image quality measurements
card metadata
relevant active grading knowledge
```

Claude should follow this sequence:

```text
1. Determine image limitations.
2. Inspect centering.
3. Inspect corners.
4. Inspect edges.
5. Inspect surface.
6. Identify defects.
7. Determine which defects are definite.
8. Determine which defects are merely possible.
9. Estimate grade probabilities.
10. Explain the grade-limiting evidence.
```

Do not begin with:

> "Does this look like a PSA 10?"

Instead behave adversarially:

> "Find every visible reason this card might NOT receive PSA 10."

This reduces optimism bias.

---

# 18. Required Grading Output

Use a strict structured schema.

Example:

```json
{
  "estimated_grade": 9,
  "grade_probabilities": {
    "PSA_10": 0.54,
    "PSA_9": 0.34,
    "PSA_8": 0.09,
    "PSA_7_or_lower": 0.03
  },
  "psa10_probability": 0.54,
  "review_confidence": 0.78,
  "image_quality": {
    "overall": 0.82,
    "front": 0.90,
    "back": 0.74
  },
  "centering": {
    "score": 9.3,
    "lr_ratio": "52/48",
    "tb_ratio": "54/46",
    "confidence": 0.91,
    "issues": []
  },
  "corners": {
    "score": 9.1,
    "confidence": 0.84,
    "issues": []
  },
  "edges": {
    "score": 8.7,
    "confidence": 0.80,
    "issues": [
      {
        "description": "Possible whitening on lower-right back edge",
        "confidence": 0.67
      }
    ]
  },
  "surface": {
    "score": 8.2,
    "confidence": 0.62,
    "issues": [
      {
        "description": "Possible vertical print line on front",
        "confidence": 0.58
      }
    ]
  },
  "grade_limiters": [
    "Possible front print line",
    "Possible back edge whitening"
  ],
  "missing_evidence": [
    "No angled-light surface photograph"
  ],
  "recommended_additional_photos": [
    "Front under angled lighting",
    "Close-up of lower-right back edge"
  ]
}
```

---

# 19. Estimated Grade vs Confidence

Do not confuse:

```text
PSA10 probability
```

with:

```text
confidence in our evaluation
```

Example:

```text
PSA10 probability: 74%
review confidence: 42%
```

because the photos are poor.

Those are different concepts and must remain separate.

---

# 20. Ungradable State

The system must be allowed to say:

```text
UNGRADABLE_FROM_LISTING
```

Examples:

```text
only front provided
card photographed through scratched top loader
severe glare
resolution insufficient
back missing
corners cropped from photograph
surface cannot be evaluated
```

Never fabricate an accurate grade from inadequate evidence.

---

# 21. Human-Readable Report

Alongside JSON, produce something like:

```text
2025-26 Topps Chrome Cooper Flagg Red Refractor /10

ESTIMATED GRADE
PSA 9

PSA 10 PROBABILITY
54%

REVIEW CONFIDENCE
78%

CENTERING
52/48 L-R
54/46 T-B
No obvious centering concern.

CORNERS
No definite defects visible.

EDGES
Possible whitening on lower-right back edge.

SURFACE
Possible vertical print line on front.
Surface inspection limited by seller lighting.

MAIN PSA 10 RISKS
1. Possible print line.
2. Possible back-edge whitening.

ADDITIONAL PHOTOS NEEDED
Front under angled lighting.
Close-up of lower-right back edge.
```

---

# 22. Card Review Skill

Create:

```text
skills/card-review/SKILL.md
```

Its purpose is to make Claude consistently operate the review engine.

The skill must instruct Claude to:

```text
- load relevant grading knowledge
- preprocess images
- run OpenCV measurements
- inspect generated crops
- look aggressively for defects
- distinguish certain vs possible defects
- consider photography limitations
- produce structured output
- never consider card value
- never recommend buying or selling
```

The skill should call the underlying Python utilities instead of duplicating algorithms inside `SKILL.md`.

---

# 23. CLI

Create a CLI such as:

```bash
card-review review \
  --front front.jpg \
  --back back.jpg \
  --metadata card.json
```

Output:

```text
output/<review-id>/
    original/
    normalized/
    corners/
    edges/
    surface/
    opencv_metrics.json
    review.json
    report.md
```

Also eventually support:

```bash
card-review learn-video lesson_video.mp4
```

but the first implementation can use `/watch` interactively for knowledge extraction.

---

# 24. eBay Integration Boundary

Do NOT tightly couple the reviewer to eBay.

Define an interface:

```text
review_card(
    card_metadata,
    image_sources
)
```

The Card App will eventually retrieve the highest-resolution marketplace images and call this interface.

This keeps the reviewer usable for:

```text
eBay
Flippah
Whatnot
Facebook
uploaded photographs
card-show photographs
```

---

# 25. Testing Strategy

Testing is mandatory.

Create golden examples where the true PSA grade is known.

Store:

```text
listing/raw images
actual PSA grade
review result
```

Track:

```text
PSA10 probability
actual outcome
image confidence
predicted grade
actual grade
false-positive PSA10 predictions
false-negative PSA10 predictions
```

The most important failure:

```text
Reviewer strongly predicts PSA10
but card receives PSA8/9.
```

Investigate these cases aggressively.

---

# 26. Calibration

Do not assume:

```text
80% predicted PSA10
```

actually means:

```text
80% gem rate.
```

Once enough real submissions exist, create calibration buckets:

```text
Predicted      Actual PSA10 Rate

0-20%
20-40%
40-60%
60-80%
80-100%
```

Use real results to calibrate probabilities.

Store every submitted card with:

```text
original images
review
predicted probability
predicted grade
actual PSA grade
```

This dataset will eventually be more valuable than the initial training videos.

---

# 27. Knowledge Versioning

Every grading review must identify which rubric generated it.

Example:

```json
{
  "rubric_version": "0.4.2"
}
```

Never alter old results when grading knowledge changes.

This makes it possible to determine whether reviewer versions improve.

---

# 28. Initial Development Phases

Build in this order.

## Phase 1 — Skeleton

Create:

```text
repository structure
Python package
schemas
CLI
tests
knowledge directories
```

## Phase 2 — OpenCV preprocessing

Implement:

```text
image validation
card boundary detection
perspective correction
image quality
corner crops
edge crops
surface views
```

## Phase 3 — Centering

Implement measurable centering and confidence.

## Phase 4 — Review schema

Implement strict Pydantic models for card-review output.

## Phase 5 — Claude card-review skill

Create `SKILL.md` and connect it to preprocessing.

## Phase 6 — Video learning

Install `/watch`.

Process several user-supplied grading videos.

Create initial knowledge base.

## Phase 7 — Golden test cards

Test known PSA 10, PSA 9 and PSA 8 cards.

## Phase 8 — Improve defect identification

Use failures to refine:

```text
surface
edges
corners
centering
image confidence
```

## Phase 9 — Application contract

Expose a stable function/API that the Card App can call.

---

# 29. Definition of MVP

The MVP is complete when I can provide:

```text
front.jpg
back.jpg
card.json
```

and receive:

```text
estimated PSA grade
PSA10 probability
probabilities for lower grades
review confidence
centering measurements
corner findings
edge findings
surface findings
defect list
image limitations
recommended additional photographs
JSON output
human-readable report
```

The MVP must NOT require the pricing application.

---

# 30. Non-Negotiable Rules

1. Never use card value when judging condition.
2. Never automatically assume PSA 10.
3. Search for reasons a card will NOT gem.
4. Never hide image limitations.
5. OpenCV measurements are evidence, not the final grader.
6. Claude visual assessment is evidence, not objective measurement.
7. Clearly distinguish definite defects from suspected defects.
8. Preserve original images.
9. Preserve knowledge provenance.
10. Preserve grading-rubric version.
11. Do not blindly learn every claim made in a training video.
12. Do not claim certainty that seller photographs cannot support.
13. Do not attempt to circumvent DRM or access training material the user is not authorized to access.
14. Do not introduce pricing, EV, buying, selling or market-value logic into this repository.

---

# 31. First Task

Begin by:

1. Inspecting the current project directory.
2. Creating the proposed repository structure.
3. Creating `CLAUDE.md` describing the architecture and non-negotiable rules.
4. Creating the Python environment.
5. Installing required Python dependencies.
6. Creating the Pydantic schemas for card metadata and review output.
7. Building the image-preprocessing/OpenCV pipeline.
8. Creating unit tests.
9. Creating the initial card-review `SKILL.md`.
10. Creating documentation explaining how to ingest the first training video.

Do not attempt to implement every advanced feature in one uncontrolled pass.

Build Phase 1 and Phase 2 with tests first, verify them, and then proceed through the phases sequentially.

At the end of each phase:

```text
run tests
document what works
document known limitations
commit the phase
continue
```

The objective is a reliable and auditable grader, not merely an impressive demo.
