"""Versioned prompt construction.

Adversarial brief, conservative evidence standard. No pricing, EV, profit or
purchase information ever reaches the prompt (non-negotiable rule 10).
"""

from __future__ import annotations

import json
from typing import Any

from ..versions import VERSIONS

__all__ = ["PROMPT_VERSION", "build_prompt"]

PROMPT_VERSION = "1.0.0"

_BRIEF = """You are assessing a raw trading card from photographs, to help decide
whether it has a realistic chance of grading PSA 10.

Find every visible reason this card might not receive a 10. Search aggressively.

But conclude conservatively. Each finding must carry a state:
  observed        - visible and confidently a real feature of the card
  suspected       - visible but could be glare, design, or artifact
  not_observed    - you looked, evidence was adequate, it is not present
  not_assessable  - the evidence was insufficient to look

A wrongly confirmed defect is the expensive error here. A suspected print
line stays suspected.

Do not re-measure centering. It is measured for you and supplied below. Your
job is what measurement cannot do: chipping versus glare, soft corners,
print lines, dimples, stains, foil and refractor artifacts, and whether a
suspected defect is part of the card's design.

An anomaly candidate marked "visible_in_original": false was surfaced only by
image enhancement. You may report it, but it cannot on its own establish a
confirmed defect — say `suspected` unless you can see it in an unenhanced
view.

For EVERY finding, cite the artifact_id values you relied on. Cite only ids
that appear in the artifact list below. Give the finding's normalized
location as {"x0","y0","x1","y1"} in card coordinates, and its severity as
one of minor / moderate / severe.

For EVERY category (centering, corners, edges, surface) state whether you
could assess it at all. "I could not judge the surface" is a first-class
answer and must not be omitted.

Finally give an independent gem view, one of: no_visible_psa10_disqualifier,
possible_psa10_disqualifier, visible_psa10_disqualifier, insufficient_evidence.

Respond with JSON only.
"""

#: Every section of the canonical payload is rendered. Silently omitting one
#: degrades the provider's answers while still looking like a working
#: integration — an invisible failure, so it is asserted in tests.
_SECTIONS = (
    ("Artifacts (cite these artifact_id values)", "artifacts", []),
    ("Measurements already taken", "measurements", {}),
    ("Detectability per region and defect type", "detectability", {}),
    ("Why detectability is limited (reason codes)", "detectability_reasons", {}),
    ("Image quality limitations", "image_limitations", []),
    ("Conflicts between photographs", "conflicts", []),
    ("Anomaly candidates (with enhancement provenance)", "anomaly_candidates", []),
    ("Applicable grading rules", "rubric_rules", []),
)


def build_prompt(manifest: dict[str, Any]) -> str:
    parts = [_BRIEF]
    for heading, key, default in _SECTIONS:
        parts.append(
            f"{heading}:\n{json.dumps(manifest.get(key, default), indent=2)}"
        )
    return "\n\n".join(parts) + "\n"
