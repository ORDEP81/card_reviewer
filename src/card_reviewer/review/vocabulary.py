"""Canonical card-type and set vocabulary.

Derived by inspecting the live subsystem B rubric, not from memory. At rubric
v4.0.0 the only product-scoped rule is SURFACE_SHINY_001, scoped to
chrome/refractor/foil; no rule is scoped by set. Adding a supported product
means editing this file and nothing else — grading logic never changes.
"""

from __future__ import annotations

from .versions import VOCABULARY_VERSION

__all__ = ["CARD_TYPE_VOCABULARY", "SET_VOCABULARY", "VOCABULARY_VERSION"]

#: alias -> canonical. Exact lookup only; see `normalize` for why.
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

#: Empty by inspection: no active rule at v4.0.0 is set-scoped. The set axis
#: exists so a future set-scoped rule has somewhere to land, and is exercised
#: by synthetic fixtures until subsystem B adds one.
SET_VOCABULARY: dict[str, str] = {}
