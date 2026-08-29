"""Load the segmentation lexicon and score a single transcript cue.

Scoring is deliberately simple and inspectable: phrase matching with weights.
A cue's score answers one question — how likely is it that the instructor is
inspecting a card right now?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEMONSTRATION = "demonstration"


@dataclass
class CueScore:
    score: float = 0.0
    categories: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    visual_cue: bool = False


@dataclass
class Lexicon:
    version: str
    categories: dict[str, dict[str, float]]
    _patterns: dict[str, tuple[re.Pattern, float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile regex patterns once at initialization for performance."""
        for category, terms in self.categories.items():
            for term, weight in terms.items():
                # Compile word-boundary regex for each term
                pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
                self._patterns[term] = (pattern, float(weight))

    def score(self, text: str) -> CueScore:
        total = 0.0
        matched: list[str] = []
        hit_categories: list[str] = []
        visual = False

        for category, terms in self.categories.items():
            category_hit = False
            for term in terms.keys():
                pattern, weight = self._patterns[term]
                if pattern.search(text):
                    # Counted once per cue: repetition does not add information.
                    total += weight
                    matched.append(term)
                    category_hit = True
            if not category_hit:
                continue
            if category == DEMONSTRATION:
                visual = True
            else:
                hit_categories.append(category)

        return CueScore(
            score=total,
            categories=hit_categories,
            matched_terms=matched,
            visual_cue=visual,
        )


def load(path: Path | str) -> Lexicon:
    data = yaml.safe_load(Path(path).read_text())
    return Lexicon(
        version=str(data.get("version", "0")),
        categories=data.get("categories", {}),
    )
