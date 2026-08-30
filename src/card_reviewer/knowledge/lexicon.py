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


class LexiconError(Exception):
    """The segmentation lexicon file is missing required structure.

    Raised by `load` when the YAML has no `categories` key, or a category's
    value isn't a mapping of term -> weight — either shape would otherwise
    load "successfully" and silently score every cue 0.0.
    """


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
    # Keyed by (category, term), not term alone: the same term appearing in
    # two categories must not share one compiled pattern/weight, or both
    # categories get credited from whichever category's entry was inserted
    # last.
    _patterns: dict[tuple[str, str], tuple[re.Pattern, float]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Compile regex patterns once at initialization for performance."""
        for category, terms in self.categories.items():
            for term, weight in terms.items():
                # Compile word-boundary regex for each term
                pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
                self._patterns[(category, term)] = (pattern, float(weight))

    def score(self, text: str) -> CueScore:
        total = 0.0
        matched: list[str] = []
        hit_categories: list[str] = []
        visual = False

        for category, terms in self.categories.items():
            category_hit = False
            for term in terms.keys():
                pattern, weight = self._patterns[(category, term)]
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
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if "categories" not in data:
        raise LexiconError(f"{path}: lexicon file is missing a 'categories' key")
    categories = data["categories"]
    for name, terms in categories.items():
        if not isinstance(terms, dict):
            raise LexiconError(
                f"{path}: category {name!r} must map terms to weights, "
                f"got {type(terms).__name__}"
            )
    return Lexicon(
        version=str(data.get("version", "0")),
        categories=categories,
    )
