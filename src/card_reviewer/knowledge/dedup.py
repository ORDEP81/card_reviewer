"""Flag pending rules that duplicate or contradict active ones.

This module only ever raises flags. Resolution belongs to the user during
`card-knowledge review` — an automated merge would silently rewrite what the
grader believes.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .models import Rule

THRESHOLD = 0.72

NEGATIONS = (
    "not",
    "never",
    "no",
    "cannot",
    "wont",
    "doesnt",
    "dont",
    "isnt",
    "wasnt",
    "arent",
    "cant",
    "does not",
    "will not",
    "rarely",
    "unlikely",
)


@dataclass
class Flag:
    kind: str  # "duplicate" | "contradiction"
    other_id: str
    score: float
    other_statement: str


def normalize(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def is_negated(text: str) -> bool:
    words = set(normalize(text).split())
    phrase = normalize(text)
    for n in NEGATIONS:
        if " " not in n:
            # Single-word negation: check against tokenized words
            if n in words:
                return True
        else:
            # Multi-word negation: check with word boundaries
            if re.search(r"\b" + re.escape(n) + r"\b", phrase):
                return True
    return False


def flags_for(
    rule: Rule, active_rules: list[Rule], threshold: float = THRESHOLD
) -> list[Flag]:
    flags: list[Flag] = []
    for other in active_rules:
        if other.category is not rule.category or other.id == rule.id:
            continue
        score = similarity(rule.statement, other.statement)
        if score < threshold:
            continue
        kind = (
            "contradiction"
            if is_negated(rule.statement) != is_negated(other.statement)
            else "duplicate"
        )
        flags.append(Flag(kind, other.id, round(score, 3), other.statement))

    flags.sort(key=lambda f: -f.score)
    return flags
