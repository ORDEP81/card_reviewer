"""Raw listing strings -> canonical CardContext.

Free-form text must never reach `Rubric.for_card`, which matches by exact set
intersection on lowercase strings. An unrecognized value becomes unknown,
never the nearest neighbour: guessing "Prizm Silver Mojo" into `chrome` would
apply a product-scoped rule the owner never sanctioned.
"""

from __future__ import annotations

import re

from .context import CardContext
from .enums import Provenance
from .versions import VOCABULARY_VERSION
from .vocabulary import CARD_TYPE_VOCABULARY, SET_VOCABULARY

__all__ = ["CardContextNormalizer"]

_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")

TITLE_INFERENCE_CONFIDENCE = 0.6
SUPPLIED_CONFIDENCE = 1.0


def _key(value: str) -> str:
    """Casefold, strip punctuation, collapse whitespace — then exact lookup."""
    return _WS.sub(" ", _PUNCT.sub(" ", value.casefold())).strip()


class CardContextNormalizer:
    version = VOCABULARY_VERSION

    def normalize(
        self,
        raw_title: str | None = None,
        supplied_card_type: str | None = None,
        supplied_set: str | None = None,
    ) -> CardContext:
        card_types, provenance, confidence = self._card_types(
            raw_title, supplied_card_type
        )
        return CardContext(
            raw_card_type=supplied_card_type,
            raw_set=supplied_set,
            raw_title=raw_title,
            canonical_card_types=card_types,
            canonical_sets=self._sets(supplied_set),
            provenance=provenance,
            confidence=confidence,
        )

    def _card_types(
        self, title: str | None, supplied: str | None
    ) -> tuple[list[str] | None, Provenance, float]:
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
        """Whole-alias matches only.

        A substring match would read "Chromedome" as `chrome` and scope the
        rubric on a product the listing never named.
        """
        key = _key(title)
        hits = {
            canonical
            for alias, canonical in CARD_TYPE_VOCABULARY.items()
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", key)
        }
        return sorted(hits) or None

    @staticmethod
    def _sets(supplied: str | None) -> list[str] | None:
        if not supplied:
            return None
        canonical = SET_VOCABULARY.get(_key(supplied))
        return [canonical] if canonical else None
