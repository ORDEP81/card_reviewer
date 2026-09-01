"""Resolved card context: raw and canonical, side by side."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import Provenance


class CardContext(BaseModel):
    """What the listing said, and what the rubric can be scoped on.

    Both are kept. The raw value is what a human recognises in a report; the
    canonical value is the only thing subsystem B's exact-match scoping may
    ever see.
    """

    raw_card_type: str | None = None
    raw_set: str | None = None
    raw_title: str | None = None
    # None means unconstrained. NEVER [] — subsystem B reads an explicit empty
    # list as "known to be empty", a real constraint that would drop every
    # scoped rule and silently narrow the rubric.
    canonical_card_types: list[str] | None = None
    canonical_sets: list[str] | None = None
    provenance: Provenance = Provenance.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}

    @property
    def is_known(self) -> bool:
        return self.canonical_card_types is not None
