"""Application service: composes adapter, pipeline and storage.

The CLI is a thin surface over this. Provider construction lives here so
SMART and DEEP have a normal path to a configured provider — `provider-smoke`
is not the only way to reach the API.
"""

from __future__ import annotations

import os
from pathlib import Path

from .enums import Mode
from .ingest.adapter import ManualAdapter
from .models import CandidateInput, CardReview
from .pipeline import ReviewPipeline
from .storage.artifacts import ArtifactStore
from .storage.migrations import connect, migrate
from .storage.repository import SqliteRepository
from .vision.provider import VisionProvider

__all__ = ["ReviewContext", "build_provider", "open_context", "review_card"]


class ReviewContext:
    """The wired-up components for one data directory."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        conn = connect(self.data_dir / "card_reviewer.db")
        migrate(conn)
        self.repo = SqliteRepository(conn)
        self.store = ArtifactStore(self.data_dir / "artifacts")
        self.pipeline = ReviewPipeline(self.repo, self.store)


def open_context(data_dir: Path | str) -> ReviewContext:
    return ReviewContext(data_dir)


def build_provider(store: ArtifactStore) -> VisionProvider | None:
    """A configured provider, or None when no credentials are present.

    Returning None is not the same as OFF: routing still records that a call
    was wanted, and the pipeline marks the categories vision would have
    judged as unassessed rather than passing the card.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    from .vision.anthropic import AnthropicVisionProvider

    return AnthropicVisionProvider(store=store, api_key=api_key)


def review_card(
    candidate: CandidateInput,
    mode: Mode = Mode.SMART,
    data_dir: Path | str = "data",
    provider: VisionProvider | None = None,
) -> CardReview:
    context = open_context(data_dir)
    resolved = ManualAdapter(context.store).resolve(candidate)
    if provider is None and mode is not Mode.OFF:
        provider = build_provider(context.store)
    return context.pipeline.review(resolved, mode, provider)
