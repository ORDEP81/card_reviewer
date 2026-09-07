"""Adapters resolve external input. The ONLY component permitted network I/O.

A future Flippah API is a new adapter and nothing else — the grading core
never learns that HTTP exists.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Protocol

from ..models import CandidateInput, ResolvedCandidate, ResolvedImage
from ..storage.artifacts import ArtifactStore

__all__ = ["CandidateAdapter", "ManualAdapter"]


class CandidateAdapter(Protocol):
    def resolve(self, candidate: CandidateInput) -> ResolvedCandidate: ...


class ManualAdapter:
    """Local files only — never opens a socket."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def resolve(self, candidate: CandidateInput) -> ResolvedCandidate:
        images = [
            ResolvedImage(
                image_hash=self._store.put_image(path.read_bytes()),
                supplied_role=candidate.supplied_roles.get(str(path)),
                ordering=index,
            )
            for index, path in enumerate(candidate.image_paths)
        ]
        # asking_price is deliberately not carried across: this is where rule
        # 10 stops being a promise and becomes a type.
        return ResolvedCandidate(
            candidate_id=self._candidate_id(candidate),
            source=candidate.source,
            title=candidate.title,
            card_type=candidate.card_type,
            set_name=candidate.set_name,
            images=images,
        )

    @staticmethod
    def _candidate_id(candidate: CandidateInput) -> str:
        """Identity of a PHYSICAL card, which a title does not establish.

        Two different copies of the same card share a title exactly, so a
        title-derived id would merge them into one candidate and overwrite
        the first one's review history. Only a listing identity is a real
        external key; without one, mint a UUID and let the caller persist it.
        """
        if candidate.candidate_id:
            return candidate.candidate_id
        listing = candidate.listing_id or candidate.listing_url
        if listing:
            return hashlib.sha256(
                f"{candidate.source}|{listing}".encode()
            ).hexdigest()[:16]
        return f"manual-{uuid.uuid4().hex[:16]}"
