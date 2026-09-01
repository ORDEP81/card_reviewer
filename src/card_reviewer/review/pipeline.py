"""Stage execution with content-addressed caching.

Cache identity is (stage, input_fingerprint, producer_signature). A row
exists ONLY for an output that ran to completion AND passed schema
validation. Failures — exceptions, timeouts, malformed provider responses,
schema violations — are recorded as attempts and can never satisfy a lookup.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .fingerprint import fingerprint, signature_for
from .storage.repository import Repository

__all__ = ["StageRunner", "StageValidationError"]


class StageValidationError(Exception):
    """A stage produced output that does not match its declared schema."""


class StageRunner:
    def __init__(self, repo: Repository) -> None:
        self._repo = repo

    def run_with_id(
        self,
        stage: str,
        inputs: dict[str, Any],
        versions: dict[str, Any],
        compute: Callable[[], dict[str, Any]],
        *,
        schema: type[BaseModel] | None = None,
        image_hash: str | None = None,
        candidate_id: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Run or reuse a stage, returning its output and its row id.

        `review` carries foreign keys to the exact `stage_result` rows that
        produced it, so the id travels with the output rather than being
        re-derived from the cache key afterwards.
        """
        fp = fingerprint(inputs)
        # Raises on an unknown stage or a missing version key: an omitted
        # version would silently make two different implementations look
        # identical, which is worse than a hard failure.
        sig = signature_for(stage, versions)

        cached = self._repo.get_stage_result(stage, fp, sig)
        if cached is not None:
            return cached.output, cached.id

        try:
            output = compute()
        except Exception as exc:
            self._repo.record_attempt(
                stage, fp, sig, error_kind=type(exc).__name__,
                error_detail=str(exc), image_hash=image_hash,
                candidate_id=candidate_id,
            )
            raise

        # "Validated successes only" is enforced here, not assumed. An output
        # that does not match its schema is a failure however cleanly the
        # stage returned it — caching it would poison every later run.
        if schema is not None:
            try:
                schema.model_validate(output)
            except ValidationError as exc:
                self._repo.record_attempt(
                    stage, fp, sig, error_kind="StageValidationError",
                    error_detail=str(exc), image_hash=image_hash,
                    candidate_id=candidate_id,
                )
                raise StageValidationError(
                    f"stage {stage!r} output failed validation: {exc}"
                ) from exc

        row_id = self._repo.put_stage_result(
            stage, fp, sig, output, versions, image_hash=image_hash,
            candidate_id=candidate_id,
        )
        return output, row_id

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """The output alone, for stages whose row id nothing references."""
        return self.run_with_id(*args, **kwargs)[0]
