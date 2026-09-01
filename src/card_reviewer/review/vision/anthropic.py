"""Anthropic implementation of VisionProvider.

A vision provider that sends only text is not a vision provider. The request
carries the actual bytes of the selected manifest artifacts as image content
blocks, each labelled with its artifact id so the provider's citations
resolve back to real evidence.

The SDK is imported lazily inside `_call`, so importing this module never
requires the dependency and tests can replace `_call` without any client
being constructed.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from ..storage.artifacts import ArtifactStore
from .prompt import PROMPT_VERSION, build_prompt
from .provider import Assessment, ProviderContractError, parse_assessment

__all__ = ["DEFAULT_MODEL", "AnthropicVisionProvider", "build_request"]

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096

#: Magic bytes for the formats the API accepts and we actually produce.
SUPPORTED_MEDIA = {b"\x89PNG": "image/png", b"\xff\xd8\xff": "image/jpeg"}


def _media_type(data: bytes) -> str:
    for magic, media in SUPPORTED_MEDIA.items():
        if data.startswith(magic):
            return media
    raise ProviderContractError(
        "artifact is not a supported image format (PNG or JPEG)"
    )


def build_request(manifest: dict[str, Any], store: ArtifactStore) -> list[dict]:
    """Content blocks for one request: the brief, then each artifact as a
    LABELLED image block.

    The label immediately precedes its image, so the provider can cite
    artifact_id values that resolve back through the manifest index.
    """
    blocks: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(manifest)}]
    for artifact in manifest.get("artifacts", []):
        artifact_id = artifact["artifact_id"]
        data = store.read(artifact_id)
        blocks.append({
            "type": "text",
            "text": (
                f"artifact_id={artifact_id} view={artifact.get('view')} "
                f"origin={artifact.get('origin')} "
                f"enhancement={artifact.get('enhancement')}"
            ),
        })
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64", "media_type": _media_type(data),
                "data": base64.b64encode(data).decode("ascii"),
            },
        })
    return blocks


class AnthropicVisionProvider:
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        store: ArtifactStore | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
    ) -> None:
        if store is None:
            raise ValueError(
                "AnthropicVisionProvider requires an ArtifactStore — the "
                "request carries image bytes, so there is nothing to send "
                "without one"
            )
        self.model = model
        # Omitted by default: sampling parameters are not accepted by every
        # current model, and this provider has no need to vary sampling. The
        # determinism that matters is the manifest, which is fingerprinted.
        self.temperature = temperature
        self._store = store
        self._api_key = api_key

    def assess(self, evidence_manifest: dict[str, Any]) -> Assessment:
        payload = self._call(build_request(evidence_manifest, self._store))
        allowed = {
            a["artifact_id"] for a in evidence_manifest.get("artifacts", [])
        }
        return parse_assessment(payload, allowed_artifact_ids=allowed)

    def _call(self, blocks: list[dict]) -> dict[str, Any]:
        import anthropic  # lazy: never imported at module load

        client = anthropic.Anthropic(api_key=self._api_key)
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = client.messages.create(
            model=self.model, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": blocks}], **kwargs,
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderContractError(
                f"provider returned non-JSON content: {exc}"
            ) from exc

    @property
    def inference_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"max_tokens": MAX_TOKENS}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return params

    def signature(self) -> dict[str, Any]:
        """The vision stage's producer signature.

        The pipeline reads this rather than knowing anything about Anthropic,
        so swapping providers changes the cache key without changing the
        caching code.
        """
        return {
            "provider": "anthropic", "model": self.model,
            "prompt_version": self.prompt_version,
            "inference_params": self.inference_params,
        }
