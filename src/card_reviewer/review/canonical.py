"""Canonical serialization for fingerprinting (spec §4).

There is no single global float precision. Each value is quantized by its
own declared semantic precision before serialization, under a versioned
scheme whose version participates in every fingerprint.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .versions import CANON_SCHEME_VERSION

__all__ = ["CANON_SCHEME_VERSION", "PRECISION_MAP", "canonicalize", "quantize"]

DEFAULT_PRECISION = 1e-4

PRECISION_MAP: dict[str, float] = {
    # Centering is reported to +/-1.5 percentage points, so that IS the
    # semantic precision. A finer step (the plan suggested 0.5) preserves
    # distinctions the method cannot support and manufactures cache misses
    # between two readings that mean the same thing — precisely what spec §4
    # says per-field quantization exists to prevent.
    "centering.horizontal": 1.5,
    "centering.vertical": 1.5,
    # Normalized coordinates drive crop extraction; 1e-3 of a card edge is
    # roughly a pixel at typical resolutions.
    "region.x0": 1e-3,
    "region.y0": 1e-3,
    "region.x1": 1e-3,
    "region.y1": 1e-3,
    # Confidences are compared against coarse thresholds, never summed.
    "confidence": 0.01,
    # Pixel-space measurements are integers in practice.
    "border_px": 1.0,
}

EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "computed_at",
        "elapsed_ms",
        "latency_ms",
        "created_at",
        "updated_at",
        "cost_usd",
        "request_id",
    }
)


def quantize(field_path: str, value: float) -> float:
    step = PRECISION_MAP.get(field_path, DEFAULT_PRECISION)
    if step <= 0:
        raise ValueError(f"precision for {field_path!r} must be positive")
    return math.floor(value / step + 0.5) * step


def _walk(node: Any, path: str) -> Any:
    if isinstance(node, dict):
        bad = [k for k in node if not isinstance(k, str)]
        if bad:
            raise TypeError(
                f"canonicalize requires string keys — {path or '<root>'} has "
                f"{bad!r}. Cached stage outputs are stored as JSON, so a "
                "tuple-keyed dict must be flattened by its own model first."
            )
        return {
            k: _walk(v, f"{path}.{k}" if path else k)
            for k, v in sorted(node.items())
            if k not in EXCLUDED_KEYS
        }
    if isinstance(node, (list, tuple)):
        return [_walk(v, path) for v in node]
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        return round(quantize(path, node), 12)
    return node


def canonicalize(obj: Any) -> str:
    """Deterministic JSON: sorted keys, quantized floats, no excluded fields."""
    return json.dumps(
        _walk(obj, ""),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
