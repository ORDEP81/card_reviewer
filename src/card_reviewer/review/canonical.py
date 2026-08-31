"""Canonical serialization for fingerprinting (spec §4).

Two properties this module exists to guarantee:

1. **Precision is semantic, not positional.** Each value is quantized by the
   precision its *meaning* declares, resolved from the tail of its field
   path — so a centering ratio quantizes the same way whether it arrives at
   the root or nested three stages deep under `assembled_evidence`.
2. **Nothing enters a cache key by accident.** An unregistered float, a
   non-string dict key, or a type JSON cannot represent is an error, not
   something silently coerced. A fingerprint that quietly stringifies an
   object is a cache that reuses results it should not.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .versions import CANON_SCHEME_VERSION

__all__ = [
    "CANON_SCHEME_VERSION",
    "PRECISION_MAP",
    "canonicalize",
    "precision_for",
    "quantize",
]

#: Declared semantic precision, keyed by the SUFFIX of a field path. Real
#: payloads wrap values under stage names, list indices and model fields, so
#: matching an absolute path would mean re-registering the same meaning at
#: every depth it can appear. Longest matching suffix wins.
PRECISION_MAP: dict[str, float] = {
    # Centering is reported to +/-1.5 percentage points, so that IS the
    # semantic precision. A finer step preserves distinctions the method
    # cannot support and manufactures cache misses between two readings that
    # mean the same thing.
    "centering.horizontal": 1.5,
    "centering.vertical": 1.5,
    # Confidences are compared against coarse thresholds, never summed.
    "confidence": 0.01,
    # Normalized card coordinates, wherever they appear — EvidenceRef.region
    # and Finding.location are both NormalizedBox. 1e-3 of a card edge is
    # roughly a pixel at typical resolutions.
    "x0": 1e-3,
    "y0": 1e-3,
    "x1": 1e-3,
    "y1": 1e-3,
    # Pixel-space measurements are integers in practice.
    "border_px": 1.0,
}

#: Non-semantic fields: they describe the run, not the evidence, so they must
#: never make identical work look different.
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


def precision_for(field_path: str) -> float:
    """The declared precision for a path, by longest matching suffix.

    Raises when nothing is registered: spec §4 says precision is *declared*,
    and a generic fallback invents one for a value nobody reasoned about.
    """
    parts = field_path.split(".")
    for start in range(len(parts)):
        suffix = ".".join(parts[start:])
        if suffix in PRECISION_MAP:
            return PRECISION_MAP[suffix]
    raise ValueError(
        f"no declared precision for float field {field_path!r} — register its "
        "semantic precision in PRECISION_MAP rather than letting a generic "
        "default decide what counts as the same measurement"
    )


def quantize(field_path: str, value: float) -> float:
    step = precision_for(field_path)
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
    if node is None or isinstance(node, (str, int)):
        return node
    raise TypeError(
        f"canonicalize cannot represent {type(node).__name__} at "
        f"{path or '<root>'}. Stringifying it would put an unstable or "
        "colliding value into a cache key; convert it to a JSON type first."
    )


def canonicalize(obj: Any) -> str:
    """Deterministic JSON: sorted keys, quantized floats, no excluded fields.

    Deliberately no `default=` fallback — an unsupported type must fail here
    rather than reach a fingerprint as its repr.
    """
    return json.dumps(
        _walk(obj, ""), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
