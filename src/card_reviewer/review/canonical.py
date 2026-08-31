"""Canonical serialization for fingerprinting (spec §4).

There are TWO canonical forms here, and conflating them is a real defect:

- `canonicalize` is for **evidence and stage inputs**. It quantizes by
  declared measurement precision, because two centering readings inside the
  same bucket are the same observation and must reuse the same result.
- `canonicalize_config` is for **producer signatures** — config, weights,
  inference parameters. These identify behaviour, not measurements, so
  floats are preserved exactly. Rounding `temperature=0.2` and
  `temperature=0.204` together would make two different configurations
  share a cache row.

Two properties both forms guarantee:

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
    "SIGNATURE_SCHEME_VERSION",
    "canonicalize",
    "canonicalize_config",
    "precision_for",
    "quantize",
]

#: Versioned independently of the evidence scheme: changing how producer
#: configuration is rendered must be a traceable invalidation of signatures
#: without disturbing evidence fingerprints, and vice versa.
SIGNATURE_SCHEME_VERSION = "1.0.0"

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


def _walk_config(node: Any, path: str) -> Any:
    """Exact rendering for producer configuration.

    Same strictness as the evidence walker — string keys, JSON-safe types
    only — but floats pass through unrounded. A configuration value is an
    identity, not a measurement, so there is no bucket it belongs to.
    """
    if isinstance(node, dict):
        bad = [k for k in node if not isinstance(k, str)]
        if bad:
            raise TypeError(
                f"canonicalize_config requires string keys — {path or '<root>'} "
                f"has {bad!r}."
            )
        return {
            k: _walk_config(v, f"{path}.{k}" if path else k)
            for k, v in sorted(node.items())
        }
    if isinstance(node, (list, tuple)):
        return [_walk_config(v, path) for v in node]
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        if not math.isfinite(node):
            raise ValueError(
                f"producer configuration must be finite — {path or '<root>'} is "
                f"{node!r}. NaN is not JSON and never equals itself, so a "
                "signature containing one could never match its own cache row."
            )
        return node
    if node is None or isinstance(node, (str, int)):
        return node
    raise TypeError(
        f"canonicalize_config cannot represent {type(node).__name__} at "
        f"{path or '<root>'}. Stringifying it would put an unstable or "
        "colliding value into a producer signature."
    )


def canonicalize_config(obj: Any) -> str:
    """Deterministic, EXACT JSON for producer signatures.

    No quantization: `PRECISION_MAP` describes what counts as the same
    observation, which says nothing about what counts as the same
    configuration.
    """
    return json.dumps(
        _walk_config(obj, ""),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
