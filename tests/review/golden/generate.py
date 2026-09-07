"""Regenerate the golden fixtures and their expectations.

    uv run python tests/review/golden/generate.py

Committed for two reasons the mutation review made concrete. Without it,
`synthetic: true` is an unverifiable claim about eight opaque PNGs, and the
fixtures cannot be regenerated when a producer legitimately changes — which
leaves the choice between hand-editing expectations and deleting the set.

The expectations are DERIVED FROM THE CURRENT IMPLEMENTATION. That makes this
a regression baseline, not ground truth: it detects change, and it cannot
detect that today's behaviour is wrong. Every value written here should be
read against the reasoning in the module that produces it, and the whole set
should be replaced by real photographs with human-checked observations as
soon as any exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from card_reviewer.review.imaging.geometry import analyze as geom  # noqa: E402
from card_reviewer.review.imaging.measure.centering import (  # noqa: E402
    measure_centering,
)
from card_reviewer.review.imaging.observability import analyze as obs  # noqa: E402
from card_reviewer.review.imaging.preflight import analyze as pre  # noqa: E402
from card_reviewer.review.imaging.synthetic import (  # noqa: E402
    CardSpec, achieved_centering, render_png,
)
from card_reviewer.review.storage.artifacts import ArtifactStore  # noqa: E402

HERE = Path(__file__).parent

#: Each case exists to drive a DIFFERENT path through the policies. The
#: comment on each is the reason it is in the set; a case that stops
#: distinguishing itself from its neighbours has stopped earning its place.
CASES: dict[str, CardSpec] = {
    # The common modern base card: a white border makes whitening
    # structurally invisible while leaving everything else assessable.
    "white_border_clean.png": CardSpec(border_color=(255, 255, 255)),
    # The control for that: a dark border CAN show whitening.
    "dark_border_clean.png": CardSpec(border_color=(20, 20, 20)),
    # No border reference at all, so centering is structurally unmeasurable.
    "borderless_chrome.png": CardSpec(borderless=True),
    # A localized specular highlight: circumstantial, and a better
    # photograph fixes it.
    "glare_heavy.png": CardSpec(glare_regions=["top_left", "bottom_right"]),
    # Warp resampling on a tilt is what used to fabricate a miscut.
    "angled_card.png": CardSpec(rotation_deg=12.0),
    # A genuine centering defect, which must survive being measured.
    "miscut.png": CardSpec(h_centering=72.0, v_centering=58.0),
    # Backs are text-heavy: busy artwork must not read as damage.
    "card_back.png": CardSpec(text_heavy=True, border_color=(255, 255, 255)),
    # Below the resolution floor: rejected before anything else runs.
    "thumbnail_too_small.png": CardSpec(card_w=120, card_h=168, border_px=8),
    # A physical surface defect, so the set is not all clean cards.
    "scratched_surface.png": CardSpec(border_color=(20, 20, 20),
                                      scratches=[1.0, 0.9]),
    # An obstruction: assessability lost for one region only.
    "occluded_corner.png": CardSpec(border_color=(20, 20, 20)),
}

#: Regions reported per case, so an expectation names the corner it is about.
REPORTED_REGIONS = ("top_left", "bottom_right")


def _occlude(data: bytes) -> bytes:
    import cv2
    import numpy as np

    from card_reviewer.review.imaging.synthetic import _draw_card, _place_on_background

    spec = CASES["occluded_corner.png"]
    card = _draw_card(spec, np.random.default_rng(spec.seed))
    card[0:260, 0:260] = 4
    return cv2.imencode(".png", _place_on_background(card, spec, cv2))[1].tobytes()


def observe(data: bytes, store: ArtifactStore) -> dict:
    """Everything the golden test asserts, for one image."""
    preflight = pre(data)
    case: dict = {
        "synthetic": True,
        "preflight_usable": preflight.usable,
    }
    if not preflight.usable:
        case["preflight_reason"] = preflight.reason_code
        return case

    image_hash = store.put_image(data)
    geometry = geom(data, store, image_hash)
    case["boundary_detected"] = geometry.boundary_confidence > 0.5
    case["has_reliable_border"] = geometry.has_reliable_border

    centering = measure_centering(geometry, store)
    case["centering_measurable"] = centering.measurable
    if centering.measurable:
        # The VALUE, not just that a value exists. Asserting only
        # `measurable: true` made a miscut fixture indistinguishable from a
        # clean one, so three of the eight cases constrained nothing.
        case["centering_horizontal"] = round(centering.horizontal, 1)
        case["centering_vertical"] = round(centering.vertical, 1)
    else:
        case["centering_reason"] = centering.reason

    observability = obs(geometry, store, image_hash)
    case["detectability"] = {
        f"{region}.{category}.{defect_type}": value.label
        for (region, category, defect_type), value
        in sorted(observability.detectability.items())
        if region in REPORTED_REGIONS
    }
    # The surface producer's actual readings. It is deliberately silent —
    # the clean and scratched populations overlap, so its thresholds sit
    # above the clean maximum — but recording the numbers keeps a scratched
    # fixture distinguishable from a clean one and gives anyone recalibrating
    # a starting point instead of a blank.
    from card_reviewer.review.imaging.geometry import load_geometry
    from card_reviewer.review.imaging.measure.surface import _local_outlier

    gray = load_geometry(geometry, store).normalized.mean(axis=2)
    case["surface_outlier"] = round(_local_outlier(gray), 1)

    case["reason_codes"] = {
        f"{region}.{category}.{defect_type}": code
        for (region, category, defect_type), code
        in sorted(observability.reason_codes.items())
        if region in REPORTED_REGIONS
    }
    return case


def main() -> None:
    import tempfile

    store = ArtifactStore(Path(tempfile.mkdtemp()))
    cases = []
    for name, spec in CASES.items():
        data = _occlude(b"") if name == "occluded_corner.png" else render_png(spec)
        (HERE / name).write_bytes(data)

        case = {"file": name}
        case.update(observe(data, store))
        if not spec.borderless and name != "thumbnail_too_small.png":
            truth = achieved_centering(spec)
            case["rendered_centering"] = [round(truth[0], 1), round(truth[1], 1)]
        cases.append(case)

    (HERE / "expectations.yaml").write_text(
        "# GENERATED by generate.py — do not hand-edit.\n"
        "#\n"
        "# Derived from the current implementation, so this is a REGRESSION\n"
        "# BASELINE and not ground truth: it detects change, and it cannot\n"
        "# detect that today's behaviour is wrong. `rendered_centering` is\n"
        "# the one exception — it comes from the generator, so a measured\n"
        "# value can be read against what was actually drawn.\n"
        + yaml.safe_dump(cases, sort_keys=False)
    )
    print(f"wrote {len(cases)} cases")


if __name__ == "__main__":
    main()
