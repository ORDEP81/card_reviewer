"""Check the engine's measurements against real, labelled photographs.

    uv run python .dev/calibrate.py

Answers one question: do the numbers the engine produces actually SEPARATE
the classes a human marked, or is a threshold fitted to a corpus that does
not resemble reality? Every threshold in the imaging layer was calibrated on
synthetic fixtures, and the first real photographs showed those fixtures
disagree with reality about the most basic property of the input.

Prints what it can measure and, more importantly, what it cannot.
"""

from __future__ import annotations

import csv
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

PHOTOS = Path(__file__).resolve().parents[3] / "training" / "photos"


def load() -> list[dict]:
    rows = list(csv.DictReader((PHOTOS / "labels.csv").open()))
    for r in rows:
        for key in ("has", "regions", "holder", "wide_border"):
            r[key] = [v for v in r.get(key, "").split("|") if v]
    return rows


def _shape(data: bytes):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img.shape[:2]


def main() -> None:
    from card_reviewer.review.imaging.geometry import analyze
    from card_reviewer.review.imaging.preflight import analyze as preflight
    from card_reviewer.review.storage.artifacts import ArtifactStore

    rows = load()
    store = ArtifactStore(Path(tempfile.mkdtemp()))
    print(f"{len(rows)} labelled photographs\n")

    # Slabs are a different object: the engine screens RAW cards.
    raw = [r for r in rows if "slab" not in r["holder"]]
    print(f"raw cards (slabs excluded): {len(raw)}\n")

    outcome = Counter()
    usable = []
    for r in raw:
        data = (PHOTOS / r["file"]).read_bytes()
        pf = preflight(data)
        if not pf.usable:
            outcome[f"preflight {pf.reason_code}"] += 1
            continue
        g = analyze(data, store, store.put_image(data))
        if not g.usable:
            outcome[f"geometry declined @{g.boundary_confidence:.2f}"] += 1
            continue
        outcome["usable"] += 1
        usable.append((r, g))

    # Every quad that is now accepted gets checked for plausibility: a card
    # is a convex quadrilateral of roughly 2.5x3.5, and a "detection" that is
    # not is worse than a refusal because it measures the wrong object
    # silently.
    from card_reviewer.review.imaging.geometry import (
        ASPECT_TOLERANCE, CARD_ASPECT, _aspect, _quad_area,
    )
    import numpy as np

    suspect = []
    for r, g in usable:
        quad = np.asarray(g.quad, dtype=float)
        w = quad[:, 0].max() - quad[:, 0].min()
        h = quad[:, 1].max() - quad[:, 1].min()
        data = (PHOTOS / r["file"]).read_bytes()
        img_h, img_w = _shape(data)
        share = _quad_area(quad) / float(img_h * img_w)
        off = abs(_aspect(w, h) - CARD_ASPECT)
        if off > ASPECT_TOLERANCE * 2 or share < 0.15:
            suspect.append((r["file"], round(_aspect(w, h), 3), round(share, 3)))

    print("what the engine does with them:")
    for k, v in outcome.most_common():
        print(f"  {v:3}  {k}")

    reachable = len(usable)
    print(f"\n{reachable} of {len(raw)} reach the measurement stages.")
    if suspect:
        print(f"\n{len(suspect)} QUESTIONABLE QUAD(S) — aspect far from a "
              "card, or a tiny share of the frame:")
        for name, asp, share in suspect:
            print(f"   {name}  aspect={asp} share={share}")
    else:
        print("no questionable quads: every accepted region is card-shaped "
              "and a plausible share of its frame.")

    lab = Counter()
    for r, _ in usable:
        for v in r["has"]:
            if v in ("glare", "photo_ok"):
                lab[v] += 1
    print(f"\nreaching measurement: {lab['glare']} labelled glare, "
          f"{lab['photo_ok']} labelled photo-ok")
    if reachable < len(raw) * 0.5:
        print("\nNOT ENOUGH TO CALIBRATE ANYTHING.")
        print("Whatever a threshold looked like on this many samples would be")
        print("fitted to whichever few photographs happened to survive, which")
        print("is the same mistake as fitting it to the synthetic corpus.")
        print("The framing assumption has to be fixed first.")
        return

    glare_check(usable, store)


def glare_check(usable, store) -> None:
    from card_reviewer.review.imaging.geometry import load_geometry
    from card_reviewer.review.imaging.observability import (
        GLARE_LUMA, REGIONS_FOR_CATEGORY, _patch,
    )

    glared, clean = [], []
    for r, g in usable:
        gray = load_geometry(g, store).normalized.mean(axis=2)
        for region in REGIONS_FOR_CATEGORY["corners"]:
            frac = float((_patch(gray, region) >= GLARE_LUMA).mean())
            marked = "glare" in r["has"] and region in r["regions"]
            (glared if marked else clean).append(frac)

    if not glared or not clean:
        print("\nno labelled glare regions among the usable photographs.")
        return
    print("\nclipped fraction, per corner region:")
    print(f"  labelled GLARED  n={len(glared):3} "
          f"min={min(glared):.3f} median={statistics.median(glared):.3f}")
    print(f"  not glared       n={len(clean):3} "
          f"max={max(clean):.3f} median={statistics.median(clean):.3f}")
    print("\n  SEPARATES" if min(glared) > max(clean)
          else "\n  OVERLAPS — no threshold on this measure divides them")


if __name__ == "__main__":
    sys.exit(main())
