"""Does a measurement separate the classes a human labelled?

    uv run python .dev/separate.py

Per PHOTO, not per region: `regions` in the label schema is shared across all
of a card's defects, so a region marked for corner wear cannot be told apart
from one marked for glare. Comparing whole photographs sidesteps that.

Reports, for each candidate measure, whether the two labelled populations
actually separate — and refuses to suggest a threshold when they do not.
"""

from __future__ import annotations

import csv
import statistics
import tempfile
from pathlib import Path

import numpy as np

PHOTOS = Path(__file__).resolve().parents[3] / "training" / "photos"


def rows():
    for r in csv.DictReader((PHOTOS / "labels.csv").open()):
        r["has"] = [v for v in r["has"].split("|") if v]
        r["holder"] = [v for v in r["holder"].split("|") if v]
        yield r


def report(name, positive, negative, higher_is_worse=True):
    if len(positive) < 4 or len(negative) < 4:
        print(f"\n{name}: too few samples ({len(positive)} vs {len(negative)})")
        return
    p, n = sorted(positive), sorted(negative)
    print(f"\n{name}")
    print(f"  labelled   n={len(p):3}  min={p[0]:8.3f} median={statistics.median(p):8.3f} max={p[-1]:8.3f}")
    print(f"  not        n={len(n):3}  min={n[0]:8.3f} median={statistics.median(n):8.3f} max={n[-1]:8.3f}")
    if higher_is_worse:
        clean_sep = min(p) > max(n)
        # How many of the labelled class clear the clean maximum?
        caught = sum(1 for x in p if x > max(n))
        false_at_best = 0
    else:
        clean_sep = max(p) < min(n)
        caught = sum(1 for x in p if x < min(n))
    overlap = statistics.median(p) == statistics.median(n)
    verdict = ("SEPARATES" if clean_sep else
               f"overlaps — {caught}/{len(p)} clear the clean extreme "
               f"({caught/len(p):.0%} recall at 0% false positives)")
    print(f"  -> {verdict}")


def main():
    from card_reviewer.review.imaging.geometry import analyze, load_geometry
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import (
        GLARE_LUMA, REGIONS_FOR_CATEGORY, _patch,
    )
    from card_reviewer.review.imaging.measure.surface import _local_outlier
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(Path(tempfile.mkdtemp()))
    measured = []
    for r in rows():
        if "slab" in r["holder"]:
            continue
        data = (PHOTOS / r["file"]).read_bytes()
        g = analyze(data, store, store.put_image(data))
        if not g.usable:
            continue
        gray = load_geometry(g, store).normalized.mean(axis=2)
        corners = REGIONS_FOR_CATEGORY["corners"]
        clip = [float((_patch(gray, c) >= GLARE_LUMA).mean()) for c in corners]
        cv = measure_all(g, store, store.put_image(data))
        measured.append({
            "row": r,
            "max_clip": max(clip),
            "clip_spread": max(clip) - float(np.median(clip)),
            "surface_outlier": _local_outlier(gray),
            "corner_anoms": len([a for a in cv.anomalies
                                 if a["category"] == "corners"]),
            "corner_worst": max([a["contrast"] for a in cv.anomalies
                                 if a["category"] == "corners"], default=0.0),
        })

    print(f"{len(measured)} photographs measured\n" + "=" * 62)

    def split(key, metric):
        pos = [m[metric] for m in measured if key in m["row"]["has"]]
        neg = [m[metric] for m in measured
               if key not in m["row"]["has"] and "photo_ok" in m["row"]["has"]]
        return pos, neg

    # The label vocabulary is NAMESPACED ("corners:rounding"), and `has` is
    # a list, so matching bare tokens against it counted every corner- and
    # surface-labelled card as clean. `clean` is its own label; use it.
    clean = [m for m in measured if "clean" in m["row"]["has"]]
    firing = [m for m in clean if m["corner_anoms"] > 0]
    print(f"\nCORNER DETECTOR FIRING RATE ON CLEAN CARDS: "
          f"{len(firing)}/{len(clean)}")
    print("  anomalies per clean card: "
          + str(sorted(m["corner_anoms"] for m in clean)))

    print("\n### GLARE")
    for metric in ("max_clip", "clip_spread"):
        report(f"glare by {metric}", *split("glare", metric))

    print("\n" + "=" * 62 + "\n### CORNER WEAR")
    worn = [m for m in measured
            if any(h.startswith("corners:") for h in m["row"]["has"])]
    clean = [m for m in measured if "clean" in m["row"]["has"]]
    for metric in ("corner_worst", "corner_anoms"):
        report(f"corner wear by {metric}",
               [m[metric] for m in worn], [m[metric] for m in clean])

    print("\n" + "=" * 62 + "\n### SURFACE")
    surf = [m for m in measured
            if any(h.startswith("surface:") for h in m["row"]["has"])]
    report("surface by local outlier",
           [m["surface_outlier"] for m in surf],
           [m["surface_outlier"] for m in clean])


if __name__ == "__main__":
    main()
