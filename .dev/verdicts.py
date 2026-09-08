"""Verdicts and rank scores over the labelled corpus.

calibrate.py measures framing and separate.py measures detector
separation; neither runs the pipeline, so the verdict and rank-score
figures in docs/superpowers/findings/ were not reproducible from
committed tooling until this existed.

    uv run python .dev/verdicts.py
"""
import csv, sys, tempfile
from collections import Counter
from pathlib import Path

PHOTOS = Path("training/photos")
sys.path.insert(0, "src")

from card_reviewer.review.enums import Mode
from card_reviewer.review.imaging.geometry import analyze
from card_reviewer.review.imaging.measure import measure_all
from card_reviewer.review.ingest.adapter import ManualAdapter
from card_reviewer.review.models import CandidateInput
from card_reviewer.review.pipeline import ReviewPipeline
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository

tmp = Path(tempfile.mkdtemp())
conn = connect(tmp / "t.db"); migrate(conn)
store = ArtifactStore(tmp / "store")
pipeline = ReviewPipeline(SqliteRepository(conn), store)

DEFECTS = ("corner_wear", "glare", "surface", "crease", "print_lines",
           "scratches", "gloss_break", "dimples", "stains", "paper_loss",
           "off_center", "miscut", "edge_wear", "whitening")

by_group = {}
buckets = {}
for r in csv.DictReader((PHOTOS / "labels.csv").open()):
    if "slab" in r.get("holder", ""):
        continue
    has = r.get("has", "")
    path = PHOTOS / r["file"]
    if not path.exists():
        continue
    tokens = [t for t in has.split("|") if t]
    clean = "clean" in tokens
    geometry = None
    if clean:
        data = path.read_bytes()
        image_hash = store.put_image(data)
        geometry = analyze(data, store, image_hash)
        if geometry.usable:
            fired = len([a for a in measure_all(geometry, store,
                                                image_hash).anomalies
                         if a["category"] == "corners"])
            buckets.setdefault(fired, [])
    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="unknown", image_paths=[path],
        supplied_roles={str(path): "front"}))
    try:
        review = pipeline.review(resolved, Mode.OFF)
    except Exception as e:
        by_group.setdefault("ERROR", Counter())[type(e).__name__] += 1
        continue
    group = "clean" if clean else ("defect-labelled"
                                   if any(":" in t for t in tokens) else "other")
    if group == "clean" and review.verdict == "REJECT":
        print("CLEAN REJECT:", r["file"], "| labels:", has)
        for f in review.defects_found:
            print("   ", {k: f.get(k) for k in
                          ("category", "defect_type", "state", "severity",
                           "confidence", "i3_demoted", "demotion_reason")})
        print("   reasoning:", review.reasoning[:400] if hasattr(review, "reasoning") else "")
    by_group.setdefault(group, Counter())[review.verdict] += 1
    if clean and geometry is not None and geometry.usable:
        buckets[fired].append(review.psa10_rank_score)
    scores = by_group.setdefault(group + ":scores", [])
    if isinstance(scores, list):
        scores.append(review.psa10_rank_score)

for group in ("clean", "defect-labelled", "other", "ERROR"):
    if group in by_group:
        print(f"\n{group}: {dict(by_group[group])}")
    s = by_group.get(group + ":scores")
    if s:
        ranked = [v for v in s if v is not None]
        print(f"  rank scores: {len(ranked)}/{len(s)} rankable"
              + (f", min={min(ranked)} median={sorted(ranked)[len(ranked)//2]} max={max(ranked)}" if ranked else ""))


# The corner detector's noise, and what it costs the ranking. The findings
# document quotes these buckets, so they belong in committed tooling.
if buckets:
    from statistics import median

    firing = sum(len(v) for k, v in buckets.items() if k > 0)
    print(f"\ncorner detector fires on {firing}/"
          f"{sum(len(v) for v in buckets.values())} clean cards")
    for fired in sorted(buckets):
        scores = [s for s in buckets[fired] if s is not None]
        print(f"  {fired} anomalies: n={len(buckets[fired]):2d}"
              + (f" median rank score {median(scores):5.1f}" if scores else ""))
