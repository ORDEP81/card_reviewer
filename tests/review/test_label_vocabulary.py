"""The human labelling tool and the engine must mean the same things.

This project's recurring failure is two halves agreeing on a concept and
disagreeing on its vocabulary: the region dimension, evidence refs across
faces, the edge region names. The labelled corpus is another such seam, and
the least guarded one — nothing executable connected `training/photos/`'s
label tokens to `taxonomy.py` until this file.

It is not a hypothetical. `separate.py` matched bare tokens like
`corner_wear` against a namespaced, list-valued `has` column, so every card
labelled `corners:rounding` counted as CLEAN, and the reported clean/defect
split was drawn from the wrong populations.
"""

import csv
from pathlib import Path

import pytest

from card_reviewer.review.taxonomy import CATEGORIES, defect_types_for

CORPUS = Path(__file__).resolve().parents[2] / "training" / "photos"
LABELS = CORPUS / "labels.csv"

#: Label tokens that are deliberately finer than the taxonomy, and the
#: taxonomy pair each one is evidence for. A collector says "miscut" for two
#: different physical faults: the card was PRINTED off-centre (borders
#: unequal), or the CUT went wrong (a sliver of the neighbouring card shows,
#: or a border is missing). The engine measures one number for both.
#:
#: Recording the mapping here rather than renaming the labels keeps the
#: distinction the photographs actually carry. If calibration later shows the
#: two behave differently against PSA outcomes, the corpus can still tell
#: them apart; a label flattened to `border_ratio` could not.
FINER_THAN_TAXONOMY = {
    "centering:off_center": "centering:border_ratio",
    "centering:miscut": "centering:border_ratio",
}

#: Tokens that describe the PHOTOGRAPH rather than the card. They have no
#: taxonomy pair by design — they are image limitations, and the engine
#: models them as reason codes, not defects.
PHOTO_TOKENS = {"glare", "blur", "underexposed", "occluded", "photo_ok",
                "clean", "screenshot", "listing_image"}


def _taxonomy() -> set[str]:
    return {f"{c}:{d}" for c in CATEGORIES for d in defect_types_for(c)}


def _tokens() -> set[str]:
    if not LABELS.exists():
        pytest.skip("no labelled corpus in this checkout")
    used: set[str] = set()
    with LABELS.open() as handle:
        for row in csv.DictReader(handle):
            used |= {t for t in (row.get("has") or "").split("|") if t}
    return used


def test_every_defect_label_names_something_the_engine_can_find():
    """A label the taxonomy cannot express is a labelled photograph no
    finding can ever be compared against — it is silently worth nothing,
    and it looks exactly like a detector with no signal."""
    unknown = {t for t in _tokens() if ":" in t} - _taxonomy() - set(FINER_THAN_TAXONOMY)
    assert not unknown, (
        f"label tokens with no taxonomy pair and no declared mapping: "
        f"{sorted(unknown)}")


def test_every_non_defect_label_is_a_declared_photograph_token():
    """The other direction: a bare token that is not a known photograph
    property is a typo or a vocabulary drift, and it would silently shrink
    whichever population a harness selects with it."""
    stray = {t for t in _tokens() if ":" not in t} - PHOTO_TOKENS
    assert not stray, f"unrecognized bare label tokens: {sorted(stray)}"


def test_each_declared_mapping_points_at_a_real_taxonomy_pair():
    """The mapping table is itself a place for the two vocabularies to
    drift. A defect type renamed in the taxonomy must break this."""
    taxonomy = _taxonomy()
    bad = {k: v for k, v in FINER_THAN_TAXONOMY.items() if v not in taxonomy}
    assert not bad, f"mappings pointing at nothing in the taxonomy: {bad}"


def test_the_corpus_and_its_labels_describe_the_same_files():
    """A label row for a deleted photograph, or a photograph nothing
    labelled, is silent sample loss — the count looks right and the
    population is not what it claims."""
    if not LABELS.exists():
        pytest.skip("no labelled corpus in this checkout")
    with LABELS.open() as handle:
        labelled = {row["file"] for row in csv.DictReader(handle)}
    on_disk = {
        str(p.relative_to(CORPUS))
        for p in CORPUS.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".heic"}
        and p.relative_to(CORPUS).parts[0] != "graded"
    }
    assert labelled - on_disk == set(), (
        f"labelled but missing from disk: {sorted(labelled - on_disk)}")
    assert on_disk - labelled == set(), (
        f"on disk but never labelled: {sorted(on_disk - labelled)}")
