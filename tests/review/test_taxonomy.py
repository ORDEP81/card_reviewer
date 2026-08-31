import pytest

from card_reviewer.review import taxonomy as tx
from card_reviewer.review.enums import UndetectabilityClass


def test_the_four_grading_categories_have_the_spec_defect_types():
    assert tx.defect_types_for("centering") == ["border_ratio"]
    assert set(tx.defect_types_for("corners")) == {"whitening", "rounding", "fraying"}
    assert set(tx.defect_types_for("edges")) == {"whitening", "chipping", "roughness"}
    assert set(tx.defect_types_for("surface")) == {
        "scratches", "print_lines", "dimples", "stains", "gloss_break",
        "crease", "paper_loss",
    }


def test_every_defect_type_declares_a_promotion_level():
    for spec in tx.DEFECT_TYPES.values():
        assert spec.promotion in (tx.Promotion.MEASUREMENT, tx.Promotion.INTERPRETIVE)


def test_measurement_types_are_exactly_the_declared_list():
    measurement = {
        k for k, v in tx.DEFECT_TYPES.items()
        if v.promotion is tx.Promotion.MEASUREMENT
    }
    assert measurement == {
        "centering:border_ratio", "corners:whitening",
        "corners:rounding", "edges:whitening",
    }


def test_white_border_is_structural_and_glare_is_circumstantial():
    assert tx.class_of("WHITE_BORDER") is UndetectabilityClass.STRUCTURAL
    assert tx.class_of("GLARE") is UndetectabilityClass.CIRCUMSTANTIAL


def test_unknown_product_context_is_metadata_resolvable_not_circumstantial():
    """Classing it circumstantial would generate a photo request that no
    photograph could satisfy."""
    assert tx.class_of("UNKNOWN_PRODUCT_CONTEXT") is (
        UndetectabilityClass.METADATA_RESOLVABLE
    )


def test_an_unknown_reason_code_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="NOT_A_CODE"):
        tx.class_of("NOT_A_CODE")


def test_promotion_of_takes_category_and_name():
    assert tx.promotion_of("corners", "whitening") is tx.Promotion.MEASUREMENT
    assert tx.promotion_of("surface", "print_lines") is tx.Promotion.INTERPRETIVE


def test_the_taxonomy_version_is_declared_and_matches_the_versions_table():
    from card_reviewer.review.versions import SUPPORTING_VERSIONS

    assert tx.TAXONOMY_VERSION == SUPPORTING_VERSIONS["taxonomy"]


def test_categories_are_the_four_grading_categories():
    assert tx.CATEGORIES == ("centering", "corners", "edges", "surface")


def test_crease_and_paper_loss_are_declared_surface_defect_types():
    """SURFACE_TECHNICAL_DEFECT_001 is an active objective rule naming a
    minor crease and paper loss as grade-limiting, and the spec leans on it
    to justify requiring the back. A defect type the rubric calls
    grade-limiting must exist in the taxonomy or nothing can ever record it."""
    surface = set(tx.defect_types_for("surface"))
    assert {"crease", "paper_loss"} <= surface


def test_crease_and_paper_loss_cannot_be_confirmed_by_cv_alone():
    """No validated deterministic measurement distinguishes a crease from a
    scan line or a fold shadow, so v1 treats both as interpretive: CV may
    raise them as candidates, only the vision layer may confirm."""
    for defect_type in ("crease", "paper_loss"):
        assert tx.promotion_of("surface", defect_type) is tx.Promotion.INTERPRETIVE


def test_the_rule_that_motivates_them_is_still_active():
    """A tripwire: if this rule is ever rejected or superseded, the reason
    these two defect types exist has changed and should be revisited."""
    from card_reviewer.knowledge import load_active_rubric

    ids = {r.id for r in load_active_rubric().rules}
    assert "SURFACE_TECHNICAL_DEFECT_001" in ids
