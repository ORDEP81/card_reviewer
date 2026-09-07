"""Builders for detectability maps.

Detectability is keyed by (face, region, category, defect_type). These keep
the region dimension visible in tests rather than hiding it behind sugar:
"one region" and "every region" are exactly what the coverage policy
distinguishes, so a test should have to say which it means.
"""
def regions_for(category):
    from card_reviewer.review.imaging.observability import REGIONS_FOR_CATEGORY

    return REGIONS_FOR_CATEGORY[category]


def detectability_map(faces, value=None):
    """Every (face, region, category, defect_type) at one value."""
    from card_reviewer.review.enums import Scale
    from card_reviewer.review.taxonomy import CATEGORIES, defect_types_for

    value = Scale.HIGH if value is None else value
    return {(face, region, category, defect_type): value
            for face in faces
            for category in CATEGORIES
            for region in regions_for(category)
            for defect_type in defect_types_for(category)}


def set_every_region(mapping, face, category, defect_type, value):
    """Set one (face, category, defect_type) across all of its regions."""
    for region in regions_for(category):
        mapping[(face, region, category, defect_type)] = value


def set_one_region(mapping, face, region, category, defect_type, value):
    mapping[(face, region, category, defect_type)] = value
