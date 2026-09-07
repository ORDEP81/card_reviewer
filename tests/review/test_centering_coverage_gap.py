"""A centering measurement that declined must become a coverage gap.

`measure_centering` returning `measurable=False` is the engine saying it
could not assess centering on this photograph. If that never reaches
detectability, the heuristic emits no centering finding and coverage counts
centering as assessed — "we could not measure it" silently becomes "there is
nothing wrong with it", which is exactly what I2 forbids.
"""

import pytest

from card_reviewer.review.enums import Scale
from card_reviewer.review.imaging.geometry import analyze as geometry_analyze
from card_reviewer.review.imaging.measure import measure_all
from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.assembly import to_image_evidence
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.taxonomy import REASON_CODES, class_of


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


def _image_evidence(spec, store):
    from card_reviewer.review.assembly import ImageStageOutputs
    from card_reviewer.review.imaging.observability import (
        analyze as observability_analyze,
    )
    from card_reviewer.review.imaging.preflight import analyze as preflight_analyze

    data = render_png(spec)
    image_hash = store.put_image(data)
    geometry = geometry_analyze(data, store, image_hash)
    observability = observability_analyze(geometry, store, image_hash)
    cv = measure_all(geometry, store, image_hash)
    outputs = ImageStageOutputs(
        image_hash=image_hash,
        preflight=preflight_analyze(data).model_dump(),
        geometry=geometry.model_dump(),
        observability=observability.model_dump(),
        cv_measurements=cv.model_dump(),
        role_features={},
    )
    return cv, to_image_evidence([outputs])[0]


def test_a_declined_centering_measurement_lowers_centering_detectability(store):
    """A steep tilt leaves no border band the ink threshold can sit outside
    of; a 10-degree one is measured correctly and is no use here."""
    cv, evidence = _image_evidence(CardSpec(rotation_deg=20.0), store)
    assert cv.centering["measurable"] is False, "fixture no longer triggers it"

    keys = [k for k in evidence.detectability if k[1] == "centering"]
    assert keys, "centering has no detectability entry at all"
    for key in keys:
        assert evidence.detectability[key] < Scale.MODERATE, (
            f"{key} still reports {evidence.detectability[key]} although "
            "centering could not be measured")
        assert evidence.reason_codes.get(key), f"{key} has no reason code"


def test_the_declining_reason_is_a_registered_reason_code(store):
    cv, evidence = _image_evidence(CardSpec(rotation_deg=20.0), store)
    for key, code in evidence.reason_codes.items():
        if key[1] == "centering":
            assert code in REASON_CODES, f"{code} is not a declared reason code"
            class_of(code)


def test_a_measurable_card_keeps_its_centering_detectability(store):
    """The gap must not be manufactured for cards we CAN measure."""
    cv, evidence = _image_evidence(CardSpec(), store)
    assert cv.centering["measurable"] is True
    keys = [k for k in evidence.detectability if k[1] == "centering"]
    assert any(evidence.detectability[k] >= Scale.MODERATE for k in keys)
