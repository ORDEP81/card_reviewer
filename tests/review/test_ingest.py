import json

import pytest

from card_reviewer.review.ingest.adapter import ManualAdapter
from card_reviewer.review.models import CandidateInput, CardReview, ResolvedCandidate
from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(b"pixels")
    return p


# --- rule 10: economics never reach the grading core -----------------------


def test_resolved_candidate_has_no_price_field_at_all():
    """Structural, not a discipline the grading path must remember."""
    forbidden = {"asking_price", "price", "cost", "value", "purchased",
                 "market_value", "profit"}
    assert not (set(ResolvedCandidate.model_fields) & forbidden)


def test_card_review_has_no_price_field_at_all():
    forbidden = {"asking_price", "price", "cost", "value", "purchased"}
    assert not (set(CardReview.model_fields) & forbidden)


def test_candidate_input_may_carry_price_as_listing_provenance():
    assert CandidateInput(source="manual", title="t",
                          asking_price="42.00").asking_price == "42.00"


def test_the_adapter_drops_price_when_resolving(store, image):
    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="t", asking_price="9999.00", image_paths=[image]))
    assert "9999" not in resolved.model_dump_json()


# --- identity --------------------------------------------------------------


def test_two_manual_copies_with_identical_titles_get_distinct_ids(store, image):
    """Two physical cards can share a title exactly. Deriving identity from
    the title would merge them and overwrite the first card's history."""
    adapter = ManualAdapter(store)
    kw = dict(source="manual", title="2023 Chrome #150", image_paths=[image])
    assert adapter.resolve(CandidateInput(**kw)).candidate_id != adapter.resolve(
        CandidateInput(**kw)
    ).candidate_id


def test_a_caller_supplied_id_is_used_verbatim_for_resubmission(store, image):
    adapter = ManualAdapter(store)
    kw = dict(source="manual", title="t", candidate_id="my-card-001",
              image_paths=[image])
    assert (
        adapter.resolve(CandidateInput(**kw)).candidate_id
        == adapter.resolve(CandidateInput(**kw)).candidate_id
        == "my-card-001"
    )


def test_a_listing_backed_candidate_is_stable_across_resolves(store, image):
    adapter = ManualAdapter(store)
    kw = dict(source="flippah", title="t", listing_id="L-42", image_paths=[image])
    assert adapter.resolve(CandidateInput(**kw)).candidate_id == adapter.resolve(
        CandidateInput(**kw)
    ).candidate_id


# --- images ----------------------------------------------------------------


def test_the_adapter_hashes_images_into_the_content_addressed_store(store, image):
    resolved = ManualAdapter(store).resolve(
        CandidateInput(source="manual", title="t", image_paths=[image]))
    assert len(resolved.images) == 1
    assert store.read(resolved.images[0].image_hash) == b"pixels"


def test_the_same_photo_in_two_listings_yields_one_image_hash(tmp_path, store):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    adapter = ManualAdapter(store)
    r1 = adapter.resolve(CandidateInput(source="manual", title="1", image_paths=[a]))
    r2 = adapter.resolve(CandidateInput(source="manual", title="2", image_paths=[b]))
    assert r1.images[0].image_hash == r2.images[0].image_hash


def test_supplied_roles_survive_resolution(store, tmp_path):
    front, back = tmp_path / "f.png", tmp_path / "b.png"
    front.write_bytes(b"front")
    back.write_bytes(b"back")
    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="t", image_paths=[front, back],
        supplied_roles={str(front): "front", str(back): "back"}))
    assert {i.supplied_role for i in resolved.images} == {"front", "back"}


def test_image_ordering_is_preserved(store, tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        p.write_bytes(f"pixels{i}".encode())
        paths.append(p)
    resolved = ManualAdapter(store).resolve(
        CandidateInput(source="manual", title="t", image_paths=paths))
    assert [i.ordering for i in resolved.images] == [0, 1, 2]


def test_the_manual_adapter_never_touches_the_network(store, image, monkeypatch):
    import socket

    def boom(*args, **kwargs):
        raise AssertionError("the core must not touch the network")

    monkeypatch.setattr(socket, "socket", boom)
    ManualAdapter(store).resolve(
        CandidateInput(source="manual", title="t", image_paths=[image]))


# --- CardReview ------------------------------------------------------------


def test_card_review_carries_every_spec_output_field():
    for field in (
        "verdict", "psa10_candidate", "psa10_rank_score", "rankable",
        "estimated_psa_grade", "review_confidence", "coverage", "categories",
        "image_quality", "roles_and_context", "defects_found", "limitations",
        "recommended_additional_photos", "card_identification_request",
        "cv_assessment", "vision_assessment", "reasoning", "versions",
    ):
        assert field in CardReview.model_fields, f"CardReview omits {field}"


def test_a_resolved_candidate_round_trips_through_json(store, image):
    resolved = ManualAdapter(store).resolve(
        CandidateInput(source="manual", title="t", image_paths=[image]))
    assert ResolvedCandidate.model_validate(
        json.loads(resolved.model_dump_json())) == resolved
