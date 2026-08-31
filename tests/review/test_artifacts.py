import pytest

from card_reviewer.review.storage.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


def test_identical_bytes_hash_to_one_image_stored_once(store):
    a = store.put_image(b"pixels")
    b = store.put_image(b"pixels")
    assert a == b
    assert len(list((store.root / "images").iterdir())) == 1


def test_geometry_and_measurement_crops_live_on_separate_paths(store):
    """Crop ownership is split by stage so each is invalidated by its own
    stage's cache and never by the other's (spec §7.4)."""
    h = store.put_image(b"pixels")
    face = store.put_derived(h, "face", "normalized.png", b"f")
    corner = store.put_derived(h, "corners", "bottom_left.png", b"c")
    assert "/face/" in str(store.path_of(face))
    assert "/corners/" in str(store.path_of(corner))


def test_a_derived_artifact_id_is_stable_for_the_same_inputs(store):
    h = store.put_image(b"pixels")
    assert store.put_derived(h, "corners", "bl.png", b"c") == store.put_derived(
        h, "corners", "bl.png", b"c"
    )


def test_originals_are_preserved_byte_for_byte(store):
    """Non-negotiable rule 6."""
    data = b"\x89PNG\r\n\x1a\n original"
    assert store.read(store.put_image(data)) == data


def test_reading_an_unknown_artifact_raises(store):
    with pytest.raises(KeyError):
        store.read("deadbeef")


def test_an_artifact_id_survives_a_new_store_over_the_same_directory(tmp_path):
    """A restart opens a fresh ArtifactStore over the same data. If ids were
    derived from anything process-local, every cached EvidenceRef would
    dangle after a crash."""
    first = ArtifactStore(tmp_path)
    h = first.put_image(b"pixels")
    aid = first.put_derived(h, "surface", "original.png", b"enhanced")

    second = ArtifactStore(tmp_path)
    assert second.read(h) == b"pixels"
    assert second.read(aid) == b"enhanced"


def test_derived_ids_do_not_depend_on_the_absolute_store_path(tmp_path):
    """Moving the data directory must not orphan every stored reference."""
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    a, b = ArtifactStore(a_root), ArtifactStore(b_root)
    h = a.put_image(b"pixels")
    b.put_image(b"pixels")
    assert a.put_derived(h, "corners", "bl.png", b"c") == b.put_derived(
        h, "corners", "bl.png", b"c"
    )


def test_different_derived_content_under_one_name_is_distinguished(store):
    """Two stages writing the same logical name must not collide silently."""
    h = store.put_image(b"pixels")
    first = store.put_derived(h, "corners", "bl.png", b"one")
    second = store.put_derived(h, "corners", "bl.png", b"two")
    assert first != second
    assert store.read(first) == b"one"
    assert store.read(second) == b"two"
