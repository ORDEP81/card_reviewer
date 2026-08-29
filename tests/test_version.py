import pytest

from card_reviewer.knowledge import version
from card_reviewer.knowledge.paths import ProjectPaths


@pytest.fixture
def paths(tmp_path):
    p = ProjectPaths(tmp_path)
    p.knowledge.mkdir(parents=True)
    return p


def test_read_defaults_to_initial_when_file_absent(paths):
    assert version.read(paths) == "0.1.0"


def test_write_then_read_roundtrips(paths):
    version.write(paths, "1.2.3")
    assert version.read(paths) == "1.2.3"


def test_read_tolerates_trailing_whitespace(paths):
    paths.version_file.write_text("0.4.2\n\n")
    assert version.read(paths) == "0.4.2"


def test_bump_patch():
    assert version.bump("0.4.2", "patch") == "0.4.3"


def test_bump_minor_resets_patch():
    assert version.bump("0.4.2", "minor") == "0.5.0"


def test_bump_major_resets_minor_and_patch():
    assert version.bump("0.4.2", "major") == "1.0.0"


def test_bump_rejects_unknown_level():
    with pytest.raises(ValueError):
        version.bump("0.4.2", "sideways")


def test_bump_rejects_malformed_version():
    with pytest.raises(ValueError):
        version.bump("nope", "patch")
