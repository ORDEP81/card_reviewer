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


# --- A8: write() didn't strip before validating. Python's `$` in SEMVER_RE
# matches just before a trailing "\n", so "1.2.3\n" passed the regex and was
# then written with a doubled newline.


def test_write_strips_trailing_whitespace_before_validating_and_writing(paths):
    version.write(paths, "1.2.3\n")
    assert paths.version_file.read_text() == "1.2.3\n"
    assert version.read(paths) == "1.2.3"


def test_write_rejects_value_that_is_only_whitespace(paths):
    with pytest.raises(ValueError):
        version.write(paths, "   ")


# --- B7: a zero-byte version file, distinct from a whitespace-only one,
# must also fall back to INITIAL.


def test_read_on_zero_byte_file_returns_initial(paths):
    paths.version_file.write_bytes(b"")
    assert version.read(paths) == "0.1.0"
