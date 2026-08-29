"""The package must be importable and the repo must refuse to track secrets."""
import pathlib
import tomllib


REPO = pathlib.Path(__file__).resolve().parents[1]


def test_package_is_importable():
    import card_reviewer.knowledge  # noqa: F401


def test_python_floor_is_3_14():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert data["project"]["requires-python"] == ">=3.14"


def test_gitignore_covers_secrets_and_media():
    body = (REPO / ".gitignore").read_text()
    for pattern in (
        "cookies.txt",
        "training/work/*/source/",
        ".env",
        "*.mp4",
    ):
        assert pattern in body, f"missing .gitignore entry: {pattern}"
