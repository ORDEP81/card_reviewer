"""CI must never make a real Anthropic call, and the guard must be real.

The previous guard was two lines and both were broken. It grepped the
RELATIVE path "tests/", so from any working directory without that directory
grep exits 1 with empty stdout and the assertion passes — a CI job with a
different cwd reported the suite API-safe while checking nothing. And it
matched only `anthropic.Anthropic(`, so a file doing the idiomatic
`from anthropic import Anthropic` / `Anthropic()` went straight through. The
reviewer demonstrated both by adding a file that would have billed.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Every spelling that reaches the wire. Assembled at runtime so this file
#: does not match its own pattern.
CONSTRUCTORS = re.compile(
    "|".join([
        r"\bAnthropic" + r"\(",
        r"\bAsyncAnthropic" + r"\(",
        r"anthropic\." + r"Anthropic" + r"\(",
        r"anthropic\." + r"AsyncAnthropic" + r"\(",
        r"anthropic\." + r"Client" + r"\(",
    ])
)


def _python_files(root: Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def test_the_repo_root_resolves_from_this_file_not_the_working_directory():
    """The bug that made the old guard vacuous."""
    assert (REPO / "tests" / "review").is_dir()
    assert (REPO / "src" / "card_reviewer").is_dir()


def test_no_test_constructs_a_live_anthropic_client():
    offenders = [
        f"{path.relative_to(REPO)}:{n}: {line.strip()}"
        for path in _python_files(REPO / "tests")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if CONSTRUCTORS.search(line) and path.name != Path(__file__).name
    ]
    assert offenders == [], "a test constructs a real client:\n" + "\n".join(
        offenders)


def test_the_guard_catches_the_idiomatic_construction(tmp_path):
    """Proves the pattern bites rather than merely returning empty."""
    for source in ("client = Anthro" + "pic()",
                   "client = AsyncAnthro" + "pic()",
                   "c = anthropic." + "Anthropic()",
                   "c = anthropic." + "Client()"):
        assert CONSTRUCTORS.search(source), f"{source!r} slipped through"


def test_the_guard_does_not_flag_the_lazy_import_the_provider_uses():
    assert not CONSTRUCTORS.search("from anthropic import Anthropic")
    assert not CONSTRUCTORS.search("import anthropic")


def test_grep_style_scans_would_have_reported_success_on_a_missing_directory():
    """The precise failure mode: exit code 1, empty stdout, assertion passes.
    Asserting on returncode is what the replacement avoids needing."""
    out = subprocess.run(
        ["grep", "-rn", "--include=*.py", "anything", "no_such_directory/"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert out.returncode == 1
    assert out.stdout == ""


def test_the_provider_imports_anthropic_lazily_so_import_alone_cannot_bill():
    """Importing the package must not construct anything."""
    source = (REPO / "src" / "card_reviewer" / "review" / "vision"
              / "anthropic.py").read_text()
    module_level = [
        line for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    assert not any("anthropic" in line and "card_reviewer" not in line
                   for line in module_level), (
        "the SDK is imported at module scope; it must be imported inside the "
        "call so importing this package cannot reach the network")
