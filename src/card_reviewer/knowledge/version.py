"""The rubric's semver, stamped onto every review subsystem A emits.

The bump level answers one question: would an already-issued review come out
differently under this change? If yes, it is at least minor, and a changed or
retracted active rule is major.
"""

from __future__ import annotations

import re

from .paths import ProjectPaths

INITIAL = "0.1.0"
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
LEVELS = ("patch", "minor", "major")


def read(paths: ProjectPaths) -> str:
    if not paths.version_file.exists():
        return INITIAL
    return paths.version_file.read_text().strip() or INITIAL


def write(paths: ProjectPaths, value: str) -> None:
    if not SEMVER_RE.match(value):
        raise ValueError(f"not a semver string: {value!r}")
    paths.version_file.parent.mkdir(parents=True, exist_ok=True)
    paths.version_file.write_text(value + "\n")


def bump(current: str, level: str) -> str:
    match = SEMVER_RE.match(current.strip())
    if not match:
        raise ValueError(f"not a semver string: {current!r}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    major, minor, patch = (int(g) for g in match.groups())
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"
