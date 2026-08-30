"""Stage 6: mechanically check what Claude wrote before a human ever sees it.

The timestamp-bounds check is the reason rules are structured rather than
prose: a citation pointing past the end of the video is caught by arithmetic,
which prompting alone does not reliably achieve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import Manifest, Rule, RuleStatus
from .paths import ProjectPaths

TIMESTAMP_RE = re.compile(r"^\d{1,2}(:\d{2}){1,2}$")


class BadTimestamp(Exception):
    """A timestamp string that is not HH:MM:SS or MM:SS."""


@dataclass
class ValidationReport:
    ok: bool = True
    errors: dict[str, list[str]] = field(default_factory=dict)
    checked: int = 0


def _to_seconds(stamp: str) -> float:
    if not TIMESTAMP_RE.match(stamp.strip()):
        raise BadTimestamp(f"not a timestamp: {stamp!r} (expected MM:SS or HH:MM:SS)")
    seconds = 0.0
    for part in stamp.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def parse_timestamp(text: str) -> tuple[float, float]:
    """'12:04-12:38' -> (724.0, 758.0); '12:04' -> (724.0, 724.0)."""
    parts = [p for p in text.split("-") if p.strip()]
    if len(parts) == 1:
        point = _to_seconds(parts[0])
        return point, point
    if len(parts) == 2:
        return _to_seconds(parts[0]), _to_seconds(parts[1])
    raise BadTimestamp(f"cannot parse timestamp range: {text!r}")


def load_pending(paths: ProjectPaths) -> list[tuple[Path, Rule]]:
    """Load parseable pending rules, silently skipping ones that don't parse.

    This is for callers who have already run `run()` and refused to proceed
    unless it reported `ok` — at that point every file already parses and a
    skip here can never hide an unreported problem. This function itself
    never raises on a malformed file; per-file error reporting is `run()`'s
    job, not this one's.
    """
    out: list[tuple[Path, Rule]] = []
    if not paths.pending_rules.exists():
        return out
    for path in sorted(paths.pending_rules.glob("*.yaml")):
        try:
            rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        except (ValidationError, yaml.YAMLError):
            continue
        out.append((path, rule))
    return out


def load_active(paths: ProjectPaths) -> list[Rule]:
    out: list[Rule] = []
    if not paths.rules.exists():
        return out
    for path in sorted(paths.rules.rglob("*.yaml")):
        rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        if rule.status is RuleStatus.ACTIVE:
            out.append(rule)
    return out


def load_all(paths: ProjectPaths) -> list[Rule]:
    """Every rule under `knowledge/rules/`, regardless of status.

    Unlike `load_active`, this is not scoped to what the grader currently
    believes — it exists so id-collision checks can see rejected and
    superseded rules too. Those statuses live at the same path an id would
    be promoted to, so a collision check that only knew about active ids
    would let a new rule silently overwrite a retired one. A file that
    doesn't parse is skipped rather than raised, matching `load_pending`:
    `run()` already reports unparseable *pending* files, and a broken file
    already sitting in `knowledge/rules/` is a pre-existing condition this
    check is not responsible for surfacing.
    """
    out: list[Rule] = []
    if not paths.rules.exists():
        return out
    for path in sorted(paths.rules.rglob("*.yaml")):
        try:
            rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        except (ValidationError, yaml.YAMLError):
            continue
        out.append(rule)
    return out


def video_durations(paths: ProjectPaths) -> dict[str, float]:
    durations: dict[str, float] = {}
    if not paths.work.exists():
        return durations
    for manifest_path in paths.work.glob("*/manifest.json"):
        m = Manifest.model_validate_json(manifest_path.read_text())
        durations[m.video_id] = m.source.duration_s
    return durations


def check_rule(
    rule: Rule,
    path: Path,
    existing_rules: dict[str, Rule],
    durations: dict[str, float],
    paths: ProjectPaths,
) -> list[str]:
    errors: list[str] = []

    if rule.status is not RuleStatus.PENDING:
        errors.append(
            f"status is {rule.status.value!r}; rules in pending_rules/ must be 'pending'"
        )

    if path.stem != rule.id:
        errors.append(
            f"filename {path.name!r} does not match id {rule.id!r}; a pending "
            f"rule must be saved as pending_rules/{rule.id}.yaml or promotion "
            "will not find this file to remove"
        )

    if rule.id in existing_rules:
        other = existing_rules[rule.id]
        errors.append(
            f"id {rule.id} already exists with status {other.status.value!r}; "
            "choose a new id"
        )

    if rule.supersedes is not None:
        errors.append(
            f"pending rule {rule.id} must not set supersedes (found "
            f"{rule.supersedes!r}); supersession is a decision made during "
            "`card-knowledge review` via `supersede <id>`, not something a "
            "pending rule can declare for itself"
        )

    for source in rule.sources:
        if not paths.lesson(source.lesson).exists():
            errors.append(f"cited lesson {source.lesson} does not exist")

        if source.video_id is not None:
            # Video-mode source: the trust boundary against fabricated
            # citations. Every check here must run exactly as before.
            if source.video_id not in durations:
                errors.append(f"cited video_id {source.video_id} has no work packet")
                continue

            duration = durations[source.video_id]
            for stamp in source.timestamps:
                try:
                    start, end = parse_timestamp(stamp)
                except BadTimestamp as exc:
                    errors.append(str(exc))
                    continue
                if end > duration:
                    errors.append(
                        f"timestamp {stamp} exceeds the {duration:.0f}s duration of "
                        f"{source.video_id} — the citation cannot be real"
                    )
                if start > end:
                    errors.append(f"timestamp {stamp} starts after it ends")
        else:
            # Reference-mode source: no duration to check against, but the
            # citation must actually point somewhere.
            if not (source.reference or "").strip():
                errors.append("reference-mode source has a blank reference")
            if not (source.locator or "").strip():
                errors.append("reference-mode source has a blank locator")

    return errors


def run(paths: ProjectPaths) -> ValidationReport:
    report = ValidationReport()
    durations = video_durations(paths)

    try:
        existing_rules = {r.id: r for r in load_all(paths)}
    except (ValidationError, yaml.YAMLError) as exc:
        report.ok = False
        report.errors["knowledge/rules"] = [f"active rules are unreadable: {exc}"]
        return report

    if not paths.pending_rules.exists():
        return report

    parsed: list[tuple[Path, Rule]] = []
    for path in sorted(paths.pending_rules.glob("*.yaml")):
        report.checked += 1
        try:
            rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        except (ValidationError, yaml.YAMLError) as exc:
            report.ok = False
            report.errors[path.name] = [f"does not parse as a Rule: {exc}"]
            continue
        parsed.append((path, rule))

    # Two pending files claiming the same id is ambiguous — promotion can
    # only write one file to that id's slot. Count first so both files in a
    # colliding pair get flagged, not just whichever sorts second.
    id_counts: dict[str, int] = {}
    for _, rule in parsed:
        id_counts[rule.id] = id_counts.get(rule.id, 0) + 1

    for path, rule in parsed:
        duplicate_pending_id = id_counts[rule.id] > 1
        errors = check_rule(rule, path, existing_rules, durations, paths)
        if duplicate_pending_id:
            errors.append(f"id {rule.id} is used by more than one file in pending_rules/")
        if errors:
            report.ok = False
            # Keying by rule.id is convenient (and what most tests/CLI output
            # read), but it silently drops one side of a filename/id mismatch
            # or a duplicate id — two files would clobber the same report
            # key. Key by filename instead whenever that ambiguity exists.
            mismatched_filename = path.stem != rule.id
            key = path.name if (mismatched_filename or duplicate_pending_id) else rule.id
            report.errors[key] = errors

    return report
