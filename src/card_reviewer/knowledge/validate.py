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
    out: list[tuple[Path, Rule]] = []
    if not paths.pending_rules.exists():
        return out
    for path in sorted(paths.pending_rules.glob("*.yaml")):
        out.append((path, Rule.model_validate(yaml.safe_load(path.read_text()))))
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
    active_ids: set[str],
    durations: dict[str, float],
    paths: ProjectPaths,
) -> list[str]:
    errors: list[str] = []

    if rule.status is not RuleStatus.PENDING:
        errors.append(
            f"status is {rule.status.value!r}; rules in pending_rules/ must be 'pending'"
        )

    if rule.id in active_ids:
        errors.append(f"id {rule.id} is already active; choose a new id")

    for source in rule.sources:
        if not paths.lesson(source.lesson).exists():
            errors.append(f"cited lesson {source.lesson} does not exist")

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

    return errors


def run(paths: ProjectPaths) -> ValidationReport:
    report = ValidationReport()
    durations = video_durations(paths)

    try:
        active_ids = {r.id for r in load_active(paths)}
    except (ValidationError, yaml.YAMLError) as exc:
        report.ok = False
        report.errors["knowledge/rules"] = [f"active rules are unreadable: {exc}"]
        return report

    if not paths.pending_rules.exists():
        return report

    for path in sorted(paths.pending_rules.glob("*.yaml")):
        report.checked += 1
        try:
            rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        except (ValidationError, yaml.YAMLError) as exc:
            report.ok = False
            report.errors[path.name] = [f"does not parse as a Rule: {exc}"]
            continue

        errors = check_rule(rule, active_ids, durations, paths)
        if errors:
            report.ok = False
            report.errors[rule.id] = errors

    return report
