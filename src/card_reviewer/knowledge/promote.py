"""Rule status transitions.

Every transition preserves the file. A rejected rule is a rejected rule on
disk forever, not a deleted one — the provenance of what the grader was told
and chose not to believe is part of the audit trail.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import Rule, RuleStatus
from .paths import ProjectPaths


class UnknownRule(Exception):
    """No rule with this id exists in knowledge/rules/."""


def rule_path(paths: ProjectPaths, rule: Rule) -> Path:
    return paths.rules / rule.category.value / f"{rule.id}.yaml"


def write_rule(path: Path, rule: Rule) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rule.model_dump(mode="json"), sort_keys=False))


def _drop_pending(paths: ProjectPaths, rule: Rule) -> None:
    (paths.pending_rules / f"{rule.id}.yaml").unlink(missing_ok=True)


def _find_active(paths: ProjectPaths, rule_id: str) -> tuple[Path, Rule]:
    for path in sorted(paths.rules.rglob("*.yaml")):
        rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        if rule.id == rule_id:
            return path, rule
    raise UnknownRule(f"no rule with id {rule_id!r} in {paths.rules}")


def accept(paths: ProjectPaths, rule: Rule, rubric_version: str) -> Path:
    promoted = rule.model_copy(
        update={"status": RuleStatus.ACTIVE, "rubric_version_added": rubric_version}
    )
    dest = rule_path(paths, promoted)
    write_rule(dest, promoted)
    _drop_pending(paths, rule)
    return dest


def reject(paths: ProjectPaths, rule: Rule, reason: str) -> Path:
    existing = f"{rule.notes}\n" if rule.notes else ""
    rejected = rule.model_copy(
        update={"status": RuleStatus.REJECTED, "notes": f"{existing}rejected: {reason}"}
    )
    dest = rule_path(paths, rejected)
    write_rule(dest, rejected)
    _drop_pending(paths, rule)
    return dest


def session_bump_level(accepted: bool, superseded: bool) -> str | None:
    """The single version bump a review session implies, or None.

    A supersede retracts an active rule — major. An accept alone is minor. A
    session with neither (all defers/rejects) leaves the rubric version
    untouched: nothing was added to what the grader believes.
    """
    if superseded:
        return "major"
    if accepted:
        return "minor"
    return None


def supersede(
    paths: ProjectPaths, new_rule: Rule, old_id: str, rubric_version: str
) -> tuple[Path, Path]:
    old_path, old_rule = _find_active(paths, old_id)
    if old_rule.status is not RuleStatus.ACTIVE:
        raise UnknownRule(
            f"cannot supersede {old_id}: its status is "
            f"{old_rule.status.value!r}, not 'active'"
        )

    retired = old_rule.model_copy(update={"status": RuleStatus.SUPERSEDED})
    write_rule(old_path, retired)

    promoted = new_rule.model_copy(
        update={
            "status": RuleStatus.ACTIVE,
            "rubric_version_added": rubric_version,
            "supersedes": old_id,
        }
    )
    new_path = rule_path(paths, promoted)
    write_rule(new_path, promoted)
    _drop_pending(paths, new_rule)
    return new_path, old_path
