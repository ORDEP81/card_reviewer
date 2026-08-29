"""Rule status transitions.

Every transition preserves the file. A rejected rule is a rejected rule on
disk forever, not a deleted one — the provenance of what the grader was told
and chose not to believe is part of the audit trail.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import Rule, RuleStatus
from .paths import ProjectPaths


class UnknownRule(Exception):
    """No rule with this id exists in knowledge/rules/."""


class RuleConflict(Exception):
    """Refusing to overwrite a stored rule with a different one under the same id."""


def rule_path(paths: ProjectPaths, rule: Rule) -> Path:
    return paths.rules / rule.category.value / f"{rule.id}.yaml"


def write_rule(path: Path, rule: Rule) -> None:
    """Persist `rule` at `path`, refusing to clobber a different rule.

    A rule id maps to exactly one filesystem path for its whole life
    (pending -> active -> rejected/superseded all write the *same* file for
    a given id). The legitimate transitions all carry the same `statement`
    forward via `model_copy`, so a stored file at this path whose id matches
    but whose statement differs is not a transition of the same rule — it is
    a reused id about to destroy the original rule's content. That must
    never happen silently; `validate.check_rule` is the primary guard, this
    is belt-and-braces at the point of the actual write.
    """
    if path.exists():
        try:
            existing = Rule.model_validate(yaml.safe_load(path.read_text()))
        except (ValidationError, yaml.YAMLError) as exc:
            raise RuleConflict(f"{path} exists but does not parse as a Rule: {exc}") from exc
        if existing.id == rule.id and existing.statement != rule.statement:
            raise RuleConflict(
                f"refusing to overwrite {path}: the stored rule {existing.id} "
                f"(status {existing.status.value!r}) has a different statement "
                f"than the rule being written. A rule id is never reused for a "
                f"different claim — choose a new id."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rule.model_dump(mode="json"), sort_keys=False))


def _drop_pending(paths: ProjectPaths, rule: Rule, pending_path: Path | None = None) -> None:
    """Remove the pending file for `rule`.

    Prefer `pending_path` — the actual path `validate.load_pending` read the
    rule from — over reconstructing `pending_rules/<id>.yaml`. A pending
    file whose name doesn't match its `id` (which `validate.check_rule` now
    rejects, but an old or hand-edited file might still have) would
    otherwise be promoted while its real file stays behind, permanently
    wedging the next `review` on a phantom "id already exists" collision.
    """
    path = pending_path if pending_path is not None else paths.pending_rules / f"{rule.id}.yaml"
    path.unlink(missing_ok=True)


def find_active(paths: ProjectPaths, rule_id: str) -> tuple[Path, Rule]:
    """Locate the rule with this id anywhere under `knowledge/rules/`.

    Public (not `_find_active`): `review_cmd` calls this to validate a
    `supersede <id>` target at prompt time, before any decision in the
    session is applied — see Important 6.
    """
    for path in sorted(paths.rules.rglob("*.yaml")):
        rule = Rule.model_validate(yaml.safe_load(path.read_text()))
        if rule.id == rule_id:
            return path, rule
    raise UnknownRule(f"no rule with id {rule_id!r} in {paths.rules}")


def accept(
    paths: ProjectPaths, rule: Rule, rubric_version: str, pending_path: Path | None = None
) -> Path:
    promoted = rule.model_copy(
        update={"status": RuleStatus.ACTIVE, "rubric_version_added": rubric_version}
    )
    dest = rule_path(paths, promoted)
    write_rule(dest, promoted)
    _drop_pending(paths, rule, pending_path)
    return dest


def reject(
    paths: ProjectPaths, rule: Rule, reason: str, pending_path: Path | None = None
) -> Path:
    existing = f"{rule.notes}\n" if rule.notes else ""
    rejected = rule.model_copy(
        update={"status": RuleStatus.REJECTED, "notes": f"{existing}rejected: {reason}"}
    )
    dest = rule_path(paths, rejected)
    write_rule(dest, rejected)
    _drop_pending(paths, rule, pending_path)
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
    paths: ProjectPaths,
    new_rule: Rule,
    old_id: str,
    rubric_version: str,
    pending_path: Path | None = None,
) -> tuple[Path, Path]:
    old_path, old_rule = find_active(paths, old_id)
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
    _drop_pending(paths, new_rule, pending_path)
    return new_path, old_path
