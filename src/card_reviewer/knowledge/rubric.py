"""The rubric: what the grader currently believes, and the contract that
subsystem A imports to read it.

`ACTIVE_RUBRIC.md` is a rendered view. The YAML files under knowledge/rules/
are the source of truth. Never parse the markdown.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

from . import validate, version
from .models import Rule
from .paths import ProjectPaths


@dataclass
class Rubric:
    version: str
    rules: list[Rule]

    def by_category(self, category: str) -> list[Rule]:
        return [r for r in self.rules if r.category.value == str(category)]

    def for_card(
        self,
        card_types: list[str] | None = None,
        sets: list[str] | None = None,
    ) -> list[Rule]:
        """Rules relevant to one card.

        An unscoped rule applies to every card. A scoped rule applies only when
        its scope intersects what the caller asked for.
        """
        if card_types is None and sets is None:
            return list(self.rules)

        wanted_types = set(card_types or [])
        wanted_sets = set(sets or [])

        def matches(rule: Rule) -> bool:
            type_ok = not rule.applies_to.card_types or bool(
                set(rule.applies_to.card_types) & wanted_types
            )
            set_ok = not rule.applies_to.sets or bool(
                set(rule.applies_to.sets) & wanted_sets
            )
            return type_ok and set_ok

        return [r for r in self.rules if matches(r)]


def load_active_rubric(root: Path | str | None = None) -> Rubric:
    """The contract subsystem A calls. `version` is stamped onto every review."""
    paths = ProjectPaths(root or Path(__file__).resolve().parents[3])
    return Rubric(version=version.read(paths), rules=validate.load_active(paths))


def render(r: Rubric) -> str:
    lines = [
        "# Active Grading Rubric",
        "",
        "<!-- GENERATED FILE — DO NOT EDIT.",
        "     Source of truth is knowledge/rules/. Regenerate with:",
        "       card-knowledge build-rubric -->",
        "",
        f"**Version:** {r.version}  ",
        f"**Rules:** {len(r.rules)} active rules  ",
        f"**Built:** {datetime.date.today().isoformat()}",
        "",
    ]

    by_category: dict[str, list[Rule]] = {}
    for rule in r.rules:
        by_category.setdefault(rule.category.value, []).append(rule)

    for category in sorted(by_category):
        lines += [f"## {category}", ""]
        for rule in sorted(by_category[category], key=lambda x: x.id):
            scope = ""
            if rule.applies_to.card_types or rule.applies_to.sets:
                bits = []
                if rule.applies_to.card_types:
                    bits.append("card types: " + ", ".join(rule.applies_to.card_types))
                if rule.applies_to.sets:
                    bits.append("sets: " + ", ".join(rule.applies_to.sets))
                scope = f" _({'; '.join(bits)})_"
            citations = ", ".join(
                f"{s.lesson} {'/'.join(s.timestamps)}" for s in rule.sources
            )
            lines += [
                f"### {rule.id}{scope}",
                "",
                rule.statement,
                "",
                f"- evidence: `{rule.evidence_type.value}`  ",
                f"- confidence: `{rule.confidence.value}`  ",
                f"- sources: {citations}",
                "",
            ]

    return "\n".join(lines) + "\n"


def build(paths: ProjectPaths) -> Path:
    r = Rubric(version=version.read(paths), rules=validate.load_active(paths))
    paths.rubric_file.parent.mkdir(parents=True, exist_ok=True)
    paths.rubric_file.write_text(render(r))
    return paths.rubric_file
