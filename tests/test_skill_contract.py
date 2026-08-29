import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
SKILL = REPO / "skills" / "learn-video" / "SKILL.md"


def test_skill_file_exists():
    assert SKILL.exists()


def test_skill_has_frontmatter_with_name_and_description():
    body = SKILL.read_text()
    assert body.startswith("---")
    assert "name: learn-video" in body
    assert "description:" in body


def test_skill_names_the_commands_it_drives():
    body = SKILL.read_text()
    for command in ("card-knowledge extract-frames", "card-knowledge validate"):
        assert command in body, f"skill does not mention: {command}"


def test_skill_states_the_prohibitions():
    """Spec §5: the skill may not promote its own output."""
    body = SKILL.read_text().lower()
    assert "knowledge/rules/" in body
    assert "active_rubric.md" in body
    assert "status: active" in body


def test_skill_requires_evidence_type_classification():
    body = SKILL.read_text()
    for value in ("objective", "experience_based", "opinion", "unverified", "contradicted"):
        assert value in body


def test_lesson_template_exists_and_covers_the_plan_sections():
    template = (REPO / "training" / "lessons" / "TEMPLATE.md").read_text()
    for heading in (
        "RULES TAUGHT",
        "DEFECTS SHOWN",
        "PSA GRADE EXAMPLES",
        "INSTRUCTOR OPINIONS",
        "POTENTIAL CONTRADICTIONS",
        "SOURCE TIMESTAMPS",
    ):
        assert heading in template
