"""A stage whose code changes must move its version constant.

This is the rule "changing a value capable of changing output must
invalidate that stage", enforced mechanically rather than by remembering.
It has been broken twice on this branch, both times silently:

- COVERAGE_POLICY_VERSION stayed at 1.0.0 while the policy started
  returning PARTIAL where it had returned INADEQUATE.
- FUSION_VERSION stayed at 1.1.0 while `_same_place` changed combine's
  output on 62 of 117 corpus photographs — so the fix did not reach any
  card already in a database, and a minor top edge kept being reported as
  severe.

Neither was caught by a test, because no test connected a constant to the
code it stands for. This one does.

The fingerprint is the module's ABSTRACT SYNTAX TREE, not its text:
comments and docstrings are excluded deliberately, so rewording an
explanation does not demand a version bump, while any change to what the
code DOES demands one.

When this fails, do not just update the hash. Ask whether the change can
alter the stage's output for inputs it has already seen. It almost always
can — that is what these modules are. Bump the constant, then update the
hash in the same commit.
"""

import ast
import hashlib
from pathlib import Path

import pytest

from card_reviewer.review import versions

SRC = Path(versions.__file__).parent

#: module path -> (version constant name, AST digest at that version)
GUARDED = {
    "assembly.py": ("ASSEMBLY_VERSION", "5596999abb728daa"),
    "fusion.py": ("FUSION_VERSION", "ed79215c86f7d1b6"),
    "heuristic.py": ("SCORER_VERSION", "f22b4d937bc61a60"),
    "imaging/geometry.py": ("GEOMETRY_VERSION", "d86f6b3b9216b7c0"),
    "imaging/observability.py": ("OBSERVABILITY_VERSION", "7f2ef7eaa4866e02"),
    "manifest.py": ("MANIFEST_BUILDER_VERSION", "d7f8d7b370e8dbe9"),
    "policies/combine_v1.py": ("COMBINATION_POLICY_VERSION", "6a0b5dea7c7aa25d"),
    "policies/coverage_v1.py": ("COVERAGE_POLICY_VERSION", "8649ac47d5601daa"),
}


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def behaviour_digest(path: Path) -> str:
    tree = _strip_docstrings(ast.parse(path.read_text()))
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


@pytest.mark.parametrize("module", sorted(GUARDED))
def test_a_stage_whose_code_changed_moved_its_version(module):
    constant, recorded = GUARDED[module]
    actual = behaviour_digest(SRC / module)
    assert actual == recorded, (
        f"{module} changed but {constant} is still "
        f"{getattr(versions, constant)!r}.\n"
        f"Can this change alter the stage's output for an input it has "
        f"already seen? If so, bump {constant} and set the digest to "
        f"{actual!r} in the same commit. If it genuinely cannot — a pure "
        f"refactor — just update the digest and say so in the message.")


def test_every_guarded_constant_exists():
    """The table is itself a place to drift: a renamed constant would make
    the guard above pass against nothing."""
    missing = [c for c, _ in GUARDED.values() if not hasattr(versions, c)]
    assert not missing, f"guarded constants that no longer exist: {missing}"


def test_every_guarded_module_exists():
    missing = [m for m in GUARDED if not (SRC / m).exists()]
    assert not missing, f"guarded modules that no longer exist: {missing}"
