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

MUTATION TESTING: this file fires on EVERY source mutation of a guarded
module, so it co-fires with whatever real test kills a mutation, and a
mutation that only trips this one has SURVIVED. CLAUDE.md's rule already
says to check which test killed a mutation rather than accepting that one
did; here that is not pedantry but the only way to read the result.
Deselect it when mutating: `-p no:cacheprovider --deselect
tests/review/test_version_bumps.py`.
"""

import ast
import hashlib
from importlib import import_module
from pathlib import Path

import pytest

from card_reviewer.review import versions

SRC = Path(versions.__file__).parent

#: module path -> (constant name, its value, AST digest at that value)
#:
#: The VALUE is recorded too, and that is the whole point. An earlier
#: version of this table held only (constant, digest), which bound the
#: module to a digest and never bound the constant to anything — so
#: reverting FUSION_VERSION to 1.1.0, the exact failure this file was
#: written for, passed the entire suite.
#:
#: Several modules share one constant on purpose — the four measure/*
#: modules and their package all ship as CV_VERSION, because they are one
#: stage. A change to any of them must move that one constant.
#:
#: With both halves recorded, a stale bump fails outright, and the weaker
#: evasion — updating the digest and leaving the version alone — becomes a
#: one-line anomaly in the diff, visible to a reviewer: a changed digest
#: beside an unchanged version is the thing to question.
GUARDED = {
    "assembly.py": ("ASSEMBLY_VERSION", "1.1.0", "5596999abb728daa"),
    # Behaviour that belongs to a stage but lives outside the module the
    # constant is named for. `roles.py` holds the density thresholds and
    # `_resolve` that ARE the role_context stage; `findings.py` holds
    # `enforce_i3`, which combine calls; `evaluability.py` decides
    # UNEVALUABLE, which the heuristic consumes.
    "vision/anthropic.py": ("PROVIDER_ADAPTER_VERSION", "1.0.0", "672c4769d2847b6e"),
    "vision/provider.py": ("PROVIDER_ADAPTER_VERSION", "1.0.0", "99223d56539b9040"),
    "roles.py": ("RESOLVER_VERSION", "1.0.0", "19d1c6fbe391afd2"),
    "findings.py": ("COMBINATION_POLICY_VERSION", "1.1.0", "f80b54635c10131b"),
    "evaluability.py": ("SCORER_VERSION", "1.2.0", "975083105bb28487"),
    "canonical.py": ("CANON_SCHEME_VERSION", "1.1.0", "00b8ab6d4331d969"),
    "fusion.py": ("FUSION_VERSION", "1.2.0", "ed79215c86f7d1b6"),
    "heuristic.py": ("SCORER_VERSION", "1.2.0", "f22b4d937bc61a60"),
    "imaging/geometry.py": ("GEOMETRY_VERSION", "1.2.0", "d86f6b3b9216b7c0"),
    "imaging/measure/__init__.py": ("CV_VERSION", "1.1.0", "8a6651730d4e2cf8"),
    "imaging/measure/centering.py": ("CV_VERSION", "1.1.0", "a32346125ae0d9ea"),
    "imaging/measure/corners.py": ("CV_VERSION", "1.1.0", "b3e814e23b9d164d"),
    "imaging/measure/edges.py": ("CV_VERSION", "1.1.0", "a600ff7a44aa99e4"),
    "imaging/measure/surface.py": ("CV_VERSION", "1.1.0", "7af402307cb20909"),
    "imaging/observability.py": ("OBSERVABILITY_VERSION", "1.1.0", "7f2ef7eaa4866e02"),
    "imaging/preflight.py": ("PREFLIGHT_VERSION", "1.1.0", "972ee68643548a61"),
    "imaging/role_features.py": ("ROLE_FEATURES_VERSION", "1.0.0", "692c039193292c2f"),
    "manifest.py": ("MANIFEST_BUILDER_VERSION", "1.4.0", "1de15491a6d078ed"),
    "normalize.py": ("VOCABULARY_VERSION", "1.0.0", "7a710b0ced1b4cf4"),
    "policies/authority_v1.py": ("AUTHORITY_POLICY_VERSION", "1.0.0", "a38e410e720a6641"),
    "policies/combine_v1.py": ("COMBINATION_POLICY_VERSION", "1.1.0", "6a0b5dea7c7aa25d"),
    "policies/coverage_v1.py": ("COVERAGE_POLICY_VERSION", "1.1.0", "8649ac47d5601daa"),
    "policies/relevance_v1.py": ("RELEVANCE_POLICY_VERSION", "1.0.0", "bde3330ebb7cd704"),
    "policies/routing_v1.py": ("ROUTING_POLICY_VERSION", "1.1.0", "adfb0a768010388b"),
    "policies/scoring_v1.py": ("SCORING_POLICY_VERSION", "1.1.0", "ed8fa6f1e35893aa"),
    "relevance.py": ("RELEVANCE_POLICY_VERSION", "1.0.0", "ea269f91d02dd57c"),
    "role_context.py": ("RESOLVER_VERSION", "1.0.0", "dd7d9e70e7ec808a"),
    "taxonomy.py": ("TAXONOMY_VERSION", "1.1.0", "cf367536bf66499e"),
    "vision/prompt.py": ("PROMPT_VERSION", "1.1.0", "56d09acc4a109eb8"),
    "vocabulary.py": ("VOCABULARY_VERSION", "1.0.0", "040b976b969da81e"),
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


def constant_value(module: str, constant: str):
    """Most stage constants live in `versions.py`; PROMPT_VERSION lives in
    the module it versions. Look in `versions` first, then in the module
    itself — the BILLED stage is the last one that should go unguarded
    because its constant is kept somewhere else.
    """
    if hasattr(versions, constant):
        return getattr(versions, constant)
    dotted = "card_reviewer.review." + module[: -len(".py")].replace("/", ".")
    try:
        return getattr(import_module(dotted), constant)
    except AttributeError:
        # A constant shared by sibling modules lives in one of them.
        package = dotted.rsplit(".", 1)[0]
        for sibling in ("anthropic", "prompt", "provider"):
            module_obj = import_module(f"{package}.{sibling}")
            if hasattr(module_obj, constant):
                return getattr(module_obj, constant)
        raise


def behaviour_digest(path: Path) -> str:
    tree = _strip_docstrings(ast.parse(path.read_text()))
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


@pytest.mark.parametrize("module", sorted(GUARDED))
def test_a_guarded_constant_still_holds_the_value_recorded_beside_its_code(
        module):
    """The half that was missing. Reverting a bump used to pass."""
    constant, recorded_value, _ = GUARDED[module]
    actual = constant_value(module, constant)
    assert actual == recorded_value, (
        f"{constant} is {actual!r} but this table records {recorded_value!r} "
        f"for the current {module}. If you are deliberately changing the "
        f"version, update the value here in the same commit; if you did not "
        f"mean to change it, a bump has been lost.")


@pytest.mark.parametrize("module", sorted(GUARDED))
def test_a_stage_whose_code_changed_moved_its_version(module):
    constant, _, recorded = GUARDED[module]
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
    missing = []
    for module, (constant, _, _) in GUARDED.items():
        try:
            constant_value(module, constant)
        except AttributeError:
            missing.append(constant)
    assert not missing, f"guarded constants that no longer exist: {missing}"


def test_every_guarded_module_exists():
    missing = [m for m in GUARDED if not (SRC / m).exists()]
    assert not missing, f"guarded modules that no longer exist: {missing}"


#: Files that carry no behaviour a cached stage depends on. Each one is a
#: claim, not a convenience: if any of these grows a threshold or a branch
#: that changes a stage's output, it belongs in GUARDED instead.
EXEMPT = {
    "__init__.py",
    # Presentation and orchestration. None of them is an input to a cached
    # stage: they read stage output and render or route it.
    "cli.py",
    "report.py",
    "service.py",
    # Test-fixture generation, never imported by production paths.
    "imaging/synthetic.py",
    # Ingest resolves a candidate into content-addressed images. What the
    # stages fingerprint is the image CONTENT, so a change here changes
    # which photographs arrive, not what a stage computes from one.
    "ingest/adapter.py",
    # Persistence: schema and row access. A change here can break storage
    # but cannot make a stage compute a different answer.
    "storage/migrations.py",
    "storage/repository.py",
    # Content-addressed store. Artifact ids are a pure function of bytes,
    # so ids stay stable unless the HASHING SCHEME changes — and that
    # invalidates every derived id at once, which is a deliberate global
    # migration rather than a stage bump. If you change the scheme, this
    # exemption is the thing to revisit.
    "storage/artifacts.py",
    "enums.py",            # names and orderings, no decisions
    "models.py",           # the output record's shape
    "provenance.py",       # EvidenceRef and NormalizedBox
    "context.py",          # the CardContext container
    "versions.py",         # the constants themselves
    "fingerprint.py",      # covered by its own declaration tests
    "pipeline.py",         # orchestration; every stage it calls is guarded
    "taxonomy.py",         # guarded above, listed for the walk below
}


def test_every_module_is_either_guarded_or_deliberately_exempt():
    """The guard's premise — "any change to what the code DOES demands a
    bump" — is only true for the modules it lists. A file added tomorrow is
    unguarded by DEFAULT, and silently so.

    This turns that silence into a failing test: a new module must be
    guarded or written down as exempt, and either way a human decides which
    rather than nobody noticing. `roles.py` reached this project's role
    resolution thresholds without either, and changing BACK_TEXT_DENSITY
    passed the entire suite.
    """
    everything = {
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    known = set(GUARDED) | EXEMPT | {
        name for name in everything if Path(name).name in EXEMPT
    }
    unclassified = sorted(everything - known)
    assert not unclassified, (
        f"modules neither guarded nor exempt: {unclassified}. Does a change "
        f"here alter a cached stage's output? If so add it to GUARDED with "
        f"its constant; if not, add it to EXEMPT and say why.")
