"""Contract: adapter -> roles -> context -> scoped rubric.

Everything Phase 3 produces, wired the way Task 36 will wire it, through the
real serialization boundary. Hand-built fixtures at each seam would pass
while the seams themselves disagreed.
"""

import json

import pytest

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.enums import Provenance, RuleEvaluability
from card_reviewer.review.evaluability import applicable, rule_content, scope_rules
from card_reviewer.review.ingest.adapter import ManualAdapter
from card_reviewer.review.models import CandidateInput, ResolvedCandidate
from card_reviewer.review.normalize import CardContextNormalizer
from card_reviewer.review.roles import ImageRole, RoleInput, resolve_roles
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.storage.migrations import connect, migrate
from card_reviewer.review.storage.repository import SqliteRepository


@pytest.fixture
def rig(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    store = ArtifactStore(tmp_path / "store")
    front, back = tmp_path / "f.png", tmp_path / "b.png"
    front.write_bytes(b"front-pixels")
    back.write_bytes(b"back-pixels")
    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="2023 Topps Chrome Julio Rodriguez #150",
        image_paths=[front, back],
        supplied_roles={str(front): "front", str(back): "back"}))
    yield SqliteRepository(conn), store, resolved
    conn.close()


def test_the_adapters_supplied_roles_drive_role_resolution(rig):
    """The supplied_role string the adapter carries must be the exact value
    resolve_roles recognises — a mismatch silently downgrades every image to
    inference."""
    _, _, resolved = rig
    roles = resolve_roles([
        RoleInput(image_hash=i.image_hash, supplied_role=i.supplied_role,
                  text_density=0.30)  # ambiguous, so only `supplied` can win
        for i in resolved.images
    ])
    assert {r.role for r in roles.values()} == {ImageRole.FRONT, ImageRole.BACK}
    assert all(r.provenance is Provenance.SUPPLIED for r in roles.values())


def test_the_adapters_title_drives_context_and_then_rubric_scoping(rig):
    """Free listing text through the normalizer into for_card, end to end."""
    _, _, resolved = rig
    context = CardContextNormalizer().normalize(
        raw_title=resolved.title, supplied_card_type=resolved.card_type)
    scoped = scope_rules(
        load_active_rubric().for_card(context.canonical_card_types,
                                      context.canonical_sets),
        context)
    by_id = {s.rule.id: s for s in scoped}
    assert context.canonical_card_types == ["chrome"]
    assert by_id["SURFACE_SHINY_001"].evaluability is RuleEvaluability.APPLICABLE


def test_a_candidate_and_its_images_persist_and_reload(rig):
    """The pipeline writes these rows before anything references them."""
    repo, store, resolved = rig
    repo.save_candidate(id=resolved.candidate_id, source=resolved.source,
                        title=resolved.title)
    for image in resolved.images:
        repo.save_image(image.image_hash, store.path_of(image.image_hash))
        repo.link_image(resolved.candidate_id, image.image_hash,
                        supplied_role=image.supplied_role,
                        ordering=image.ordering)
    rows = repo._conn.execute(
        "SELECT image_hash, supplied_role FROM candidate_image"
        " WHERE candidate_id=? ORDER BY ordering", (resolved.candidate_id,)
    ).fetchall()
    assert [r[1] for r in rows] == ["front", "back"]
    assert {r[0] for r in rows} == {i.image_hash for i in resolved.images}


def test_an_image_hash_from_the_adapter_resolves_in_the_store(rig):
    """A stored path must lead back to the exact bytes, or every later
    EvidenceRef points at nothing."""
    _, store, resolved = rig
    assert store.read(resolved.images[0].image_hash) == b"front-pixels"


def test_a_resolved_candidate_survives_the_cache_boundary_and_still_scopes(rig):
    """role_context fingerprints supplied metadata, so the candidate crosses
    JSON before scoping happens on the far side."""
    _, _, resolved = rig
    revived = ResolvedCandidate.model_validate(
        json.loads(resolved.model_dump_json()))
    context = CardContextNormalizer().normalize(raw_title=revived.title)
    assert context.canonical_card_types == ["chrome"]
    assert revived.images == resolved.images


def test_rule_content_from_a_real_listing_is_fingerprintable(rig):
    """The whole chain feeding a cache key: title -> context -> rules ->
    canonical form."""
    from card_reviewer.review.fingerprint import fingerprint

    _, _, resolved = rig
    context = CardContextNormalizer().normalize(raw_title=resolved.title)
    scoped = scope_rules(
        load_active_rubric().for_card(context.canonical_card_types, None), context)
    assert fingerprint({"applicable_rubric_rules": rule_content(scoped)})


def test_an_unidentifiable_listing_still_produces_a_usable_scoping(rig):
    """The recall-safe path: unknown product, every rule still present, the
    product-scoped one withheld rather than applied."""
    context = CardContextNormalizer().normalize(raw_title="mystery lot of 4")
    rubric = load_active_rubric()
    scoped = scope_rules(rubric.for_card(None, None), context)
    assert len(scoped) == len(rubric.rules)
    assert len(applicable(scoped)) < len(rubric.rules)
