import json

import pytest

from card_reviewer.knowledge import load_active_rubric
from card_reviewer.review.assembly import Assembled
from card_reviewer.review.enums import Mode, Scale
from card_reviewer.review.manifest import BUDGETS, BuiltManifest, build_manifest
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef
from card_reviewer.review.roles import ImageRole


def _refs(n, origin=EvidenceOrigin.NORMALIZED):
    return [EvidenceRef(artifact_id=f"a{i}", image_hash="h", origin=origin,
                        enhancement="clahe:clip=2.0"
                        if origin is EvidenceOrigin.ENHANCED else None,
                        view=f"corner_{i}")
            for i in range(n)]


def _assembled(refs, **kw):
    base = dict(evidence_refs={"corners:rounding": refs},
                detectability_flat={
                    Assembled.key(ImageRole.FRONT, "top_left", "corners", "rounding"):
                    Scale.HIGH.label},
                reason_codes_flat={},
                centering={"measurable": True, "horizontal": 52.0},
                anomalies=[], conflicts=[], limitations=[])
    return Assembled(**(base | kw))


def test_smart_and_deep_have_different_declared_budgets():
    assert BUDGETS[Mode.SMART] < BUDGETS[Mode.DEEP]


def test_selection_respects_the_mode_budget():
    m = build_manifest(_assembled(_refs(40)), Mode.SMART, []).payload
    assert len(m["artifacts"]) <= BUDGETS[Mode.SMART]


def test_deep_selects_more_than_smart_but_not_everything():
    """DEEP means maximum USEFUL evidence, not mechanically every artifact."""
    deep = build_manifest(_assembled(_refs(40)), Mode.DEEP, []).payload
    assert BUDGETS[Mode.SMART] < len(deep["artifacts"]) <= BUDGETS[Mode.DEEP]
    assert len(deep["artifacts"]) < 40


def test_duplicate_artifact_ids_are_eliminated():
    m = build_manifest(_assembled(_refs(3) + _refs(3)), Mode.DEEP, []).payload
    ids = [a["artifact_id"] for a in m["artifacts"]]
    assert len(ids) == len(set(ids))


def test_selection_is_deterministic_for_the_same_inputs():
    assert build_manifest(_assembled(_refs(30)), Mode.SMART, []).payload == (
        build_manifest(_assembled(_refs(30)), Mode.SMART, []).payload)


def test_selection_follows_the_declared_view_priority():
    """Determinism alone is not enough — the ORDER has to be the declared
    one, or which evidence survives the budget becomes an accident of
    dictionary iteration. The refs below deliberately vary by image hash so
    a hash-ordered sort would produce a different, still-deterministic,
    selection.
    """
    refs = [
        EvidenceRef(artifact_id="c1", image_hash="zzz",
                    origin=EvidenceOrigin.NORMALIZED, view="corner_top_left"),
        EvidenceRef(artifact_id="s1", image_hash="aaa",
                    origin=EvidenceOrigin.NORMALIZED, view="surface_original"),
        EvidenceRef(artifact_id="e1", image_hash="mmm",
                    origin=EvidenceOrigin.NORMALIZED, view="edge_top"),
    ]
    built = build_manifest(_assembled(refs), Mode.SMART, [])
    assert [a["view"] for a in built.payload["artifacts"]] == [
        "surface_original", "corner_top_left", "edge_top"]


def test_a_tight_budget_keeps_the_highest_priority_evidence():
    """The point of the ordering: when the budget bites, what survives is
    the most useful evidence rather than whatever sorted first."""
    refs = [
        EvidenceRef(artifact_id=f"e{i}", image_hash="aaa",
                    origin=EvidenceOrigin.NORMALIZED, view=f"edge_{i}")
        for i in range(BUDGETS[Mode.SMART])
    ] + [
        EvidenceRef(artifact_id="s1", image_hash="zzz",
                    origin=EvidenceOrigin.NORMALIZED, view="surface_original")
    ]
    built = build_manifest(_assembled(refs), Mode.SMART, [])
    assert "surface_original" in [a["view"] for a in built.payload["artifacts"]]


def test_the_index_resolves_every_sent_artifact_back_to_its_ref():
    """Without this the provider's citations cannot be resolved and
    provenance is lost at the round trip."""
    built = build_manifest(_assembled(_refs(5)), Mode.SMART, [])
    for artifact in built.payload["artifacts"]:
        ref = built.index[artifact["artifact_id"]]
        assert ref.origin.value == artifact["origin"]
        assert ref.image_hash


def test_the_index_contains_exactly_what_was_sent():
    built = build_manifest(_assembled(_refs(40)), Mode.SMART, [])
    assert set(built.index) == {a["artifact_id"]
                                for a in built.payload["artifacts"]}


def test_enhanced_artifacts_declare_their_enhancement_to_the_provider():
    """The provider must be able to tell an enhanced view from an original,
    or it cannot honour the conservative evidence standard."""
    built = build_manifest(_assembled(_refs(3, EvidenceOrigin.ENHANCED)),
                           Mode.DEEP, [])
    assert all(a["enhancement"] for a in built.payload["artifacts"])


def test_the_builder_version_is_not_in_the_provider_payload():
    """It would otherwise enter the vision fingerprint and re-bill every card
    on a builder bump the provider cannot see."""
    built = build_manifest(_assembled(_refs(3)), Mode.SMART, [])
    assert "builder_version" not in built.payload
    assert built.builder_meta["builder_version"]


def test_a_builder_bump_leaves_the_provider_payload_untouched(monkeypatch):
    import card_reviewer.review.manifest as mod

    before = build_manifest(_assembled(_refs(3)), Mode.SMART, []).payload
    monkeypatch.setattr(mod, "MANIFEST_BUILDER_VERSION", "9.9.9")
    assert build_manifest(_assembled(_refs(3)), Mode.SMART, []).payload == before


def test_the_manifest_carries_every_field_the_design_promised(rubric_rules):
    """A silently thinned payload makes the provider's answers worse while
    still looking like a working integration."""
    a = _assembled(
        _refs(3),
        reason_codes_flat={
            Assembled.key(ImageRole.FRONT, "top_left", "corners", "whitening"):
            "WHITE_BORDER"},
        conflicts=[{"field": "centering.horizontal", "values": [52.0, 61.0]}],
        limitations=["front is glared"],
        anomalies=[{"category": "surface", "defect_type": "scratches"}])
    m = build_manifest(a, Mode.DEEP, rubric_rules).payload
    for field in ("artifacts", "measurements", "detectability",
                  "detectability_reasons", "image_limitations", "conflicts",
                  "anomaly_candidates", "rubric_rules"):
        assert field in m, f"manifest omits {field}"
        # Present AND populated: an empty section is a silently thinned
        # payload, which degrades the provider's answers while still looking
        # like a working integration.
        assert m[field], f"manifest section {field} is empty"


def test_anomaly_candidates_carry_enhancement_provenance():
    a = _assembled(_refs(2), anomalies=[
        {"category": "surface", "defect_type": "scratches",
         "surfaced_by": "clahe", "visible_in_original": False,
         "artifact_id": "x"}])
    m = build_manifest(a, Mode.DEEP, []).payload
    assert m["anomaly_candidates"][0]["visible_in_original"] is False
    assert m["anomaly_candidates"][0]["surfaced_by"] == "clahe"


def test_the_manifest_carries_rubric_rule_content_not_a_version_string(
        rubric_rules):
    m = build_manifest(_assembled(_refs(3)), Mode.SMART, rubric_rules[:2]).payload
    assert isinstance(m["rubric_rules"], list)
    assert "statement" in m["rubric_rules"][0]


def test_no_pricing_information_reaches_the_manifest(rubric_rules):
    import re

    m = build_manifest(_assembled(_refs(3)), Mode.DEEP, rubric_rules).payload
    blob = repr(m).lower()
    for word in ("price", "cost", "profit", "purchase", "resale", "ev", "roi"):
        assert not re.search(rf"\b{word}\b", blob), f"pricing term {word!r} leaked"


def test_the_built_manifest_serializes_for_the_cache():
    """`manifest` is a cached stage, so its output must round-trip as JSON
    with the index intact — that index resolves provider citations."""
    built = build_manifest(_assembled(_refs(4)), Mode.SMART, [])
    revived = BuiltManifest.model_validate(json.loads(built.model_dump_json()))
    assert revived.payload == built.payload
    assert set(revived.index) == set(built.index)


def test_detectability_keys_are_stable_strings_not_python_reprs():
    """The manifest payload IS the vision input fingerprint.

    Formatting the tuple key gave "(<ImageRole.FRONT: 'front'>, 'corners',
    'whitening')" — a CPython enum repr. That is not a declared part of this
    system's cache identity and has changed between releases, so a Python
    upgrade would silently re-bill every card. It is also unreadable for the
    provider that has to act on it.
    """
    from card_reviewer.review.assembly import Assembled
    from card_reviewer.review.enums import Mode, Scale
    from card_reviewer.review.roles import ImageRole

    key = Assembled.key(ImageRole.FRONT, "top_left", "corners", "whitening")
    assembled = Assembled(detectability_flat={key: Scale.LOW.label},
                          reason_codes_flat={key: "GLARE"})
    payload = build_manifest(assembled, Mode.SMART, []).payload

    assert set(payload["detectability"]) == {"front|top_left|corners|whitening"}
    assert set(payload["detectability_reasons"]) == {
        "front|top_left|corners|whitening"}
    for section in ("detectability", "detectability_reasons"):
        for k in payload[section]:
            assert "<" not in k and "ImageRole" not in k, (
                f"{section} key {k!r} carries a Python repr")


def test_the_provider_receives_the_picture_of_every_anomaly_it_is_told_about():
    """Selection ranked by generic view name, so an anomaly's own artifact
    could fall outside the budget while the payload still cited it.

    The provider was told "an anomaly candidate at artifact a37" and never
    given a37 — a dangling id, against the rule that artifact ids map
    deterministically to the exact image blocks sent, and against sending
    the relevant anomaly views. `corner_` sorts before `edge_` and then
    alphabetically, so an anomaly on a later corner or any edge was the
    normal case, not an edge case.
    """
    refs = _refs(40)
    anomalies = [{"category": "corners", "defect_type": "rounding",
                  "region": "top_left", "artifact_id": refs[-1].artifact_id,
                  "surfaced_by": "original", "visible_in_original": True},
                 {"category": "edges", "defect_type": "chipping",
                  "region": "left", "artifact_id": refs[-2].artifact_id,
                  "surfaced_by": "original", "visible_in_original": True}]
    payload = build_manifest(_assembled(refs, anomalies=anomalies),
                             Mode.SMART, []).payload

    sent = {a["artifact_id"] for a in payload["artifacts"]}
    cited = {a["artifact_id"] for a in payload["anomaly_candidates"]
             if a["artifact_id"]}
    assert cited <= sent, (
        f"the provider is told about anomalies whose pictures it never "
        f"receives: {sorted(cited - sent)}")
    assert len(payload["artifacts"]) <= BUDGETS[Mode.SMART], (
        "the budget was abandoned rather than reprioritized")


def _mixed_pool():
    """Overviews plus per-region crops. `_refs` is corner views only, which
    is why the starvation below was invisible to the test that introduced
    it.

    `front_face` and `back_face` are in VIEW_PRIORITY but NO producer emits
    them, so this pool is a superset of what the engine makes rather than a
    copy of it — see
    `test_a_real_assembled_card_produces_a_usable_manifest` for the real
    thing."""
    views = ["surface_original", "front_face", "back_face"] + [
        f"corner_{c}" for c in ("bottom_left", "bottom_right",
                                "top_left", "top_right")
    ] + [f"edge_{e}" for e in ("bottom", "left", "right", "top")] + [
        "surface_clahe", "surface_sharpen"]
    return [EvidenceRef(artifact_id=f"a{i}", image_hash="h",
                        origin=EvidenceOrigin.NORMALIZED, view=v)
            for i, v in enumerate(views)]


def _anomaly(ref, category="edges", defect_type="chipping"):
    return {"category": category, "defect_type": defect_type,
            "region": "left", "artifact_id": ref.artifact_id,
            "surfaced_by": "original", "visible_in_original": True}


def test_the_card_itself_is_still_sent_when_every_region_has_an_anomaly():
    """Prioritizing anomaly crops took the whole budget on a card with eight
    candidates: SMART sent eight crops and NO view of the card at all.

    The provider is asked for print lines, dimples, stains and foil
    artifacts and must answer `category_assessability["surface"]`, and it
    was being given no overview and no surface view — so surface became
    structurally unassessable. It fails safe, since unassessable surface
    blocks PASS, but it is self-inflicted evidence removal, and the
    provider payload is required to carry the useful originals.
    """
    refs = _mixed_pool()
    crops = [r for r in refs if r.view.startswith(("corner_", "edge_"))]
    payload = build_manifest(
        _assembled(refs, anomalies=[_anomaly(r) for r in crops]),
        Mode.SMART, []).payload

    views = {a["view"] for a in payload["artifacts"]}
    assert "surface_original" in views, (
        f"SMART sent no view of the card as a whole: {sorted(views)}")


def test_no_anomaly_claims_an_artifact_that_was_not_sent():
    """The other half, which must hold at the SAME time: an id in the
    payload names a block the provider actually received. When the budget
    cannot carry an anomaly's crop, the candidate is still described — its
    category, region and provenance are real information — but it stops
    claiming a picture that is not there.
    """
    refs = _mixed_pool()
    crops = [r for r in refs if r.view.startswith(("corner_", "edge_"))]
    payload = build_manifest(
        _assembled(refs, anomalies=[_anomaly(r) for r in crops]),
        Mode.SMART, []).payload

    sent = {a["artifact_id"] for a in payload["artifacts"]}
    cited = {a["artifact_id"] for a in payload["anomaly_candidates"]
             if a["artifact_id"]}
    assert cited <= sent, f"dangling anomaly artifact ids: {sorted(cited - sent)}"
    assert len(payload["anomaly_candidates"]) == len(crops), (
        "an anomaly candidate was deleted rather than un-cited")


def _multi_image_pool(n_images):
    """One overview plus four corner crops per photograph, which is what a
    real multi-image listing assembles to."""
    refs = []
    for i in range(n_images):
        refs.append(EvidenceRef(artifact_id=f"ov{i}", image_hash=f"h{i}",
                                origin=EvidenceOrigin.NORMALIZED,
                                view="surface_original"))
        for corner in ("bottom_left", "bottom_right", "top_left", "top_right"):
            refs.append(EvidenceRef(
                artifact_id=f"c{i}_{corner}", image_hash=f"h{i}",
                origin=EvidenceOrigin.NORMALIZED, view=f"corner_{corner}"))
    return refs


def test_overviews_do_not_multiply_with_the_number_of_photographs():
    """Pinning whole-card views fixed single-photo starvation and broke the
    multi-photo case: overviews are per IMAGE, so a six-photograph listing
    spent six of SMART's eight slots on six copies of the same view and
    sent ONE distinct region crop.

    Measured on six real photographs before this cap: 6 overviews, 1
    distinct region view, 39 of 41 candidates un-cited. Listings with many
    photographs are the case the engine exists for, so the cap belongs
    here rather than in the single-photo fixture that missed it.
    """
    refs = _multi_image_pool(6)
    crops = [r for r in refs if r.view.startswith("corner_")]
    payload = build_manifest(
        _assembled(refs, anomalies=[_anomaly(r, "corners", "rounding")
                                    for r in crops]),
        Mode.SMART, []).payload

    views = [a["view"] for a in payload["artifacts"]]
    overviews = [v for v in views if not v.startswith(("corner_", "edge_"))]
    assert 1 <= len(overviews) <= 2, (
        f"{len(overviews)} whole-card views in an 8-slot budget: {views}")
    assert len({a["artifact_id"] for a in payload["artifacts"]
                if a["view"].startswith("corner_")}) >= 5, (
        f"crops were crowded out by duplicate overviews: {views}")


def test_the_crops_that_fit_cover_the_regions_rather_than_repeating_one():
    """Sorting crops by view name clustered them alphabetically, so a
    six-photograph listing spent every crop slot on `corner_bottom_left` —
    the same corner six times, and no view at all of the other three.

    The provider is asked to assess all four corners and can only answer
    from what it was sent. Covering each region once before showing any
    region twice is the same budget spent on evidence that differs.
    """
    refs = _multi_image_pool(6)
    crops = [r for r in refs if r.view.startswith("corner_")]
    payload = build_manifest(
        _assembled(refs, anomalies=[_anomaly(r, "corners", "rounding")
                                    for r in crops]),
        Mode.SMART, []).payload

    corner_views = {a["view"] for a in payload["artifacts"]
                    if a["view"].startswith("corner_")}
    assert len(corner_views) == 4, (
        f"the crop budget went to {len(corner_views)} of the four corners: "
        f"{sorted(corner_views)}")


def test_one_photograph_still_gets_its_overview():
    """The cap must not undo the fix it is capping."""
    refs = _multi_image_pool(1)
    crops = [r for r in refs if r.view.startswith("corner_")]
    payload = build_manifest(
        _assembled(refs, anomalies=[_anomaly(r, "corners", "rounding")
                                    for r in crops]),
        Mode.SMART, []).payload
    assert "surface_original" in {a["view"] for a in payload["artifacts"]}


def test_a_real_assembled_card_produces_a_usable_manifest(tmp_path):
    """Every other test in this file hands `build_manifest` hand-built
    refs, and one of those pools claims to be "as a real card produces"
    while containing `front_face` and `back_face` views that NO producer in
    this repository emits.

    That gap is how the starvation bug hid: a fixture of corner views only
    cannot show an overview being crowded out. This runs the real geometry,
    observability and measurement producers into `assemble` and then into
    the manifest, so the payload is checked against evidence the engine
    actually makes.

    What it does NOT do is guard the selection POLICY: its fixture has two
    anomalies against an eight-slot budget, so `cited <= sent` holds by
    construction and every mutation of the cap or the ordering leaves it
    green. It catches shape, type and view-name drift between the producers
    and the manifest. The budget rules are guarded by the tests above,
    which is where a cap mutation is meant to fail.
    """
    from card_reviewer.review.assembly import (
        ImageStageOutputs, assemble, to_image_evidence,
    )
    from card_reviewer.review.enums import Provenance
    from card_reviewer.review.imaging.geometry import analyze
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import analyze as observe
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.roles import ResolvedRole
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    outputs, roles = [], {}
    for spec, role in ((CardSpec(border_color=(20, 20, 20),
                                 corner_damage={"top_left": 0.9}),
                        ImageRole.FRONT),
                       (CardSpec(text_heavy=True), ImageRole.BACK)):
        data = render_png(spec)
        image_hash = store.put_image(data)
        geometry = analyze(data, store, image_hash)
        outputs.append(ImageStageOutputs(
            image_hash=image_hash, preflight={"global_sharpness": 120.0},
            geometry=geometry.model_dump(),
            observability=observe(geometry, store, image_hash).model_dump(),
            cv_measurements=measure_all(geometry, store,
                                        image_hash).model_dump()))
        roles[image_hash] = ResolvedRole(
            image_hash=image_hash, role=role,
            provenance=Provenance.SUPPLIED, confidence=1.0)

    assembled = assemble(to_image_evidence(outputs), roles)
    built = build_manifest(assembled, Mode.SMART, [])
    payload = built.payload

    sent = {a["artifact_id"] for a in payload["artifacts"]}
    assert sent, "a real two-photograph card produced an empty payload"
    assert len(sent) <= BUDGETS[Mode.SMART]

    cited = {a["artifact_id"] for a in payload["anomaly_candidates"]
             if a["artifact_id"]}
    assert cited <= sent, f"dangling ids on real evidence: {sorted(cited - sent)}"

    views = {a["view"] for a in payload["artifacts"]}
    assert any(not v.startswith(("corner_", "edge_")) for v in views), (
        f"no whole-card view of a real card reached the provider: {sorted(views)}")

    # The index is what resolves a provider citation back to an artifact
    # after a restart, so every id offered must be in it.
    assert sent <= set(built.index), (
        f"ids sent with no index entry: {sorted(sent - set(built.index))}")


def test_both_faces_get_a_whole_card_view_when_both_are_present():
    """Capping pinned overviews at two fixed duplication and introduced a
    worse failure on an ordinary listing: the two pinned views could be the
    SAME face, leaving the other with no whole-card view at all.

    Reproduced end to end on four real photographs — two fronts, two backs
    — where both overviews came from one face in SMART and in DEEP alike.
    The face that decides PSA 10 can be the one that loses its view, and
    its categories then come back not_assessable: recall-safe, but a billed
    call spent on half a card.

    `EvidenceRef` carries no role, so the roles have to be handed in. Two
    pinned views mean one per face wherever both exist.
    """
    refs, roles = [], {}
    for i, face in enumerate((ImageRole.FRONT, ImageRole.FRONT,
                              ImageRole.BACK, ImageRole.BACK)):
        roles[f"h{i}"] = face
        refs.append(EvidenceRef(artifact_id=f"ov{i}", image_hash=f"h{i}",
                                origin=EvidenceOrigin.NORMALIZED,
                                view="surface_original"))
        for corner in ("bottom_left", "bottom_right", "top_left", "top_right"):
            refs.append(EvidenceRef(
                artifact_id=f"c{i}_{corner}", image_hash=f"h{i}",
                origin=EvidenceOrigin.NORMALIZED, view=f"corner_{corner}"))

    crops = [r for r in refs if r.view.startswith("corner_")]
    payload = build_manifest(
        _assembled(refs, anomalies=[_anomaly(r, "corners", "rounding")
                                    for r in crops]),
        Mode.SMART, [], image_roles=roles).payload

    by_id = {r.artifact_id: r for r in refs}
    faces = sorted(roles[by_id[a["artifact_id"]].image_hash].value
                   for a in payload["artifacts"]
                   if not a["view"].startswith(("corner_", "edge_")))
    assert faces == ["back", "front"], (
        f"the two whole-card views sent were {faces}; one face has none")


def test_one_face_present_still_gets_two_photographs_pinned():
    """The fallback: with only fronts, "one per face" must not collapse to
    a single view — two photographs of the front are still worth more than
    one, and the cap is two."""
    # Four corners each, not two: with only two the budget had slack, so
    # the second overview arrived on ordinary competition and removing the
    # distinct-image clause changed nothing the test could see.
    refs, roles = [], {}
    for i in range(3):
        roles[f"h{i}"] = ImageRole.FRONT
        refs.append(EvidenceRef(artifact_id=f"ov{i}", image_hash=f"h{i}",
                                origin=EvidenceOrigin.NORMALIZED,
                                view="surface_original"))
        for corner in ("bottom_left", "bottom_right", "top_left", "top_right"):
            refs.append(EvidenceRef(
                artifact_id=f"c{i}_{corner}", image_hash=f"h{i}",
                origin=EvidenceOrigin.NORMALIZED, view=f"corner_{corner}"))

    # Anomalies on every crop, so the crops compete for the budget — with
    # nothing competing, overviews win on ordinary priority and the pin is
    # not what is being measured.
    crops = [r for r in refs if r.view.startswith("corner_")]
    payload = build_manifest(
        _assembled(refs, anomalies=[_anomaly(r, "corners", "rounding")
                                    for r in crops]),
        Mode.SMART, [], image_roles=roles).payload
    overviews = [a for a in payload["artifacts"]
                 if not a["view"].startswith(("corner_", "edge_"))]
    assert len(overviews) == 2, (
        f"a front-only listing pinned {len(overviews)} whole-card views")
    # Documentation, not a second guard: each image emits exactly one
    # `surface_original`, so two pinned views ARE two photographs and this
    # cannot fail while the assertion above passes. It states the property
    # the pin is for, so a producer that ever emits two whole-card views
    # for one image finds the claim already written down.
    by_id = {r.artifact_id: r for r in refs}
    assert len({by_id[a["artifact_id"]].image_hash for a in overviews}) == 2


def test_an_unresolved_photograph_does_not_take_a_real_faces_pinned_slot():
    """`unknown` is not a face, and the rest of the engine already says so:
    `assemble` gives an unknown-role image its anomalies but never lets it
    "claim a face", and `faces_present` excludes it.

    The pin did not, so a listing whose third photograph could not be
    resolved spent a pinned slot on it and left the FRONT — the face that
    decides PSA 10 — with no whole-card view. That is a live path, not a
    corner: `screen` supplies no roles, so every role is inferred and
    `roles.py` returns UNKNOWN for the whole ambiguous band by design.

    An unresolved photograph still competes for the remaining budget; it
    just cannot displace a face that was actually identified.
    """
    # Artifact ids are content hashes, so the unresolved photograph can
    # sort FIRST — which is exactly how this reproduced on real evidence.
    refs, roles = [], {}
    for tag, face in (("a", ImageRole.UNKNOWN), ("m", ImageRole.FRONT),
                      ("z", ImageRole.BACK)):
        roles[f"h{tag}"] = face
        refs.append(EvidenceRef(artifact_id=f"{tag}_ov", image_hash=f"h{tag}",
                                origin=EvidenceOrigin.NORMALIZED,
                                view="surface_original"))
        for corner in ("bottom_left", "bottom_right", "top_left", "top_right"):
            refs.append(EvidenceRef(
                artifact_id=f"{tag}_c_{corner}", image_hash=f"h{tag}",
                origin=EvidenceOrigin.NORMALIZED, view=f"corner_{corner}"))

    crops = [r for r in refs if r.view.startswith("corner_")]
    payload = build_manifest(
        _assembled(refs, anomalies=[_anomaly(r, "corners", "rounding")
                                    for r in crops]),
        Mode.SMART, [], image_roles=roles).payload

    by_id = {r.artifact_id: r for r in refs}
    faces = sorted(roles[by_id[a["artifact_id"]].image_hash].value
                   for a in payload["artifacts"]
                   if not a["view"].startswith(("corner_", "edge_")))
    assert faces == ["back", "front"], (
        f"an unresolved photograph displaced an identified face: {faces}")


def test_every_view_a_producer_emits_has_a_declared_priority(tmp_path):
    """`_rank` gives an unrecognized view the LOWEST priority silently.

    So a producer adding a view name lands it at the bottom of the budget
    with no signal — which is how `front_face` and `back_face` sat in
    VIEW_PRIORITY for months while nothing emitted them, the mirror image
    of the same gap. A vocabulary is a contract only if both halves are
    checked against each other.
    """
    from card_reviewer.review.assembly import (
        ImageStageOutputs, assemble, to_image_evidence,
    )
    from card_reviewer.review.enums import Provenance
    from card_reviewer.review.imaging.geometry import analyze
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import analyze as observe
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.manifest import VIEW_PRIORITY, _rank
    from card_reviewer.review.roles import ResolvedRole
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    outputs, roles = [], {}
    for spec, role in (
        (CardSpec(border_color=(20, 20, 20), scratches=[0.8],
                  corner_damage={"top_left": 0.9}), ImageRole.FRONT),
        (CardSpec(text_heavy=True), ImageRole.BACK),
    ):
        data = render_png(spec)
        image_hash = store.put_image(data)
        geometry = analyze(data, store, image_hash)
        outputs.append(ImageStageOutputs(
            image_hash=image_hash, preflight={"global_sharpness": 120.0},
            geometry=geometry.model_dump(),
            observability=observe(geometry, store, image_hash).model_dump(),
            cv_measurements=measure_all(geometry, store,
                                        image_hash).model_dump()))
        roles[image_hash] = ResolvedRole(
            image_hash=image_hash, role=role,
            provenance=Provenance.SUPPLIED, confidence=1.0)

    assembled = assemble(to_image_evidence(outputs), roles)
    emitted = {ref.view for refs in assembled.evidence_refs.values()
               for ref in refs}
    assert emitted, "the producers emitted no views at all"

    unranked = sorted(v for v in emitted if _rank(v) == len(VIEW_PRIORITY))
    assert not unranked, (
        f"views the producers emit that VIEW_PRIORITY does not rank, so they "
        f"sort last with no signal: {unranked}")
