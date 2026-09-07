"""A finding belongs to a face, and both I1 and fusion must know which.

The key is (face, region, category, defect_type) and `detectability_for`
discarded the face before taking the minimum — so a FRONT finding's I1
adequacy was judged against the worse of the two faces. Its own docstring
cites "a sharp front vouch for a blown-out back" as the motivation and
achieved the exact mirror image: a positively measured 78/22 miscut front
dropped from REJECT to REVIEW purely because the BACK was a borderless
design, a structural property of a face that says nothing about the front's
border measurement.

Fusion had the same omission: it correlates on category, defect type and
overlapping normalized box, with no face discriminator, and boxes are
normalized per card — so a defect on the front and a different defect at the
same corner of the BACK merged into one. "Do not double-penalize
corroboration" became "do not count the second face".
"""

import pytest

from card_reviewer.review.enums import FindingState, Scale
from card_reviewer.review.findings import Finding, FindingProducer, Severity
from card_reviewer.review.fusion import fuse
from card_reviewer.review.heuristic import detectability_for
from card_reviewer.review.provenance import (
    EvidenceOrigin, EvidenceRef, NormalizedBox,
)
from card_reviewer.review.roles import ImageRole

BOX = NormalizedBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)


def _finding(image_hash, producer=FindingProducer.HEURISTIC,
             severity=Severity.MODERATE):
    return Finding(
        defect_type="rounding", category="corners",
        state=FindingState.OBSERVED, producer=producer, confidence=0.95,
        psa10_relevant=True, severity=severity, location=BOX,
        evidence=[EvidenceRef(artifact_id=f"a-{image_hash}",
                              image_hash=image_hash,
                              origin=EvidenceOrigin.ORIGINAL,
                              view="corner_top_left", region=BOX)])


DETECTABILITY = {
    (ImageRole.FRONT, "top_left", "corners", "rounding"): Scale.HIGH,
    (ImageRole.BACK, "top_left", "corners", "rounding"): Scale.LOW,
}


def test_a_front_finding_is_judged_against_the_front():
    assert detectability_for(DETECTABILITY, "corners", "rounding",
                             "top_left", ImageRole.FRONT) is Scale.HIGH


def test_a_back_finding_is_judged_against_the_back():
    assert detectability_for(DETECTABILITY, "corners", "rounding",
                             "top_left", ImageRole.BACK) is Scale.LOW


def test_an_unlocated_finding_still_takes_the_weakest():
    """Unchanged where the face is genuinely unknown: whatever we have not
    narrowed to is somewhere the finding might be."""
    assert detectability_for(DETECTABILITY, "corners", "rounding") is Scale.LOW


def test_a_blown_out_back_does_not_weaken_a_front_measurement(rubric_scoped):
    """The reported consequence, end to end."""
    from card_reviewer.review.enums import Coverage, Verdict
    from card_reviewer.review.heuristic import HeuristicResult
    from card_reviewer.review.policies.combine_v1 import combine
    from card_reviewer.review.policies.coverage_v1 import CoverageResult

    front_finding = _finding("front-hash", severity=Severity.SEVERE)
    roles = {"front-hash": ImageRole.FRONT, "back-hash": ImageRole.BACK}

    def verdict(detectability):
        return combine(
            HeuristicResult(findings=[front_finding]), None,
            CoverageResult(outcome=Coverage.SUFFICIENT, rankable=True),
            card_context_known=True, scoped_rules=rubric_scoped,
            detectability=detectability, image_roles=roles,
        ).verdict

    good_back = dict(DETECTABILITY)
    good_back[(ImageRole.BACK, "top_left", "corners", "rounding")] = Scale.HIGH

    assert verdict(good_back) is verdict(DETECTABILITY), (
        "the back's detectability changed the verdict on a front finding")


def test_the_same_corner_on_two_faces_is_two_defects():
    """Normalized boxes are per card, so the front's top-left corner and the
    back's occupy the same box. Without a face discriminator a card damaged
    on both faces reported one defect."""
    roles = {"front-hash": ImageRole.FRONT, "back-hash": ImageRole.BACK}
    fused = fuse([_finding("front-hash"), _finding("back-hash")], roles)
    assert len(fused) == 2, "damage on two faces was counted once"


def test_two_producers_on_the_same_face_still_fuse():
    """The behaviour fusion exists for must survive."""
    roles = {"front-hash": ImageRole.FRONT}
    fused = fuse([_finding("front-hash", FindingProducer.HEURISTIC),
                  _finding("front-hash", FindingProducer.VISION)], roles)
    assert len(fused) == 1
    assert len(fused[0].sources) == 2


def test_fusion_without_roles_does_not_merge_across_photographs():
    """`>= 1` asserted nothing — it held whether fusion merged or not.

    Without a role map `_face_of` returns None for both findings, and
    `None != None` is false, so two findings from DIFFERENT photographs
    correlated and merged: exactly the bug the face dimension exists to
    prevent, still reachable through every caller that has no roles.

    Merging requires positive evidence that two findings are the same
    thing. Different images with no map to say they are the same face is
    not that evidence.
    """
    fused = fuse([_finding("front-hash"), _finding("back-hash")])
    assert len(fused) == 2, "two photographs' findings merged with no roles"


def test_two_producers_on_one_photograph_fuse_without_roles():
    """The other half: refusing to merge without roles must not become
    refusing to merge at all. Two producers describing one defect on ONE
    photograph are the same defect whether or not anything named its face,
    and splitting them would double-penalize corroboration."""
    fused = fuse([_finding("front-hash", FindingProducer.HEURISTIC),
                  _finding("front-hash", FindingProducer.VISION)])
    assert len(fused) == 1
    assert len(fused[0].sources) == 2


def test_the_pipeline_supplies_the_role_map(tmp_path):
    """Threading the face through combine and fusion is decorative unless
    the pipeline actually hands over the roles. A card damaged at the same
    corner of BOTH faces is two defects; with no role map the normalized
    boxes coincide and it reports one.
    """
    from card_reviewer.review.enums import Mode
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.ingest.adapter import ManualAdapter
    from card_reviewer.review.models import CandidateInput
    from card_reviewer.review.pipeline import ReviewPipeline
    from card_reviewer.review.storage.artifacts import ArtifactStore
    from card_reviewer.review.storage.migrations import connect, migrate
    from card_reviewer.review.storage.repository import SqliteRepository

    store = ArtifactStore(tmp_path / "store")
    conn = connect(tmp_path / "t.db")
    migrate(conn)

    damaged = CardSpec(border_color=(20, 20, 20),
                       corner_damage={"bottom_left": 0.9})
    paths = []
    for i, spec in enumerate((damaged,
                              damaged.model_copy(update={"text_heavy": True}))):
        path = tmp_path / f"{i}.png"
        path.write_bytes(render_png(spec))
        paths.append(path)
    resolved = ManualAdapter(store).resolve(CandidateInput(
        source="manual", title="2023 Topps Chrome", candidate_id="c",
        image_paths=paths,
        supplied_roles={str(paths[0]): "front", str(paths[1]): "back"}))

    review = ReviewPipeline(SqliteRepository(conn), store).review(
        resolved, Mode.OFF)
    conn.close()

    corners = [f for f in review.defects_found if f["category"] == "corners"]
    assert len(corners) >= 2, (
        "the same corner damaged on both faces was reported as one defect")


def test_the_heuristic_judges_a_front_finding_against_the_front(tmp_path):
    """The face was threaded into `combine` and not into `evaluate`.

    `_state_for`'s promotion floor runs in the HEURISTIC, before combine ever
    sees the finding — and it took the minimum across both faces. Once a
    finding is `suspected` no face-aware lookup downstream can recover it,
    because I1 requires OBSERVED. Verified end to end: a positively measured
    78/22 miscut front was REJECT with a normal back and dropped to REVIEW
    purely because the BACK was borderless.

    It bites hardest on centering, which is the ONLY defect type CV can
    promote to observed at all — so the face fix changed no heuristic
    outcome whatsoever until this.
    """
    from card_reviewer.review.assembly import ImageStageOutputs, assemble, to_image_evidence
    from card_reviewer.review.enums import FindingState, Provenance
    from card_reviewer.review.heuristic import evaluate
    from card_reviewer.review.imaging.geometry import analyze
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import analyze as observe
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.roles import ResolvedRole
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")

    def outputs(spec):
        data = render_png(spec)
        image_hash = store.put_image(data)
        geometry = analyze(data, store, image_hash)
        return image_hash, ImageStageOutputs(
            image_hash=image_hash, preflight={"global_sharpness": 120.0},
            geometry=geometry.model_dump(),
            observability=observe(geometry, store, image_hash).model_dump(),
            cv_measurements=measure_all(geometry, store, image_hash).model_dump())

    miscut = CardSpec(border_color=(20, 20, 20), h_centering=78.0)
    states = {}
    for label, back_spec in (("normal back", CardSpec(border_color=(20, 20, 20))),
                             ("borderless back", CardSpec(borderless=True))):
        front_hash, front = outputs(miscut)
        back_hash, back = outputs(back_spec)
        roles = {
            front_hash: ResolvedRole(image_hash=front_hash, role=ImageRole.FRONT,
                                     provenance=Provenance.SUPPLIED, confidence=1.0),
            back_hash: ResolvedRole(image_hash=back_hash, role=ImageRole.BACK,
                                    provenance=Provenance.SUPPLIED, confidence=1.0),
        }
        assembled = assemble(to_image_evidence([front, back]), roles)
        result = evaluate(assembled, [], image_roles=roles)
        centering = [f for f in result.findings if f.category == "centering"]
        states[label] = centering[0].state if centering else None

    assert states["normal back"] is FindingState.OBSERVED, (
        "a measured 78/22 miscut front was not promoted even with a good back")
    assert states["borderless back"] == states["normal back"], (
        f"the back changed the front's promotion: {states}")


def _ref(image_hash, view="corner_bottom_left"):
    from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef, NormalizedBox
    return EvidenceRef(artifact_id=f"a-{image_hash}", image_hash=image_hash,
                       origin=EvidenceOrigin.NORMALIZED, view=view,
                       region=NormalizedBox(x0=0.0, y0=0.8, x1=0.2, y1=1.0))


def test_a_finding_never_borrows_another_photographs_evidence():
    """`_own` ended in `mine or refs`, so an anomaly whose OWN image had no
    refs under that key silently inherited every other image's.

    The docstring already claimed to narrow by image. The fallback defeated
    it in exactly the case it existed for: the finding then belongs to no
    single face, which breaks I1's per-face adequacy prong and fusion's
    per-face separation — and it attributes the back's evidence to a front
    finding, which is fabricated provenance.

    A finding with no evidence from its own photograph has no evidence.
    `evaluate` already drops findings with empty refs, so the honest
    outcome is no finding at all.
    """
    from card_reviewer.review.assembly import Assembled
    from card_reviewer.review.enums import Scale
    from card_reviewer.review.heuristic import evaluate
    from detectability_helpers import regions_for

    flat = {Assembled.key(r, region, "corners", "whitening"): Scale.HIGH.label
            for r in (ImageRole.FRONT, ImageRole.BACK)
            for region in regions_for("corners")}
    assembled = Assembled(
        detectability_flat=flat, faces_present=["front", "back"],
        centering={"measurable": False},
        anomalies=[{"category": "corners", "defect_type": "whitening",
                    "region": "bottom_left", "confidence": 0.9,
                    "image_hash": "back-hash"}],
        # Refs exist for the FRONT only. The anomaly is on the back.
        evidence_refs={"corners:whitening": [_ref("front-hash")]})

    findings = evaluate(assembled, []).findings
    borrowed = [f for f in findings
                if any(r.image_hash != "back-hash" for r in f.evidence)]
    assert not borrowed, (
        f"a back anomaly is carrying the front's evidence: "
        f"{[[r.image_hash for r in f.evidence] for f in borrowed]}")

    # The assertion above also holds if the finding is simply deleted, so
    # say what should happen when its own evidence IS present: the finding
    # survives, citing only its own photograph. Deletion is safe only
    # because assembly never emits an anomaly without own-image evidence —
    # see test_every_anomaly_assembly_emits_carries_its_own_photographs_evidence.
    assembled.evidence_refs["corners:whitening"].append(_ref("back-hash"))
    kept = evaluate(assembled, []).findings
    assert kept, "a back anomaly with its own evidence was dropped"
    assert {r.image_hash for f in kept for r in f.evidence} == {"back-hash"}


def test_the_centering_finding_rests_only_on_the_photo_it_was_measured_from():
    """Centering is measured on ONE image — assembly records which, in
    `best_for["centering"]` — but the finding took every ref filed under
    `centering:border_ratio`, unioned across images.

    So a front measurement carried the back's refs, and the face used for
    its promotion floor was whichever ref happened to sort first. A finding
    that spans faces satisfies I1's adequacy prong at no face in particular.
    """
    from card_reviewer.review.assembly import Assembled
    from card_reviewer.review.enums import Scale
    from card_reviewer.review.heuristic import evaluate
    from detectability_helpers import regions_for

    flat = {Assembled.key(r, region, "centering", "border_ratio"): Scale.HIGH.label
            for r in (ImageRole.FRONT, ImageRole.BACK)
            for region in regions_for("centering")}
    assembled = Assembled(
        detectability_flat=flat, faces_present=["front", "back"],
        centering={"horizontal": 78.0, "vertical": 50.0, "measurable": True},
        best_for={"centering": "front-hash"},
        evidence_refs={"centering:border_ratio": [_ref("back-hash", "surface_original"),
                                                  _ref("front-hash", "surface_original")]})

    findings = [f for f in evaluate(assembled, []).findings
                if f.category == "centering"]
    assert findings, "a 78/22 card produced no centering finding"
    assert {r.image_hash for f in findings for r in f.evidence} == {"front-hash"}, (
        "the centering finding carries evidence from a photo it was not "
        "measured from")


def test_a_finding_whose_evidence_spans_photos_still_fuses_with_a_front_one():
    """A face can be unknown even WITH a role map: `face_of_finding` returns
    None when a finding's evidence spans more than one image.

    So the two branches of `_same_face` are not "roles or no roles" — they
    are "both faces known" against "at least one unknown", and requiring
    only ONE known face would send this pair down the equality branch,
    where `front != None` refuses the merge. Two producers describing one
    corner would then be counted twice, which is the double-penalty rule
    the correlation exists to honour.
    """
    spanning = Finding(
        defect_type="rounding", category="corners",
        state=FindingState.OBSERVED, producer=FindingProducer.VISION,
        confidence=0.9, psa10_relevant=True, severity=Severity.MODERATE,
        location=BOX,
        evidence=[EvidenceRef(artifact_id=f"a-{h}", image_hash=h,
                              origin=EvidenceOrigin.ORIGINAL,
                              view="corner_top_left", region=BOX)
                  for h in ("front-hash", "back-hash")])
    roles = {"front-hash": ImageRole.FRONT, "back-hash": ImageRole.BACK}

    fused = fuse([_finding("front-hash"), spanning], roles)
    assert len(fused) == 1, "corroboration on one corner was counted twice"
    assert len(fused[0].sources) == 2


def _outputs(store, spec):
    from card_reviewer.review.assembly import ImageStageOutputs
    from card_reviewer.review.imaging.geometry import analyze
    from card_reviewer.review.imaging.measure import measure_all
    from card_reviewer.review.imaging.observability import analyze as observe
    from card_reviewer.review.imaging.synthetic import render_png

    data = render_png(spec)
    image_hash = store.put_image(data)
    geometry = analyze(data, store, image_hash)
    return image_hash, ImageStageOutputs(
        image_hash=image_hash, preflight={"global_sharpness": 120.0},
        geometry=geometry.model_dump(),
        observability=observe(geometry, store, image_hash).model_dump(),
        cv_measurements=measure_all(geometry, store, image_hash).model_dump())


def test_a_miscut_survives_an_unmeasurable_photograph_listed_first(tmp_path):
    """`best_for["centering"]` is NOT the image the measurement came from.

    `_centering` deliberately carries the WORST measurable front, precisely
    because `fronts[0]` "made the answer depend on the order the photographs
    happened to be listed in: with an unmeasurable photo first, a 78/22
    miscut DISAPPEARED". `_best_for` still returns `fronts[0]`.

    So narrowing the finding's refs to `best_for` reintroduced that bug
    through the consumer: refs under `centering:border_ratio` are written
    only for photographs whose centering was measurable, so the borderless
    first photo contributes none, the narrowing yields nothing, and
    `evaluate` drops the finding. A measured 80/20 miscut disappears and
    the card reads clean.
    """
    from card_reviewer.review.assembly import assemble, to_image_evidence
    from card_reviewer.review.enums import Provenance
    from card_reviewer.review.heuristic import evaluate
    from card_reviewer.review.imaging.synthetic import CardSpec
    from card_reviewer.review.roles import ResolvedRole
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    borderless_hash, borderless = _outputs(store, CardSpec(borderless=True))
    miscut_hash, miscut = _outputs(
        store, CardSpec(border_color=(20, 20, 20), h_centering=80.0))

    roles = {h: ResolvedRole(image_hash=h, role=ImageRole.FRONT,
                             provenance=Provenance.SUPPLIED, confidence=1.0)
             for h in (borderless_hash, miscut_hash)}
    assembled = assemble(to_image_evidence([borderless, miscut]), roles)
    assert assembled.centering.get("measurable"), "the miscut was not measured"

    findings = [f for f in evaluate(assembled, []).findings
                if f.category == "centering"]
    assert findings, "a measured 80/20 miscut disappeared behind a borderless photo"
    assert {r.image_hash for f in findings for r in f.evidence} == {miscut_hash}, (
        "the finding cites a photograph the measurement did not come from")


def test_damage_in_a_photo_whose_role_is_unknown_is_not_silently_dropped(tmp_path):
    """An unknown-role image contributed anomalies but no evidence refs, so
    once findings stopped borrowing another photograph's evidence its
    anomalies resolved to nothing and were dropped by `evaluate`.

    A photograph whose role could not be resolved then contributed NOTHING
    and the card read clean — missing role metadata manufacturing a cleaner
    result, which is I2. Its refs are keyed by category and region, never by
    face, so they can be carried without claiming a face; detectability
    still has no face key for it and so falls back to the weakest, which is
    the conservative direction.
    """
    from card_reviewer.review.assembly import assemble, to_image_evidence
    from card_reviewer.review.enums import Provenance
    from card_reviewer.review.imaging.synthetic import CardSpec
    from card_reviewer.review.roles import ResolvedRole
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    front_hash, front = _outputs(store, CardSpec(border_color=(20, 20, 20)))
    odd_hash, odd = _outputs(store, CardSpec(
        border_color=(20, 20, 20),
        corner_damage={"top_left": 0.9, "bottom_right": 0.9}))

    roles = {
        front_hash: ResolvedRole(image_hash=front_hash, role=ImageRole.FRONT,
                                 provenance=Provenance.SUPPLIED, confidence=1.0),
        odd_hash: ResolvedRole(image_hash=odd_hash, role=ImageRole.UNKNOWN,
                               provenance=Provenance.INFERRED, confidence=0.2),
    }
    assembled = assemble(to_image_evidence([front, odd]), roles)

    from_unknown = [a for a in assembled.anomalies
                    if a.get("image_hash") == odd_hash]
    # Asserted, not skipped. A skip here would silently disarm the test the
    # day the fixture stopped producing anomalies, which is exactly when it
    # would stop protecting anything.
    assert from_unknown, "fixture produced no anomaly on the unknown-role image"

    carried = [r for refs in assembled.evidence_refs.values() for r in refs
               if r.image_hash == odd_hash]
    assert carried, (
        "an unknown-role photograph raised anomalies but carries no evidence, "
        "so every one of them is dropped and the card reads clean")


@pytest.mark.parametrize("role", [ImageRole.FRONT, ImageRole.BACK,
                                  ImageRole.UNKNOWN])
def test_every_anomaly_assembly_emits_carries_its_own_photographs_evidence(
        tmp_path, role):
    """The invariant that keeps the silent drop unreachable.

    `evaluate` drops a finding whose own photograph contributed no evidence
    refs. That is the right call for a finding with no evidence, but it is
    only safe while assembly never emits such an anomaly — otherwise real
    damage disappears with nothing recorded, which is I2.

    Measured on the corpus at the time of writing: 0 of 422 anomalies
    across 88 real photographs lacked own-image evidence. This holds that
    line for every role, including UNKNOWN, which is where it was broken.
    """
    from card_reviewer.review.assembly import assemble, to_image_evidence
    from card_reviewer.review.enums import Provenance
    from card_reviewer.review.imaging.synthetic import CardSpec
    from card_reviewer.review.roles import ResolvedRole
    from card_reviewer.review.storage.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    image_hash, outputs = _outputs(store, CardSpec(
        border_color=(20, 20, 20), scratches=[0.8],
        corner_damage={"top_left": 0.9, "bottom_right": 0.9}))
    roles = {image_hash: ResolvedRole(image_hash=image_hash, role=role,
                                      provenance=Provenance.SUPPLIED,
                                      confidence=1.0)}
    assembled = assemble(to_image_evidence([outputs]), roles)
    assert assembled.anomalies, "fixture produced no anomalies to check"

    orphans = []
    for anomaly in assembled.anomalies:
        key = f"{anomaly.get('category')}:{anomaly.get('defect_type')}"
        refs = (assembled.evidence_refs.get(f"{key}:{anomaly.get('region')}")
                or assembled.evidence_refs.get(key) or [])
        if not [r for r in refs if r.image_hash == anomaly.get("image_hash")]:
            orphans.append(anomaly)

    assert not orphans, (
        f"{len(orphans)} anomalies carry no evidence from their own "
        f"photograph, so `evaluate` deletes them without a trace: "
        f"{[(a.get('category'), a.get('defect_type')) for a in orphans]}")


def test_a_spanning_finding_cannot_bridge_two_faces_into_one_defect():
    """`fuse` compares each finding against `group[0]` only, so a finding
    whose evidence spans BOTH photographs can act as a bridge: it shares a
    hash with the front finding and a hash with the back one, and all three
    land in a single group.

    That inverts the rule the face dimension exists for — the same corner on
    two faces is two defects — and it is order-dependent, so it appears only
    when the spanning finding happens to sort first. `raw` is heuristic
    findings followed by vision findings, so a vision-only defect class puts
    a spanning finding at the head of its group.
    """
    spanning = Finding(
        defect_type="rounding", category="corners",
        state=FindingState.OBSERVED, producer=FindingProducer.VISION,
        confidence=0.9, psa10_relevant=True, severity=Severity.MODERATE,
        location=BOX,
        evidence=[EvidenceRef(artifact_id=f"a-{h}", image_hash=h,
                              origin=EvidenceOrigin.ORIGINAL,
                              view="corner_top_left", region=BOX)
                  for h in ("front-hash", "back-hash")])
    roles = {"front-hash": ImageRole.FRONT, "back-hash": ImageRole.BACK}

    fused = fuse([spanning, _finding("front-hash"), _finding("back-hash")],
                 roles)
    assert len(fused) >= 2, (
        "a finding spanning both photographs merged the front's damage and "
        "the back's into one defect")
