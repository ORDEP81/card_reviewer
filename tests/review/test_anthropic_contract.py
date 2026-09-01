"""Offline contract tests for the Anthropic provider.

No test here makes a network call. The request is constructed and inspected;
responses come from saved fixtures.
"""

import json
from pathlib import Path

import pytest

from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.vision.anthropic import AnthropicVisionProvider, build_request
from card_reviewer.review.vision.prompt import PROMPT_VERSION, build_prompt
from card_reviewer.review.vision.provider import ProviderContractError

FIXTURES = Path(__file__).parent / "fixtures" / "vision"


def _fixture(name, artifact_id="a1"):
    return json.loads(FIXTURES.joinpath(name).read_text()
                      .replace("__ARTIFACT__", artifact_id))


@pytest.fixture
def rig(tmp_path):
    """A provider whose store actually holds the artifact the manifest cites."""
    store = ArtifactStore(tmp_path)
    image_hash = store.put_image(render_png(CardSpec()))
    artifact_id = store.put_derived(image_hash, "surface", "original.png",
                                    render_png(CardSpec()))
    provider = AnthropicVisionProvider(model="m", store=store, api_key="unused")
    manifest = {
        "artifacts": [{"artifact_id": artifact_id, "view": "surface_original",
                       "origin": "normalized", "enhancement": None,
                       "region": None}],
        "rubric_rules": [], "measurements": {},
    }
    return provider, manifest, artifact_id


# --- the prompt ------------------------------------------------------------

def test_the_prompt_is_adversarial_but_demands_conservative_evidence():
    """Both halves matter, and each is asserted separately.

    Checking only that the four state words appear would pass a brief that
    told the model to report whatever it sees — the conservative framing is
    what keeps an adversarial search from manufacturing defects.
    """
    text = " ".join(build_prompt({}).lower().split())
    assert "every visible reason" in text
    assert "conclude conservatively" in text
    assert "wrongly confirmed defect is the expensive error" in text
    for state in ("observed", "suspected", "not_observed", "not_assessable"):
        assert state in text


def test_the_prompt_tells_the_provider_enhancement_alone_cannot_confirm():
    """I3 has to reach the model, or it will confirm defects from CLAHE."""
    # Collapse whitespace: the brief wraps, so a phrase can straddle a line
    # break and a naive substring check would miss it.
    text = " ".join(build_prompt({}).lower().split())
    assert "visible_in_original" in text
    assert "cannot on its own establish a confirmed defect" in text


def test_the_prompt_never_mentions_price_or_market_value():
    import re

    text = build_prompt({}).lower()
    for word in ("price", "prices", "profit", "worth", "resale", "roi",
                 "market", "purchase"):
        assert not re.search(rf"\b{word}\b", text), f"prompt mentions {word!r}"


def test_the_prompt_does_not_ask_the_provider_to_restate_centering():
    assert "do not re-measure" in build_prompt(
        {"measurements": {"horizontal": 54.0}}).lower()


def test_the_prompt_renders_every_canonical_payload_section():
    text = build_prompt({
        "artifacts": [{"artifact_id": "a1"}], "rubric_rules": [{"id": "R"}],
        "measurements": {"horizontal": 54.0}, "detectability": {"x": "low"},
        "detectability_reasons": {"x": "GLARE"},
        "image_limitations": ["front is glared"], "conflicts": [{"f": 1}],
        "anomaly_candidates": [{"defect_type": "scratches"}]}).lower()
    for section in ("detectability", "limitation", "conflict", "anomaly"):
        assert section in text


def test_the_prompt_declares_its_version():
    assert PROMPT_VERSION


# --- the request carries images -------------------------------------------

def test_the_request_contains_real_image_blocks(rig):
    _, manifest, _ = rig
    provider, _, _ = rig
    blocks = build_request(manifest, provider._store)
    images = [b for b in blocks if b["type"] == "image"]
    assert images, "the request carries no image content at all"
    assert images[0]["source"]["type"] == "base64"
    assert len(images[0]["source"]["data"]) > 100


def test_each_image_block_is_labelled_with_its_artifact_id(rig):
    """Without a deterministic id-to-image mapping the provider cannot cite
    evidence we can resolve back.

    The label must be the block IMMEDIATELY BEFORE its image. Searching all
    text blocks is not enough: the brief already renders the manifest, so
    the id appears there whether or not the image is labelled at all.
    """
    provider, manifest, artifact_id = rig
    blocks = build_request(manifest, provider._store)
    image_positions = [i for i, b in enumerate(blocks) if b["type"] == "image"]
    assert image_positions
    for position in image_positions:
        label = blocks[position - 1]
        assert label["type"] == "text"
        assert artifact_id in label["text"]


def test_image_block_order_is_deterministic(rig):
    provider, manifest, _ = rig
    assert build_request(manifest, provider._store) == build_request(
        manifest, provider._store)


def test_building_the_request_never_opens_a_socket(rig, monkeypatch):
    import socket

    provider, manifest, _ = rig

    def boom(*args, **kwargs):
        raise AssertionError("build_request must not touch the network")

    monkeypatch.setattr(socket, "socket", boom)
    assert build_request(manifest, provider._store)


def test_an_unsupported_artifact_format_is_rejected(tmp_path):
    store = ArtifactStore(tmp_path)
    h = store.put_image(b"pixels")
    bad = store.put_derived(h, "surface", "notes.txt", b"plain text")
    with pytest.raises(ProviderContractError, match="supported image"):
        build_request({"artifacts": [{"artifact_id": bad, "view": "v",
                                      "origin": "normalized"}]}, store)


# --- responses -------------------------------------------------------------

def test_a_well_formed_saved_response_parses(monkeypatch, rig):
    provider, manifest, artifact_id = rig
    monkeypatch.setattr(provider, "_call",
                        lambda blocks: _fixture("valid.json", artifact_id))
    assert provider.assess(manifest).gem_view.value == (
        "possible_psa10_disqualifier")


@pytest.mark.parametrize("name", [
    "missing_gem_view.json", "unknown_artifact.json",
    "missing_assessability.json", "malformed_state.json",
])
def test_malformed_saved_responses_raise_a_contract_error(monkeypatch, name, rig):
    provider, manifest, artifact_id = rig
    monkeypatch.setattr(provider, "_call",
                        lambda blocks: _fixture(name, artifact_id))
    with pytest.raises(ProviderContractError):
        provider.assess(manifest)


def test_non_json_content_is_a_contract_error(monkeypatch, rig):
    provider, manifest, _ = rig

    def not_json(blocks):
        raise ProviderContractError("provider returned non-JSON content")

    monkeypatch.setattr(provider, "_call", not_json)
    with pytest.raises(ProviderContractError):
        provider.assess(manifest)


# --- signature and safety --------------------------------------------------

def test_the_signature_carries_the_four_declared_inputs(rig):
    provider, _, _ = rig
    sig = provider.signature()
    assert set(sig) == {"provider", "model", "prompt_version", "inference_params"}
    assert sig["provider"] == "anthropic"


def test_changing_the_model_changes_the_signature(tmp_path):
    store = ArtifactStore(tmp_path)
    a = AnthropicVisionProvider(model="m1", store=store, api_key="x").signature()
    b = AnthropicVisionProvider(model="m2", store=store, api_key="x").signature()
    assert a != b


def test_a_provider_built_without_a_store_is_rejected_at_construction():
    """Failing here beats an AttributeError on the first real call."""
    with pytest.raises(ValueError, match="ArtifactStore"):
        AnthropicVisionProvider(model="m", api_key="unused")


def test_the_sdk_is_not_imported_at_module_load():
    import sys

    assert "anthropic" not in sys.modules


def test_no_test_in_this_suite_constructs_a_real_client():
    """CI must never make a real API call.

    The pattern is assembled at runtime so this test cannot match its own
    source — the first version searched for a literal it contained, and
    failed against itself.
    """
    import subprocess

    pattern = "anthropic." + "Anthropic("
    out = subprocess.run(["grep", "-rn", "--include=*.py", pattern, "tests/"],
                         capture_output=True, text=True)
    assert out.stdout == "", f"a test constructs a real client:\n{out.stdout}"
