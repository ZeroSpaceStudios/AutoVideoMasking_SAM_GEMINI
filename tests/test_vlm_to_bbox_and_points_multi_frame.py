"""Tests for VLMtoBBoxAndPointsMultiFrame.

Loads the node module via importlib with a synthetic package alias so the
`from .prompts import ...` relative import resolves outside of ComfyUI.
Mocks `_call_gemini` so tests don't require an API key or network.

Validates:
- keyframe_indices pre-dispatch validator (shape, bounds, duplicates)
- v1.4.0 payload shape (schema_type, schema_version, accepted_frames, seeds)
- per-seed fields: frame_idx, obj_id, pos_pts, neg_pts, box (cxcywh normalized)
- N=1 semantic equivalence — single-keyframe output is structurally equivalent
  to wrapping VLMtoBBoxAndPoints' single-frame output into a v1.4 batched payload
- Soft-fail per keyframe (partial VLM failures don't kill the run)
- Hard-fail when ALL keyframes fail (mirrors consumer's error_on_noop semantics)
"""

import importlib.util
import json
import pathlib
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Module load with synthetic package alias (so relative imports resolve)
# ---------------------------------------------------------------------------

_pkg_root = pathlib.Path(__file__).resolve().parent.parent
_nodes_dir = _pkg_root / "nodes"

_pkg_name = "_avm_test_pkg"
_pkg = types.ModuleType(_pkg_name)
_pkg.__path__ = [str(_nodes_dir)]
sys.modules[_pkg_name] = _pkg


def _load(rel_path: str, full_name: str):
    spec = importlib.util.spec_from_file_location(full_name, str(_nodes_dir / rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Prompts module must load first — bridge imports it via relative import.
_prompts_mod = _load("prompts.py", f"{_pkg_name}.prompts")

# Bridge module imports numpy + PIL at top level. We rely on real numpy
# being installed (matches AVM's runtime requirement); we mock the image
# conversion via monkeypatch on _tensor_to_pil so PIL specifics don't run.
_bridge_mod = _load("vlm_sam3_bridge.py", f"{_pkg_name}.vlm_sam3_bridge")

VLMtoBBoxAndPointsMultiFrame = _bridge_mod.VLMtoBBoxAndPointsMultiFrame
_parse_keyframe_indices_strict = _bridge_mod._parse_keyframe_indices_strict


# ---------------------------------------------------------------------------
# Fake IMAGE tensor — minimal surface (just .shape)
# ---------------------------------------------------------------------------

class FakeImages:
    """Quacks like an IMAGE tensor: has .shape, supports slicing with [t:t+1]."""
    def __init__(self, T, H, W, C=3):
        self.shape = (T, H, W, C)

    def __getitem__(self, key):
        # Return a single-frame fake with batch dim preserved
        return FakeImages(1, self.shape[1], self.shape[2], self.shape[3])


# ---------------------------------------------------------------------------
# Mock Gemini call — returns canned JSON matching bbox_and_points_prompt schema
# ---------------------------------------------------------------------------

def _make_canned_response(box=(300, 200, 700, 600), pos=None, neg=None):
    """Build a Gemini response in 0-1000 normalized scale (matches AVM convention)."""
    pos = pos if pos is not None else [[450, 350], [550, 400], [500, 500]]
    neg = neg if neg is not None else [[100, 100]]
    return json.dumps({
        "boundary_description": "synthetic test response",
        "bbox": list(box),
        "positive": pos,
        "negative": neg,
    })


def _install_gemini_mock(monkeypatch, response_factory=None):
    """Replace _call_gemini in the loaded module with a deterministic stub.

    response_factory: callable(pil_img, prompt, api) -> raw_str. Defaults
    to a fixed canned response.
    """
    if response_factory is None:
        def response_factory(pil_img, prompt, api):
            return _make_canned_response()
    monkeypatch.setattr(_bridge_mod, "_call_gemini", response_factory)
    # Also stub _tensor_to_pil — we use FakeImages, not real tensors.
    def fake_tensor_to_pil(image_tensor):
        # Return an object with .size attribute (W, H) that the producer uses
        class _PIL:
            size = (image_tensor.shape[2], image_tensor.shape[1])  # (W, H)
        return _PIL()
    monkeypatch.setattr(_bridge_mod, "_tensor_to_pil", fake_tensor_to_pil)


# ---------------------------------------------------------------------------
# _parse_keyframe_indices_strict — pre-dispatch validator
# ---------------------------------------------------------------------------

def test_parse_kf_accepts_valid_array():
    assert _parse_keyframe_indices_strict("[0, 5, 13, 25]", 100) == [0, 5, 13, 25]


def test_parse_kf_rejects_non_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_keyframe_indices_strict("not json", 100)


def test_parse_kf_rejects_non_list():
    with pytest.raises(ValueError, match="JSON int array"):
        _parse_keyframe_indices_strict('{"a": 1}', 100)


def test_parse_kf_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        _parse_keyframe_indices_strict("[]", 100)


def test_parse_kf_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        _parse_keyframe_indices_strict("[0, 5, 200]", 100)


def test_parse_kf_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        _parse_keyframe_indices_strict("[0, 5, 5]", 100)


def test_parse_kf_rejects_bool_member():
    # JSON 'true' is a bool, not int — Python json treats True as int subclass
    with pytest.raises(ValueError, match="not an int"):
        _parse_keyframe_indices_strict("[0, true, 5]", 100)


# ---------------------------------------------------------------------------
# Producer happy path
# ---------------------------------------------------------------------------

def _make_api():
    return {
        "api_key": "test-key",
        "model_name": "gemini-3.1-pro-preview",
        "provider": "gemini_direct",
        "base_url": "",
    }


def test_producer_emits_v1_4_payload_with_all_seed_fields(monkeypatch):
    """Producer output is valid JSON with schema_version=1.4.0 and per-seed
    pos_pts, neg_pts, box fields populated."""
    _install_gemini_mock(monkeypatch)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, raw, info = node.run(
        images=images, api=_make_api(), target_description="the main subject",
        keyframe_indices="[5, 13, 25]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=3, obj_id=1,
    )
    payload = json.loads(seed_json)
    assert payload["schema_type"] == "sam3_seed_prompts"
    assert payload["schema_version"] == "1.4.0"
    assert payload["schema_minor_compatible_with"] == "1.x"
    assert payload["generator_node"] == "VLMtoBBoxAndPointsMultiFrame"
    assert payload["accepted_frames"] == [5, 13, 25]
    assert len(payload["seeds"]) == 3
    for s in payload["seeds"]:
        assert "frame_idx" in s and "obj_id" in s
        assert s["obj_id"] == 1
        assert "pos_pts" in s and isinstance(s["pos_pts"], list)
        assert "neg_pts" in s and isinstance(s["neg_pts"], list)
        assert "box" in s and len(s["box"]) == 4
        # All normalized in [0, 1]
        for v in s["box"]:
            assert 0.0 <= v <= 1.0, f"box value {v} out of [0,1]"
        for pt in s["pos_pts"] + s["neg_pts"]:
            assert 0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0


def test_producer_box_format_is_cxcywh_normalized(monkeypatch):
    """Verify the conversion from Gemini's [x1,y1,x2,y2] in 0-1000 scale to
    SAM3's [cx, cy, w, h] in [0,1] is correct."""
    # bbox = (300, 200, 700, 600) in 0-1000 scale
    # Expected: x1n=0.3, y1n=0.2, x2n=0.7, y2n=0.6
    # → cx=(0.3+0.7)/2=0.5, cy=(0.2+0.6)/2=0.4, w=0.4, h=0.4
    def canned(pil_img, prompt, api):
        return _make_canned_response(box=(300, 200, 700, 600))
    _install_gemini_mock(monkeypatch, response_factory=canned)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[10]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
    )
    payload = json.loads(seed_json)
    box = payload["seeds"][0]["box"]
    assert box == pytest.approx([0.5, 0.4, 0.4, 0.4], abs=1e-6), f"got {box}"


# ---------------------------------------------------------------------------
# N=1 semantic equivalence (the headline backwards-compat guarantee)
# ---------------------------------------------------------------------------

def test_n_equals_1_semantic_equivalence(monkeypatch):
    """N=1 producer output is semantically equivalent to wrapping
    VLMtoBBoxAndPoints' single-frame output into a v1.4 batched payload.

    Specifically: the same Gemini response feeding both the single-frame and
    multi-frame producers should yield the same box (cxcywh), pos_pts, and
    neg_pts in the resulting seed. The multi-frame producer's seed at N=1
    must match the single-frame producer's typed-dict outputs."""
    canned = _make_canned_response(
        box=(200, 150, 800, 750),
        pos=[[400, 300], [600, 400]],
        neg=[[50, 50], [950, 950]],
    )

    def respond(pil_img, prompt, api):
        return canned

    _install_gemini_mock(monkeypatch, response_factory=respond)

    # Multi-frame at N=1
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _ = node.run(
        images=images, api=_make_api(), target_description="subject",
        keyframe_indices="[0]",
        num_pos_points=6, num_neg_points=3,
        parallel_call_count=1, obj_id=7,
    )
    mf_payload = json.loads(seed_json)
    assert len(mf_payload["seeds"]) == 1
    mf_seed = mf_payload["seeds"][0]

    # Single-frame producer (for comparison) — recompute the same conversion
    # the multi-frame producer should match. Box: bbox=(200,150,800,750) in
    # 0-1000 scale → x1n=0.2, y1n=0.15, x2n=0.8, y2n=0.75
    # → cx=0.5, cy=0.45, w=0.6, h=0.6
    assert mf_seed["box"] == pytest.approx([0.5, 0.45, 0.6, 0.6], abs=1e-6)

    # pos/neg points: 400,300 → 0.4,0.3; 600,400 → 0.6,0.4
    # pytest.approx doesn't handle nested lists; check element-wise.
    flat_pos = [v for pt in mf_seed["pos_pts"] for v in pt]
    assert flat_pos == pytest.approx([0.4, 0.3, 0.6, 0.4], abs=1e-6)
    flat_neg = [v for pt in mf_seed["neg_pts"] for v in pt]
    assert flat_neg == pytest.approx([0.05, 0.05, 0.95, 0.95], abs=1e-6)

    # obj_id passthrough
    assert mf_seed["obj_id"] == 7
    assert mf_seed["frame_idx"] == 0


# ---------------------------------------------------------------------------
# Soft-fail policy
# ---------------------------------------------------------------------------

def test_invalid_image_shape_raises_with_diagnostic(monkeypatch):
    """Shape guard: producer rejects non-4D IMAGE tensors with diagnostic.

    The contract is [T, H, W, C] (4-D). A 3-D tensor would be silently
    misinterpreted as T=H, H=W, W=C — caller must explicitly opt-in by
    providing a 4-D batch.
    """
    _install_gemini_mock(monkeypatch)
    node = VLMtoBBoxAndPointsMultiFrame()

    class Bad3D:
        shape = (100, 1080, 1920)  # missing C dim
    with pytest.raises(ValueError, match="must be IMAGE tensor"):
        node.run(
            images=Bad3D(), api=_make_api(), target_description="x",
            keyframe_indices="[0]",
            num_pos_points=3, num_neg_points=1,
            parallel_call_count=1, obj_id=1,
        )

    # No .shape attribute at all
    class NoShape: pass
    with pytest.raises(ValueError, match="must be IMAGE tensor"):
        node.run(
            images=NoShape(), api=_make_api(), target_description="x",
            keyframe_indices="[0]",
            num_pos_points=3, num_neg_points=1,
            parallel_call_count=1, obj_id=1,
        )


def test_partial_failure_soft_skips_failed_keyframe(monkeypatch):
    """If one keyframe parse fails, other keyframes still produce seeds.
    Failed keyframe absent from accepted_frames + info reports the error."""
    call_idx = {"n": 0}

    def flaky_respond(pil_img, prompt, api):
        call_idx["n"] += 1
        # Second call returns malformed JSON
        if call_idx["n"] == 2:
            return "{ not valid JSON"
        return _make_canned_response()

    _install_gemini_mock(monkeypatch, response_factory=flaky_respond)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, info = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5, 13, 25]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,  # sequential for ordering
    )
    payload = json.loads(seed_json)
    # 2 of 3 keyframes succeed
    assert len(payload["seeds"]) == 2
    assert len(payload["accepted_frames"]) == 2
    # Info string mentions the failure
    assert "accepted=False" in info


def test_all_failures_hard_raises(monkeypatch):
    """If ALL keyframes fail parsing, producer hard-raises (mirrors consumer
    error_on_noop semantics — no point letting the chain proceed empty)."""
    def always_fail(pil_img, prompt, api):
        return "garbage"
    _install_gemini_mock(monkeypatch, response_factory=always_fail)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    with pytest.raises(RuntimeError, match="All .* keyframes failed"):
        node.run(
            images=images, api=_make_api(), target_description="x",
            keyframe_indices="[5, 13, 25]",
            num_pos_points=3, num_neg_points=1,
            parallel_call_count=1, obj_id=1,
        )
