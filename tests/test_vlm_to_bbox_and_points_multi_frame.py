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
_resolve_target_subject = _bridge_mod._resolve_target_subject
_AVM_MF_TARGET_PRESETS = _bridge_mod._AVM_MF_TARGET_PRESETS
_AVM_MF_DEFAULT_TARGET_DESCRIPTION = _bridge_mod._AVM_MF_DEFAULT_TARGET_DESCRIPTION


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


class _FakePIL:
    """Quacks like a PIL.Image: has .size (W, H) and supports .crop((l,t,r,b)).

    The crop-in path calls `pil_img.crop((cx1, cy1, cx2, cy2))` and then reads
    `.size` on the cropped result. Stubbing both keeps the test from needing
    real PIL/numpy in the bridge's PIL path.
    """
    def __init__(self, W, H):
        self.size = (int(W), int(H))

    def crop(self, box):
        left, top, right, bottom = box
        return _FakePIL(right - left, bottom - top)


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
        # Return an object with .size attribute (W, H) and .crop() so the
        # crop-in path can call pil_img.crop((l,t,r,b)) and read .size on
        # the result.
        return _FakePIL(image_tensor.shape[2], image_tensor.shape[1])
    monkeypatch.setattr(_bridge_mod, "_tensor_to_pil", fake_tensor_to_pil)


def _make_stage1_response(bbox=(300, 200, 700, 600)):
    """Build a face_region_stage1_prompt response (bbox only)."""
    return json.dumps({
        "boundary_description": "synthetic stage1 test response",
        "bbox": list(bbox),
    })


def _make_stage2_response(fg=None, bg=None, confidence=None):
    """Build a face_region_stage2_prompt response (foreground/background points in crop space)."""
    fg = fg if fg is not None else [[450, 350], [550, 400], [500, 500]]
    bg = bg if bg is not None else [[100, 100]]
    out = {
        "anchor_plan": "synthetic stage2 test response",
        "foreground": fg,
        "background": bg,
    }
    if confidence is not None:
        out["confidence"] = confidence
    return json.dumps(out)


def _make_canned_response_with_confidence(box=(300, 200, 700, 600), pos=None, neg=None, confidence=0.95):
    """Single-stage canned response with the new v1.5 confidence field."""
    pos = pos if pos is not None else [[450, 350], [550, 400], [500, 500]]
    neg = neg if neg is not None else [[100, 100]]
    return json.dumps({
        "boundary_description": "synthetic test response",
        "bbox": list(box),
        "positive": pos,
        "negative": neg,
        "confidence": confidence,
    })


def _make_two_stage_responder(stage1_bbox=(300, 200, 700, 600),
                              stage2_fg=None, stage2_bg=None,
                              stage2_confidence=None):
    """Build a Gemini-stub that returns stage1 then stage2 responses based on
    which prompt was sent. Detects stage by the presence of unique prompt text.
    """
    def respond(pil_img, prompt, api):
        # face_region_stage1_prompt: contains "Localize one target region"
        # face_region_stage2_prompt: contains "Place segmentation prompt points"
        if "Place segmentation prompt points" in prompt:
            return _make_stage2_response(
                fg=stage2_fg, bg=stage2_bg, confidence=stage2_confidence,
            )
        if "Localize one target region" in prompt:
            return _make_stage1_response(bbox=stage1_bbox)
        # Single-stage bbox_and_points_prompt
        return _make_canned_response()
    return respond


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


def test_producer_emits_v1_5_payload_with_all_seed_fields(monkeypatch):
    """Producer output is valid JSON with schema_version=1.5.0, per-seed
    pos_pts/neg_pts/box populated, plus v1.5 additive fields (crop_in_applied,
    crop_meta, confidence). Top-level v1.5 additions also present."""
    _install_gemini_mock(monkeypatch)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, raw, info, _ibp, _ibsp, _ipp, _inp, _ifi = node.run(
        images=images, api=_make_api(), target_description="the main subject",
        keyframe_indices="[5, 13, 25]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=3, obj_id=1,
    )
    payload = json.loads(seed_json)
    assert payload["schema_type"] == "sam3_seed_prompts"
    assert payload["schema_version"] == "1.5.0"
    assert payload["schema_minor_compatible_with"] == "1.x"
    assert payload["generator_node"] == "VLMtoBBoxAndPointsMultiFrame"
    assert payload["accepted_frames"] == [5, 13, 25]
    assert len(payload["seeds"]) == 3
    # v1.5 top-level additions
    assert payload["crop_in_enabled"] is False
    assert payload["crop_padding"] == 24  # widget default
    assert payload["confidence_threshold"] == 0.0  # gating disabled by default
    assert payload["low_confidence_skipped"] == []
    for s in payload["seeds"]:
        assert "frame_idx" in s and "obj_id" in s
        assert s["obj_id"] == 1
        assert "pos_pts" in s and isinstance(s["pos_pts"], list)
        assert "neg_pts" in s and isinstance(s["neg_pts"], list)
        assert "box" in s and len(s["box"]) == 4
        # v1.5 per-seed additions
        assert s["crop_in_applied"] is False
        assert s["crop_meta"] is None
        # Confidence defaults to 1.0 when the model omits the field
        assert "confidence" in s and 0.0 <= s["confidence"] <= 1.0
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
    seed_json, _, _, _ibp, _ibsp, _ipp, _inp, _ifi = node.run(
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
    seed_json, _, _, _ibp, _ibsp, _ipp, _inp, _ifi = node.run(
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


# ---------------------------------------------------------------------------
# Init outputs (Strategy C — single-node init + batch)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# target_preset resolution (quick-pick subject presets)
# ---------------------------------------------------------------------------

def test_resolve_custom_returns_description_verbatim():
    """custom preset = back-compat — target_description used verbatim."""
    assert _resolve_target_subject("custom", "the boxer in red jacket") == "the boxer in red jacket"


def test_resolve_custom_with_empty_description_falls_back_to_default():
    """custom + empty description → the default 'the main subject' baseline."""
    assert _resolve_target_subject("custom", "") == _AVM_MF_DEFAULT_TARGET_DESCRIPTION
    assert _resolve_target_subject("custom", "   ") == _AVM_MF_DEFAULT_TARGET_DESCRIPTION


def test_resolve_preset_with_default_description_uses_preset_only():
    """preset + default description (user didn't override) → preset text alone."""
    # default "the main subject" treated as 'no user detail'
    assert _resolve_target_subject("face", _AVM_MF_DEFAULT_TARGET_DESCRIPTION) == "the subject's face"
    assert _resolve_target_subject("hair", "") == "the subject's hair"


def test_resolve_preset_with_user_detail_composes():
    """preset + meaningful description → composed prompt for refinement."""
    out = _resolve_target_subject("face", "wearing red sunglasses")
    assert out == "the subject's face, wearing red sunglasses"
    out2 = _resolve_target_subject("hair", "dyed pink, shoulder-length")
    assert out2 == "the subject's hair, dyed pink, shoulder-length"


def test_resolve_unknown_preset_raises():
    """Unknown preset name raises with a diagnostic list of valid choices."""
    with pytest.raises(ValueError, match="unknown target_preset"):
        _resolve_target_subject("not_a_preset", "ignored")


def test_target_preset_table_starts_with_custom_sentinel():
    """D-349 sentinel-FIRST: 'custom' must be at index 0 of the choices list
    so legacy saved workflows without this widget default to the back-compat
    free-form behavior."""
    keys = list(_AVM_MF_TARGET_PRESETS.keys())
    assert keys[0] == "custom", f"expected 'custom' at index 0, got {keys[:3]}"


def test_producer_run_with_preset_overrides_description(monkeypatch):
    """End-to-end: producer with target_preset='face' + default description
    sends 'the subject's face' to Gemini (verified via the canned prompt that
    gets passed through). The payload's effective target_description should
    reflect the preset-resolved text, not the raw widget value."""
    captured_prompts = []

    def capture_prompt(pil_img, prompt, api):
        captured_prompts.append(prompt)
        return _make_canned_response()

    _install_gemini_mock(monkeypatch, response_factory=capture_prompt)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    out = node.run(
        images=images, api=_make_api(),
        target_description=_AVM_MF_DEFAULT_TARGET_DESCRIPTION,  # the default
        target_preset="face",
        keyframe_indices="[5]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
    )
    # The prompt sent to Gemini contains the preset-resolved subject text
    assert len(captured_prompts) == 1
    assert "the subject's face" in captured_prompts[0]
    # Payload records BOTH the raw widget value AND the effective resolved text
    # (avoids semantic drift for any downstream consumer of target_description).
    payload = json.loads(out[0])
    assert payload["target_description"] == _AVM_MF_DEFAULT_TARGET_DESCRIPTION, (
        "raw target_description should record the widget value, not the resolved text"
    )
    assert payload["effective_target_description"] == "the subject's face"
    assert payload["target_preset"] == "face"


def test_producer_run_with_preset_and_refinement_composes(monkeypatch):
    """End-to-end: target_preset='hair' + target_description='dyed pink' →
    composed prompt 'the subject's hair, dyed pink'."""
    captured_prompts = []
    def capture(pil_img, prompt, api):
        captured_prompts.append(prompt)
        return _make_canned_response()
    _install_gemini_mock(monkeypatch, response_factory=capture)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    out = node.run(
        images=images, api=_make_api(),
        target_description="dyed pink",
        target_preset="hair",
        keyframe_indices="[10]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
    )
    assert "the subject's hair, dyed pink" in captured_prompts[0]
    payload = json.loads(out[0])
    # Raw widget value preserved; effective records the composed result
    assert payload["target_description"] == "dyed pink"
    assert payload["effective_target_description"] == "the subject's hair, dyed pink"


def test_resolve_duplicate_preset_text_in_description_pins_intentional_no_dedupe():
    """Edge case pin: when description text matches/contains the preset text,
    the resolver does NOT silently deduplicate. Users get exactly what they
    typed plus the preset baseline. This is deliberate — silent dedup would
    surprise users who genuinely want emphasis (e.g., 'face' + 'face only, not
    hair' is a valid refinement). Tooltip on the widget documents this."""
    # preset=face, desc=preset-equivalent text → composes verbatim (caller's
    # choice, no dedup). The Gemini prompt will read "the subject's face,
    # the subject's face" — accepted as user intent.
    out = _resolve_target_subject("face", "the subject's face")
    assert out == "the subject's face, the subject's face"


def test_hands_preset_resolves_correctly():
    """Hands preset (folded in per R3): one of the most-requested capabilities
    in VFX object-interaction workflows."""
    assert _resolve_target_subject("hands", _AVM_MF_DEFAULT_TARGET_DESCRIPTION) == "the subject's hands, wrists, and fingers"
    assert _resolve_target_subject("hands", "holding a knife") == "the subject's hands, wrists, and fingers, holding a knife"


def test_init_outputs_derived_from_first_accepted_seed(monkeypatch):
    """Init outputs are reformatted from the first accepted seed (lowest
    frame_idx). They feed SAM3VideoSegmentation directly so a separate
    single-frame VLMtoBBoxAndPoints call is not required."""
    canned = _make_canned_response(
        box=(200, 150, 800, 750),
        pos=[[400, 300], [600, 400]],
        neg=[[50, 50], [950, 950]],
    )
    def respond(pil_img, prompt, api):
        return canned
    _install_gemini_mock(monkeypatch, response_factory=respond)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    out = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5, 13, 25]",
        num_pos_points=6, num_neg_points=3,
        parallel_call_count=1, obj_id=7,
    )
    assert len(out) == 8, f"expected 8 outputs, got {len(out)}"
    (seed_json, _raw, _info,
     init_box_prompt, init_boxes_prompt,
     init_positive_points, init_negative_points,
     init_frame_idx) = out

    # init_frame_idx = first accepted keyframe (lowest frame_idx)
    assert init_frame_idx == 5

    # init_box_prompt (singular): matches the first seed's box
    payload = json.loads(seed_json)
    first_seed = payload["seeds"][0]
    assert init_box_prompt == {"box": first_seed["box"], "label": True}

    # init_boxes_prompt (plural): same box, wrapped in batch shape
    assert init_boxes_prompt == {
        "boxes": [first_seed["box"]],
        "labels": [True],
    }

    # init_positive_points: same pts as first seed, labels = [1, 1, ...]
    assert init_positive_points["points"] == first_seed["pos_pts"]
    assert init_positive_points["labels"] == [1] * len(first_seed["pos_pts"])

    # init_negative_points: same pts as first seed, labels = [0, 0, ...]
    assert init_negative_points["points"] == first_seed["neg_pts"]
    assert init_negative_points["labels"] == [0] * len(first_seed["neg_pts"])


def test_init_frame_idx_when_zero_in_keyframes(monkeypatch):
    """When keyframe_indices includes 0, init_frame_idx is 0 (works with
    SAM3VideoSegmentation's default frame_idx=0; no wiring change needed)."""
    _install_gemini_mock(monkeypatch)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    out = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[0, 5, 13]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
    )
    init_frame_idx = out[7]
    assert init_frame_idx == 0


def test_init_outputs_skip_failed_first_keyframe(monkeypatch):
    """If the first requested keyframe fails Gemini parsing, init_frame_idx
    advances to the first SUCCESSFUL keyframe — matches the same fallback
    semantics SAM3VideoSegmentation needs to see."""
    call_idx = {"n": 0}
    def flaky(pil_img, prompt, api):
        call_idx["n"] += 1
        # First keyframe (t=5) fails; subsequent succeed
        if call_idx["n"] == 1:
            return "{ malformed json"
        return _make_canned_response()
    _install_gemini_mock(monkeypatch, response_factory=flaky)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    out = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5, 13, 25]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,  # sequential so first call IS t=5
    )
    # accepted_frames was [13, 25], first accepted = 13
    init_frame_idx = out[7]
    assert init_frame_idx == 13, f"expected init_frame_idx=13, got {init_frame_idx}"


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
    seed_json, _, info, _ibp, _ibsp, _ipp, _inp, _ifi = node.run(
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


# ---------------------------------------------------------------------------
# D-381: confidence-gating (v1.5)
# ---------------------------------------------------------------------------

def test_confidence_default_is_one_when_field_missing(monkeypatch):
    """Back-compat: pre-1.5 prompts / older Gemini models omit the confidence
    field. Per-seed confidence defaults to 1.0 so gating disabled is a true
    no-op vs v1.4 behavior."""
    _install_gemini_mock(monkeypatch)  # default canned has NO confidence field
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _, _ibp, _ibsp, _ipp, _inp, _ifi = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5, 13]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
    )
    payload = json.loads(seed_json)
    for s in payload["seeds"]:
        assert s["confidence"] == 1.0


def test_confidence_extracted_from_response(monkeypatch):
    """When the model emits a confidence field, it's extracted into the seed."""
    def respond(pil_img, prompt, api):
        return _make_canned_response_with_confidence(confidence=0.62)
    _install_gemini_mock(monkeypatch, response_factory=respond)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _, _ibp, _ibsp, _ipp, _inp, _ifi = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
    )
    payload = json.loads(seed_json)
    assert payload["seeds"][0]["confidence"] == pytest.approx(0.62, abs=1e-6)


def test_confidence_gate_disabled_when_threshold_zero(monkeypatch):
    """confidence_threshold=0.0 (default) = back-compat: every keyframe accepted
    regardless of confidence value."""
    def low_conf(pil_img, prompt, api):
        return _make_canned_response_with_confidence(confidence=0.1)
    _install_gemini_mock(monkeypatch, response_factory=low_conf)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, info, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5, 13, 25]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
        confidence_threshold=0.0,
    )
    payload = json.loads(seed_json)
    # All 3 keyframes accepted despite low confidence
    assert payload["accepted_frames"] == [5, 13, 25]
    assert payload["low_confidence_skipped"] == []
    assert payload["confidence_threshold"] == 0.0
    assert "confidence_gate=OFF" in info


def test_confidence_gate_drops_low_confidence_seeds(monkeypatch):
    """When threshold > 0, keyframes scoring below threshold are dropped from
    accepted_frames + seeds; their frame_idx values appear in low_confidence_skipped."""
    # Alternate confidence per frame_idx: 0.95, 0.45, 0.85
    confidences = {5: 0.95, 13: 0.45, 25: 0.85}
    call_idx = {"n": 0}
    sequence = [5, 13, 25]
    def respond(pil_img, prompt, api):
        t = sequence[call_idx["n"]]
        call_idx["n"] += 1
        return _make_canned_response_with_confidence(confidence=confidences[t])
    _install_gemini_mock(monkeypatch, response_factory=respond)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, info, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5, 13, 25]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,  # sequential so call order is deterministic
        confidence_threshold=0.6,
    )
    payload = json.loads(seed_json)
    # 0.45 < 0.6 → frame 13 dropped; 0.95 and 0.85 pass
    assert payload["accepted_frames"] == [5, 25]
    assert payload["low_confidence_skipped"] == [13]
    assert payload["confidence_threshold"] == 0.6
    # Surfaced in info string
    assert "skipped_low_confidence" in info
    assert "confidence=0.450" in info


def test_confidence_gate_all_dropped_raises_specific_error(monkeypatch):
    """When the gate drops EVERY keyframe, the error message names the gate
    (not 'all parses failed') so users know the right knob to adjust."""
    def all_low(pil_img, prompt, api):
        return _make_canned_response_with_confidence(confidence=0.1)
    _install_gemini_mock(monkeypatch, response_factory=all_low)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    with pytest.raises(RuntimeError, match="dropped by confidence gate"):
        node.run(
            images=images, api=_make_api(), target_description="x",
            keyframe_indices="[5, 13, 25]",
            num_pos_points=3, num_neg_points=1,
            parallel_call_count=1, obj_id=1,
            confidence_threshold=0.8,
        )


def test_confidence_coercion_handles_out_of_range_and_missing():
    """_coerce_confidence accepts None, NaN, negatives, >1.0, and 0-100 scale
    misuse — always returns a value in [0.0, 1.0]."""
    _coerce = _bridge_mod._coerce_confidence
    assert _coerce(None) == 1.0  # missing → confident default
    assert _coerce("not a number") == 1.0  # parse fail → confident default
    assert _coerce(-0.5) == 0.0  # negative → clamp
    assert _coerce(1.5) == pytest.approx(0.015, abs=1e-6)  # treated as 0-100 scale
    assert _coerce(85) == pytest.approx(0.85, abs=1e-6)  # 0-100 scale misuse rescaled
    assert _coerce(999) == 1.0  # absurdly high → clamp to 1.0
    assert _coerce(0.62) == pytest.approx(0.62, abs=1e-6)  # normal
    assert _coerce(float("nan")) == 1.0  # NaN → default
    assert _coerce(0.0) == 0.0  # zero is valid


# ---------------------------------------------------------------------------
# D-381: crop-in two-stage mode (v1.5)
# ---------------------------------------------------------------------------

def test_crop_in_disabled_by_default(monkeypatch):
    """Without enable_crop_in=True, behavior is identical to v1.4 single-stage:
    seeds carry crop_in_applied=False / crop_meta=None and Gemini is called
    exactly once per keyframe."""
    call_count = {"n": 0}
    def count_calls(pil_img, prompt, api):
        call_count["n"] += 1
        return _make_canned_response()
    _install_gemini_mock(monkeypatch, response_factory=count_calls)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _, *_ = node.run(
        images=images, api=_make_api(), target_description="the boxer",
        keyframe_indices="[5, 13, 25]",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
    )
    # 3 keyframes × 1 call each (single-stage) = 3 calls
    assert call_count["n"] == 3
    payload = json.loads(seed_json)
    assert payload["crop_in_enabled"] is False
    for s in payload["seeds"]:
        assert s["crop_in_applied"] is False
        assert s["crop_meta"] is None


def test_crop_in_enabled_doubles_gemini_calls(monkeypatch):
    """With crop-in enabled, each keyframe triggers Stage 1 + Stage 2 = 2 calls."""
    call_count = {"n": 0}
    responder = _make_two_stage_responder()
    def count_calls(pil_img, prompt, api):
        call_count["n"] += 1
        return responder(pil_img, prompt, api)
    _install_gemini_mock(monkeypatch, response_factory=count_calls)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5, 13, 25]", target_preset="face",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
        enable_crop_in=True, crop_padding=24,
    )
    # 3 keyframes × 2 calls (stage1 + stage2) = 6 calls
    assert call_count["n"] == 6
    payload = json.loads(seed_json)
    assert payload["crop_in_enabled"] is True
    assert payload["crop_padding"] == 24
    for s in payload["seeds"]:
        assert s["crop_in_applied"] is True
        # crop_meta populated with sane pixel-space bounds
        meta = s["crop_meta"]
        assert meta is not None
        assert meta["orig_w"] == 1920 and meta["orig_h"] == 1080
        assert 0 <= meta["x1"] < meta["x2"] <= 1920
        assert 0 <= meta["y1"] < meta["y2"] <= 1080


def test_crop_in_seed_box_is_tight_stage1_bbox_not_padded_crop(monkeypatch):
    """Crop-in's seed `box` must be Stage 1's tight bbox in full-frame coords,
    NOT the padded crop bounds. SAM3 uses box as a strong region prior; feeding
    the padded crop would bleed segmentation into the padding zone."""
    # Stage 1 bbox: [300, 200, 700, 600] in 0-1000 → [0.3, 0.2, 0.7, 0.6] full-frame
    # → cx=0.5, cy=0.4, w=0.4, h=0.4
    responder = _make_two_stage_responder(stage1_bbox=(300, 200, 700, 600))
    _install_gemini_mock(monkeypatch, response_factory=responder)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5]", target_preset="face",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
        enable_crop_in=True, crop_padding=24,
    )
    payload = json.loads(seed_json)
    box = payload["seeds"][0]["box"]
    # Tight stage1 box in full-frame normalized coords — NOT the padded crop bounds
    assert box == pytest.approx([0.5, 0.4, 0.4, 0.4], abs=1e-3)


def test_crop_in_stage2_points_projected_to_full_frame(monkeypatch):
    """Stage 2 emits points in CROP space (0-1000 relative to crop). They must
    be projected back to FULL-FRAME [0,1] before going into the seed —
    otherwise downstream SAM3 (running on the full 89-frame video) would apply
    crop-space coords as if they were full-frame coords and place every point
    in the top-left corner."""
    # Stage 1 bbox [300,200,700,600] in 0-1000 → pixel space [576,216,1344,648]
    # After +24 padding clamped: crop [552, 192, 1368, 672], crop_w=816, crop_h=480
    # Stage 2 point [500, 500] in 0-1000 = crop coord (0.5, 0.5)
    # = pixel (552 + 0.5*816, 192 + 0.5*480) = (960, 432)
    # Full-frame normalized: (960/1920, 432/1080) = (0.5, 0.4)
    responder = _make_two_stage_responder(
        stage1_bbox=(300, 200, 700, 600),
        stage2_fg=[[500, 500]],
        stage2_bg=[[0, 0]],
    )
    _install_gemini_mock(monkeypatch, response_factory=responder)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5]", target_preset="face",
        num_pos_points=1, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
        enable_crop_in=True, crop_padding=24,
    )
    payload = json.loads(seed_json)
    pos_pt = payload["seeds"][0]["pos_pts"][0]
    assert pos_pt == pytest.approx([0.5, 0.4], abs=1e-2)


def test_crop_in_stage1_inverted_corners_handled(monkeypatch):
    """Gemini occasionally swaps corners (x2 < x1 etc). Crop-in path must
    sort corners before computing crop bounds — otherwise crop bounds collapse
    to zero area and the keyframe soft-fails."""
    # Stage 1 returns inverted corners (right < left, bottom < top)
    responder = _make_two_stage_responder(stage1_bbox=(700, 600, 300, 200))
    _install_gemini_mock(monkeypatch, response_factory=responder)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    # Should not raise — corners get sorted internally and Stage 2 fires normally
    seed_json, _, _, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5]", target_preset="face",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
        enable_crop_in=True, crop_padding=24,
    )
    payload = json.loads(seed_json)
    assert len(payload["seeds"]) == 1
    # Sorted corners yield same final box as the non-inverted variant
    box = payload["seeds"][0]["box"]
    assert box == pytest.approx([0.5, 0.4, 0.4, 0.4], abs=1e-3)


def test_crop_in_stage1_off_image_bbox_soft_skips_keyframe(monkeypatch):
    """When Stage 1 returns a bbox completely off-image so that clamping
    collapses crop bounds to zero area, the crop-in path raises per-keyframe
    (caught by the fan-out's soft-fail handler) so other keyframes proceed.

    Note: a tiny zero-area-PRE-padding bbox (e.g. cx==same point) is RESCUED
    by padding into a small valid crop, and ships a degenerate but valid seed
    — that's intentional (the user can detect & skip downstream via the tiny
    crop_meta area). Only truly off-image bboxes trigger the soft-skip path.
    """
    call_idx = {"n": 0}
    def responder(pil_img, prompt, api):
        call_idx["n"] += 1
        if "Place segmentation prompt points" in prompt:
            return _make_stage2_response()
        if "Localize one target region" in prompt:
            # First keyframe (t=5) — bbox completely past right edge. After
            # _maybe_normalize_corners divides by 1000, pixel coords are
            # (2880, 0, 3840, 216) — clamped to image bounds (1920, 0, 1920, 240)
            # which has cx1==cx2 → soft-skip.
            if call_idx["n"] == 1:
                return _make_stage1_response(bbox=(1500, 0, 2000, 100))
            return _make_stage1_response()
        return _make_canned_response()
    _install_gemini_mock(monkeypatch, response_factory=responder)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, info, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5, 13, 25]", target_preset="face",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,  # sequential — keyframe 5 fails FIRST
        enable_crop_in=True, crop_padding=24,
    )
    payload = json.loads(seed_json)
    # 2 of 3 keyframes succeed
    assert len(payload["seeds"]) == 2
    assert payload["accepted_frames"] == [13, 25]
    assert "accepted=False" in info


def test_crop_in_confidence_from_stage2(monkeypatch):
    """Per-seed confidence in crop-in mode comes from Stage 2's response
    (the points placement step). Stage 1 confidence is not consumed for gating."""
    responder = _make_two_stage_responder(stage2_confidence=0.55)
    _install_gemini_mock(monkeypatch, response_factory=responder)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5]", target_preset="face",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
        enable_crop_in=True, crop_padding=24,
    )
    payload = json.loads(seed_json)
    assert payload["seeds"][0]["confidence"] == pytest.approx(0.55, abs=1e-6)


def test_crop_in_plus_confidence_gate_drops_low_confidence(monkeypatch):
    """Combined: crop-in produces high-quality stage2 points, then the
    confidence gate filters anyway. A keyframe with Stage2 confidence below
    threshold gets dropped despite crop-in being enabled."""
    # Three keyframes get stage2 confidences 0.9, 0.3, 0.7
    confidences = {5: 0.9, 13: 0.3, 25: 0.7}
    call_state = {"current_t": None, "kf_order": [5, 13, 25], "kf_idx": 0}
    def responder(pil_img, prompt, api):
        if "Localize one target region" in prompt:
            # Track which keyframe is being processed — stage1 starts each keyframe
            call_state["current_t"] = call_state["kf_order"][call_state["kf_idx"]]
            call_state["kf_idx"] += 1
            return _make_stage1_response()
        if "Place segmentation prompt points" in prompt:
            t = call_state["current_t"]
            return _make_stage2_response(confidence=confidences[t])
        return _make_canned_response()
    _install_gemini_mock(monkeypatch, response_factory=responder)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, info, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5, 13, 25]", target_preset="face",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,  # sequential so kf_order tracking is valid
        enable_crop_in=True, crop_padding=24,
        confidence_threshold=0.5,
    )
    payload = json.loads(seed_json)
    # 0.3 < 0.5 → frame 13 dropped; 0.9 and 0.7 pass
    assert payload["accepted_frames"] == [5, 25]
    assert payload["low_confidence_skipped"] == [13]
    # Crop-in was applied to the accepted keyframes
    for s in payload["seeds"]:
        assert s["crop_in_applied"] is True


def test_crop_in_non_head_preset_emits_soft_notice(monkeypatch, capsys):
    """When crop-in is enabled with a non-head target_preset, a soft NOTICE
    is printed. The FACE_RULES still apply; the call still goes through."""
    responder = _make_two_stage_responder()
    _install_gemini_mock(monkeypatch, response_factory=responder)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5]", target_preset="full_body",  # non-head preset
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
        enable_crop_in=True, crop_padding=24,
    )
    captured = capsys.readouterr()
    assert "NOTICE: crop-in enabled with target_preset='full_body'" in captured.out
    # Call still succeeds
    payload = json.loads(seed_json)
    assert len(payload["seeds"]) == 1


def test_crop_in_head_preset_no_notice(monkeypatch, capsys):
    """Head-class presets (face/head/hair/head_and_shoulders/custom) don't
    trigger the NOTICE — FACE_RULES reads naturally for those targets."""
    responder = _make_two_stage_responder()
    _install_gemini_mock(monkeypatch, response_factory=responder)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5]", target_preset="head",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
        enable_crop_in=True, crop_padding=24,
    )
    captured = capsys.readouterr()
    assert "NOTICE: crop-in enabled" not in captured.out


def test_crop_in_zero_padding_works(monkeypatch):
    """crop_padding=0 is a valid edge case — no padding, crop is exactly
    Stage 1's bbox. Should not crash."""
    responder = _make_two_stage_responder(stage1_bbox=(300, 200, 700, 600))
    _install_gemini_mock(monkeypatch, response_factory=responder)
    node = VLMtoBBoxAndPointsMultiFrame()
    images = FakeImages(T=100, H=1080, W=1920)
    seed_json, _, _, *_ = node.run(
        images=images, api=_make_api(), target_description="x",
        keyframe_indices="[5]", target_preset="face",
        num_pos_points=3, num_neg_points=1,
        parallel_call_count=1, obj_id=1,
        enable_crop_in=True, crop_padding=0,
    )
    payload = json.loads(seed_json)
    assert payload["crop_padding"] == 0
    meta = payload["seeds"][0]["crop_meta"]
    # With padding=0, crop bounds equal the tight stage1 bbox in pixel space
    # Stage1 bbox [300,200,700,600] in 0-1000 → pixel [576,216,1344,648]
    assert meta["x1"] == 576 and meta["y1"] == 216
    assert meta["x2"] == 1344 and meta["y2"] == 648


def test_schema_version_bumped_to_1_5_0():
    """The constant moved from 1.4.0 to 1.5.0 — verify here so a drift between
    payload and the source-of-truth constant is caught."""
    assert _bridge_mod._AVM_MF_SCHEMA_VERSION == "1.5.0"
    # Forward-compat string unchanged — SAM3 consumer's 1.x family covers us
    assert _bridge_mod._AVM_MF_SCHEMA_MINOR_COMPATIBLE_WITH == "1.x"


def test_two_stage_seed_builder_unit():
    """Direct unit test of _seed_from_crop_in_two_stage helper.

    Pinning the math so future refactors don't accidentally break the
    crop-space-to-full-frame point projection.
    """
    raw1 = _make_stage1_response(bbox=(300, 200, 700, 600))
    raw2 = _make_stage2_response(
        fg=[[500, 500]],  # crop-space center
        bg=[[100, 100]],  # crop-space upper-left
        confidence=0.85,
    )
    seed, meta = _bridge_mod._seed_from_crop_in_two_stage(
        raw1, raw2, frame_idx=5, obj_id=7,
        num_pos=1, num_neg=1,
        pW=1920, pH=1080, crop_padding=24,
    )
    assert seed["frame_idx"] == 5
    assert seed["obj_id"] == 7
    assert seed["crop_in_applied"] is True
    assert seed["confidence"] == 0.85
    # Tight bbox in full-frame normalized coords (no padding)
    assert seed["box"] == pytest.approx([0.5, 0.4, 0.4, 0.4], abs=1e-3)
    # crop_meta is pixel-space with padding applied
    assert meta["orig_w"] == 1920 and meta["orig_h"] == 1080
    # Crop bounds with padding: max(0, 576-24)=552, max(0, 216-24)=192,
    # min(1920, 1344+24)=1368, min(1080, 648+24)=672
    assert meta["x1"] == 552 and meta["y1"] == 192
    assert meta["x2"] == 1368 and meta["y2"] == 672
    # Point projection: crop center (500/1000) → crop midpoint
    # crop_w=816, crop_h=480; pixel (552 + 0.5*816, 192 + 0.5*480) = (960, 432)
    # Full-frame [0,1]: (960/1920, 432/1080) = (0.5, 0.4)
    assert seed["pos_pts"][0] == pytest.approx([0.5, 0.4], abs=1e-3)
