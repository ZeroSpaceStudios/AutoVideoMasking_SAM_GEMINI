"""Tests for VLMMultiFrameBBoxPreview (D-380).

Per-keyframe debug overlay node. Iterates v1.5 seed_prompts JSON and renders
one preview frame per accepted seed with box + positives + negatives + crop_box.

Loads the module via the synthetic-package-alias pattern used by
test_vlm_to_bbox_and_points_multi_frame.py.
"""

import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Module load with synthetic package alias
# ---------------------------------------------------------------------------

_pkg_root = pathlib.Path(__file__).resolve().parent.parent
_nodes_dir = _pkg_root / "nodes"

_pkg_name = "_avm_test_pkg_preview"
_pkg = types.ModuleType(_pkg_name)
_pkg.__path__ = [str(_nodes_dir)]
sys.modules[_pkg_name] = _pkg


def _load(rel_path: str, full_name: str):
    spec = importlib.util.spec_from_file_location(full_name, str(_nodes_dir / rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_prompts_mod = _load("prompts.py", f"{_pkg_name}.prompts")
_bridge_mod = _load("vlm_sam3_bridge.py", f"{_pkg_name}.vlm_sam3_bridge")

VLMMultiFrameBBoxPreview = _bridge_mod.VLMMultiFrameBBoxPreview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_batch(T=20, H=120, W=200, channels=3):
    """Real torch.FloatTensor in [0,1] with shape [T,H,W,C]. Uses real arrays
    because the node calls _tensor_to_pil which uses .numpy()."""
    arr = np.random.rand(T, H, W, channels).astype(np.float32)
    return torch.from_numpy(arr)


def _make_v1_5_seeds(frames=(5, 10, 15), with_crop_in=False, with_confidence=True):
    """Build a v1.5 seed_prompts payload with the given keyframe indices."""
    seeds = []
    for f in frames:
        s = {
            "frame_idx": f,
            "obj_id": 1,
            "pos_pts": [[0.3, 0.3], [0.5, 0.5], [0.7, 0.7]],
            "neg_pts": [[0.1, 0.1]],
            "box": [0.5, 0.5, 0.4, 0.4],
        }
        if with_confidence:
            s["confidence"] = 0.85
        if with_crop_in:
            s["crop_in_applied"] = True
            s["crop_meta"] = {"x1": 40, "y1": 30, "x2": 160, "y2": 90, "orig_w": 200, "orig_h": 120}
        else:
            s["crop_in_applied"] = False
            s["crop_meta"] = None
        seeds.append(s)
    return {
        "schema_type": "sam3_seed_prompts",
        "schema_version": "1.5.0",
        "schema_minor_compatible_with": "1.x",
        "generator_node": "VLMtoBBoxAndPointsMultiFrame",
        "accepted_frames": list(frames),
        "seeds": seeds,
        "crop_in_enabled": with_crop_in,
        "confidence_threshold": 0.0,
        "low_confidence_skipped": [],
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_emits_one_preview_per_seed():
    """N seeds → IMAGE batch of size N."""
    images = _make_image_batch(T=20)
    seeds = _make_v1_5_seeds(frames=(5, 10, 15))
    node = VLMMultiFrameBBoxPreview()
    preview, info = node.draw(images=images, seed_prompts=json.dumps(seeds))
    assert preview.shape[0] == 3
    assert preview.shape[1:] == images.shape[1:]  # H, W, C preserved
    assert "rendered 3 previews" in info


def test_output_is_image_tensor_in_0_1_range():
    """Output dtype + value range matches IMAGE contract."""
    images = _make_image_batch(T=20)
    seeds = _make_v1_5_seeds(frames=(5,))
    node = VLMMultiFrameBBoxPreview()
    preview, _ = node.draw(images=images, seed_prompts=json.dumps(seeds))
    assert preview.dtype == torch.float32
    assert preview.min().item() >= 0.0
    assert preview.max().item() <= 1.0


def test_previews_sorted_by_frame_idx():
    """Seeds emitted out of order must be sorted by frame_idx in output."""
    images = _make_image_batch(T=30)
    seeds = _make_v1_5_seeds(frames=(20, 5, 15, 10))
    node = VLMMultiFrameBBoxPreview()
    _preview, info = node.draw(images=images, seed_prompts=json.dumps(seeds))
    # Info reports the range from sorted seeds: lowest to highest
    assert "5-20" in info


# ---------------------------------------------------------------------------
# Frame-index out-of-range tolerance
# ---------------------------------------------------------------------------


def test_seeds_with_out_of_range_frame_idx_are_skipped():
    """Seeds whose frame_idx exceeds the image batch size are soft-skipped."""
    images = _make_image_batch(T=10)
    seeds = _make_v1_5_seeds(frames=(5, 50, 7))  # 50 > T
    node = VLMMultiFrameBBoxPreview()
    preview, info = node.draw(images=images, seed_prompts=json.dumps(seeds))
    assert preview.shape[0] == 2  # only frames 5 and 7
    assert "1 skipped" in info
    assert "frame_idx=50" in info


def test_all_out_of_range_returns_placeholder():
    """When all seeds skip, output is a single placeholder frame (not crash)."""
    images = _make_image_batch(T=5)
    seeds = _make_v1_5_seeds(frames=(20, 30, 40))
    node = VLMMultiFrameBBoxPreview()
    preview, info = node.draw(images=images, seed_prompts=json.dumps(seeds))
    assert preview.shape[0] == 1
    assert "no seeds rendered" in info


# ---------------------------------------------------------------------------
# Schema tolerance
# ---------------------------------------------------------------------------


def test_empty_seed_prompts_returns_placeholder():
    images = _make_image_batch(T=5)
    node = VLMMultiFrameBBoxPreview()
    preview, info = node.draw(images=images, seed_prompts="")
    assert preview.shape[0] == 1
    assert "empty seed_prompts" in info


def test_malformed_json_raises_with_diagnostic():
    images = _make_image_batch(T=5)
    node = VLMMultiFrameBBoxPreview()
    with pytest.raises(ValueError, match="not valid JSON"):
        node.draw(images=images, seed_prompts="{ malformed")


def test_v1_4_payload_renders_without_crop_in_features():
    """v1.4 seeds (no crop_in_applied / crop_meta / confidence) render cleanly."""
    images = _make_image_batch(T=10)
    payload = {
        "schema_type": "sam3_seed_prompts",
        "schema_version": "1.4.0",
        "seeds": [
            {"frame_idx": 5, "obj_id": 1, "pos_pts": [[0.5, 0.5]], "neg_pts": [],
             "box": [0.5, 0.5, 0.3, 0.3]}
        ],
    }
    node = VLMMultiFrameBBoxPreview()
    preview, info = node.draw(images=images, seed_prompts=json.dumps(payload))
    assert preview.shape[0] == 1
    assert "rendered 1 previews" in info


def test_payload_without_seeds_key_returns_placeholder():
    images = _make_image_batch(T=5)
    node = VLMMultiFrameBBoxPreview()
    preview, info = node.draw(images=images, seed_prompts=json.dumps({"schema_type": "x"}))
    assert preview.shape[0] == 1
    assert "no seeds in payload" in info


def test_seed_missing_frame_idx_is_skipped():
    """A seed dict without frame_idx is silently skipped (not crash)."""
    images = _make_image_batch(T=10)
    payload = {
        "seeds": [
            {"frame_idx": 5, "pos_pts": [[0.5, 0.5]], "neg_pts": [], "box": [0.5, 0.5, 0.3, 0.3]},
            {"pos_pts": [[0.5, 0.5]], "neg_pts": [], "box": [0.5, 0.5, 0.3, 0.3]},  # no frame_idx
        ],
    }
    node = VLMMultiFrameBBoxPreview()
    preview, _ = node.draw(images=images, seed_prompts=json.dumps(payload))
    assert preview.shape[0] == 1  # only the seed with frame_idx renders


# ---------------------------------------------------------------------------
# Image shape validation
# ---------------------------------------------------------------------------


def test_bad_image_shape_raises():
    """Non-4D images tensor rejected with clear message."""
    bad = torch.zeros((10, 100, 100))  # 3D, missing C
    seeds = _make_v1_5_seeds(frames=(5,))
    node = VLMMultiFrameBBoxPreview()
    with pytest.raises(ValueError, match="must be IMAGE tensor"):
        node.draw(images=bad, seed_prompts=json.dumps(seeds))


# ---------------------------------------------------------------------------
# Visual content sanity — pixels at the overlay positions actually changed
# ---------------------------------------------------------------------------


def test_overlay_actually_paints_pixels():
    """Sanity check: at least one pixel in the preview differs from the
    source frame (rectangles + circles got drawn)."""
    images = _make_image_batch(T=10, H=200, W=300)
    seeds = _make_v1_5_seeds(frames=(5,))
    node = VLMMultiFrameBBoxPreview()
    preview, _ = node.draw(images=images, seed_prompts=json.dumps(seeds))
    # Source frame at idx 5 vs preview frame at idx 0 (only one seed)
    source = images[5]
    overlay = preview[0]
    diff = (source - overlay).abs().sum().item()
    assert diff > 0.0, "overlay didn't paint any pixels"


def test_crop_box_only_rendered_when_show_crop_box_true():
    """show_crop_box=False suppresses the yellow crop outline. Verified by
    comparing pixel diff between two renders of the SAME crop_in seed —
    one with show_crop_box on, one off."""
    images = _make_image_batch(T=10, H=200, W=300)
    seeds = _make_v1_5_seeds(frames=(5,), with_crop_in=True)
    node = VLMMultiFrameBBoxPreview()
    p_on, _ = node.draw(images=images, seed_prompts=json.dumps(seeds), show_crop_box=True)
    p_off, _ = node.draw(images=images, seed_prompts=json.dumps(seeds), show_crop_box=False)
    diff = (p_on - p_off).abs().sum().item()
    assert diff > 0.0, "crop_box rendering had no visible effect when toggled"


def test_labels_off_reduces_pixel_diff_vs_source():
    """show_labels=False removes the top-left annotation bar.
    Confirms the toggle is wired."""
    images = _make_image_batch(T=10, H=200, W=300)
    seeds = _make_v1_5_seeds(frames=(5,))
    node = VLMMultiFrameBBoxPreview()
    p_on, _ = node.draw(images=images, seed_prompts=json.dumps(seeds), show_labels=True)
    p_off, _ = node.draw(images=images, seed_prompts=json.dumps(seeds), show_labels=False)
    # Top-left bar (0:24 rows) should be ALL different between the two
    bar_on = p_on[0, :24, :, :]
    bar_off = p_off[0, :24, :, :]
    bar_diff = (bar_on - bar_off).abs().sum().item()
    assert bar_diff > 0.0
