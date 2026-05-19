"""
VLM -> SAM3 Bridge Node
Calls Gemini to auto-generate bbox or point prompts,
then outputs native SAM3_BOX_PROMPT / SAM3_POINTS_PROMPT types that wire
directly into SAM3Segmentation or SAM3Grounding.

Author: Hera Kang

Coordinate conventions (must match segmentation.py):
  SAM3_BOX_PROMPT   : {"box": [cx, cy, w, h],  "label": bool}   - normalized [0,1]
  SAM3_BOXES_PROMPT : {"boxes": [...], "labels": [...]}
  SAM3_POINT_PROMPT : {"point": [x, y], "label": int}           - normalized [0,1]
  SAM3_POINTS_PROMPT: {"points": [...], "labels": [...]}
"""

import os
import re
import json
import io
import numpy as np
from PIL import Image
from .prompts import (
    DESCRIBE_IMAGE,
    bbox_prompt, points_prompt, multi_bbox_prompt, bbox_and_points_prompt,
    face_parts_bbox_prompt, face_precise_points_prompt,
    face_region_stage1_prompt, face_region_stage2_prompt,
    layer_discovery_prompt, layer_localize_prompt,
    reference_match_prompt,
    autocrop_discovery_prompt, autocrop_localize_prompt,
)

GEMINI_MODELS = ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]
OPENROUTER_MODELS = [
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
]
AVAILABLE_MODELS = GEMINI_MODELS + [f"openrouter:{m}" for m in OPENROUTER_MODELS]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Path to .env file in the package root (one level up from nodes/)
_ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")


def _resolve_api_key(ui_key: str, provider: str = "gemini_direct") -> str:
    """Tiered API key lookup: env var → .env file → UI input. Provider-aware."""
    env_var = "OPENROUTER_API_KEY" if provider == "openrouter" else "GEMINI_API_KEY"
    label = "OpenRouter" if provider == "openrouter" else "Gemini"

    key = os.environ.get(env_var, "").strip()
    if key:
        print(f"[AVM] {label} API key loaded from environment variable.")
        return key
    env_path = os.path.normpath(_ENV_FILE)
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{env_var}="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        print(f"[AVM] {label} API key loaded from .env file.")
                        return key
    if ui_key.strip():
        print(f"[AVM] {label} API key loaded from node UI input.")
        return ui_key.strip()
    raise ValueError(
        f"[AVM] No API key found. Set {env_var} env var, add it to .env, or enter it in the node."
    )


# =============================================================================
# AVMAPIConfig — set credentials once, connect api slot to all nodes
# =============================================================================

class AVMAPIConfig:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (AVAILABLE_MODELS, {"default": DEFAULT_MODEL,
                               "tooltip": "Gemini direct or openrouter:<slug> for VLM inference"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False,
                            "tooltip": "Leave blank to use GEMINI_API_KEY / OPENROUTER_API_KEY env var or .env file"}),
            }
        }

    RETURN_TYPES  = ("AVM_API",)
    RETURN_NAMES  = ("api",)
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, model_name, api_key=""):
        if model_name.startswith("openrouter:"):
            provider = "openrouter"
            actual_model = model_name.split(":", 1)[1]
        else:
            provider = "gemini_direct"
            actual_model = model_name
        resolved_key = _resolve_api_key(api_key, provider)
        return ({
            "api_key": resolved_key,
            "model_name": actual_model,
            "provider": provider,
            "base_url": OPENROUTER_BASE_URL,
        },)


# -- helpers ------------------------------------------------------------------

def _tensor_to_pil(image_tensor):
    arr = (image_tensor[0].numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    return json.loads(text)

def _maybe_normalize_corners(x1, y1, x2, y2, W, H):
    # Gemini returns 0-1000 normalized scale, not pixel coords.
    # If any value > 2.0 assume 0-1000 scale and divide by 1000.
    # Note: values may slightly overshoot 1000 (e.g., 1071 when subject reaches
    # image edge); clamp is applied downstream, don't switch to pixel path here.
    if any(v > 2.0 for v in [x1, y1, x2, y2]):
        return x1/1000, y1/1000, x2/1000, y2/1000
    return x1, y1, x2, y2

def normalize_points(pts_raw, label_val, W=1000, H=1000):
    """Clamp Gemini points to [0,1]. Divides by W/H if value > 1.5, else treats as already normalized."""
    result, lbls = [], []
    for pt in pts_raw:
        nx = max(0.0, min(1.0, pt[0] / W if pt[0] > 1.5 else pt[0]))
        ny = max(0.0, min(1.0, pt[1] / H if pt[1] > 1.5 else pt[1]))
        result.append([nx, ny])
        lbls.append(label_val)
    return {"points": result, "labels": lbls}

def normalize_points_crop_to_full(pts_raw, label_val, crop_w, crop_h, crop_x1, crop_y1, full_W, full_H):
    """Map crop-space Gemini points into full-image [0,1] coords."""
    result, lbls = [], []
    for pt in pts_raw:
        abs_x = (pt[0] / 1000 * crop_w + crop_x1) if pt[0] > 1.5 else (pt[0] * crop_w + crop_x1)
        abs_y = (pt[1] / 1000 * crop_h + crop_y1) if pt[1] > 1.5 else (pt[1] * crop_h + crop_y1)
        result.append([max(0.0, min(1.0, abs_x / full_W)), max(0.0, min(1.0, abs_y / full_H))])
        lbls.append(label_val)
    return {"points": result, "labels": lbls}


def normalize_points_auto(pts_raw, label_val):
    """Point normalizer for Gemini 0-1000 scale output.

    Gemini's grounding tokens are 0-1000 scale; values may slightly overshoot
    1000 when points are near image edges. Dividing by 1000 keeps overshoot
    handling cheap: the clamp to [0,1] absorbs any excess.
    Values <= 1.5 are treated as already 0-1 normalized.
    """
    result, lbls = [], []
    for pt in pts_raw:
        x, y = pt[0], pt[1]
        nx = x / 1000.0 if x > 1.5 else x
        ny = y / 1000.0 if y > 1.5 else y
        result.append([max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))])
        lbls.append(label_val)
    return {"points": result, "labels": lbls}

def _call_gemini_direct(pil_imgs, prompt, api):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    client = genai.Client(api_key=api["api_key"])
    parts = []
    for img in pil_imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    parts.append(types.Part.from_text(text=prompt))
    response = client.models.generate_content(model=api["model_name"], contents=parts)
    return response.text


def _call_openrouter(pil_imgs, prompt, api):
    import base64
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed. Run: pip install requests")

    content = [{"type": "text", "text": prompt}]
    for img in pil_imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})

    payload = {
        "model": api["model_name"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/neonvoid/ComfyUI-AutoVideoMasking",
        "X-Title": "ComfyUI-AutoVideoMasking",
    }
    url = api.get("base_url", OPENROUTER_BASE_URL).rstrip("/") + "/chat/completions"

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=180)
    except requests.Timeout as e:
        raise RuntimeError("[AVM/OpenRouter] Request timed out after 180s") from e
    except requests.RequestException as e:
        raise RuntimeError(f"[AVM/OpenRouter] Network error: {e}") from e

    if not r.ok:
        raise RuntimeError(f"[AVM/OpenRouter] HTTP {r.status_code}: {r.text[:500]}")

    try:
        data = r.json()
        text = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"[AVM/OpenRouter] Malformed response ({e}): {r.text[:500]}") from e

    if not isinstance(text, str):
        raise RuntimeError(f"[AVM/OpenRouter] Expected string content, got {type(text).__name__}: {str(text)[:300]}")

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _call_gemini(pil_img_or_list, prompt, api):
    """Provider-dispatching VLM call. Accepts a single PIL image or list of PIL images."""
    pil_imgs = pil_img_or_list if isinstance(pil_img_or_list, list) else [pil_img_or_list]
    provider = api.get("provider", "gemini_direct")
    if provider == "openrouter":
        return _call_openrouter(pil_imgs, prompt, api)
    return _call_gemini_direct(pil_imgs, prompt, api)


def _find_sam3_nodes_dir() -> str:
    """Locate the ComfyUI-SAM3/nodes directory.

    Search order:
      1. AVM_SAM3_DIR environment variable (explicit override)
      2. sys.modules — ComfyUI may have already imported SAM3 modules
      3. ComfyUI folder_paths custom_nodes base
      4. Hardcoded relative path (../../ComfyUI-SAM3/nodes)

    Raises ImportError with actionable guidance if nothing is found.
    """
    import os, sys

    env = os.environ.get("AVM_SAM3_DIR", "").strip()
    if env:
        path = os.path.normpath(env)
        if os.path.isdir(path):
            return path
        raise ImportError(
            f"[AVM] AVM_SAM3_DIR is set to '{env}' but that directory does not exist."
        )

    for mod in sys.modules.values():
        f = getattr(mod, "__file__", None)
        if f and "ComfyUI-SAM3" in f:
            candidate = os.path.normpath(os.path.dirname(f))
            if os.path.isfile(os.path.join(candidate, "video_state.py")):
                return candidate

    try:
        import folder_paths
        for base in folder_paths.get_folder_paths("custom_nodes"):
            candidate = os.path.normpath(os.path.join(base, "ComfyUI-SAM3", "nodes"))
            if os.path.isdir(candidate):
                return candidate
    except Exception:
        pass

    candidate = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "ComfyUI-SAM3", "nodes")
    )
    if os.path.isdir(candidate):
        return candidate

    raise ImportError(
        "[AVM] ComfyUI-SAM3 not found. Install it into your custom_nodes directory, "
        "or set the AVM_SAM3_DIR environment variable to its 'nodes' folder path. "
        "Expected to find: video_state.py, sam3_video_nodes.py"
    )


def _load_sam3_modules():
    """Load video_state and sam3_video_nodes from ComfyUI-SAM3. Returns (vs_mod, vn_mod)."""
    import importlib.util, os as _os

    sam3_dir = _find_sam3_nodes_dir()

    def _load(fname):
        path = _os.path.join(sam3_dir, fname)
        if not _os.path.exists(path):
            raise ImportError(f"[AVM] {fname} not found in SAM3 nodes dir: {sam3_dir}")
        spec = importlib.util.spec_from_file_location(f"_avm_sam3_{fname[:-3]}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return _load("video_state.py"), _load("sam3_video_nodes.py")


# =============================================================================
# VLMImageTest — verify Gemini is receiving the image correctly
# =============================================================================

class VLMImageTest:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api":   ("AVM_API",),
            }
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("description",)
    FUNCTION      = "run"
    CATEGORY      = "AVM"
    OUTPUT_NODE   = True

    def run(self, image, api):
        pil_img = _tensor_to_pil(image)
        print(f"[VLMImageTest] Image size: {pil_img.size}, model: {api['model_name']}")
        prompt = DESCRIBE_IMAGE
        raw = _call_gemini(pil_img, prompt, api)
        print(f"[VLMImageTest] Response: {raw}")
        return (raw,)


# =============================================================================
# VLMtoBBox
# =============================================================================

class VLMtoBBox:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":              ("IMAGE",),
                "api":                ("AVM_API",),
                "target_description": ("STRING", {"default": "the main subject", "multiline": False}),
                "is_positive":        ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "few_shot_examples": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES  = ("SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT", "STRING")
    RETURN_NAMES  = ("box_prompt", "boxes_prompt", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, image, api, target_description, is_positive, few_shot_examples=""):
        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = "\n\nExamples:\n" + few_shot_examples.strip() + "\n\nApply same quality to the new image."

        prompt = bbox_prompt(target_description, W, H, few_shot_block)

        raw = _call_gemini(pil_img, prompt, api)
        print(f"[VLMtoBBox] Raw: {raw}")

        try:
            data = _parse_json(raw)
            x1, y1, x2, y2 = data["bbox"]
        except Exception as e:
            raise RuntimeError(f"[VLMtoBBox] Failed to parse Gemini response: {e}\nRaw: {raw}") from e

        x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
        cx = (x1n + x2n) / 2;  cy = (y1n + y2n) / 2
        bw = x2n - x1n;        bh = y2n - y1n

        box_prompt   = {"box": [cx, cy, bw, bh], "label": is_positive}
        boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [is_positive]}
        print(f"[VLMtoBBox] (cx,cy,w,h): [{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}]")
        return (box_prompt, boxes_prompt, raw)


# =============================================================================
# VLMtoPoints
# =============================================================================

class VLMtoPoints:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":              ("IMAGE",),
                "api":                ("AVM_API",),
                "target_description": ("STRING", {"default": "the main subject", "multiline": False}),
                "num_pos_points":     ("INT", {"default": 6, "min": 1, "max": 12}),
                "num_neg_points":     ("INT", {"default": 3, "min": 0, "max": 6}),
            },
            "optional": {
                "bbox_context":      ("SAM3_BOXES_PROMPT",),
                "few_shot_examples": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES  = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "STRING")
    RETURN_NAMES  = ("positive_points", "negative_points", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, image, api, target_description, num_pos_points, num_neg_points,
            bbox_context=None, few_shot_examples=""):
        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = "\n\nGuidance:\n" + few_shot_examples.strip()

        size_note = "This image is cropped to the target." if (bbox_context and bbox_context.get("boxes")) else f"Image: {W}x{H} pixels."

        prompt = points_prompt(target_description, size_note, num_pos_points, num_neg_points, few_shot_block)

        crop_x1, crop_y1, crop_w, crop_h = 0, 0, W, H
        send_img = pil_img

        if bbox_context and bbox_context.get("boxes"):
            cx_n, cy_n, bw_n, bh_n = bbox_context["boxes"][0]
            cx1 = max(0, int((cx_n - bw_n/2) * W) - 10)
            cy1 = max(0, int((cy_n - bh_n/2) * H) - 10)
            cx2 = min(W, int((cx_n + bw_n/2) * W) + 10)
            cy2 = min(H, int((cy_n + bh_n/2) * H) + 10)
            send_img = pil_img.crop((cx1, cy1, cx2, cy2))
            crop_x1, crop_y1 = cx1, cy1
            crop_w, crop_h = cx2 - cx1, cy2 - cy1

        raw = _call_gemini(send_img, prompt, api)
        print(f"[VLMtoPoints] Raw: {raw}")

        try:
            data = _parse_json(raw)
            pos_raw = data.get("positive", [[crop_w//2, crop_h//2]])
            neg_raw = data.get("negative", [])
        except Exception as e:
            raise RuntimeError(f"[VLMtoPoints] Failed to parse Gemini response: {e}\nRaw: {raw}") from e

        pos_raw = pos_raw[:num_pos_points]
        neg_raw = neg_raw[:num_neg_points]

        positive_points = normalize_points_crop_to_full(pos_raw, 1, crop_w, crop_h, crop_x1, crop_y1, W, H)
        negative_points = normalize_points_crop_to_full(neg_raw, 0, crop_w, crop_h, crop_x1, crop_y1, W, H)
        return (positive_points, negative_points, raw)


# =============================================================================
# VLMtoMultiBBox
# =============================================================================

class VLMtoMultiBBox:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":              ("IMAGE",),
                "api":                ("AVM_API",),
                "target_description": ("STRING", {"default": "all bags", "multiline": False}),
                "max_objects":        ("INT", {"default": 3, "min": 1, "max": 5}),
            },
            "optional": {
                "few_shot_examples": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES  = (
        "SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT",
        "SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT",
        "SAM3_BOXES_PROMPT", "STRING"
    )
    RETURN_NAMES  = ("box_1", "box_2", "box_3", "box_4", "box_5", "all_boxes", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, image, api, target_description, max_objects, few_shot_examples=""):
        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = "\n\nExamples:\n" + few_shot_examples.strip() if few_shot_examples.strip() else ""

        prompt = multi_bbox_prompt(target_description, W, H, max_objects, few_shot_block)

        raw = _call_gemini(pil_img, prompt, api)
        print(f"[VLMtoMultiBBox] Raw: {raw}")

        try:
            objects = _parse_json(raw).get("objects", [])[:max_objects]
        except Exception as e:
            print(f"[AVM ERROR] VLMtoMultiBBox failed to parse response: {e}\nRaw: {raw}")
            objects = []

        def to_boxes(obj):
            x1, y1, x2, y2 = obj["bbox"]
            x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
            cx = (x1n+x2n)/2; cy = (y1n+y2n)/2
            return {"boxes": [[cx, cy, x2n-x1n, y2n-y1n]], "labels": [True]}

        empty = {"boxes": [], "labels": []}
        box_outputs = [to_boxes(o) for o in objects]
        while len(box_outputs) < 5:
            box_outputs.append(empty)

        all_boxes = {
            "boxes":  [b for bp in box_outputs for b in bp["boxes"]],
            "labels": [l for bp in box_outputs for l in bp["labels"]],
        }
        return (*box_outputs, all_boxes, raw)


# =============================================================================
# VLMtoBBoxAndPoints — single call: bbox + points together
# =============================================================================

class VLMtoBBoxAndPoints:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":              ("IMAGE",),
                "api":                ("AVM_API",),
                "target_description": ("STRING", {"default": "the main subject", "multiline": False}),
                "num_pos_points":     ("INT", {"default": 6, "min": 1, "max": 12}),
                "num_neg_points":     ("INT", {"default": 3, "min": 0, "max": 6}),
                "is_positive":        ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "few_shot_examples": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES  = ("SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "SAM3_BOX_AND_POINT", "STRING")
    RETURN_NAMES  = ("box_prompt", "boxes_prompt", "positive_points", "negative_points", "box_and_point", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, image, api, target_description, num_pos_points, num_neg_points,
            is_positive, few_shot_examples=""):
        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = "\n\nExamples:\n" + few_shot_examples.strip() if few_shot_examples.strip() else ""

        prompt = bbox_and_points_prompt(target_description, W, H, num_pos_points, num_neg_points, few_shot_block)

        raw = _call_gemini(pil_img, prompt, api)
        print(f"[VLMtoBBoxAndPoints] Raw: {raw}")

        try:
            data = _parse_json(raw)
            x1, y1, x2, y2 = data["bbox"]
            pos_raw = data.get("positive", [[W//2, H//2]])
            neg_raw = data.get("negative", [])
        except Exception as e:
            raise RuntimeError(f"[VLMtoBBoxAndPoints] Failed to parse Gemini response: {e}\nRaw: {raw}") from e

        pos_raw = pos_raw[:num_pos_points]
        neg_raw = neg_raw[:num_neg_points]

        print(f"[VLMtoBBoxAndPoints] Image size received: {W}x{H}, "
              f"bbox pixel: {x1},{y1},{x2},{y2}, "
              f"normalized: {x1/W:.3f},{y1/H:.3f},{x2/W:.3f},{y2/H:.3f}")

        x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
        cx = (x1n+x2n)/2; cy = (y1n+y2n)/2
        bw = x2n-x1n;     bh = y2n-y1n

        box_prompt   = {"box":   [cx, cy, bw, bh], "label": is_positive}
        boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [is_positive]}

        positive_points = normalize_points(pos_raw, 1)
        negative_points = normalize_points(neg_raw, 0)

        box_and_point = {
            "boxes":    boxes_prompt,
            "positive": positive_points,
            "negative": negative_points,
        }

        print(f"[VLMtoBBoxAndPoints] box:[{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}] pos:{len(positive_points['points'])} neg:{len(negative_points['points'])}")
        return (box_prompt, boxes_prompt, positive_points, negative_points, box_and_point, raw)


# =============================================================================
# VLMtoBBoxAndPointsMultiFrame — multi-frame extension feeding SAM3MultiFrameAddPrompt
# =============================================================================

# Schema version emitted by the multi-frame producer. The SAM3 fork's
# SAM3MultiFrameAddPrompt accepts the 1.x family via schema_minor_compatible_with.
# Independent constant per R2 plan-review judgment (Codex+Gemini agreed on
# option a: no cross-repo shared module).
#
# v1.5.0 (D-381): adds optional crop-in mode + confidence-gating.
#   Top-level additions:
#     crop_in_enabled: bool
#     crop_padding: int
#     confidence_threshold: float (0.0 = gating disabled)
#     low_confidence_skipped: [int]  (frame_idx values dropped by the gate)
#   Per-seed additions:
#     crop_in_applied: bool
#     crop_meta: {x1, y1, x2, y2, orig_w, orig_h} | null
#     confidence: float  (Gemini self-assessment, 0.0-1.0; defaults to 1.0
#                         when the model omits the field)
# Forward-compat: consumer's `schema_minor_compatible_with: "1.x"` absorbs
# the bump (WARN, not raise). All new fields are additive — v1.4 consumers
# silently ignore them.
_AVM_MF_SCHEMA_VERSION = "1.5.0"
_AVM_MF_SCHEMA_TYPE = "sam3_seed_prompts"
_AVM_MF_SCHEMA_MINOR_COMPATIBLE_WITH = "1.x"

# Face-anatomy boundary rules for the two-stage crop-in mode (D-381).
# Duplicated from VLMFaceRegion._FACE_RULES — kept in sync manually since
# VLMFaceRegion is defined later in the file. Both nodes target face/head
# regions so the same anatomical guidance applies.
_AVM_FACE_REGION_RULES = (
    "CRITICAL boundary rules:\n"
    "  • FACE: bbox must include full chin, jaw underside, and any open mouth /\n"
    "    teeth / tongue interior. Never cut at the lips.\n"
    "  • NECK: bbox must extend fully to where neck meets collar or shoulders.\n"
    "    Never cut at the chin.\n"
    "  • HAIR: bbox extends to hair tips, not just the scalp.\n"
    "  • Foreground points must be deep inside the region — never on its border.\n"
    "  • Background points must be just outside the region boundary.\n"
    "  • Spread points across the WHOLE region, do NOT cluster them.\n"
)

# Presets where the two-stage crop-in rules read naturally (face-anatomy focus).
# Other presets still work but emit a soft notice that the rules will read awkwardly.
_AVM_CROP_IN_HEAD_PRESETS = frozenset({
    "face", "head", "head_and_shoulders", "hair", "custom",
})

# Quick-pick subject presets for VLMtoBBoxAndPointsMultiFrame.
# Sentinel "custom" at index 0 per D-349 (sentinel-FIRST cross-suite rule)
# so legacy saved workflows that didn't have this widget default to the
# back-compat free-form path (target_description is used verbatim).
#
# Resolution at run():
#   target_preset == "custom" -> send target_description as-is to Gemini
#   target_preset != "custom" + target_description is empty/default ->
#     send the preset's canned text only
#   target_preset != "custom" + target_description has user text ->
#     compose: "<preset_canned>, <target_description>" so refinements
#     append cleanly to the preset baseline.
_AVM_MF_TARGET_PRESETS = {
    "custom":           None,  # sentinel — uses target_description verbatim
    "face":             "the subject's face",
    "head":             "the subject's entire head, including hair",
    "head_and_shoulders": "the subject's head and shoulders",
    "upper_body":       "the subject's upper body, torso, and arms",
    "full_body":        "the subject's full body",
    "hair":             "the subject's hair",
    "hands":            "the subject's hands, wrists, and fingers",
    "clothing_top":     "the subject's shirt, jacket, or top garment",
    "clothing_bottom":  "the subject's pants, skirt, or bottom garment",
    "footwear":         "the subject's shoes or footwear",
    "prop_held":        "the prop or object the subject is holding",
}
# Default target_description value — used to detect "user hasn't overridden"
# when composing preset + description. Keep in sync with the widget default
# (line cross-ref: target_description STRING default in INPUT_TYPES below).
_AVM_MF_DEFAULT_TARGET_DESCRIPTION = "the main subject"


def _resolve_target_subject(target_preset: str, target_description: str) -> str:
    """Apply target_preset + target_description composition rules.

    See _AVM_MF_TARGET_PRESETS docstring for resolution table.
    Raises ValueError on unknown preset.
    """
    if target_preset not in _AVM_MF_TARGET_PRESETS:
        raise ValueError(
            f"[VLMtoBBoxAndPointsMultiFrame] unknown target_preset "
            f"'{target_preset}'. Supported: {list(_AVM_MF_TARGET_PRESETS.keys())}"
        )
    preset_text = _AVM_MF_TARGET_PRESETS[target_preset]
    desc = (target_description or "").strip()
    if preset_text is None:
        # custom mode — verbatim description (current/default behavior)
        return desc if desc else _AVM_MF_DEFAULT_TARGET_DESCRIPTION
    # Preset path — compose if user added meaningful detail beyond the default
    has_user_detail = bool(desc) and desc != _AVM_MF_DEFAULT_TARGET_DESCRIPTION
    if has_user_detail:
        return f"{preset_text}, {desc}"
    return preset_text


def _parse_keyframe_indices_strict(s: str, total_frames: int):
    """Validate keyframe_indices STRING pre-dispatch.

    Required: JSON int array, every value in [0, total_frames), no duplicates.
    Raises ValueError with diagnostic on any malformation.
    """
    if not isinstance(s, str) or not s.strip():
        raise ValueError("[VLMtoBBoxAndPointsMultiFrame] keyframe_indices must be non-empty STRING JSON")
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"[VLMtoBBoxAndPointsMultiFrame] keyframe_indices invalid JSON: {e}")
    if not isinstance(parsed, list):
        raise ValueError(
            f"[VLMtoBBoxAndPointsMultiFrame] keyframe_indices must be a JSON int array, "
            f"got {type(parsed).__name__}"
        )
    if not parsed:
        raise ValueError("[VLMtoBBoxAndPointsMultiFrame] keyframe_indices is empty — nothing to process")
    out = []
    seen = set()
    for i, v in enumerate(parsed):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"[VLMtoBBoxAndPointsMultiFrame] keyframe_indices[{i}]={v!r} is not an int"
            )
        if v < 0 or v >= total_frames:
            raise ValueError(
                f"[VLMtoBBoxAndPointsMultiFrame] keyframe_indices[{i}]={v} out of range "
                f"[0, {total_frames})"
            )
        if v in seen:
            raise ValueError(
                f"[VLMtoBBoxAndPointsMultiFrame] keyframe_indices has duplicate value {v}"
            )
        seen.add(v)
        out.append(v)
    return out


def _coerce_confidence(raw_val) -> float:
    """Coerce a model-reported confidence to [0.0, 1.0].

    Tolerates: missing field (None), numeric str, out-of-range floats, NaN.
    Default behavior when unparseable / missing: 1.0 (treat as confident — keeps
    behavior identical to pre-confidence v1.4 when the model omits the field).
    """
    if raw_val is None:
        return 1.0
    try:
        v = float(raw_val)
    except (TypeError, ValueError):
        return 1.0
    if not (v == v):  # NaN check
        return 1.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        # Some models emit 0-100 scale despite instructions; rescale.
        if v <= 100.0:
            return v / 100.0
        return 1.0
    return v


def _seed_from_bbox_and_points_response(
    raw: str, frame_idx: int, obj_id: int, num_pos: int, num_neg: int, W: int, H: int
):
    """Parse one Gemini bbox_and_points response into a v1.5 seed dict.

    Returns seed_dict. Raises ValueError on parse failure for caller
    to catch and soft-fail per keyframe.

    v1.5 adds `confidence` (model self-assessment) and `crop_in_applied` /
    `crop_meta` (always False / None for the single-stage path). Confidence
    defaults to 1.0 when the model omits the field — preserves v1.4 behavior.
    """
    data = _parse_json(raw)
    x1, y1, x2, y2 = data["bbox"]
    pos_raw = data.get("positive", []) or []
    neg_raw = data.get("negative", []) or []
    pos_raw = pos_raw[:num_pos]
    neg_raw = neg_raw[:num_neg]

    x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
    cx = (x1n + x2n) / 2
    cy = (y1n + y2n) / 2
    bw = x2n - x1n
    bh = y2n - y1n

    positive_pts = normalize_points(pos_raw, 1)["points"]
    negative_pts = normalize_points(neg_raw, 0)["points"]

    confidence = _coerce_confidence(data.get("confidence"))

    return {
        "frame_idx": int(frame_idx),
        "obj_id": int(obj_id),
        "pos_pts": positive_pts,
        "neg_pts": negative_pts,
        "box": [float(cx), float(cy), float(bw), float(bh)],
        # v1.5 (D-381) — crop-in metadata + confidence self-assessment.
        # Single-stage path never crops; confidence defaults to 1.0 if the
        # model omits the field (back-compat with pre-1.5 prompts).
        "crop_in_applied": False,
        "crop_meta": None,
        "confidence": confidence,
    }


def _seed_from_crop_in_two_stage(
    raw1: str, raw2: str, frame_idx: int, obj_id: int,
    num_pos: int, num_neg: int,
    pW: int, pH: int, crop_padding: int,
):
    """Build a v1.5 seed from two Gemini calls (Stage 1 bbox + Stage 2 points on crop).

    Raises ValueError on parse failure for caller to catch and soft-fail
    per keyframe — matches the single-stage behavior.

    Returns:
        (seed_dict, crop_meta_dict) — caller uses crop_meta for logging.
    """
    # Stage 1 — tight bbox in full-frame coords
    d1 = _parse_json(raw1)
    bx1, by1, bx2, by2 = d1["bbox"]
    bx1, by1, bx2, by2 = _maybe_normalize_corners(bx1, by1, bx2, by2, pW, pH)
    # Project normalized [0,1] back to pixel space for crop math
    bx1_px = bx1 * pW
    by1_px = by1 * pH
    bx2_px = bx2 * pW
    by2_px = by2 * pH
    # Gemini occasionally swaps corners; ensure x1<x2, y1<y2
    if bx1_px > bx2_px:
        bx1_px, bx2_px = bx2_px, bx1_px
    if by1_px > by2_px:
        by1_px, by2_px = by2_px, by1_px
    # Apply padding + clamp to image bounds
    cx1 = max(0, int(bx1_px) - crop_padding)
    cy1 = max(0, int(by1_px) - crop_padding)
    cx2 = min(pW, int(bx2_px) + crop_padding)
    cy2 = min(pH, int(by2_px) + crop_padding)
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(
            f"Stage1 produced empty crop: "
            f"[{cx1},{cy1},{cx2},{cy2}] from bbox=[{bx1_px:.1f},{by1_px:.1f},"
            f"{bx2_px:.1f},{by2_px:.1f}] padding={crop_padding} frame={pW}x{pH}. "
            f"Stage1 raw: {raw1}"
        )
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    # Stage 2 — points placed on the crop
    d2 = _parse_json(raw2)
    fg_raw = (d2.get("foreground", []) or [])[:num_pos]
    bg_raw = (d2.get("background", []) or [])[:num_neg]

    # Project Stage-2 crop-space points to full-frame [0,1]
    positive_pts = normalize_points_crop_to_full(fg_raw, 1, crop_w, crop_h, cx1, cy1, pW, pH)["points"]
    negative_pts = normalize_points_crop_to_full(bg_raw, 0, crop_w, crop_h, cx1, cy1, pW, pH)["points"]

    # Confidence is taken from Stage 2 — that's the call that placed the
    # points feeding SAM3. Stage 1's bbox alone is rarely the failure mode.
    confidence = _coerce_confidence(d2.get("confidence"))

    # Seed bbox = Stage-1 tight bbox (no padding) in full-frame normalized coords
    tx1 = max(0.0, min(1.0, bx1_px / pW))
    ty1 = max(0.0, min(1.0, by1_px / pH))
    tx2 = max(0.0, min(1.0, bx2_px / pW))
    ty2 = max(0.0, min(1.0, by2_px / pH))
    cx = (tx1 + tx2) / 2.0
    cy = (ty1 + ty2) / 2.0
    bw = tx2 - tx1
    bh = ty2 - ty1

    crop_meta = {
        "x1": int(cx1), "y1": int(cy1),
        "x2": int(cx2), "y2": int(cy2),
        "orig_w": int(pW), "orig_h": int(pH),
    }

    seed = {
        "frame_idx": int(frame_idx),
        "obj_id": int(obj_id),
        "pos_pts": positive_pts,
        "neg_pts": negative_pts,
        "box": [float(cx), float(cy), float(bw), float(bh)],
        "crop_in_applied": True,
        "crop_meta": crop_meta,
        "confidence": confidence,
    }
    return seed, crop_meta


class VLMtoBBoxAndPointsMultiFrame:
    """Multi-frame extension of VLMtoBBoxAndPoints.

    Calls Gemini once (or twice, in crop-in mode) per keyframe — parallel-fanned
    via ThreadPoolExecutor — each call reuses the existing single-frame
    `bbox_and_points_prompt` and the same parser. Emits a v1.5.0 batched payload
    (schema_type="sam3_seed_prompts") containing one seed per keyframe with
    per-frame `pos_pts`, `neg_pts`, and `box` ([cx,cy,w,h] normalized).

    Crop-in mode (D-381, enable_crop_in=True): per keyframe, Stage 1 detects a
    tight target bbox on the full frame, then Stage 2 places points on the
    cropped region (~512px instead of ~150px effective face resolution).
    Mirrors VLMFaceRegion's two-stage pattern. Doubles per-keyframe Gemini
    call count. Output coords are always full-frame regardless of mode —
    seeds wire directly into SAM3MultiFrameAddPrompt as before.

    Confidence gating (D-381, confidence_threshold > 0.0): the Gemini prompt
    now asks for a confidence self-assessment per keyframe (1.0 = certain,
    0.4 = blurry/occluded). Keyframes below threshold are DROPPED from the
    payload entirely. Rationale: one low-confidence keyframe injects noisy
    point prompts that corrupt SAM3's memory propagation for the surrounding
    ~15-frame neighborhood (forward+backward). Skipping is usually better
    than including. Threshold 0.0 disables the gate (v1.4 behavior).

    Downstream consumer: SAM3MultiFrameAddPrompt (ComfyUI-SAM3_nvFork).
    The consumer accepts the 1.x family via schema_minor_compatible_with;
    payloads from this node carry both points and box, and the consumer
    applies them as a chained prompt per seed. v1.5 adds top-level
    `crop_in_enabled` / `crop_padding` / `confidence_threshold` /
    `low_confidence_skipped` plus per-seed `crop_in_applied` / `crop_meta` /
    `confidence`; the consumer's existing 1.x forward-compat absorbs these
    as unknown fields.

    Single-frame VLMtoBBoxAndPoints is UNTOUCHED — its 5 outputs continue to
    emit unchanged for existing workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":             ("IMAGE", {"tooltip": "Image batch [T,H,W,C] — wire from VHS_LoadVideoPath or equivalent."}),
                "api":                ("AVM_API",),
                "target_description": ("STRING", {"default": "the main subject", "multiline": False,
                                                    "tooltip": "Subject description passed to Gemini per keyframe."}),
                "keyframe_indices":   ("STRING", {"default": "[0]", "forceInput": True,
                                                    "tooltip": "JSON int array, e.g. [5,13,25,60,72,88]. Wire from NV_KeyframeSampler.keyframe_indices."}),
                "num_pos_points":     ("INT", {"default": 6, "min": 1, "max": 12}),
                "num_neg_points":     ("INT", {"default": 3, "min": 0, "max": 6}),
                "parallel_call_count": ("INT", {"default": 3, "min": 1, "max": 7,
                                                    "tooltip": "Max concurrent Gemini calls. One call per keyframe; fanned out in batches."}),
                "obj_id":             ("INT", {"default": 1, "min": 1, "max": 100,
                                                    "tooltip": "SAM3 obj_id for every emitted seed. Match downstream Multi-Frame Add Prompt."}),
                # Appended at END of required block per D-201 — inserting
                # mid-schema would scramble widgets_values on saved workflows.
                # 'custom' sentinel is at index 0 of _AVM_MF_TARGET_PRESETS so
                # legacy workflows default to the free-form behavior.
                "target_preset":      (list(_AVM_MF_TARGET_PRESETS.keys()), {
                    "default": "custom",
                    "tooltip": (
                        "Quick-pick subject for common workflows. "
                        "'custom' uses target_description verbatim (default / "
                        "back-compat). Other presets supply a canned subject "
                        "phrase; if target_description has user-added detail "
                        "beyond the default, it gets appended for refinement "
                        "(e.g. 'face' + 'wearing glasses' -> 'the subject's "
                        "face, wearing glasses'). NOTE: the resolver does NOT "
                        "deduplicate — if your description repeats the preset "
                        "text, the prompt will too (intentional, lets you "
                        "emphasize)."
                    ),
                }),
                # D-381 crop-in extension — APPENDED at END of required block
                # per D-201 to preserve widgets_values for legacy workflows.
                # When False (default), behavior is identical to v1.4 single-stage.
                "enable_crop_in":     ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Two-stage crop-in mode (mirrors VLMFaceRegion). When "
                        "True: per keyframe, Stage 1 detects a tight target "
                        "bbox on the full frame, then Stage 2 places points on "
                        "the cropped region at higher effective resolution "
                        "(~512px instead of ~150px in 1920×1080). DOUBLES the "
                        "Gemini call count per keyframe. Best on face/head/hair "
                        "presets — uses face-anatomy boundary rules. Output "
                        "coords are always full-frame regardless of mode."
                    ),
                }),
                "crop_padding":       ("INT", {
                    "default": 24, "min": 0, "max": 80, "step": 1,
                    "tooltip": (
                        "Pixels of padding around the Stage-1 tight bbox "
                        "before Stage-2 crop. Larger padding gives Stage 2 "
                        "more context (better for crops at face boundaries / "
                        "hair / chin); smaller padding zooms further into the "
                        "target. Ignored when enable_crop_in=False."
                    ),
                }),
                # D-381 confidence gating — APPENDED at END of required block
                # per D-201. The Gemini prompt now requests a confidence self-
                # assessment per keyframe (1.0 = certain, 0.4 = blurry/occluded).
                # Setting threshold > 0 drops low-confidence keyframes from
                # the payload entirely, preventing them from corrupting SAM3's
                # memory propagation. Default 0.0 = disabled (back-compat).
                "confidence_threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Drop keyframes where Gemini's self-reported confidence "
                        "is below this threshold. 0.0 = disabled (accept all "
                        "keyframes, same as v1.4). 0.5 = drop heavily-blurred / "
                        "occluded frames. 0.7 = only accept high-confidence "
                        "anchors. Low-confidence keyframes inject noisy point "
                        "prompts that corrupt SAM3's memory propagation for a "
                        "~15-frame neighborhood (forward+backward); skipping "
                        "them is usually better than including them. Skipped "
                        "frame indices appear in the info STRING and in the "
                        "payload's `low_confidence_skipped` field for audit."
                    ),
                }),
            },
            "optional": {
                "few_shot_examples": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = (
        "STRING",                # seed_prompts (batched v1.4 JSON)
        "STRING",                # raw_vlm_responses
        "STRING",                # info
        # The five "init_*" outputs below are derived from the first accepted
        # keyframe and reformatted as the typed dicts SAM3VideoSegmentation
        # expects, so this single node can drive BOTH init and batch-add
        # without paying for an extra single-frame VLMtoBBoxAndPoints call.
        "SAM3_BOX_PROMPT",       # init_box_prompt (singular, for SAM3AddPrompt)
        "SAM3_BOXES_PROMPT",     # init_boxes_prompt (plural, for SAM3VideoSegmentation.positive_boxes)
        "SAM3_POINTS_PROMPT",    # init_positive_points
        "SAM3_POINTS_PROMPT",    # init_negative_points
        "INT",                   # init_frame_idx (the first accepted keyframe; wire to SAM3VideoSegmentation.frame_idx)
    )
    RETURN_NAMES = (
        "seed_prompts", "raw_vlm_responses", "info",
        "init_box_prompt", "init_boxes_prompt",
        "init_positive_points", "init_negative_points",
        "init_frame_idx",
    )
    FUNCTION     = "run"
    CATEGORY     = "AVM"
    DESCRIPTION  = (
        "Multi-frame Gemini → SAM3 prompt generator. Mirrors VLMtoBBoxAndPoints "
        "per keyframe, batches into a v1.4 sam3_seed_prompts payload for "
        "SAM3MultiFrameAddPrompt downstream. Also emits init_* outputs derived "
        "from the first accepted keyframe so SAM3VideoSegmentation init can be "
        "driven from the same Gemini call set without a separate single-frame "
        "VLMtoBBoxAndPoints node. Wire init_frame_idx to "
        "SAM3VideoSegmentation.frame_idx when the keyframe sampler does not "
        "include frame 0."
    )

    def run(self, images, api, target_description, keyframe_indices,
            num_pos_points, num_neg_points, parallel_call_count,
            obj_id, target_preset="custom",
            enable_crop_in=False, crop_padding=24,
            confidence_threshold=0.0,
            few_shot_examples=""):
        import concurrent.futures

        # ----- Input shape + validation -----
        if not hasattr(images, "shape") or len(images.shape) != 4:
            raise ValueError(
                f"[VLMtoBBoxAndPointsMultiFrame] images must be IMAGE tensor "
                f"[T,H,W,C] (4-D), got shape="
                f"{tuple(getattr(images, 'shape', ()))!r} type={type(images).__name__}"
            )
        total_frames = int(images.shape[0])
        H, W = int(images.shape[1]), int(images.shape[2])

        kf_list = _parse_keyframe_indices_strict(keyframe_indices, total_frames)

        # Resolve effective subject text from target_preset + target_description.
        # custom -> verbatim description; preset -> preset text (with optional
        # description appended as refinement). See _resolve_target_subject for
        # the composition table.
        effective_target = _resolve_target_subject(target_preset, target_description)

        few_shot_block = "\n\nExamples:\n" + few_shot_examples.strip() if few_shot_examples.strip() else ""

        # Soft notice when crop-in is enabled on a non-head preset — FACE_RULES
        # still applies but reads awkwardly for body/clothing/prop targets.
        if enable_crop_in and target_preset not in _AVM_CROP_IN_HEAD_PRESETS:
            print(
                f"[VLMtoBBoxAndPointsMultiFrame] NOTICE: crop-in enabled with "
                f"target_preset={target_preset!r} (non-head). The two-stage "
                f"prompts use face-anatomy boundary rules — they will still "
                f"work but may not be optimal for this target class."
            )

        # Clamp confidence_threshold to valid range — Comfy widget bounds should
        # already enforce this but defensive clamp protects against API callers.
        confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))

        print(
            f"[VLMtoBBoxAndPointsMultiFrame] T={total_frames}, frame={W}x{H}, "
            f"kf={kf_list}, parallel={parallel_call_count}, model={api.get('model_name')}, "
            f"target_preset={target_preset!r}, effective_target={effective_target!r}, "
            f"crop_in={enable_crop_in} (padding={crop_padding}px), "
            f"conf_gate={'OFF' if confidence_threshold == 0.0 else f'>={confidence_threshold:.2f}'}"
        )

        # ----- Build per-keyframe call tasks -----
        def _call_one(t: int):
            # images[t:t+1] preserves the [1,H,W,C] batch shape that AVM's
            # _tensor_to_pil expects (it indexes [0] internally — passing
            # images[t] would strip the batch dim and break the helper).
            single_batch = images[t:t+1]
            pil_img = _tensor_to_pil(single_batch)
            pW, pH = pil_img.size

            if not enable_crop_in:
                # Single-stage path (v1.4 behavior, unchanged)
                prompt = bbox_and_points_prompt(effective_target, pW, pH, num_pos_points, num_neg_points, few_shot_block)
                raw = _call_gemini(pil_img, prompt, api)
                seed = _seed_from_bbox_and_points_response(
                    raw, frame_idx=t, obj_id=obj_id, num_pos=num_pos_points, num_neg=num_neg_points, W=pW, H=pH
                )
                return t, seed, raw

            # Two-stage crop-in path (D-381, v1.5)
            # Stage 1 — tight bbox on full frame
            prompt1 = face_region_stage1_prompt(pW, pH, effective_target, _AVM_FACE_REGION_RULES)
            raw1 = _call_gemini(pil_img, prompt1, api)

            # Parse Stage 1 to compute crop bounds, then crop & call Stage 2.
            # The seed builder re-parses raw1; that's fine (cheap) and keeps the
            # seed builder a pure function of the two raw strings.
            d1 = _parse_json(raw1)
            bx1, by1, bx2, by2 = d1["bbox"]
            bx1n, by1n, bx2n, by2n = _maybe_normalize_corners(bx1, by1, bx2, by2, pW, pH)
            bx1_px, by1_px = bx1n * pW, by1n * pH
            bx2_px, by2_px = bx2n * pW, by2n * pH
            if bx1_px > bx2_px:
                bx1_px, bx2_px = bx2_px, bx1_px
            if by1_px > by2_px:
                by1_px, by2_px = by2_px, by1_px
            cx1 = max(0, int(bx1_px) - crop_padding)
            cy1 = max(0, int(by1_px) - crop_padding)
            cx2 = min(pW, int(bx2_px) + crop_padding)
            cy2 = min(pH, int(by2_px) + crop_padding)
            if cx2 <= cx1 or cy2 <= cy1:
                raise ValueError(
                    f"[crop-in t={t}] Stage1 produced empty crop: "
                    f"[{cx1},{cy1},{cx2},{cy2}] padding={crop_padding} frame={pW}x{pH}. "
                    f"Stage1 raw: {raw1}"
                )
            pil_crop = pil_img.crop((cx1, cy1, cx2, cy2))
            cW, cH = pil_crop.size

            # Stage 2 — high-density points placed on the crop
            prompt2 = face_region_stage2_prompt(cW, cH, effective_target, _AVM_FACE_REGION_RULES, num_pos_points, num_neg_points)
            raw2 = _call_gemini(pil_crop, prompt2, api)

            seed, _crop_meta = _seed_from_crop_in_two_stage(
                raw1, raw2, frame_idx=t, obj_id=obj_id,
                num_pos=num_pos_points, num_neg=num_neg_points,
                pW=pW, pH=pH, crop_padding=crop_padding,
            )
            raw_combined = f"=== Stage1 (bbox) ===\n{raw1}\n=== Stage2 (points) ===\n{raw2}"
            return t, seed, raw_combined

        # ----- Fan-out with ThreadPoolExecutor; soft-fail per keyframe -----
        results_by_t = {}
        raw_by_t = {}
        info_lines = []
        low_confidence_skipped = []  # list of frame_idx dropped by confidence gate
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_call_count) as pool:
            future_to_t = {pool.submit(_call_one, t): t for t in kf_list}
            for fut in concurrent.futures.as_completed(future_to_t):
                t = future_to_t[fut]
                try:
                    _t, seed, raw = fut.result()
                    raw_by_t[_t] = raw
                    n_pos = len(seed.get("pos_pts", []))
                    n_neg = len(seed.get("neg_pts", []))
                    crop_tag = "true" if seed.get("crop_in_applied") else "false"
                    conf = float(seed.get("confidence", 1.0))
                    # Confidence gate (D-381) — drop low-confidence keyframes
                    # entirely rather than feeding noisy hints to SAM3. Disabled
                    # when threshold == 0.0 (back-compat with v1.4 behavior).
                    if confidence_threshold > 0.0 and conf < confidence_threshold:
                        low_confidence_skipped.append(int(_t))
                        info_lines.append(
                            f"t={_t} accepted=False pos=0 neg=0 box=skipped_low_confidence "
                            f"confidence={conf:.3f} threshold={confidence_threshold:.2f}"
                        )
                        print(
                            f"[VLMtoBBoxAndPointsMultiFrame] keyframe t={_t} DROPPED by "
                            f"confidence gate: conf={conf:.3f} < threshold={confidence_threshold:.2f} "
                            f"(was pos={n_pos} neg={n_neg} crop_in={crop_tag})"
                        )
                        continue
                    results_by_t[_t] = seed
                    info_lines.append(
                        f"t={_t} accepted=True pos={n_pos} neg={n_neg} box=true "
                        f"crop_in={crop_tag} confidence={conf:.3f}"
                    )
                except Exception as e:
                    err_str = f"{type(e).__name__}: {e}"
                    print(f"[VLMtoBBoxAndPointsMultiFrame] WARN keyframe t={t} failed: {err_str}")
                    info_lines.append(f"t={t} accepted=False pos=0 neg=0 box=skipped_malformed error={err_str}")
                    raw_by_t[t] = f"[ERROR t={t}] {err_str}"

        if not results_by_t:
            # Distinguish "all parse-failed" from "all dropped by confidence gate"
            # so the error message points to the right knob.
            n_gated = len(low_confidence_skipped)
            n_failed = len(kf_list) - n_gated
            if n_gated > 0 and n_failed == 0:
                raise RuntimeError(
                    f"[VLMtoBBoxAndPointsMultiFrame] All {len(kf_list)} keyframes "
                    f"dropped by confidence gate (threshold={confidence_threshold:.2f}, "
                    f"skipped_frames={low_confidence_skipped}). Lower the threshold or "
                    f"disable gating (set to 0.0) to ship at least one keyframe. "
                    f"info:\n" + "\n".join(info_lines)
                )
            raise RuntimeError(
                f"[VLMtoBBoxAndPointsMultiFrame] All {len(kf_list)} keyframes failed "
                f"(parse_failures={n_failed}, confidence_gated={n_gated}). "
                f"Chain is broken — check raw_vlm_responses output for Gemini response details. "
                f"info:\n" + "\n".join(info_lines)
            )

        accepted_frames = sorted(results_by_t.keys())
        seeds_sorted = [results_by_t[t] for t in accepted_frames]

        # ----- Assemble v1.5 payload -----
        payload = {
            "schema_type": _AVM_MF_SCHEMA_TYPE,
            "schema_version": _AVM_MF_SCHEMA_VERSION,
            "schema_minor_compatible_with": _AVM_MF_SCHEMA_MINOR_COMPATIBLE_WITH,
            "generator_node": "VLMtoBBoxAndPointsMultiFrame",
            # `target_description` records the RAW widget value (semantic
            # contract: it's what the user typed). `effective_target_description`
            # records what was actually sent to Gemini after target_preset
            # resolution. Emitting both avoids silent contract drift if a
            # downstream consumer interpreted `target_description` as raw
            # input — they get the raw value, plus a clearly-named effective
            # field for the resolved prompt.
            "target_description": target_description,
            "effective_target_description": effective_target,
            "target_preset": target_preset,
            "frame_width": W,
            "frame_height": H,
            "total_frames": total_frames,
            "accepted_frames": accepted_frames,
            "seeds": seeds_sorted,
            # v1.5 (D-381) — top-level crop-in flag for downstream tooling.
            # Per-seed `crop_in_applied` + `crop_meta` live inside each seed.
            "crop_in_enabled": bool(enable_crop_in),
            "crop_padding": int(crop_padding),
            # v1.5 (D-381) — confidence gating audit trail. `confidence_threshold`
            # is the value used this run (0.0 = gating disabled). `low_confidence_skipped`
            # lists frame_idx values that Gemini self-scored below threshold and
            # were therefore excluded from `seeds`. Per-seed `confidence` lives
            # inside each entry of `seeds` for downstream weighting tooling.
            "confidence_threshold": float(confidence_threshold),
            "low_confidence_skipped": sorted(low_confidence_skipped),
        }

        # ----- Compose info STRING (compact, multi-line) -----
        n_accepted = len(accepted_frames)
        n_requested = len(kf_list)
        n_gated = len(low_confidence_skipped)
        # ThreadPoolExecutor doesn't expose true batch count; approximate as
        # ceil(n_requested / parallel_call_count) for telemetry.
        n_batches = (n_requested + parallel_call_count - 1) // parallel_call_count
        gate_summary = (
            f"confidence_gated={n_gated}/{n_requested}"
            if confidence_threshold > 0.0 else "confidence_gate=OFF"
        )
        summary = (
            f"effective_keyframes={n_accepted}/{n_requested} "
            f"parallel_batches={n_batches} {gate_summary} "
            f"schema_version={_AVM_MF_SCHEMA_VERSION}"
        )
        info_str = "\n".join(info_lines) + "\n" + summary

        # ----- Concatenate raw responses for debug -----
        raw_concat = "\n".join(
            f"=== t={t} ===\n{raw_by_t.get(t, '')}" for t in kf_list
        )

        # ----- Init outputs: derive from first accepted seed (lowest frame_idx) -----
        # Same Gemini data the batched payload already paid for. The
        # SAM3VideoSegmentation node expects positive_boxes as the typed dict
        # {boxes: [[cx,cy,w,h]], labels: [bool]}; SAM3AddPrompt expects the
        # singular {box: [cx,cy,w,h], label: bool}. We emit both so either
        # downstream node wires cleanly.
        init_seed = seeds_sorted[0]
        init_frame_idx = int(init_seed["frame_idx"])
        init_box_vec = list(init_seed["box"])
        init_pos_pts = list(init_seed.get("pos_pts", []))
        init_neg_pts = list(init_seed.get("neg_pts", []))

        init_box_prompt = {"box": init_box_vec, "label": True}
        init_boxes_prompt = {"boxes": [init_box_vec], "labels": [True]}
        init_positive_points = {
            "points": init_pos_pts,
            "labels": [1] * len(init_pos_pts),
        }
        init_negative_points = {
            "points": init_neg_pts,
            "labels": [0] * len(init_neg_pts),
        }

        print(
            f"[VLMtoBBoxAndPointsMultiFrame] {summary}, "
            f"init_frame_idx={init_frame_idx}, init_pos={len(init_pos_pts)}, "
            f"init_neg={len(init_neg_pts)}"
        )

        return (
            json.dumps(payload), raw_concat, info_str,
            init_box_prompt, init_boxes_prompt,
            init_positive_points, init_negative_points,
            init_frame_idx,
        )


# =============================================================================
# VLMPromptEditor — inspect and override the Gemini prompt inside the node
# =============================================================================

class VLMPromptEditor:
    """
    Drop-in replacement for VLMtoBBoxAndPoints.
    - Auto-builds the same prompt from parameters
    - Outputs prompt_used (STRING) so you can wire it to a text display node
    - override_prompt text area lets you edit the prompt directly in the node
      (leave empty to use auto-generated prompt)
    - Identical outputs to VLMtoBBoxAndPoints — fully compatible
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":              ("IMAGE",),
                "api":                ("AVM_API",),
                "target_description": ("STRING", {"default": "the main subject", "multiline": False}),
                "num_pos_points":     ("INT", {"default": 6, "min": 1, "max": 12}),
                "num_neg_points":     ("INT", {"default": 3, "min": 0, "max": 6}),
                "is_positive":        ("BOOLEAN", {"default": True}),
                "override_prompt":    ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Leave empty to use the auto-generated prompt. "
                               "Edit here to override what gets sent to Gemini.",
                }),
            }
        }

    RETURN_TYPES  = ("SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT",
                     "SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT",
                     "STRING", "STRING")
    RETURN_NAMES  = ("box_prompt", "boxes_prompt",
                     "positive_points", "negative_points",
                     "prompt_used", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, image, api, target_description, num_pos_points,
            num_neg_points, is_positive, override_prompt=""):
        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        auto_prompt = bbox_and_points_prompt(target_description, W, H, num_pos_points, num_neg_points)

        final_prompt = override_prompt.strip() if override_prompt.strip() else auto_prompt
        mode = "OVERRIDE" if override_prompt.strip() else "AUTO"

        print(f"[VLMPromptEditor] Image: {W}x{H} | mode: {mode}")
        print(f"[VLMPromptEditor] Prompt sent:\n{final_prompt}")

        raw = _call_gemini(pil_img, final_prompt, api)
        print(f"[VLMPromptEditor] Raw response: {raw}")

        try:
            data   = _parse_json(raw)
            x1, y1, x2, y2 = data["bbox"]
            pos_raw = data.get("positive", [[W//2, H//2]])
            neg_raw = data.get("negative", [])
        except Exception as e:
            raise RuntimeError(f"[VLMPromptEditor] Failed to parse Gemini response: {e}\nRaw: {raw}") from e

        print(f"[VLMPromptEditor] Image size received: {W}x{H}, "
              f"bbox pixel: {x1},{y1},{x2},{y2}, "
              f"normalized: {x1/W:.3f},{y1/H:.3f},{x2/W:.3f},{y2/H:.3f}")

        pos_raw = pos_raw[:num_pos_points]
        neg_raw = neg_raw[:num_neg_points]

        x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
        cx = (x1n+x2n)/2; cy = (y1n+y2n)/2
        bw = x2n-x1n;     bh = y2n-y1n

        box_prompt   = {"box":   [cx, cy, bw, bh], "label": is_positive}
        boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [is_positive]}

        positive_points = normalize_points(pos_raw, 1)
        negative_points = normalize_points(neg_raw, 0)

        print(f"[VLMPromptEditor] box:[{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}] "
              f"pos:{len(positive_points['points'])} neg:{len(negative_points['points'])}")
        return (box_prompt, boxes_prompt, positive_points, negative_points, final_prompt, raw)


# =============================================================================
# VLMBBoxPreview
# =============================================================================

class VLMBBoxPreview:

    COLORS = [(255,80,80),(80,220,80),(80,120,255),(255,200,50),(200,80,255)]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "boxes_prompt": ("SAM3_BOXES_PROMPT",),
            },
            "optional": {
                "line_width": ("INT", {"default": 3, "min": 1, "max": 10}),
                "show_index": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("preview",)
    FUNCTION      = "draw"
    CATEGORY      = "AVM"

    def draw(self, image, boxes_prompt, line_width=3, show_index=True):
        import torch
        from PIL import ImageDraw
        pil_img = _tensor_to_pil(image).copy()
        W, H = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        for i, box in enumerate(boxes_prompt.get("boxes", [])):
            cx, cy, bw, bh = box
            x1 = int((cx-bw/2)*W); y1 = int((cy-bh/2)*H)
            x2 = int((cx+bw/2)*W); y2 = int((cy+bh/2)*H)
            color = self.COLORS[i % len(self.COLORS)]
            draw.rectangle([x1,y1,x2,y2], outline=color, width=line_width)
            if show_index:
                draw.rectangle([x1, max(0,y1-18), x1+28, y1], fill=color)
                draw.text((x1+3, max(0,y1-16)), f"#{i+1}", fill=(255,255,255))
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)


# =============================================================================
# VLMDebugPreview
# =============================================================================

class VLMDebugPreview:

    BBOX_COLORS = [(255,80,80),(80,220,80),(80,120,255),(255,200,50),(200,80,255)]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "box_and_point":   ("SAM3_BOX_AND_POINT",),
                "boxes_prompt":    ("SAM3_BOXES_PROMPT",),
                "positive_points": ("SAM3_POINTS_PROMPT",),
                "negative_points": ("SAM3_POINTS_PROMPT",),
                "line_width":   ("INT",     {"default": 3, "min": 1, "max": 10}),
                "point_radius": ("INT",     {"default": 8, "min": 2, "max": 30}),
                "show_labels":  ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("debug_preview",)
    FUNCTION      = "draw"
    CATEGORY      = "AVM"

    def draw(self, image, box_and_point=None, boxes_prompt=None, positive_points=None,
             negative_points=None, line_width=3, point_radius=8, show_labels=True):
        if box_and_point is not None:
            boxes_prompt    = box_and_point.get("boxes")
            positive_points = box_and_point.get("positive")
            negative_points = box_and_point.get("negative")
        import torch
        from PIL import ImageDraw
        pil_img = _tensor_to_pil(image).copy()
        W, H = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        r = point_radius

        if boxes_prompt:
            for i, box in enumerate(boxes_prompt.get("boxes", [])):
                cx, cy, bw, bh = box
                x1 = int((cx-bw/2)*W); y1 = int((cy-bh/2)*H)
                x2 = int((cx+bw/2)*W); y2 = int((cy+bh/2)*H)
                color = self.BBOX_COLORS[i % len(self.BBOX_COLORS)]
                draw.rectangle([x1,y1,x2,y2], outline=color, width=line_width)
                if show_labels:
                    draw.rectangle([x1, max(0,y1-18), x1+28, y1], fill=color)
                    draw.text((x1+3, max(0,y1-16)), f"#{i+1}", fill=(255,255,255))

        if positive_points:
            for i, pt in enumerate(positive_points.get("points", [])):
                px = int(pt[0]*W); py = int(pt[1]*H)
                draw.ellipse([px-r-2,py-r-2,px+r+2,py+r+2], fill=(255,255,255))
                draw.ellipse([px-r,py-r,px+r,py+r], fill=(50,210,50))
                draw.ellipse([px-2,py-2,px+2,py+2], fill=(255,255,255))
                if show_labels:
                    draw.text((px+r+4, py-6), f"fg{i+1}", fill=(50,210,50))

        if negative_points:
            for i, pt in enumerate(negative_points.get("points", [])):
                px = int(pt[0]*W); py = int(pt[1]*H)
                draw.ellipse([px-r-2,py-r-2,px+r+2,py+r+2], fill=(255,255,255))
                draw.ellipse([px-r,py-r,px+r,py+r], fill=(210,50,50))
                draw.line([px-r//2,py-r//2,px+r//2,py+r//2], fill=(255,255,255), width=2)
                draw.line([px+r//2,py-r//2,px-r//2,py+r//2], fill=(255,255,255), width=2)
                if show_labels:
                    draw.text((px+r+4, py-6), f"bg{i+1}", fill=(210,50,50))

        arr = np.array(pil_img).astype(np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)


# =============================================================================
# VLMMultiFrameBBoxPreview (D-380) — per-keyframe debug overlay
# =============================================================================

class VLMMultiFrameBBoxPreview:
    """Debug overlay node — renders box + positives + negatives per accepted
    keyframe from a VLMtoBBoxAndPointsMultiFrame v1.5 seed_prompts payload.

    Output is an IMAGE batch with one preview frame per accepted seed (sorted
    by frame_idx), so you can wire it into NV Preview Animation or video
    saver and scrub through what Gemini placed at each keyframe.

    Reuses the drawing primitives from VLMDebugPreview but iterates the v1.5
    seeds list and pulls each seed's frame from the full IMAGE batch.

    When seeds carry v1.5 crop_meta (i.e. crop_in_applied=True), the crop_box
    is also rendered (yellow outline) so you can verify which area Gemini was
    looking at during Stage 2 — useful for diagnosing why crop-in helps OR
    hurts on a given clip class.
    """

    FG_COLOR = (50, 210, 50)         # positive points — green
    BG_COLOR = (210, 50, 50)         # negative points — red
    BBOX_COLOR = (80, 120, 255)      # detection bbox — blue
    CROP_COLOR = (255, 220, 50)      # crop_box (stage 2 area) — yellow

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Full image batch [T,H,W,C] — the same batch fed "
                               "to VLMtoBBoxAndPointsMultiFrame. Each accepted "
                               "seed's frame_idx is used to pick the right "
                               "source frame.",
                }),
                "seed_prompts": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Wire from VLMtoBBoxAndPointsMultiFrame.seed_prompts "
                               "(v1.4 or v1.5 JSON payload).",
                }),
            },
            "optional": {
                "line_width": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Bbox outline thickness (px)."
                }),
                "point_radius": ("INT", {
                    "default": 12, "min": 3, "max": 40, "step": 1,
                    "tooltip": "Per-point overlay radius (px). Default larger than "
                               "single-frame VLMDebugPreview for easier visual "
                               "inspection of per-keyframe placement patterns.",
                }),
                "show_labels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Annotate each preview with frame_idx + confidence "
                               "+ crop_in flag at the top-left.",
                }),
                "show_crop_box": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When the seed has crop_meta (crop_in_applied=True), "
                               "render the crop bounds as a yellow outline. Lets you "
                               "verify which area Gemini was looking at during "
                               "Stage 2 point placement.",
                }),
                "show_fg_bg_indices": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Number each point ('fg1', 'fg2', 'bg1', ...). Off "
                               "by default — busy at 8 pos + 4 neg per frame.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_batch", "info")
    FUNCTION = "draw"
    CATEGORY = "AVM"
    DESCRIPTION = (
        "Per-keyframe debug overlay for VLMtoBBoxAndPointsMultiFrame seeds. "
        "Renders one preview frame per accepted seed showing box (blue) + "
        "positive points (green) + negative points (red) + crop_box (yellow, "
        "when crop_in was on). Output is IMAGE batch — wire to NV Preview "
        "Animation to scrub through and visually verify what Gemini placed "
        "at each keyframe."
    )

    def draw(self, images, seed_prompts,
             line_width=3, point_radius=12, show_labels=True,
             show_crop_box=True, show_fg_bg_indices=False):
        import torch
        from PIL import ImageDraw

        # Shape + payload validation
        if not hasattr(images, "shape") or len(images.shape) != 4:
            raise ValueError(
                f"[VLMMultiFrameBBoxPreview] images must be IMAGE tensor "
                f"[T,H,W,C] (4-D), got shape="
                f"{tuple(getattr(images, 'shape', ()))!r}"
            )
        T = int(images.shape[0])
        H = int(images.shape[1])
        W = int(images.shape[2])

        if not seed_prompts or not seed_prompts.strip():
            # Empty input — return an empty single-frame placeholder image
            # so downstream IMAGE consumers don't crash.
            placeholder = torch.zeros((1, H, W, 3), dtype=torch.float32)
            return (placeholder, "[VLMMultiFrameBBoxPreview] empty seed_prompts — placeholder emitted")

        try:
            payload = json.loads(seed_prompts)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[VLMMultiFrameBBoxPreview] seed_prompts is not valid JSON: {e}"
            )

        seeds = payload.get("seeds") if isinstance(payload, dict) else None
        if not isinstance(seeds, list) or not seeds:
            placeholder = torch.zeros((1, H, W, 3), dtype=torch.float32)
            return (placeholder, "[VLMMultiFrameBBoxPreview] no seeds in payload — placeholder emitted")

        # Sort by frame_idx so the output batch is temporally ordered
        seeds_sorted = sorted(
            (s for s in seeds if isinstance(s, dict) and "frame_idx" in s),
            key=lambda s: int(s["frame_idx"])
        )

        previews = []
        info_lines = []
        skipped = 0
        r = int(point_radius)
        lw = int(line_width)

        for seed in seeds_sorted:
            frame_idx = int(seed["frame_idx"])
            if frame_idx < 0 or frame_idx >= T:
                skipped += 1
                info_lines.append(
                    f"  seed frame_idx={frame_idx} out of range [0, {T}); skipped"
                )
                continue

            # Pull the source frame: images[frame_idx] is [H, W, C].
            # _tensor_to_pil indexes [0] internally, so wrap in batch dim.
            single = images[frame_idx:frame_idx + 1]
            pil = _tensor_to_pil(single).copy()
            draw = ImageDraw.Draw(pil)

            # ----- Bbox (blue) — from seed["box"] (cxcywh normalized) -----
            box = seed.get("box")
            if isinstance(box, list) and len(box) == 4:
                cx, cy, bw, bh = box
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)
                draw.rectangle([x1, y1, x2, y2], outline=self.BBOX_COLOR, width=lw)

            # ----- crop_box (yellow) — v1.5 crop_meta if crop_in was on -----
            if show_crop_box:
                cm = seed.get("crop_meta")
                if isinstance(cm, dict):
                    cx1, cy1 = int(cm.get("x1", 0)), int(cm.get("y1", 0))
                    cx2, cy2 = int(cm.get("x2", 0)), int(cm.get("y2", 0))
                    if cx2 > cx1 and cy2 > cy1:
                        # Dashed-look approximation — outline is solid; draw
                        # corner ticks inside for visual distinction from bbox.
                        draw.rectangle([cx1, cy1, cx2, cy2],
                                       outline=self.CROP_COLOR, width=max(1, lw - 1))

            # ----- Positive points (green) -----
            for i, pt in enumerate(seed.get("pos_pts", []) or []):
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    continue
                px = int(pt[0] * W)
                py = int(pt[1] * H)
                draw.ellipse([px - r - 2, py - r - 2, px + r + 2, py + r + 2], fill=(255, 255, 255))
                draw.ellipse([px - r, py - r, px + r, py + r], fill=self.FG_COLOR)
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 255))
                if show_fg_bg_indices:
                    draw.text((px + r + 4, py - 6), f"fg{i + 1}", fill=self.FG_COLOR)

            # ----- Negative points (red) -----
            for i, pt in enumerate(seed.get("neg_pts", []) or []):
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    continue
                px = int(pt[0] * W)
                py = int(pt[1] * H)
                draw.ellipse([px - r - 2, py - r - 2, px + r + 2, py + r + 2], fill=(255, 255, 255))
                draw.ellipse([px - r, py - r, px + r, py + r], fill=self.BG_COLOR)
                # X marker
                draw.line([px - r // 2, py - r // 2, px + r // 2, py + r // 2],
                          fill=(255, 255, 255), width=2)
                draw.line([px + r // 2, py - r // 2, px - r // 2, py + r // 2],
                          fill=(255, 255, 255), width=2)
                if show_fg_bg_indices:
                    draw.text((px + r + 4, py - 6), f"bg{i + 1}", fill=self.BG_COLOR)

            # ----- Top-left annotation header -----
            if show_labels:
                conf = seed.get("confidence")
                crop_on = bool(seed.get("crop_in_applied"))
                conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
                header = (
                    f"frame={frame_idx}  conf={conf_str}  "
                    f"crop_in={'Y' if crop_on else 'N'}  "
                    f"obj={seed.get('obj_id', '?')}"
                )
                # Background bar for readability
                draw.rectangle([0, 0, min(W, 480), 24], fill=(0, 0, 0))
                draw.text((6, 6), header, fill=(255, 255, 255))

            arr = np.array(pil).astype(np.float32) / 255.0
            previews.append(torch.from_numpy(arr))

        if not previews:
            placeholder = torch.zeros((1, H, W, 3), dtype=torch.float32)
            return (placeholder, f"[VLMMultiFrameBBoxPreview] no seeds rendered "
                                 f"({skipped} skipped out of range)")

        preview_batch = torch.stack(previews, dim=0)
        info_str = (
            f"[VLMMultiFrameBBoxPreview] rendered {preview_batch.shape[0]} previews "
            f"({skipped} skipped). Frame range: "
            f"{int(seeds_sorted[0]['frame_idx'])}-{int(seeds_sorted[-1]['frame_idx'])}. "
            f"Source batch: T={T}, frame={W}x{H}."
        )
        if info_lines:
            info_str += "\n" + "\n".join(info_lines)

        return (preview_batch, info_str)


# =============================================================================
# AVMCropByBox
# =============================================================================

class AVMCropByBox:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "boxes_prompt": ("SAM3_BOX_AND_POINT",),
            },
            "optional": {
                "label":          ("STRING",  {"default": "", "multiline": False,
                                   "tooltip": "Label to display inside the node."}),
                "padding":        ("INT",     {"default": 16,   "min": 0,   "max": 128}),
                "box_index":      ("INT",     {"default": 0,    "min": 0,   "max": 4}),
                "normalize_size": ("BOOLEAN", {"default": True}),
                "target_long_side": ("INT",   {"default": 1008, "min": 256, "max": 4096}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "CROP_META")
    RETURN_NAMES  = ("cropped_image", "crop_meta")
    OUTPUT_NODE   = True
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, image, boxes_prompt, label="", padding=16, box_index=0,
            normalize_size=True, target_long_side=1008):
        import torch
        import torch.nn.functional as F
        import folder_paths, uuid, os
        B, H, W, C = image.shape
        boxes = boxes_prompt["boxes"].get("boxes", [])

        if not boxes or box_index >= len(boxes):
            meta = {"x1": 0, "y1": 0, "x2": W, "y2": H,
                    "orig_w": W, "orig_h": H, "scale": 1.0,
                    "resized_w": W, "resized_h": H}
            return {"ui": {"images": [], "text": [label]}, "result": (image, meta)}

        cx, cy, bw, bh = boxes[box_index]
        x1 = max(0, int((cx - bw / 2) * W) - padding)
        y1 = max(0, int((cy - bh / 2) * H) - padding)
        x2 = min(W, int((cx + bw / 2) * W) + padding)
        y2 = min(H, int((cy + bh / 2) * H) + padding)

        cropped = image[:, y1:y2, x1:x2, :]
        crop_h, crop_w = y2 - y1, x2 - x1

        scale = 1.0
        out_h, out_w = crop_h, crop_w

        if normalize_size:
            scale = target_long_side / max(crop_h, crop_w)
            if scale != 1.0:
                out_h = round(crop_h * scale)
                out_w = round(crop_w * scale)
                cropped = F.interpolate(
                    cropped.permute(0, 3, 1, 2).float(),
                    size=(out_h, out_w),
                    mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)

        meta = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "orig_w": W, "orig_h": H,
            "scale": scale, "resized_w": out_w, "resized_h": out_h,
        }
        print(f"[AVMCropByBox] crop=[{x1},{y1},{x2},{y2}] {crop_w}x{crop_h}"
              f" → {out_w}x{out_h} (scale={scale:.3f})")

        # Save preview to temp with label bar
        fname = f"avm_crop_{uuid.uuid4().hex[:8]}.png"
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        fpath = os.path.join(temp_dir, fname)
        preview = _tensor_to_pil(cropped).copy()
        if label:
            from PIL import ImageDraw, ImageFont
            d   = ImageDraw.Draw(preview)
            pw, ph = preview.size
            font_size = max(20, ph // 18)
            try:
                font = ImageFont.load_default(size=font_size)
            except TypeError:
                font = ImageFont.load_default()
            bar_h = font_size + 14
            d.rectangle([0, 0, pw, bar_h], fill=(0, 0, 0))
            d.text((8, 7), label, fill=(255, 255, 255), font=font)
        else:
            preview = _tensor_to_pil(cropped)
        preview.save(fpath)

        return {
            "ui": {
                "images": [{"filename": fname, "subfolder": "", "type": "temp"}],
                "text":   [label],
            },
            "result": (cropped, meta),
        }


# =============================================================================
# AVMPasteBackMask
# =============================================================================

class AVMPasteBackMask:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks":     ("MASK",),
                "crop_meta": ("CROP_META",),
            },
            "optional": {
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 32}),
            }
        }

    RETURN_TYPES  = ("MASK",)
    RETURN_NAMES  = ("full_masks",)
    FUNCTION      = "run"
    CATEGORY      = "AVM/Face"

    def run(self, masks, crop_meta, feather_px=0):
        import torch
        import torch.nn.functional as F

        x1, y1, x2, y2 = crop_meta["x1"], crop_meta["y1"], crop_meta["x2"], crop_meta["y2"]
        orig_w, orig_h  = crop_meta["orig_w"], crop_meta["orig_h"]
        # Target crop size in original image space (before any resize)
        exp_h, exp_w = y2 - y1, x2 - x1

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Resize mask back to original crop dimensions (undoes normalize_size scale)
        N, mask_h, mask_w = masks.shape
        if mask_h != exp_h or mask_w != exp_w:
            masks = F.interpolate(masks.unsqueeze(1).float(), size=(exp_h, exp_w),
                                  mode="bilinear", align_corners=False).squeeze(1)

        full = torch.zeros((N, orig_h, orig_w), dtype=masks.dtype, device=masks.device)
        full[:, y1:y2, x1:x2] = masks

        if feather_px > 0:
            k = feather_px * 2 + 1
            full = F.avg_pool2d(full.unsqueeze(1).float(), kernel_size=k, stride=1,
                                padding=feather_px).squeeze(1)
            full = torch.clamp(full, 0.0, 1.0)

        print(f"[AVMPasteBackMask] {N} masks -> {orig_w}x{orig_h}")
        return (full,)


# =============================================================================
# AVMAddFramePrompt
# =============================================================================

class AVMAddFramePrompt:

    PROMPT_MODES = ["point", "box"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_state": ("SAM3_VIDEO_STATE",),
                "prompt_mode": (cls.PROMPT_MODES, {"default": "point"}),
                "frame_idx":   ("INT", {"default": 15, "min": 0,
                                "tooltip": "Frame to anchor. 30-frame clip: 14=mid, 29=end."}),
                "obj_id":      ("INT", {"default": 1, "min": 1}),
            },
            "optional": {
                "positive_points": ("SAM3_POINTS_PROMPT",),
                "negative_points": ("SAM3_POINTS_PROMPT",),
                "positive_boxes":  ("SAM3_BOXES_PROMPT",),
                "negative_boxes":  ("SAM3_BOXES_PROMPT",),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "add_frame_prompt"
    CATEGORY      = "AVM"

    def add_frame_prompt(self, video_state, prompt_mode, frame_idx, obj_id,
                         positive_points=None, negative_points=None,
                         positive_boxes=None, negative_boxes=None):

        import importlib.util, os as _os
        _base = _os.path.join(_find_sam3_nodes_dir(), "video_state.py")
        _spec = importlib.util.spec_from_file_location("sam3_video_state", _base)
        _mod  = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        VideoPrompt = _mod.VideoPrompt

        if prompt_mode == "point":
            all_points, all_labels = [], []
            if positive_points and positive_points.get("points"):
                for pt in positive_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])]); all_labels.append(1)
            if negative_points and negative_points.get("points"):
                for pt in negative_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])]); all_labels.append(0)
            if not all_points:
                return (video_state,)
            video_state = video_state.with_prompt(VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels))
            print(f"[AVMAddFramePrompt] {len(all_points)} points at frame {frame_idx}")

        elif prompt_mode == "box":
            if positive_boxes and positive_boxes.get("boxes"):
                cx, cy, w, h = positive_boxes["boxes"][0]
                video_state = video_state.with_prompt(
                    VideoPrompt.create_box(frame_idx, obj_id, [cx-w/2, cy-h/2, cx+w/2, cy+h/2], is_positive=True))
            if negative_boxes and negative_boxes.get("boxes"):
                cx, cy, w, h = negative_boxes["boxes"][0]
                video_state = video_state.with_prompt(
                    VideoPrompt.create_box(frame_idx, obj_id, [cx-w/2, cy-h/2, cx+w/2, cy+h/2], is_positive=False))

        return (video_state,)


# =============================================================================
# VLMFacePartsBBox
# =============================================================================

FACE_PART_PROMPTS = {
    "hair":      "The person's hair only — scalp to hairline tips. Exclude forehead.",
    "face":      "The person's face skin only — forehead, cheeks, nose, lips, chin. Exclude hair, neck.",
    "neck":      "The person's neck only — below chin to collar. Exclude face and clothing.",
    "face_neck": "The person's face AND neck combined — forehead to collar. Exclude hair.",
    "clothing":  "The person's clothing — shirt, dress, jacket etc. Exclude skin, hair.",
}

class VLMFacePartsBBox:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),
                "api":        ("AVM_API",),
                "person_box": ("SAM3_BOXES_PROMPT",),
            },
            "optional": {
                "score_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "padding_px":      ("INT",   {"default": 8,   "min": 0,   "max": 40}),
            }
        }

    RETURN_TYPES  = ("SAM3_BOXES_PROMPT","SAM3_BOXES_PROMPT","SAM3_BOXES_PROMPT","SAM3_BOXES_PROMPT","SAM3_BOXES_PROMPT","STRING")
    RETURN_NAMES  = ("hair", "face", "neck", "face_neck", "clothing", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "AVM/Face"

    def run(self, image, api, person_box, score_threshold=0.5, padding_px=8):
        pil_full = _tensor_to_pil(image)
        W, H = pil_full.size

        pil_img, crop_x1, crop_y1 = pil_full, 0, 0
        crop_w, crop_h = W, H
        if person_box and person_box.get("boxes"):
            cx_n, cy_n, bw_n, bh_n = person_box["boxes"][0]
            cx1 = max(0, int((cx_n-bw_n/2)*W) - 20); cy1 = max(0, int((cy_n-bh_n/2)*H) - 20)
            cx2 = min(W, int((cx_n+bw_n/2)*W) + 20); cy2 = min(H, int((cy_n+bh_n/2)*H) + 20)
            pil_img = pil_full.crop((cx1, cy1, cx2, cy2))
            crop_x1, crop_y1 = cx1, cy1
            crop_w, crop_h = cx2-cx1, cy2-cy1

        cW, cH = pil_img.size
        parts_desc = "\n".join(f'  "{k}": {v}' for k, v in FACE_PART_PROMPTS.items())
        prompt = face_parts_bbox_prompt(cW, cH, parts_desc)

        raw = _call_gemini(pil_img, prompt, api)
        print(f"[VLMFacePartsBBox] Raw: {raw}")

        try:
            data = _parse_json(raw)
        except Exception as e:
            print(f"[AVM ERROR] VLMFacePartsBBox failed to parse response: {e}\nRaw: {raw}")
            data = {}

        empty = {"boxes": [], "labels": []}

        def _to_box(key):
            entry = data.get(key, {})
            if not entry or not entry.get("bbox"):
                return empty
            if float(entry.get("confidence", 1.0)) < score_threshold:
                return empty
            x1, y1, x2, y2 = entry["bbox"]
            # Normalize Gemini's coord scale (typically 0-1000) to [0,1] relative to crop
            x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, cW, cH)
            # Sort corners defensively (Gemini occasionally swaps)
            if x1n > x2n: x1n, x2n = x2n, x1n
            if y1n > y2n: y1n, y2n = y2n, y1n
            # Convert to crop-space pixels so padding_px makes sense
            x1p = x1n * cW; y1p = y1n * cH
            x2p = x2n * cW; y2p = y2n * cH
            # Apply padding in pixel space
            x1p = max(0, x1p - padding_px); y1p = max(0, y1p - padding_px)
            x2p = min(cW, x2p + padding_px); y2p = min(cH, y2p + padding_px)
            # Guard against degenerate box after clamp
            if x2p <= x1p or y2p <= y1p:
                return empty
            # Map to full-frame normalized [0,1]
            ax1 = (x1p + crop_x1) / W; ay1 = (y1p + crop_y1) / H
            ax2 = (x2p + crop_x1) / W; ay2 = (y2p + crop_y1) / H
            cx = (ax1 + ax2) / 2; cy = (ay1 + ay2) / 2
            return {"boxes": [[cx, cy, ax2-ax1, ay2-ay1]], "labels": [True]}

        return (_to_box("hair"), _to_box("face"), _to_box("neck"), _to_box("face_neck"), _to_box("clothing"), raw)


# =============================================================================
# VLMFacePrecisePoints
# =============================================================================

# Per-target: what foreground covers, which zones to sample, what background is
_FACE_TARGET_CONFIG = {
    "face_skin": {
        "fg_desc":   "face skin only — forehead, both cheeks, nose bridge, nose tip, lips, chin. Strictly exclude hair, eyebrows, eyelashes, glasses, and neck.",
        "fg_zones":  "forehead center, left cheek mid, right cheek mid, nose tip, nose bridge, chin center, left cheek near jaw, right cheek near jaw",
        "bg_desc":   "hair (above hairline), neck and chin underside, background behind head, shoulders",
        "bg_zones":  "hair above left temple, hair above right temple, neck below chin center, background left of face, background right of face",
    },
    "face_with_hair": {
        "fg_desc":   "full face AND complete head of hair — from hair crown to chin",
        "fg_zones":  "forehead center, left cheek, right cheek, crown of hair, left side of hair, right side of hair, chin center",
        "bg_desc":   "neck/collar area, background behind head, shoulders, body",
        "bg_zones":  "neck below chin, background top-left, background top-right, left shoulder, right shoulder",
    },
    "face_with_neck": {
        "fg_desc":   "face skin AND neck — forehead down to collar. Exclude hair.",
        "fg_zones":  "forehead center, left cheek, right cheek, nose tip, chin center, left neck side, right neck side, lower neck center",
        "bg_desc":   "hair above forehead, background, clothing collar, shoulders",
        "bg_zones":  "hair top-left, hair top-right, background left, collar left, collar right",
    },
    "full_head": {
        "fg_desc":   "entire head — face, hair, ears, top of neck. Exclude background and clothing.",
        "fg_zones":  "forehead, left cheek, right cheek, crown of hair, left ear area, right ear area, upper neck",
        "bg_desc":   "background behind head, clothing, body, anything below collar",
        "bg_zones":  "background top-center, background left, background right, clothing collar, lower body",
    },
}

class VLMFacePrecisePoints:
    """
    Generates SAM3-ready bbox + points specifically for precise face masking.
    Uses face-anatomy-aware prompting to spread foreground points across all
    major face zones and place background points at exact face boundaries.

    Workflow:
        VLMtoBBox -> VLMFacePartsBBox -> VLMFacePrecisePoints -> SAM3
        or
        VLMtoBBox -> VLMFacePrecisePoints (no face_bbox, crops to person)
    """

    FACE_TARGETS = ["face_skin", "face_with_hair", "face_with_neck", "full_head"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":          ("IMAGE",),
                "api":            ("AVM_API",),
                "face_target":    (cls.FACE_TARGETS, {"default": "face_skin"}),
                "num_fg_points":  ("INT", {"default": 8, "min": 4, "max": 16}),
                "num_bg_points":  ("INT", {"default": 4, "min": 0, "max": 8}),
            },
            "optional": {
                "face_bbox":      ("SAM3_BOXES_PROMPT",),  # from VLMFacePartsBBox face output
                "include_beard":  ("BOOLEAN", {"default": True}),
                "include_ears":   ("BOOLEAN", {"default": False}),
                "crop_padding":   ("INT", {"default": 20, "min": 0, "max": 80}),
            }
        }

    RETURN_TYPES  = ("SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "STRING")
    RETURN_NAMES  = ("box_prompt", "boxes_prompt", "positive_points", "negative_points", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "AVM/Face"

    def run(self, image, api, face_target, num_fg_points, num_bg_points,
            face_bbox=None, include_beard=True, include_ears=False, crop_padding=20):
        pil_full = _tensor_to_pil(image)
        W, H = pil_full.size

        # Crop to face bbox if provided
        crop_x1, crop_y1 = 0, 0
        pil_img = pil_full
        if face_bbox and face_bbox.get("boxes"):
            cx_n, cy_n, bw_n, bh_n = face_bbox["boxes"][0]
            cx1 = max(0, int((cx_n - bw_n/2) * W) - crop_padding)
            cy1 = max(0, int((cy_n - bh_n/2) * H) - crop_padding)
            cx2 = min(W, int((cx_n + bw_n/2) * W) + crop_padding)
            cy2 = min(H, int((cy_n + bh_n/2) * H) + crop_padding)
            pil_img = pil_full.crop((cx1, cy1, cx2, cy2))
            crop_x1, crop_y1 = cx1, cy1

        cW, cH = pil_img.size
        cfg = _FACE_TARGET_CONFIG[face_target]

        # Build modifier notes
        modifiers = []
        if face_target == "face_skin":
            if include_beard:
                modifiers.append("If beard/stubble is present, include it as foreground.")
            else:
                modifiers.append("Exclude any beard or stubble — treat as background.")
            if include_ears:
                modifiers.append("Include ears as foreground.")
            else:
                modifiers.append("Exclude ears — treat as background.")
        modifier_block = ("\n" + "\n".join(modifiers)) if modifiers else ""

        prompt = face_precise_points_prompt(cW, cH, cfg, num_fg_points, num_bg_points, modifier_block)

        raw = _call_gemini(pil_img, prompt, api)
        print(f"[VLMFacePrecisePoints] target={face_target} crop={cW}x{cH} | raw: {raw}")

        # Parse response
        try:
            data = _parse_json(raw)
            x1, y1, x2, y2 = data["bbox"]
            fg_raw = data.get("foreground", [[cW//2, cH//2]])
            bg_raw = data.get("background", [])
        except Exception as e:
            raise RuntimeError(f"[VLMFacePrecisePoints] Failed to parse Gemini response: {e}\nRaw: {raw}") from e

        fg_raw = fg_raw[:num_fg_points]
        bg_raw = bg_raw[:num_bg_points]

        # Normalize bbox back to full image coords
        x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, cW, cH)
        # Offset crop origin
        ax1 = (x1n * cW + crop_x1) / W
        ay1 = (y1n * cH + crop_y1) / H
        ax2 = (x2n * cW + crop_x1) / W
        ay2 = (y2n * cH + crop_y1) / H
        cx = (ax1 + ax2) / 2;  cy = (ay1 + ay2) / 2
        bw = ax2 - ax1;        bh = ay2 - ay1

        box_prompt   = {"box":   [cx, cy, bw, bh], "label": True}
        boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [True]}

        positive_points = normalize_points_crop_to_full(fg_raw, 1, cW, cH, crop_x1, crop_y1, W, H)
        negative_points = normalize_points_crop_to_full(bg_raw, 0, cW, cH, crop_x1, crop_y1, W, H)

        print(f"[VLMFacePrecisePoints] box:[{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}] "
              f"fg:{len(positive_points['points'])} bg:{len(negative_points['points'])}")
        return (box_prompt, boxes_prompt, positive_points, negative_points, raw)


# =============================================================================
# VLMFaceRegion
# =============================================================================

class VLMFaceRegion:
    """
    One-stop node for precise face region masking.

    Stage 1 (full image) — VLM detects a tight bbox for your free-text region.
    Stage 2 (crop)       — VLM places anatomy-aware points on the crop at higher
                           effective resolution.

    Solves common face masking problems:
      • Open mouth / teeth cutoff → prompt explicitly extends bbox to include
        mouth interior, teeth, tongue when visible.
      • Neck truncation           → prompt extends bbox to collar/shoulder junction.
      • Low point density on face → SAM3 works on the crop, not the full image.

    Typical workflow:
        Image → VLMFaceRegion → [cropped_image + crop_meta + points] → SAM3
                                       └─ AVMPasteBackMask → full-size mask

    region examples:
        "face including open mouth and teeth"
        "face and full neck down to collarbone"
        "neck and upper clothing, exclude face"
        "face skin only, exclude hair and neck"
    """

    _FACE_RULES = (
        "CRITICAL boundary rules:\n"
        "  • FACE: bbox must include full chin, jaw underside, and any open mouth /\n"
        "    teeth / tongue interior. Never cut at the lips.\n"
        "  • NECK: bbox must extend fully to where neck meets collar or shoulders.\n"
        "    Never cut at the chin.\n"
        "  • HAIR: bbox extends to hair tips, not just the scalp.\n"
        "  • Foreground points must be deep inside the region — never on its border.\n"
        "  • Background points must be just outside the region boundary.\n"
        "  • Spread points across the WHOLE region, do NOT cluster them.\n"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":         ("IMAGE",),
                "api":           ("AVM_API",),
                "region":        ("STRING", {
                    "default":   "face including open mouth and teeth",
                    "multiline": True,
                }),
                "num_fg_points": ("INT", {"default": 8, "min": 4, "max": 16}),
                "num_bg_points": ("INT", {"default": 4, "min": 0, "max": 8}),
                "crop_padding":  ("INT", {"default": 24, "min": 0, "max": 80}),
                "output_space":  (["crop", "full_frame"], {
                    "default": "crop",
                    "tooltip": (
                        "crop (default): bbox+points emitted in crop-space — for "
                        "static-image workflows where SAM3 runs on cropped_image "
                        "and AVMPasteBackMask stitches the mask back to the full "
                        "frame. full_frame: bbox+points emitted in full-image "
                        "coords — correct for SAM3-Video tracking a moving subject "
                        "across frames (wire SAM3 to the full video, not to "
                        "cropped_image)."
                    ),
                }),
            },
            "optional": {
                "person_bbox":   ("SAM3_BOXES_PROMPT",),
            }
        }

    RETURN_TYPES  = ("IMAGE", "CROP_META",
                     "SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT",
                     "SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "STRING")
    RETURN_NAMES  = ("cropped_image", "crop_meta",
                     "box_prompt", "boxes_prompt",
                     "positive_points", "negative_points", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "AVM/Face"

    def run(self, image, api, region, num_fg_points, num_bg_points,
            crop_padding=24, output_space="crop", person_bbox=None):
        import torch
        pil_full = _tensor_to_pil(image)
        W, H = pil_full.size

        # Optionally restrict search to person bbox
        search_img = pil_full
        search_x1, search_y1 = 0, 0
        if person_bbox and person_bbox.get("boxes"):
            cx_n, cy_n, bw_n, bh_n = person_bbox["boxes"][0]
            px1 = max(0, int((cx_n - bw_n/2) * W) - 10)
            py1 = max(0, int((cy_n - bh_n/2) * H) - 10)
            px2 = min(W, int((cx_n + bw_n/2) * W) + 10)
            py2 = min(H, int((cy_n + bh_n/2) * H) + 10)
            search_img = pil_full.crop((px1, py1, px2, py2))
            search_x1, search_y1 = px1, py1

        sW, sH = search_img.size

        # ── Stage 1: detect region bbox ───────────────────────────────
        prompt1 = face_region_stage1_prompt(sW, sH, region, self._FACE_RULES)
        raw1 = _call_gemini(search_img, prompt1, api)
        print(f"[VLMFaceRegion] Stage1: {raw1}")

        try:
            bx1, by1, bx2, by2 = _parse_json(raw1)["bbox"]
        except Exception as e:
            raise RuntimeError(f"[VLMFaceRegion] Stage1 failed to parse Gemini response: {e}\nRaw: {raw1}") from e

        # Normalize Gemini coord scale to [0,1] (prompt requests 0-1000 scale;
        # values may overshoot slightly, that's fine — clamp happens on crop).
        if any(v > 2.0 for v in [bx1, by1, bx2, by2]):
            bx1, by1 = bx1 / 1000.0, by1 / 1000.0
            bx2, by2 = bx2 / 1000.0, by2 / 1000.0
        # Map to full-image pixel space via search window
        bx1 = bx1 * sW + search_x1
        by1 = by1 * sH + search_y1
        bx2 = bx2 * sW + search_x1
        by2 = by2 * sH + search_y1
        # Gemini occasionally swaps corners; ensure x1<x2, y1<y2
        if bx1 > bx2:
            bx1, bx2 = bx2, bx1
        if by1 > by2:
            by1, by2 = by2, by1

        cx1 = max(0, int(bx1) - crop_padding)
        cy1 = max(0, int(by1) - crop_padding)
        cx2 = min(W, int(bx2) + crop_padding)
        cy2 = min(H, int(by2) + crop_padding)
        if cx2 <= cx1 or cy2 <= cy1:
            raise RuntimeError(
                f"[VLMFaceRegion] Invalid crop after Stage1 mapping: "
                f"[{cx1},{cy1},{cx2},{cy2}] (search={sW}x{sH}, full={W}x{H}). Raw: {raw1}"
            )
        crop_meta = {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2, "orig_w": W, "orig_h": H}

        cropped_tensor = image[:, cy1:cy2, cx1:cx2, :]
        pil_crop = pil_full.crop((cx1, cy1, cx2, cy2))
        cW, cH = pil_crop.size

        # ── Stage 2: precise points on the crop ───────────────────────
        prompt2 = face_region_stage2_prompt(cW, cH, region, self._FACE_RULES, num_fg_points, num_bg_points)
        raw2 = _call_gemini(pil_crop, prompt2, api)
        print(f"[VLMFaceRegion] Stage2: {raw2}")

        try:
            d2 = _parse_json(raw2)
            fg_raw = d2.get("foreground", [[cW//2, cH//2]])
            bg_raw = d2.get("background", [])
        except Exception as e:
            raise RuntimeError(f"[VLMFaceRegion] Stage2 failed to parse Gemini response: {e}\nRaw: {raw2}") from e

        fg_raw = fg_raw[:num_fg_points]
        bg_raw = bg_raw[:num_bg_points]

        if output_space == "full_frame":
            # Full-frame bbox = tight Stage-1 target box (NOT the padded crop).
            # SAM3 treats the box as a strong prior — feeding it padded-crop
            # bounds would bleed the segmentation into the padding zone.
            tx1 = max(0.0, min(float(W), bx1))
            ty1 = max(0.0, min(float(H), by1))
            tx2 = max(0.0, min(float(W), bx2))
            ty2 = max(0.0, min(float(H), by2))
            bcx = (tx1 + tx2) / 2.0 / W
            bcy = (ty1 + ty2) / 2.0 / H
            bbw = (tx2 - tx1) / W
            bbh = (ty2 - ty1) / H
            box_prompt   = {"box":   [bcx, bcy, bbw, bbh], "label": True}
            boxes_prompt = {"boxes": [[bcx, bcy, bbw, bbh]], "labels": [True]}
            # Points from Stage 2 are in crop-space — project to full-frame [0,1]
            positive_points = normalize_points_crop_to_full(fg_raw, 1, cW, cH, cx1, cy1, W, H)
            negative_points = normalize_points_crop_to_full(bg_raw, 0, cW, cH, cx1, cy1, W, H)
        else:
            # Legacy crop-space output: box covers the whole crop, points in crop coords.
            # Use when SAM3 runs on cropped_image and AVMPasteBackMask stitches back.
            box_prompt   = {"box":   [0.5, 0.5, 1.0, 1.0], "label": True}
            boxes_prompt = {"boxes": [[0.5, 0.5, 1.0, 1.0]], "labels": [True]}
            positive_points = normalize_points_auto(fg_raw, 1)
            negative_points = normalize_points_auto(bg_raw, 0)

        print(f"[VLMFaceRegion] output_space={output_space} crop=[{cx1},{cy1},{cx2},{cy2}] "
              f"fg:{len(positive_points['points'])} bg:{len(negative_points['points'])}")

        raw_combined = f"=== Stage1 (bbox) ===\n{raw1}\n=== Stage2 (points) ===\n{raw2}"
        return (cropped_tensor, crop_meta, box_prompt, boxes_prompt,
                positive_points, negative_points, raw_combined)


# =============================================================================
# AVMAutoLayer / AVMMultiFrameAutoLayer shared helpers
# =============================================================================

def _build_guidance_line(layer_preset, custom_prompt=""):
    if layer_preset == "auto":
        return ""
    if layer_preset == "custom":
        guidance = custom_prompt.strip() or "any distinct visual elements"
        return f"Focus on: {guidance}"
    preset_guidance = {
        "portrait":  "face skin, hair, eyes, mouth/lips, neck, clothing, accessories, background",
        "full_body": "face/head, hair, upper clothing, lower clothing, shoes, hands/arms, accessories, background",
        "product":   "main product, packaging, labels/text, shadow, props, background",
    }
    return f"This is a {layer_preset} image. Focus on: {preset_guidance.get(layer_preset, 'distinct visual regions')}"


def _run_discovery_and_localize(pil_img, api, layer_preset, guidance_line, W, H,
                                num_pos_points, num_neg_points, log_prefix="AVM"):
    """Run the two-stage Gemini pipeline (discovery → localize+points).
    Returns (layers, raw1, raw2); callers format their own raw string.
    """
    raw1 = _call_gemini(pil_img, layer_discovery_prompt(guidance_line), api)
    print(f"[{log_prefix}] Discovery: {raw1}")

    try:
        data1 = _parse_json(raw1)
        discovered = [l.strip() for l in data1.get("layers", []) if isinstance(l, str) and l.strip()]
        if not discovered:
            raise ValueError("empty layers list")
    except Exception as e:
        print(f"[{log_prefix}] Discovery parse error: {e} — falling back to preset")
        discovered = LAYER_PRESETS.get(layer_preset, LAYER_PRESETS["portrait"])

    labels_json = json.dumps(discovered, indent=2)
    raw2 = _call_gemini(pil_img, layer_localize_prompt(W, H, num_pos_points, num_neg_points, labels_json), api)
    print(f"[{log_prefix}] Localize+Points: {raw2}")

    try:
        data = _parse_json(raw2)
        layers = data.get("layers", [])[:8]
    except Exception as e:
        print(f"[{log_prefix}] Localize parse error: {e}")
        layers = []

    return layers, raw1, raw2


def _build_layer_bundle(entry, W, H, num_pos_points, num_neg_points):
    x1, y1, x2, y2 = entry["bbox"]
    x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
    cx = (x1n + x2n) / 2; cy = (y1n + y2n) / 2
    boxes = {"boxes": [[cx, cy, x2n - x1n, y2n - y1n]], "labels": [True]}
    pos   = normalize_points(entry.get("positive", [])[:num_pos_points], 1)
    neg   = normalize_points(entry.get("negative", [])[:num_neg_points], 0)
    return {"boxes": boxes, "positive": pos, "negative": neg}


# =============================================================================
# AVMAutoLayer — one Gemini call → up to 8 layer boxes + layer set
# =============================================================================

LAYER_PRESETS = {
    "portrait": [
        "face (skin region: forehead, cheeks, nose, chin — exclude hair, neck, ears)",
        "hair (scalp hair only — top and sides of head, eyebrows if thick)",
        "eyes and eye area (both eye sockets including eyelids, lashes, brows)",
        "mouth and lips (upper lip, lower lip, chin dimple if present)",
        "neck and upper chest",
        "clothing and garments (shirt, jacket, top — anything worn on the body)",
        "accessories (glasses, earrings, hat, necklace, jewelry)",
        "background (everything not part of the person)",
    ],
    "full_body": [
        "face and head skin",
        "hair",
        "upper body clothing (shirt, jacket, top)",
        "lower body clothing (pants, skirt, shorts)",
        "shoes and footwear",
        "hands and arms skin",
        "accessories (bag, jewelry, glasses, hat)",
        "background",
    ],
    "product": [
        "main product item",
        "product packaging or container",
        "product labels and printed text",
        "product shadow",
        "supporting props or context objects",
        "background",
    ],
}


class AVMAutoLayer:

    LAYER_PRESETS_LIST = ["auto", "portrait", "full_body", "product", "custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "api":          ("AVM_API",),
                "layer_preset": (cls.LAYER_PRESETS_LIST, {"default": "portrait"}),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "One layer description per line. Used when layer_preset='custom'.",
                }),
                "num_pos_points": ("INT", {"default": 4, "min": 1, "max": 12,
                    "tooltip": "Positive points per layer (Call 3)."}),
                "num_neg_points": ("INT", {"default": 2, "min": 0, "max": 6,
                    "tooltip": "Negative points per layer (Call 3)."}),
            }
        }

    RETURN_TYPES = (
        "SAM3_BOX_AND_POINT", "SAM3_BOX_AND_POINT", "SAM3_BOX_AND_POINT", "SAM3_BOX_AND_POINT",
        "SAM3_BOX_AND_POINT", "SAM3_BOX_AND_POINT", "SAM3_BOX_AND_POINT", "SAM3_BOX_AND_POINT",
        "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING",
        "AVM_LAYER_SET", "STRING", "STRING",
    )
    RETURN_NAMES = (
        "layer_1", "layer_2", "layer_3", "layer_4",
        "layer_5", "layer_6", "layer_7", "layer_8",
        "label_1", "label_2", "label_3", "label_4",
        "label_5", "label_6", "label_7", "label_8",
        "layer_set", "label_list", "raw_response",
    )
    FUNCTION = "run"
    CATEGORY      = "AVM"

    def run(self, image, api, layer_preset, custom_prompt="",
            num_pos_points=4, num_neg_points=2):
        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        guidance_line = _build_guidance_line(layer_preset, custom_prompt)
        layers, raw1, raw2 = _run_discovery_and_localize(
            pil_img, api, layer_preset, guidance_line, W, H,
            num_pos_points, num_neg_points, log_prefix="AVMAutoLayer",
        )
        raw = f"=== Discovery ===\n{raw1}\n\n=== Localize+Points ===\n{raw2}"

        empty_boxes  = {"boxes": [], "labels": []}
        empty_pts    = {"points": [], "labels": []}
        empty_bundle = {"boxes": empty_boxes, "positive": empty_pts, "negative": empty_pts}

        bundles = [_build_layer_bundle(l, W, H, num_pos_points, num_neg_points) for l in layers]
        labels  = [l.get("label", f"layer_{i+1}") for i, l in enumerate(layers)]

        while len(bundles) < 8:
            bundles.append(empty_bundle)
            labels.append("")

        layer_set  = {lbl: b["boxes"] for lbl, b in zip(labels, bundles) if lbl}
        label_list = "\n".join(f"{i+1}. {lb}" for i, lb in enumerate(labels) if lb)

        print(f"[AVMAutoLayer] Detected {len(layers)} layers: {[l for l in labels if l]}")
        return (*bundles, *labels, layer_set, label_list, raw)


# =============================================================================
# AVMLayerPropagate — propagate every layer through all video frames
# =============================================================================

class AVMLayerPropagate:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "layer_set":    ("AVM_LAYER_SET",),
                "sam3_model":   ("SAM3_MODEL",),
                "frame_idx":    ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Frame index the layer boxes were detected on.",
                }),
            }
        }

    RETURN_TYPES = ("AVM_LAYER_SET",)
    RETURN_NAMES = ("propagated_layer_set",)
    FUNCTION = "run"
    CATEGORY      = "AVM"

    def run(self, video_frames, layer_set, sam3_model, frame_idx):
        vs_mod, vn_mod = _load_sam3_modules()
        create_video_state = vs_mod.create_video_state
        VideoPrompt = vs_mod.VideoPrompt
        SAM3Propagate = vn_mod.SAM3Propagate

        propagator = SAM3Propagate()
        propagated = {}

        for label, boxes_prompt in layer_set.items():
            if not boxes_prompt or not boxes_prompt.get("boxes"):
                print(f"[AVMLayerPropagate] Skipping '{label}' — no boxes")
                continue

            print(f"[AVMLayerPropagate] Propagating layer: '{label}'")
            try:
                video_state = create_video_state(video_frames)
                cx, cy, bw, bh = boxes_prompt["boxes"][0]
                box_corners = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                prompt = VideoPrompt.create_box(frame_idx, 1, box_corners, is_positive=True)
                video_state = video_state.with_prompt(prompt)
                result = propagator.propagate(sam3_model, video_state)
                propagated[label] = result[0]  # SAM3_VIDEO_MASKS
                print(f"[AVMLayerPropagate] '{label}' propagation complete")
            except Exception as e:
                print(f"[AVMLayerPropagate] '{label}' failed: {e}")
                propagated[label] = None

        print(f"[AVMLayerPropagate] Done. {len(propagated)} layers propagated.")
        return (propagated,)


# =============================================================================
# AVMMultiFrameAutoLayer — run Auto Layer Detect on multiple keyframes at once
# =============================================================================

class AVMMultiFrameAutoLayer:
    """
    Like AVMAutoLayer but accepts a batch of keyframe images with their frame indices.
    Outputs a AVM_MULTI_FRAME_LAYER_SET — a list of per-frame detections — which can
    be fed directly into AVMMultiFrameLayerPropagate for multi-anchor propagation.
    """

    LAYER_PRESETS_LIST = ["auto", "portrait", "full_body", "product", "custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":        ("IMAGE",),
                "frame_indices": ("STRING", {
                    "default": "0",
                    "tooltip": "Comma-separated frame indices matching each image in the batch. E.g. '0,15,45'",
                }),
                "api":           ("AVM_API",),
                "layer_preset":  (cls.LAYER_PRESETS_LIST, {"default": "portrait"}),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "One layer description per line. Used when layer_preset='custom'.",
                }),
                "num_pos_points": ("INT", {"default": 4, "min": 1, "max": 12,
                    "tooltip": "Positive points per layer."}),
                "num_neg_points": ("INT", {"default": 2, "min": 0, "max": 6,
                    "tooltip": "Negative points per layer."}),
                "max_concurrent": ("INT", {"default": 8, "min": 1, "max": 16,
                    "tooltip": "Max parallel Gemini requests. Keep ≤ 25 to respect Gemini's RPM limit."}),
            }
        }

    RETURN_TYPES = ("AVM_MULTI_FRAME_LAYER_SET", "STRING", "STRING")
    RETURN_NAMES = ("multi_frame_layer_set", "label_list", "raw_response")
    FUNCTION = "run"
    CATEGORY      = "AVM"

    def run(self, images, frame_indices, api, layer_preset, custom_prompt="",
            num_pos_points=4, num_neg_points=2, max_concurrent=8):

        # Parse frame indices
        raw_parts = [x.strip() for x in frame_indices.split(",")]
        indices = []
        for p in raw_parts:
            try:
                indices.append(int(p))
            except ValueError:
                pass

        N = images.shape[0]
        if len(indices) != N:
            print(f"[AVMMultiFrameAutoLayer] {len(indices)} indices for {N} images — adjusting.")
            while len(indices) < N:
                indices.append((indices[-1] + 1) if indices else 0)
            indices = indices[:N]

        guidance_line = _build_guidance_line(layer_preset, custom_prompt)

        def _detect_frame(i):
            frame_idx = indices[i]
            pil_img = _tensor_to_pil(images[i:i+1])
            W, H = pil_img.size
            print(f"[AVMMultiFrameAutoLayer] Frame {frame_idx} ({i+1}/{N}) — {W}x{H}")

            try:
                layers, raw1, raw2 = _run_discovery_and_localize(
                    pil_img, api, layer_preset, guidance_line, W, H,
                    num_pos_points, num_neg_points, log_prefix="AVMMultiFrameAutoLayer",
                )
            except Exception as e:
                print(f"[AVM ERROR] AVMMultiFrameAutoLayer frame {frame_idx} failed: {e}")
                return frame_idx, {}, {}, f"=== Frame {frame_idx} ===\nERROR: {e}"

            layer_set = {}
            bundles = {}
            for entry in layers:
                label = entry.get("label", "").strip()
                if not label:
                    continue
                bundle = _build_layer_bundle(entry, W, H, num_pos_points, num_neg_points)
                bundles[label] = bundle
                layer_set[label] = bundle["boxes"]

            print(f"[AVMMultiFrameAutoLayer] Frame {frame_idx}: {list(bundles.keys())}")
            return frame_idx, layer_set, bundles, f"=== Frame {frame_idx} ===\nDiscovery: {raw1}\nLocalize: {raw2}"

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            frame_results = list(executor.map(_detect_frame, range(N)))

        multi_frame_results = []
        all_raw = []
        all_labels = set()

        for frame_idx, layer_set, bundles, raw_entry in frame_results:
            all_raw.append(raw_entry)
            all_labels.update(bundles.keys())
            multi_frame_results.append({
                "frame_idx": frame_idx,
                "layer_set": layer_set,
                "bundles":   bundles,
            })

        label_list = "\n".join(sorted(all_labels))
        raw_combined = "\n\n".join(all_raw)
        print(f"[AVMMultiFrameAutoLayer] Done. {N} frames, {len(all_labels)} unique labels.")
        return (multi_frame_results, label_list, raw_combined)


# =============================================================================
# AVMMultiFrameLayerPropagate — propagate layers with multi-frame anchors
# =============================================================================

class AVMMultiFrameLayerPropagate:
    """
    Like AVMLayerPropagate but uses a AVM_MULTI_FRAME_LAYER_SET so each label
    gets box prompts at every keyframe it was detected, giving SAM3 multiple anchors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames":          ("IMAGE",),
                "multi_frame_layer_set": ("AVM_MULTI_FRAME_LAYER_SET",),
                "sam3_model":            ("SAM3_MODEL",),
            }
        }

    RETURN_TYPES = ("AVM_LAYER_SET",)
    RETURN_NAMES = ("propagated_layer_set",)
    FUNCTION = "run"
    CATEGORY      = "AVM"

    def run(self, video_frames, multi_frame_layer_set, sam3_model):
        vs_mod, vn_mod = _load_sam3_modules()
        create_video_state = vs_mod.create_video_state
        VideoPrompt       = vs_mod.VideoPrompt
        SAM3Propagate     = vn_mod.SAM3Propagate

        # Collect all unique labels across all keyframes
        all_labels = set()
        for frame_data in multi_frame_layer_set:
            all_labels.update(frame_data["layer_set"].keys())

        propagator = SAM3Propagate()
        propagated = {}

        for label in all_labels:
            # Gather box prompts for this label at every keyframe it appears
            anchors = []
            for frame_data in multi_frame_layer_set:
                boxes_prompt = frame_data["layer_set"].get(label)
                if not boxes_prompt or not boxes_prompt.get("boxes"):
                    continue
                frame_idx = frame_data["frame_idx"]
                cx, cy, bw, bh = boxes_prompt["boxes"][0]
                box_corners = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                anchors.append((frame_idx, box_corners))

            if not anchors:
                print(f"[AVMMultiFrameLayerPropagate] '{label}' — no boxes in any frame, skipping")
                continue

            print(f"[AVMMultiFrameLayerPropagate] '{label}' — {len(anchors)} anchor(s): frames {[a[0] for a in anchors]}")
            try:
                video_state = create_video_state(video_frames)
                for frame_idx, box_corners in anchors:
                    prompt = VideoPrompt.create_box(frame_idx, 1, box_corners, is_positive=True)
                    video_state = video_state.with_prompt(prompt)
                result = propagator.propagate(sam3_model, video_state)
                propagated[label] = result[0]
                print(f"[AVMMultiFrameLayerPropagate] '{label}' done")
            except Exception as e:
                print(f"[AVMMultiFrameLayerPropagate] '{label}' failed: {e}")
                propagated[label] = None

        print(f"[AVMMultiFrameLayerPropagate] Done. {len(propagated)} layers propagated.")
        return (propagated,)


# =============================================================================
# VLMReferenceMatch — find a subject from a reference image in a target frame
# =============================================================================

class VLMReferenceMatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "target_frame":    ("IMAGE",),
                "api":             ("AVM_API",),
            },
            "optional": {
                "subject_description": ("STRING", {
                    "default": "the person",
                    "multiline": False,
                    "tooltip": "What to find — e.g. 'the person', 'the red bag', 'the cat'.",
                }),
            }
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "STRING")
    RETURN_NAMES = ("boxes_prompt", "raw_response")
    FUNCTION = "run"
    CATEGORY      = "AVM"

    def run(self, reference_image, target_frame, api, subject_description="the person"):
        ref_pil = _tensor_to_pil(reference_image)
        tgt_pil = _tensor_to_pil(target_frame)
        W, H = tgt_pil.size

        prompt = reference_match_prompt(subject_description, W, H)
        raw = _call_gemini([ref_pil, tgt_pil], prompt, api)
        print(f"[VLMReferenceMatch] Raw: {raw}")

        empty = {"boxes": [], "labels": []}
        try:
            data = _parse_json(raw)
            bbox = data.get("bbox")
            if not bbox:
                print(f"[VLMReferenceMatch] Subject not found in target frame.")
                return (empty, raw)
            x1, y1, x2, y2 = bbox
            x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
            cx = (x1n + x2n) / 2; cy = (y1n + y2n) / 2
            bw = x2n - x1n;       bh = y2n - y1n
            boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [True]}
            print(f"[VLMReferenceMatch] box (cx,cy,w,h): [{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}]")
        except Exception as e:
            print(f"[AVM ERROR] VLMReferenceMatch failed to parse response: {e}\nRaw: {raw}")
            boxes_prompt = empty

        return (boxes_prompt, raw)


# =============================================================================
# AVMLayerSelector — extract a single layer from a AVM_LAYER_SET
# =============================================================================

def _extract_mask_from_video_masks(video_masks):
    """Convert SAM3_VIDEO_MASKS {frame_idx: {"mask": [N,H,W]} | tensor} → MASK [F,H,W]."""
    import torch
    frame_indices = sorted(k for k in video_masks if isinstance(k, int))
    if not frame_indices:
        return torch.zeros(1, 8, 8)

    frame_tensors = []
    ref_h, ref_w = None, None

    for idx in frame_indices:
        data = video_masks[idx]
        m = data.get("mask") if isinstance(data, dict) else data

        if m is None:
            h, w = ref_h or 8, ref_w or 8
            frame_tensors.append(torch.zeros(h, w))
            continue

        # m: [N, H, W] or [H, W]
        if m.dim() == 3:
            m = m[0]  # first object (obj_id=1 → index 0)

        ref_h, ref_w = m.shape[-2], m.shape[-1]
        frame_tensors.append(m.float())

    return torch.stack(frame_tensors, dim=0)  # [F, H, W]


class AVMLayerSelector:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_set":  ("AVM_LAYER_SET",),
                "layer_name": ("STRING", {"default": "face", "multiline": False,
                               "tooltip": "Exact label or case-insensitive substring match."}),
            }
        }

    RETURN_TYPES = ("MASK", "SAM3_BOXES_PROMPT", "STRING")
    RETURN_NAMES = ("mask", "boxes_prompt", "available_layers")
    FUNCTION = "run"
    CATEGORY      = "AVM"

    def run(self, layer_set, layer_name):
        import torch

        available = list(layer_set.keys())
        available_str = ", ".join(available)
        empty_boxes = {"boxes": [], "labels": []}
        empty_mask  = torch.zeros(1, 8, 8)

        # Resolve key: exact → case-insensitive substring
        value, matched = None, None
        if layer_name in layer_set:
            value, matched = layer_set[layer_name], layer_name
        else:
            needle = layer_name.lower()
            for key in layer_set:
                if needle in key.lower() or key.lower() in needle:
                    value, matched = layer_set[key], key
                    break

        if value is None:
            print(f"[AVMLayerSelector] '{layer_name}' not found. Available: {available_str}")
            return (empty_mask, empty_boxes, available_str)

        print(f"[AVMLayerSelector] Matched '{matched}' for query '{layer_name}'")

        # SAM3_BOXES_PROMPT: dict with "boxes" key  (from AVMAutoLayer)
        if isinstance(value, dict) and "boxes" in value:
            print(f"[AVMLayerSelector] Type: SAM3_BOXES_PROMPT — returning boxes, empty mask")
            return (empty_mask, value, available_str)

        # SAM3_VIDEO_MASKS: dict with int frame-index keys (from AVMLayerPropagate)
        if isinstance(value, dict) and any(isinstance(k, int) for k in value):
            print(f"[AVMLayerSelector] Type: SAM3_VIDEO_MASKS — extracting mask tensor")
            try:
                mask = _extract_mask_from_video_masks(value)
                print(f"[AVMLayerSelector] Mask shape: {mask.shape}")
                return (mask, empty_boxes, available_str)
            except Exception as e:
                print(f"[AVMLayerSelector] Extraction failed: {e}")
                return (empty_mask, empty_boxes, available_str)

        print(f"[AVMLayerSelector] '{matched}' has no usable data (None or unknown type)")
        return (empty_mask, empty_boxes, available_str)


# =============================================================================
# AVMAddFramePromptBundle — add box + points to video_state in one node
# =============================================================================

class AVMAddFramePromptBundle:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_state":   ("SAM3_VIDEO_STATE",),
                "box_and_point": ("SAM3_BOX_AND_POINT",),
                "frame_idx":     ("INT", {"default": 15, "min": 0,
                                  "tooltip": "Frame to anchor prompts on."}),
                "obj_id":        ("INT", {"default": 1, "min": 1}),
            }
        }

    RETURN_TYPES  = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES  = ("video_state",)
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, video_state, box_and_point, frame_idx, obj_id):
        import importlib.util, os as _os
        _base = _os.path.join(_find_sam3_nodes_dir(), "video_state.py")
        _spec = importlib.util.spec_from_file_location("sam3_video_state", _base)
        _mod  = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        VideoPrompt = _mod.VideoPrompt

        boxes_prompt    = box_and_point.get("boxes")
        positive_points = box_and_point.get("positive")
        negative_points = box_and_point.get("negative")

        # Add bounding box
        if boxes_prompt and boxes_prompt.get("boxes"):
            cx, cy, w, h = boxes_prompt["boxes"][0]
            video_state = video_state.with_prompt(
                VideoPrompt.create_box(frame_idx, obj_id,
                                       [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                                       is_positive=True)
            )

        # Add points (positive + negative merged)
        all_points, all_labels = [], []
        if positive_points and positive_points.get("points"):
            for pt in positive_points["points"]:
                all_points.append([float(pt[0]), float(pt[1])]); all_labels.append(1)
        if negative_points and negative_points.get("points"):
            for pt in negative_points["points"]:
                all_points.append([float(pt[0]), float(pt[1])]); all_labels.append(0)
        if all_points:
            video_state = video_state.with_prompt(
                VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels)
            )

        print(f"[AVMAddFramePromptBundle] frame={frame_idx} obj={obj_id} "
              f"box={'yes' if boxes_prompt and boxes_prompt.get('boxes') else 'no'} "
              f"pts={len(all_points)}")
        return (video_state,)


# =============================================================================
# AVMUnpackBundle — split SAM3_BOX_AND_POINT into SAM3 native types
# =============================================================================

class AVMUnpackBundle:
    """Splits a SAM3_BOX_AND_POINT bundle into the three types that
    SAM3VideoSegmentation already accepts as separate inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"box_and_point": ("SAM3_BOX_AND_POINT",)}}

    RETURN_TYPES  = ("SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT")
    RETURN_NAMES  = ("boxes_prompt", "positive_points", "negative_points")
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, box_and_point):
        return (
            box_and_point.get("boxes",    {"boxes": [],  "labels": []}),
            box_and_point.get("positive", {"points": [], "labels": []}),
            box_and_point.get("negative", {"points": [], "labels": []}),
        )


# =============================================================================
# VLMAutoCrop — presence/discovery call + localization call + crop
# =============================================================================

class VLMAutoCrop:
    """
    Two-call Gemini workflow that produces cropped images without a preset:
      Call 1 (Presence / Discovery): Gemini freely identifies what regions exist.
      Call 2 (Localization): Gemini returns a tight bounding box for each region.
    Each detected region is then cropped from the input image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api":   ("AVM_API",),
            },
            "optional": {
                "focus_hint": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Optional hint to guide discovery "
                        "(e.g. 'person and clothing'). "
                        "Leave blank for fully automatic detection."
                    ),
                }),
                "max_regions":      ("INT",     {"default": 8,    "min": 1,   "max": 8}),
                "padding":          ("INT",     {"default": 16,   "min": 0,   "max": 128}),
                "normalize_size":   ("BOOLEAN", {"default": True}),
                "target_long_side": ("INT",     {"default": 1008, "min": 256, "max": 4096}),
            },
        }

    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING",
        "AVM_LAYER_SET", "STRING", "STRING",
    )
    RETURN_NAMES = (
        "crop_1", "crop_2", "crop_3", "crop_4",
        "crop_5", "crop_6", "crop_7", "crop_8",
        "label_1", "label_2", "label_3", "label_4",
        "label_5", "label_6", "label_7", "label_8",
        "layer_set", "label_list", "raw_response",
    )
    FUNCTION  = "run"
    CATEGORY      = "AVM"

    def run(self, image, api, focus_hint="", max_regions=8, padding=16,
            normalize_size=True, target_long_side=1008):
        import torch
        import torch.nn.functional as F

        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        # ── Call 1: Presence / Discovery ─────────────────────────────────
        hint_line = f"Focus on: {focus_hint.strip()}\n\n" if focus_hint.strip() else ""
        raw1 = _call_gemini(pil_img, autocrop_discovery_prompt(hint_line, max_regions), api)
        print(f"[VLMAutoCrop] Discovery: {raw1}")

        try:
            data1 = _parse_json(raw1)
            discovered = [
                r.strip() for r in data1.get("regions", [])
                if isinstance(r, str) and r.strip()
            ]
            if not discovered:
                raise ValueError("empty regions list")
        except Exception as e:
            print(f"[VLMAutoCrop] Discovery parse error: {e}")
            empty = ""
            return (*([image] * 8), *([empty] * 8), {}, "", raw1)

        # ── Call 2: Localization ──────────────────────────────────────────
        labels_json = json.dumps(discovered[:max_regions], indent=2)
        raw2 = _call_gemini(pil_img, autocrop_localize_prompt(W, H, labels_json), api)
        print(f"[VLMAutoCrop] Localization: {raw2}")
        raw = f"=== Discovery ===\n{raw1}\n\n=== Localization ===\n{raw2}"

        try:
            data2 = _parse_json(raw2)
            regions = data2.get("regions", [])[:8]
        except Exception as e:
            print(f"[VLMAutoCrop] Localization parse error: {e}")
            regions = []

        # ── Crop each region ──────────────────────────────────────────────
        _B, iH, iW, _C = image.shape

        def _crop(entry):
            x1p, y1p, x2p, y2p = entry["bbox"]
            x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1p, y1p, x2p, y2p, W, H)
            px1 = max(0, int(x1n * iW) - padding)
            py1 = max(0, int(y1n * iH) - padding)
            px2 = min(iW, int(x2n * iW) + padding)
            py2 = min(iH, int(y2n * iH) + padding)
            cropped = image[:, py1:py2, px1:px2, :]
            ch, cw = py2 - py1, px2 - px1
            if normalize_size and max(ch, cw) > 0:
                scale = target_long_side / max(ch, cw)
                if abs(scale - 1.0) > 0.01:
                    oh, ow = round(ch * scale), round(cw * scale)
                    cropped = F.interpolate(
                        cropped.permute(0, 3, 1, 2).float(),
                        size=(oh, ow), mode="bilinear", align_corners=False,
                    ).permute(0, 2, 3, 1)
            return cropped

        crops, labels, layer_set = [], [], {}
        for entry in regions:
            try:
                crop = _crop(entry)
                lbl  = entry.get("label", f"region_{len(crops) + 1}")
                crops.append(crop)
                labels.append(lbl)
                x1p, y1p, x2p, y2p = entry["bbox"]
                x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1p, y1p, x2p, y2p, W, H)
                cx = (x1n + x2n) / 2;  cy = (y1n + y2n) / 2
                layer_set[lbl] = {
                    "boxes":  [[cx, cy, x2n - x1n, y2n - y1n]],
                    "labels": [True],
                }
            except Exception as e:
                print(f"[VLMAutoCrop] Crop error for '{entry.get('label', '?')}': {e}")

        while len(crops) < 8:
            crops.append(image)
            labels.append("")

        label_list = "\n".join(f"{i+1}. {lb}" for i, lb in enumerate(labels) if lb)
        print(f"[VLMAutoCrop] Cropped {len(regions)} regions: {[l for l in labels if l]}")
        return (*crops[:8], *labels[:8], layer_set, label_list, raw)


# =============================================================================
# Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "AVMAPIConfig":                    AVMAPIConfig,
    "VLMImageTest":                    VLMImageTest,
    "VLMtoBBoxAndPoints":              VLMtoBBoxAndPoints,
    "VLMtoBBoxAndPointsMultiFrame":    VLMtoBBoxAndPointsMultiFrame,
    "VLMtoBBox":                       VLMtoBBox,
    "VLMtoPoints":                     VLMtoPoints,
    "VLMtoMultiBBox":                  VLMtoMultiBBox,
    "VLMBBoxPreview":                  VLMBBoxPreview,
    "VLMDebugPreview":                 VLMDebugPreview,
    "VLMMultiFrameBBoxPreview":        VLMMultiFrameBBoxPreview,
    "AVMAddFramePrompt":               AVMAddFramePrompt,
    "VLMFacePartsBBox":                VLMFacePartsBBox,
    "VLMFacePrecisePoints":            VLMFacePrecisePoints,
    "VLMFaceRegion":                   VLMFaceRegion,
    "AVMCropByBox":                    AVMCropByBox,
    "AVMPasteBackMask":                AVMPasteBackMask,
    "AVMAutoLayer":                    AVMAutoLayer,
    "AVMMultiFrameAutoLayer":          AVMMultiFrameAutoLayer,
    "AVMLayerPropagate":               AVMLayerPropagate,
    "AVMMultiFrameLayerPropagate":     AVMMultiFrameLayerPropagate,
    "VLMReferenceMatch":               VLMReferenceMatch,
    "AVMLayerSelector":                AVMLayerSelector,
    "VLMPromptEditor":                 VLMPromptEditor,
    "VLMAutoCrop":                     VLMAutoCrop,
    "AVMAddFramePromptBundle":         AVMAddFramePromptBundle,
    "AVMUnpackBundle":                 AVMUnpackBundle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AVMAPIConfig":                    "AVM API Config",
    "VLMImageTest":                    "AVM VLM Test",
    "VLMtoBBoxAndPoints":              "AVM VLM → BBox + Points",
    "VLMtoBBoxAndPointsMultiFrame":    "AVM VLM → BBox + Points (Multi-Frame)",
    "VLMtoBBox":                       "AVM VLM → BBox",
    "VLMtoPoints":                     "AVM VLM → Points",
    "VLMtoMultiBBox":                  "AVM VLM → Multi BBox",
    "VLMBBoxPreview":                  "AVM BBox Preview",
    "VLMDebugPreview":                 "AVM Debug Preview",
    "VLMMultiFrameBBoxPreview":        "AVM Multi-Frame BBox Preview",
    "AVMAddFramePrompt":               "AVM Add Frame Prompt",
    "VLMFacePartsBBox":                "AVM VLM → Face Parts BBox",
    "VLMFacePrecisePoints":            "AVM VLM → Face Points",
    "VLMFaceRegion":                   "AVM Face Region",
    "AVMCropByBox":                    "AVM Crop by Box",
    "AVMPasteBackMask":                "AVM Paste Back Mask",
    "AVMAutoLayer":                    "AVM Auto Layer Detect",
    "AVMMultiFrameAutoLayer":          "AVM Multi-Frame Layer Detect",
    "AVMLayerPropagate":               "AVM Layer Propagate",
    "AVMMultiFrameLayerPropagate":     "AVM Multi-Frame Layer Propagate",
    "VLMReferenceMatch":               "AVM Reference Match",
    "AVMLayerSelector":                "AVM Layer Selector",
    "VLMPromptEditor":                 "AVM Prompt Editor",
    "VLMAutoCrop":                     "AVM Auto Crop",
    "AVMAddFramePromptBundle":         "AVM Frame Prompt Bundle",
    "AVMUnpackBundle":                 "AVM Unpack Bundle",
}
