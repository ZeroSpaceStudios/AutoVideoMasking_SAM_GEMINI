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

AVAILABLE_MODELS = ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]
DEFAULT_MODEL = AVAILABLE_MODELS[0]

# Path to .env file in the package root (one level up from nodes/)
_ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")


def _resolve_api_key(ui_key: str) -> str:
    """Tiered API key lookup: env var → .env file → UI input."""
    # 1. System environment variable
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        print("[AVM] API key loaded from environment variable.")
        return key
    # 2. .env file
    env_path = os.path.normpath(_ENV_FILE)
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        print("[AVM] API key loaded from .env file.")
                        return key
    # 3. UI input
    if ui_key.strip():
        print("[AVM] API key loaded from node UI input.")
        return ui_key.strip()
    raise ValueError(
        "[AVM] No API key found. Set GEMINI_API_KEY env var, add it to .env, or enter it in the node."
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
                               "tooltip": "Gemini model to use for VLM inference"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False,
                            "tooltip": "Leave blank to use GEMINI_API_KEY env var or .env file"}),
            }
        }

    RETURN_TYPES  = ("AVM_API",)
    RETURN_NAMES  = ("api",)
    FUNCTION      = "run"
    CATEGORY      = "AVM"

    def run(self, model_name, api_key=""):
        resolved_key = _resolve_api_key(api_key)
        return ({"api_key": resolved_key, "model_name": model_name},)


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

def _call_gemini(pil_img, prompt, api):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    client = genai.Client(api_key=api["api_key"])
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    response = client.models.generate_content(
        model=api["model_name"],
        contents=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ]
    )
    return response.text


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
        prompt = (
            "Describe exactly what you see in this image. "
            "List every object, their positions and colors."
        )
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

        prompt = (
            f"Locate: {target_description}\n"
            f"Image dimensions: {W}x{H} pixels.\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"bbox": [x1, y1, x2, y2], "label": "<short name>"}\n'
            "Pixel coordinates, tight box, x1<x2, y1<y2."
            + few_shot_block
        )

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

        prompt = (
            f"Segment: {target_description}\n{size_note}\n"
            f"Place {num_pos_points} positive point(s) ON the {target_description} — spread across, deep inside, never on edges.\n"
            f"Place {num_neg_points} negative point(s) on anything NOT {target_description} — near boundary.\n"
            "Return ONLY JSON:\n"
            '{"positive": [[x, y], ...], "negative": [[x, y], ...]}'
            + few_shot_block
        )

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

        prompt = (
            f"Detect: {target_description}\n"
            f"Image: {W}x{H} px. Find up to {max_objects} instances.\n"
            "Return ONLY JSON:\n"
            '{"objects": [{"bbox": [x1,y1,x2,y2], "label": "name"}, ...]}\n'
            "Pixel coords, tight boxes, sorted by confidence."
            + few_shot_block
        )

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

        prompt = (
            f"Segment: {target_description}\nImage: {W}x{H} pixels.\n\n"
            "1. Tight bounding box around the target.\n"
            f"2. {num_pos_points} positive point(s) ON the {target_description} — spread across, deep inside, never on edges.\n"
            f"3. {num_neg_points} negative point(s) on anything NOT {target_description} — near boundary.\n\n"
            "Return ONLY JSON:\n"
            '{"bbox": [x1, y1, x2, y2], "positive": [[x, y], ...], "negative": [[x, y], ...]}'
            + few_shot_block
        )

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

        auto_prompt = (
            f"Segment: {target_description}\nImage: {W}x{H} pixels.\n\n"
            "1. Tight bounding box around the target.\n"
            f"2. {num_pos_points} positive point(s) ON the {target_description}"
            " — spread across, deep inside, never on edges.\n"
            f"3. {num_neg_points} negative point(s) on anything NOT {target_description}"
            " — near boundary.\n\n"
            "Return ONLY JSON:\n"
            '{"bbox": [x1, y1, x2, y2], "positive": [[x, y], ...], "negative": [[x, y], ...]}'
        )

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
        prompt = (
            f"Image: {cW}x{cH} px (cropped to person).\n"
            "Return tight bounding boxes for each region (pixel coords in cropped image).\n\n"
            "Regions:\n" + parts_desc + "\n\n"
            "Return ONLY JSON:\n{\n"
            '  "hair":      {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0},\n'
            '  "face":      {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0},\n'
            '  "neck":      {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0},\n'
            '  "face_neck": {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0},\n'
            '  "clothing":  {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0}\n}\n'
            "Rules: x1<x2 y1<y2, face+hair must NOT overlap, neck BELOW chin."
        )

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
            x1 = max(0, x1-padding_px); y1 = max(0, y1-padding_px)
            x2 = min(cW, x2+padding_px); y2 = min(cH, y2+padding_px)
            ax1=(x1+crop_x1)/W; ay1=(y1+crop_y1)/H
            ax2=(x2+crop_x1)/W; ay2=(y2+crop_y1)/H
            cx=(ax1+ax2)/2; cy=(ay1+ay2)/2
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

        prompt = (
            f"Image: {cW}x{cH} px (cropped to face region).\n\n"
            f"TARGET: {cfg['fg_desc']}{modifier_block}\n\n"
            f"Place {num_fg_points} FOREGROUND points spread across these zones:\n"
            f"  {cfg['fg_zones']}\n"
            "  → Points must be deep inside the region, never on edges or boundaries.\n\n"
            f"Place {num_bg_points} BACKGROUND points on: {cfg['bg_desc']}\n"
            f"  Preferred zones: {cfg['bg_zones']}\n"
            "  → Points should be close to but outside the target boundary.\n\n"
            "Also return a tight bounding box around the target region.\n\n"
            "Return ONLY JSON (pixel coordinates in this cropped image):\n"
            '{"bbox": [x1, y1, x2, y2], "foreground": [[x, y], ...], "background": [[x, y], ...]}\n'
            "Rules: x1<x2 y1<y2, spread points — do NOT cluster them."
        )

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
            crop_padding=24, person_bbox=None):
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
        prompt1 = (
            f"Image: {sW}x{sH} px.\n"
            f"TARGET: {region}\n\n"
            + self._FACE_RULES +
            "\nReturn ONLY JSON (pixel coords):\n"
            '{"bbox": [x1, y1, x2, y2]}\n'
            "Tight box. x1<x2 y1<y2."
        )
        raw1 = _call_gemini(search_img, prompt1, api)
        print(f"[VLMFaceRegion] Stage1: {raw1}")

        try:
            bx1, by1, bx2, by2 = _parse_json(raw1)["bbox"]
        except Exception as e:
            raise RuntimeError(f"[VLMFaceRegion] Stage1 failed to parse Gemini response: {e}\nRaw: {raw1}") from e

        # Map to full-image pixel space
        if any(v > 2.0 for v in [bx1, by1, bx2, by2]):
            bx1 += search_x1;  by1 += search_y1
            bx2 += search_x1;  by2 += search_y1
        else:
            bx1 = bx1 * sW + search_x1;  by1 = by1 * sH + search_y1
            bx2 = bx2 * sW + search_x1;  by2 = by2 * sH + search_y1

        cx1 = max(0, int(bx1) - crop_padding)
        cy1 = max(0, int(by1) - crop_padding)
        cx2 = min(W, int(bx2) + crop_padding)
        cy2 = min(H, int(by2) + crop_padding)
        crop_meta = {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2, "orig_w": W, "orig_h": H}

        cropped_tensor = image[:, cy1:cy2, cx1:cx2, :]
        pil_crop = pil_full.crop((cx1, cy1, cx2, cy2))
        cW, cH = pil_crop.size

        # ── Stage 2: precise points on the crop ───────────────────────
        prompt2 = (
            f"Image: {cW}x{cH} px — cropped tightly to: {region}.\n"
            f"TARGET: {region}\n\n"
            + self._FACE_RULES +
            f"\nPlace {num_fg_points} FOREGROUND points spread across the entire target.\n"
            f"Place {num_bg_points} BACKGROUND points just outside the target boundary.\n\n"
            "Return ONLY JSON (pixel coords in this cropped image):\n"
            '{"foreground": [[x, y], ...], "background": [[x, y], ...]}'
        )
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

        # box_prompt covers the whole crop (SAM3 works on cropped_image)
        box_prompt   = {"box":   [0.5, 0.5, 1.0, 1.0], "label": True}
        boxes_prompt = {"boxes": [[0.5, 0.5, 1.0, 1.0]], "labels": [True]}

        positive_points = normalize_points(fg_raw, 1, W=cW, H=cH)
        negative_points = normalize_points(bg_raw, 0, W=cW, H=cH)

        print(f"[VLMFaceRegion] crop=[{cx1},{cy1},{cx2},{cy2}] "
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
    discovery_prompt = (
        (f"{guidance_line}\n\n" if guidance_line else "")
        + "Look at the image and list every distinct visual layer or region you can clearly see. "
        "Give each a SHORT, SPECIFIC label (e.g. 'black turtleneck', 'curly brown hair', 'gold hoop earrings'). "
        "Max 8 layers. Skip anything not clearly visible.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"layers": ["label1", "label2", ...]}'
    )
    raw1 = _call_gemini(pil_img, discovery_prompt, api)
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
    localize_prompt = (
        f"Image: {W}x{H} pixels. All coordinates are pixel values in this image.\n\n"
        f"For each region in the list below, return:\n"
        f"  • A tight bounding box (x1,y1,x2,y2, pixel coords, x1<x2, y1<y2)\n"
        f"  • {num_pos_points} positive points INSIDE the region "
        f"(spread across, deep inside, never on edges)\n"
        f"  • {num_neg_points} negative points OUTSIDE the region "
        f"(just beyond its boundary)\n\n"
        f"Regions:\n{labels_json}\n\n"
        "Skip any region not clearly visible. Confidence 0.0-1.0, omit below 0.3.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"layers": [\n'
        '  {"label": "<exact label>", "bbox": [x1,y1,x2,y2], "confidence": 0.9,\n'
        '   "positive": [[x,y],...], "negative": [[x,y],...]},\n'
        "  ...\n]}"
    )
    raw2 = _call_gemini(pil_img, localize_prompt, api)
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
            }
        }

    RETURN_TYPES = ("AVM_MULTI_FRAME_LAYER_SET", "STRING", "STRING")
    RETURN_NAMES = ("multi_frame_layer_set", "label_list", "raw_response")
    FUNCTION = "run"
    CATEGORY      = "AVM"

    def run(self, images, frame_indices, api, layer_preset, custom_prompt="",
            num_pos_points=4, num_neg_points=2):

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

        multi_frame_results = []
        all_raw = []
        all_labels = set()

        for i in range(N):
            frame_idx = indices[i]
            pil_img = _tensor_to_pil(images[i:i+1])
            W, H = pil_img.size
            print(f"[AVMMultiFrameAutoLayer] Frame {frame_idx} ({i+1}/{N}) — {W}x{H}")

            layers, raw1, raw2 = _run_discovery_and_localize(
                pil_img, api, layer_preset, guidance_line, W, H,
                num_pos_points, num_neg_points, log_prefix="AVMMultiFrameAutoLayer",
            )
            all_raw.append(f"=== Frame {frame_idx} ===\nDiscovery: {raw1}\nLocalize: {raw2}")

            layer_set = {}
            bundles = {}
            for entry in layers:
                label = entry.get("label", "").strip()
                if not label:
                    continue
                bundle = _build_layer_bundle(entry, W, H, num_pos_points, num_neg_points)
                bundles[label] = bundle
                layer_set[label] = bundle["boxes"]
                all_labels.add(label)

            multi_frame_results.append({
                "frame_idx": frame_idx,
                "layer_set": layer_set,
                "bundles":   bundles,
            })
            print(f"[AVMMultiFrameAutoLayer] Frame {frame_idx}: {list(bundles.keys())}")

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
        import io as _io
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai not installed. Run: pip install google-genai")

        ref_pil = _tensor_to_pil(reference_image)
        tgt_pil = _tensor_to_pil(target_frame)
        W, H = tgt_pil.size

        prompt = (
            f"LEFT image: reference showing {subject_description}.\n"
            f"RIGHT image: target frame, {W}x{H} pixels.\n\n"
            f"Find {subject_description} from the LEFT image in the RIGHT image. "
            "Return a tight bounding box in the RIGHT image coordinate space.\n\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}\n'
            "Pixel coordinates of the RIGHT image only. x1<x2, y1<y2. "
            'If the subject is not found, return {"bbox": null, "confidence": 0.0}.'
        )

        client = genai.Client(api_key=api["api_key"])
        buf_ref = _io.BytesIO(); ref_pil.save(buf_ref, format="PNG")
        buf_tgt = _io.BytesIO(); tgt_pil.save(buf_tgt, format="PNG")

        response = client.models.generate_content(
            model=api["model_name"],
            contents=[
                types.Part.from_bytes(data=buf_ref.getvalue(), mime_type="image/png"),
                types.Part.from_bytes(data=buf_tgt.getvalue(), mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ]
        )
        raw = response.text
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
        discovery_prompt = (
            f"{hint_line}"
            "Look at this image and list every distinct visual region you can clearly see. "
            "Give each a SHORT, SPECIFIC label (e.g. 'red jacket', 'person face', 'wooden table'). "
            f"Return at most {max_regions} regions. Skip anything not clearly visible.\n\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"regions": ["label1", "label2", ...]}'
        )

        raw1 = _call_gemini(pil_img, discovery_prompt, api)
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
        localize_prompt = (
            f"Image: {W}x{H} pixels.\n\n"
            f"Return a tight bounding box for each of these regions:\n{labels_json}\n\n"
            "Pixel coordinates, x1<x2, y1<y2. Skip any region not visible. "
            "Confidence 0.0-1.0. Omit entries below 0.3.\n\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"regions": [\n'
            '  {"label": "<exact label from list>", "bbox": [x1, y1, x2, y2], "confidence": 0.9},\n'
            "  ...\n]}"
        )

        raw2 = _call_gemini(pil_img, localize_prompt, api)
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
    "VLMtoBBox":                       VLMtoBBox,
    "VLMtoPoints":                     VLMtoPoints,
    "VLMtoMultiBBox":                  VLMtoMultiBBox,
    "VLMBBoxPreview":                  VLMBBoxPreview,
    "VLMDebugPreview":                 VLMDebugPreview,
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
    "VLMtoBBox":                       "AVM VLM → BBox",
    "VLMtoPoints":                     "AVM VLM → Points",
    "VLMtoMultiBBox":                  "AVM VLM → Multi BBox",
    "VLMBBoxPreview":                  "AVM BBox Preview",
    "VLMDebugPreview":                 "AVM Debug Preview",
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
