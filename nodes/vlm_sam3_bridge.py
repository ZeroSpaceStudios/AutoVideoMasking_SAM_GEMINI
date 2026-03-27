"""
VLM -> SAM3 Bridge Node
Calls Gemini to auto-generate bbox or point prompts,
then outputs native SAM3_BOX_PROMPT / SAM3_POINTS_PROMPT types that wire
directly into SAM3Segmentation or SAM3Grounding.

Author: SAMhera

Coordinate conventions (must match segmentation.py):
  SAM3_BOX_PROMPT   : {"box": [cx, cy, w, h],  "label": bool}   - normalized [0,1]
  SAM3_BOXES_PROMPT : {"boxes": [...], "labels": [...]}
  SAM3_POINT_PROMPT : {"point": [x, y], "label": int}           - normalized [0,1]
  SAM3_POINTS_PROMPT: {"points": [...], "labels": [...]}
"""

import re
import json
import io
import numpy as np
from PIL import Image

DEFAULT_MODEL = "gemini-2.5-pro"


# =============================================================================
# SAMheraAPIKey — set credentials once, connect api slot to all nodes
# =============================================================================

class SAMheraAPIKey:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":    ("STRING", {"default": "", "multiline": False}),
                "model_name": ("STRING", {"default": DEFAULT_MODEL, "multiline": False,
                               "tooltip": "e.g. gemini-2.5-pro or gemini-3.1-pro-preview"}),
            }
        }

    RETURN_TYPES  = ("SAMHERA_API",)
    RETURN_NAMES  = ("api",)
    FUNCTION      = "run"
    CATEGORY      = "SAMhera"

    def run(self, api_key, model_name):
        return ({"api_key": api_key, "model_name": model_name},)


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


# =============================================================================
# VLMImageTest — verify Gemini is receiving the image correctly
# =============================================================================

class VLMImageTest:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api":   ("SAMHERA_API",),
            }
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("description",)
    FUNCTION      = "run"
    CATEGORY      = "SAMhera"
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
                "api":                ("SAMHERA_API",),
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
    CATEGORY      = "SAMhera"

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
            print(f"[VLMtoBBox] Parse error: {e} -- full-image fallback")
            x1, y1, x2, y2 = 0, 0, W, H

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
                "api":                ("SAMHERA_API",),
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
    CATEGORY      = "SAMhera"

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
            print(f"[VLMtoPoints] Parse error: {e}")
            pos_raw = [[crop_w//2, crop_h//2]]; neg_raw = []

        pos_raw = pos_raw[:num_pos_points]
        neg_raw = neg_raw[:num_neg_points]

        def to_norm(pts_raw, label_val):
            pts, lbls = [], []
            for pt in pts_raw:
                x, y = pt[0], pt[1]
                # Gemini uses 0-1000 scale; values <= 1.5 are already normalized 0-1
                abs_x = (x / 1000 * crop_w + crop_x1) if x > 1.5 else (x * crop_w + crop_x1)
                abs_y = (y / 1000 * crop_h + crop_y1) if y > 1.5 else (y * crop_h + crop_y1)
                pts.append([max(0.0, min(1.0, abs_x / W)), max(0.0, min(1.0, abs_y / H))])
                lbls.append(label_val)
            return {"points": pts, "labels": lbls}

        positive_points = to_norm(pos_raw, 1)
        negative_points = to_norm(neg_raw, 0)
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
                "api":                ("SAMHERA_API",),
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
    CATEGORY      = "SAMhera"

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
            print(f"[VLMtoMultiBBox] Parse error: {e}"); objects = []

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
                "api":                ("SAMHERA_API",),
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
    CATEGORY      = "SAMhera"

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
            print(f"[VLMtoBBoxAndPoints] Parse error: {e}")
            x1, y1, x2, y2 = 0, 0, W, H
            pos_raw = [[W//2, H//2]]; neg_raw = []

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

        def to_norm(pts, label_val):
            result, lbls = [], []
            for pt in pts:
                # Gemini uses 0-1000 scale; values <= 1.5 are already normalized 0-1
                nx = max(0.0, min(1.0, pt[0]/1000 if pt[0] > 1.5 else pt[0]))
                ny = max(0.0, min(1.0, pt[1]/1000 if pt[1] > 1.5 else pt[1]))
                result.append([nx, ny]); lbls.append(label_val)
            return {"points": result, "labels": lbls}

        positive_points = to_norm(pos_raw, 1)
        negative_points = to_norm(neg_raw, 0)

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
                "api":                ("SAMHERA_API",),
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
    CATEGORY      = "SAMhera"

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
            print(f"[VLMPromptEditor] Parse error: {e}")
            x1, y1, x2, y2 = 0, 0, W, H
            pos_raw = [[W//2, H//2]]; neg_raw = []

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

        def to_norm(pts, label_val):
            result, lbls = [], []
            for pt in pts:
                # Gemini uses 0-1000 scale; values <= 1.5 are already normalized 0-1
                nx = max(0.0, min(1.0, pt[0]/1000 if pt[0] > 1.5 else pt[0]))
                ny = max(0.0, min(1.0, pt[1]/1000 if pt[1] > 1.5 else pt[1]))
                result.append([nx, ny]); lbls.append(label_val)
            return {"points": result, "labels": lbls}

        positive_points = to_norm(pos_raw, 1)
        negative_points = to_norm(neg_raw, 0)

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
    CATEGORY      = "SAMhera"

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
    CATEGORY      = "SAMhera"

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
# SAMheraCropByBox
# =============================================================================

class SAMheraCropByBox:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "boxes_prompt": ("SAM3_BOX_AND_POINT",),
            },
            "optional": {
                "padding":        ("INT",     {"default": 16,   "min": 0,   "max": 128}),
                "box_index":      ("INT",     {"default": 0,    "min": 0,   "max": 4}),
                "normalize_size": ("BOOLEAN", {"default": True}),
                "target_long_side": ("INT",   {"default": 1008, "min": 256, "max": 4096}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "CROP_META")
    RETURN_NAMES  = ("cropped_image", "crop_meta")
    FUNCTION      = "run"
    CATEGORY      = "SAMhera/Face"

    def run(self, image, boxes_prompt, padding=16, box_index=0,
            normalize_size=True, target_long_side=1008):
        import torch
        import torch.nn.functional as F
        B, H, W, C = image.shape
        boxes = boxes_prompt["boxes"].get("boxes", [])

        if not boxes or box_index >= len(boxes):
            meta = {"x1": 0, "y1": 0, "x2": W, "y2": H,
                    "orig_w": W, "orig_h": H, "scale": 1.0,
                    "resized_w": W, "resized_h": H}
            return (image, meta)

        cx, cy, bw, bh = boxes[box_index]
        x1 = max(0, int((cx - bw / 2) * W) - padding)
        y1 = max(0, int((cy - bh / 2) * H) - padding)
        x2 = min(W, int((cx + bw / 2) * W) + padding)
        y2 = min(H, int((cy + bh / 2) * H) + padding)

        cropped = image[:, y1:y2, x1:x2, :]   # [B, ch, cw, C]
        crop_h, crop_w = y2 - y1, x2 - x1

        scale = 1.0
        out_h, out_w = crop_h, crop_w

        if normalize_size:
            scale = target_long_side / max(crop_h, crop_w)
            if scale != 1.0:
                out_h = round(crop_h * scale)
                out_w = round(crop_w * scale)
                # interpolate expects [B, C, H, W]
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
        print(f"[SAMheraCropByBox] crop=[{x1},{y1},{x2},{y2}] {crop_w}x{crop_h}"
              f" → {out_w}x{out_h} (scale={scale:.3f})")
        return (cropped, meta)


# =============================================================================
# SAMheraPasteBackMask
# =============================================================================

class SAMheraPasteBackMask:

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
    CATEGORY      = "SAMhera/Face"

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

        print(f"[SAMheraPasteBackMask] {N} masks -> {orig_w}x{orig_h}")
        return (full,)


# =============================================================================
# SAMheraAddFramePrompt
# =============================================================================

class SAMheraAddFramePrompt:

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
    CATEGORY = "SAMhera"

    def add_frame_prompt(self, video_state, prompt_mode, frame_idx, obj_id,
                         positive_points=None, negative_points=None,
                         positive_boxes=None, negative_boxes=None):

        import importlib.util, os as _os
        _base = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), "..", "..", "ComfyUI-SAM3", "nodes", "video_state.py"))
        if not _os.path.exists(_base):
            raise ImportError(f"[SAMheraAddFramePrompt] video_state.py not found at {_base}")
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
            print(f"[SAMheraAddFramePrompt] {len(all_points)} points at frame {frame_idx}")

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
                "api":        ("SAMHERA_API",),
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
    CATEGORY      = "SAMhera/Face"

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
            print(f"[VLMFacePartsBBox] Parse error: {e}"); data = {}

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
                "api":            ("SAMHERA_API",),
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
    CATEGORY      = "SAMhera/Face"

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
            print(f"[VLMFacePrecisePoints] Parse error: {e} — fallback to full crop")
            x1, y1, x2, y2 = 0, 0, cW, cH
            fg_raw = [[cW//2, cH//2]]
            bg_raw = []

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

        # Normalize points back to full image
        def _norm_pts(pts_raw, label_val):
            pts, lbls = [], []
            for pt in pts_raw:
                px, py = pt[0], pt[1]
                # Handle if Gemini returns normalized [0-1] instead of pixels
                # Gemini uses 0-1000 scale; values <= 1.5 are already normalized 0-1
                abs_x = (px / 1000 * cW) if px > 1.5 else (px * cW)
                abs_y = (py / 1000 * cH) if py > 1.5 else (py * cH)
                nx = max(0.0, min(1.0, (abs_x + crop_x1) / W))
                ny = max(0.0, min(1.0, (abs_y + crop_y1) / H))
                pts.append([nx, ny])
                lbls.append(label_val)
            return {"points": pts, "labels": lbls}

        positive_points = _norm_pts(fg_raw, 1)
        negative_points = _norm_pts(bg_raw, 0)

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
                                       └─ SAMheraPasteBackMask → full-size mask

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
                "api":           ("SAMHERA_API",),
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
    CATEGORY      = "SAMhera/Face"

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
            print(f"[VLMFaceRegion] Stage1 parse error: {e}")
            bx1, by1, bx2, by2 = 0, 0, sW, sH

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
            print(f"[VLMFaceRegion] Stage2 parse error: {e}")
            fg_raw = [[cW//2, cH//2]]; bg_raw = []

        fg_raw = fg_raw[:num_fg_points]
        bg_raw = bg_raw[:num_bg_points]

        # box_prompt covers the whole crop (SAM3 works on cropped_image)
        box_prompt   = {"box":   [0.5, 0.5, 1.0, 1.0], "label": True}
        boxes_prompt = {"boxes": [[0.5, 0.5, 1.0, 1.0]], "labels": [True]}

        def _to_crop_norm(pts_raw, label_val):
            pts, lbls = [], []
            for pt in pts_raw:
                px, py = pt[0], pt[1]
                nx = max(0.0, min(1.0, (px / cW) if px > 1.5 else px))
                ny = max(0.0, min(1.0, (py / cH) if py > 1.5 else py))
                pts.append([nx, ny]); lbls.append(label_val)
            return {"points": pts, "labels": lbls}

        positive_points = _to_crop_norm(fg_raw, 1)
        negative_points = _to_crop_norm(bg_raw, 0)

        print(f"[VLMFaceRegion] crop=[{cx1},{cy1},{cx2},{cy2}] "
              f"fg:{len(positive_points['points'])} bg:{len(negative_points['points'])}")

        raw_combined = f"=== Stage1 (bbox) ===\n{raw1}\n=== Stage2 (points) ===\n{raw2}"
        return (cropped_tensor, crop_meta, box_prompt, boxes_prompt,
                positive_points, negative_points, raw_combined)


# =============================================================================
# SAMheraAutoLayer — one Gemini call → up to 8 layer boxes + layer set
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


class SAMheraAutoLayer:

    LAYER_PRESETS_LIST = ["auto", "portrait", "full_body", "product", "custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "api":          ("SAMHERA_API",),
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
        "SAMHERA_LAYER_SET", "STRING", "STRING",
    )
    RETURN_NAMES = (
        "layer_1", "layer_2", "layer_3", "layer_4",
        "layer_5", "layer_6", "layer_7", "layer_8",
        "label_1", "label_2", "label_3", "label_4",
        "label_5", "label_6", "label_7", "label_8",
        "layer_set", "label_list", "raw_response",
    )
    FUNCTION = "run"
    CATEGORY = "SAMhera"

    def run(self, image, api, layer_preset, custom_prompt="",
            num_pos_points=4, num_neg_points=2):
        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        # ── Call 1: Discovery ────────────────────────────────────────
        # Let Gemini freely name what it actually sees, guided by preset category.
        if layer_preset == "auto":
            guidance_line = ""
        elif layer_preset == "custom":
            guidance = custom_prompt.strip() or "any distinct visual elements"
            guidance_line = f"Focus on: {guidance}"
        else:
            preset_guidance = {
                "portrait":  "face skin, hair, eyes, mouth/lips, neck, clothing, accessories, background",
                "full_body": "face/head, hair, upper clothing, lower clothing, shoes, hands/arms, accessories, background",
                "product":   "main product, packaging, labels/text, shadow, props, background",
            }
            guidance_line = f"This is a {layer_preset} image. Focus on: {preset_guidance.get(layer_preset, 'distinct visual regions')}"

        discovery_prompt = (
            (f"{guidance_line}\n\n" if guidance_line else "")
            + "Look at the image and list every distinct visual layer or region you can clearly see. "
            "Give each a SHORT, SPECIFIC label (e.g. 'black turtleneck', 'curly brown hair', 'gold hoop earrings'). "
            "Max 8 layers. Skip anything not clearly visible.\n\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"layers": ["label1", "label2", ...]}'
        )

        raw1 = _call_gemini(pil_img, discovery_prompt, api)
        print(f"[SAMheraAutoLayer] Discovery: {raw1}")

        try:
            data1 = _parse_json(raw1)
            discovered = [l.strip() for l in data1.get("layers", []) if isinstance(l, str) and l.strip()]
            if not discovered:
                raise ValueError("empty layers list")
        except Exception as e:
            print(f"[SAMheraAutoLayer] Discovery parse error: {e} — falling back to preset")
            discovered = LAYER_PRESETS.get(layer_preset, LAYER_PRESETS["portrait"])

        # ── Call 2: Localization + Points ─────────────────────────────
        # Return bbox AND points for every label in one shot, all in original coords.
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
        print(f"[SAMheraAutoLayer] Localize+Points: {raw2}")
        raw = f"=== Discovery ===\n{raw1}\n\n=== Localize+Points ===\n{raw2}"

        try:
            data = _parse_json(raw2)
            layers = data.get("layers", [])[:8]
        except Exception as e:
            print(f"[SAMheraAutoLayer] Localize+Points parse error: {e}"); layers = []

        # ── Build outputs ─────────────────────────────────────────────
        empty_boxes  = {"boxes": [], "labels": []}
        empty_pts    = {"points": [], "labels": []}
        empty_bundle = {"boxes": empty_boxes, "positive": empty_pts, "negative": empty_pts}

        def _norm_pts(pts, label_val):
            result, lbls = [], []
            for pt in pts:
                nx = max(0.0, min(1.0, pt[0] / 1000 if pt[0] > 1.5 else pt[0]))
                ny = max(0.0, min(1.0, pt[1] / 1000 if pt[1] > 1.5 else pt[1]))
                result.append([nx, ny]); lbls.append(label_val)
            return {"points": result, "labels": lbls}

        def _to_bundle(entry):
            x1, y1, x2, y2 = entry["bbox"]
            x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
            cx = (x1n + x2n) / 2; cy = (y1n + y2n) / 2
            boxes = {"boxes": [[cx, cy, x2n - x1n, y2n - y1n]], "labels": [True]}
            pos   = _norm_pts(entry.get("positive", [])[:num_pos_points], 1)
            neg   = _norm_pts(entry.get("negative", [])[:num_neg_points], 0)
            return {"boxes": boxes, "positive": pos, "negative": neg}

        bundles = [_to_bundle(l) for l in layers]
        labels  = [l.get("label", f"layer_{i+1}") for i, l in enumerate(layers)]

        while len(bundles) < 8:
            bundles.append(empty_bundle)
            labels.append("")

        # layer_set keeps SAM3_BOXES_PROMPT for LayerPropagate / LayerSelector
        layer_set  = {lbl: b["boxes"] for lbl, b in zip(labels, bundles) if lbl}
        label_list = "\n".join(f"{i+1}. {lb}" for i, lb in enumerate(labels) if lb)

        print(f"[SAMheraAutoLayer] Detected {len(layers)} layers: {[l for l in labels if l]}")
        return (*bundles, *labels, layer_set, label_list, raw)


# =============================================================================
# SAMheraLayerPropagate — propagate every layer through all video frames
# =============================================================================

class SAMheraLayerPropagate:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "layer_set":    ("SAMHERA_LAYER_SET",),
                "sam3_model":   ("SAM3_MODEL",),
                "frame_idx":    ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Frame index the layer boxes were detected on.",
                }),
            }
        }

    RETURN_TYPES = ("SAMHERA_LAYER_SET",)
    RETURN_NAMES = ("propagated_layer_set",)
    FUNCTION = "run"
    CATEGORY = "SAMhera"

    def _load_sam3_modules(self):
        import importlib.util, os as _os
        sam3_dir = _os.path.normpath(
            _os.path.join(_os.path.dirname(__file__), "..", "..", "ComfyUI-SAM3", "nodes")
        )

        def _load(fname):
            path = _os.path.join(sam3_dir, fname)
            if not _os.path.exists(path):
                raise ImportError(f"[SAMheraLayerPropagate] Not found: {path}")
            spec = importlib.util.spec_from_file_location(
                f"_samhera_sam3_{fname.replace('.py', '')}", path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        vs_mod = _load("video_state.py")
        vn_mod = _load("sam3_video_nodes.py")
        return vs_mod, vn_mod

    def run(self, video_frames, layer_set, sam3_model, frame_idx):
        vs_mod, vn_mod = self._load_sam3_modules()
        create_video_state = vs_mod.create_video_state
        VideoPrompt = vs_mod.VideoPrompt
        SAM3Propagate = vn_mod.SAM3Propagate

        propagator = SAM3Propagate()
        propagated = {}

        for label, boxes_prompt in layer_set.items():
            if not boxes_prompt or not boxes_prompt.get("boxes"):
                print(f"[SAMheraLayerPropagate] Skipping '{label}' — no boxes")
                continue

            print(f"[SAMheraLayerPropagate] Propagating layer: '{label}'")
            try:
                video_state = create_video_state(video_frames)
                cx, cy, bw, bh = boxes_prompt["boxes"][0]
                box_corners = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                prompt = VideoPrompt.create_box(frame_idx, 1, box_corners, is_positive=True)
                video_state = video_state.with_prompt(prompt)
                result = propagator.propagate(sam3_model, video_state)
                propagated[label] = result[0]  # SAM3_VIDEO_MASKS
                print(f"[SAMheraLayerPropagate] '{label}' propagation complete")
            except Exception as e:
                print(f"[SAMheraLayerPropagate] '{label}' failed: {e}")
                propagated[label] = None

        print(f"[SAMheraLayerPropagate] Done. {len(propagated)} layers propagated.")
        return (propagated,)


# =============================================================================
# SAMheraReferenceMatch — find a subject from a reference image in a target frame
# =============================================================================

class SAMheraReferenceMatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "target_frame":    ("IMAGE",),
                "api":             ("SAMHERA_API",),
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
    CATEGORY = "SAMhera"

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
        print(f"[SAMheraReferenceMatch] Raw: {raw}")

        empty = {"boxes": [], "labels": []}
        try:
            data = _parse_json(raw)
            bbox = data.get("bbox")
            if not bbox:
                print(f"[SAMheraReferenceMatch] Subject not found in target frame.")
                return (empty, raw)
            x1, y1, x2, y2 = bbox
            x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
            cx = (x1n + x2n) / 2; cy = (y1n + y2n) / 2
            bw = x2n - x1n;       bh = y2n - y1n
            boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [True]}
            print(f"[SAMheraReferenceMatch] box (cx,cy,w,h): [{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}]")
        except Exception as e:
            print(f"[SAMheraReferenceMatch] Parse error: {e}"); boxes_prompt = empty

        return (boxes_prompt, raw)


# =============================================================================
# SAMheraLayerSelector — extract a single layer from a SAMHERA_LAYER_SET
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


class SAMheraLayerSelector:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_set":  ("SAMHERA_LAYER_SET",),
                "layer_name": ("STRING", {"default": "face", "multiline": False,
                               "tooltip": "Exact label or case-insensitive substring match."}),
            }
        }

    RETURN_TYPES = ("MASK", "SAM3_BOXES_PROMPT", "STRING")
    RETURN_NAMES = ("mask", "boxes_prompt", "available_layers")
    FUNCTION = "run"
    CATEGORY = "SAMhera"

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
            print(f"[SAMheraLayerSelector] '{layer_name}' not found. Available: {available_str}")
            return (empty_mask, empty_boxes, available_str)

        print(f"[SAMheraLayerSelector] Matched '{matched}' for query '{layer_name}'")

        # SAM3_BOXES_PROMPT: dict with "boxes" key  (from SAMheraAutoLayer)
        if isinstance(value, dict) and "boxes" in value:
            print(f"[SAMheraLayerSelector] Type: SAM3_BOXES_PROMPT — returning boxes, empty mask")
            return (empty_mask, value, available_str)

        # SAM3_VIDEO_MASKS: dict with int frame-index keys (from SAMheraLayerPropagate)
        if isinstance(value, dict) and any(isinstance(k, int) for k in value):
            print(f"[SAMheraLayerSelector] Type: SAM3_VIDEO_MASKS — extracting mask tensor")
            try:
                mask = _extract_mask_from_video_masks(value)
                print(f"[SAMheraLayerSelector] Mask shape: {mask.shape}")
                return (mask, empty_boxes, available_str)
            except Exception as e:
                print(f"[SAMheraLayerSelector] Extraction failed: {e}")
                return (empty_mask, empty_boxes, available_str)

        print(f"[SAMheraLayerSelector] '{matched}' has no usable data (None or unknown type)")
        return (empty_mask, empty_boxes, available_str)


# =============================================================================
# SAMheraAddFramePromptBundle — add box + points to video_state in one node
# =============================================================================

class SAMheraAddFramePromptBundle:

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
    CATEGORY      = "SAMhera"

    def run(self, video_state, box_and_point, frame_idx, obj_id):
        import importlib.util, os as _os
        _base = _os.path.normpath(
            _os.path.join(_os.path.dirname(__file__), "..", "..", "ComfyUI-SAM3", "nodes", "video_state.py")
        )
        if not _os.path.exists(_base):
            raise ImportError(f"[SAMheraAddFramePromptBundle] video_state.py not found at {_base}")
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

        print(f"[SAMheraAddFramePromptBundle] frame={frame_idx} obj={obj_id} "
              f"box={'yes' if boxes_prompt and boxes_prompt.get('boxes') else 'no'} "
              f"pts={len(all_points)}")
        return (video_state,)


# =============================================================================
# SAMheraAutoCrop — presence/discovery call + localization call + crop
# =============================================================================

class SAMheraAutoCrop:
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
                "api":   ("SAMHERA_API",),
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
        "SAMHERA_LAYER_SET", "STRING", "STRING",
    )
    RETURN_NAMES = (
        "crop_1", "crop_2", "crop_3", "crop_4",
        "crop_5", "crop_6", "crop_7", "crop_8",
        "label_1", "label_2", "label_3", "label_4",
        "label_5", "label_6", "label_7", "label_8",
        "layer_set", "label_list", "raw_response",
    )
    FUNCTION  = "run"
    CATEGORY  = "SAMhera"

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
        print(f"[SAMheraAutoCrop] Discovery: {raw1}")

        try:
            data1 = _parse_json(raw1)
            discovered = [
                r.strip() for r in data1.get("regions", [])
                if isinstance(r, str) and r.strip()
            ]
            if not discovered:
                raise ValueError("empty regions list")
        except Exception as e:
            print(f"[SAMheraAutoCrop] Discovery parse error: {e}")
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
        print(f"[SAMheraAutoCrop] Localization: {raw2}")
        raw = f"=== Discovery ===\n{raw1}\n\n=== Localization ===\n{raw2}"

        try:
            data2 = _parse_json(raw2)
            regions = data2.get("regions", [])[:8]
        except Exception as e:
            print(f"[SAMheraAutoCrop] Localization parse error: {e}")
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
                print(f"[SAMheraAutoCrop] Crop error for '{entry.get('label', '?')}': {e}")

        while len(crops) < 8:
            crops.append(image)
            labels.append("")

        label_list = "\n".join(f"{i+1}. {lb}" for i, lb in enumerate(labels) if lb)
        print(f"[SAMheraAutoCrop] Cropped {len(regions)} regions: {[l for l in labels if l]}")
        return (*crops[:8], *labels[:8], layer_set, label_list, raw)


# =============================================================================
# Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SAMheraAPIKey":          SAMheraAPIKey,
    "VLMImageTest":           VLMImageTest,
    "VLMtoBBoxAndPoints":     VLMtoBBoxAndPoints,
    "VLMtoBBox":              VLMtoBBox,
    "VLMtoPoints":            VLMtoPoints,
    "VLMtoMultiBBox":         VLMtoMultiBBox,
    "VLMBBoxPreview":         VLMBBoxPreview,
    "VLMDebugPreview":        VLMDebugPreview,
    "SAMheraAddFramePrompt":  SAMheraAddFramePrompt,
    "VLMFacePartsBBox":       VLMFacePartsBBox,
    "VLMFacePrecisePoints":   VLMFacePrecisePoints,
    "VLMFaceRegion":          VLMFaceRegion,
    "SAMheraCropByBox":       SAMheraCropByBox,
    "SAMheraPasteBackMask":   SAMheraPasteBackMask,
    "SAMheraAutoLayer":       SAMheraAutoLayer,
    "SAMheraLayerPropagate":  SAMheraLayerPropagate,
    "SAMheraReferenceMatch":  SAMheraReferenceMatch,
    "SAMheraLayerSelector":   SAMheraLayerSelector,
    "VLMPromptEditor":        VLMPromptEditor,
    "SAMheraAutoCrop":              SAMheraAutoCrop,
    "SAMheraAddFramePromptBundle":  SAMheraAddFramePromptBundle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMheraAPIKey":          "SAMhera API Key",
    "VLMImageTest":           "VLM Image Test (SAMhera)",
    "VLMtoBBoxAndPoints":     "VLM -> BBox + Points (SAMhera)",
    "VLMtoBBox":              "VLM -> BBox (SAMhera)",
    "VLMtoPoints":            "VLM -> Points (SAMhera)",
    "VLMtoMultiBBox":         "VLM -> Multi-BBox (SAMhera)",
    "VLMBBoxPreview":         "VLM BBox Preview (SAMhera)",
    "VLMDebugPreview":        "VLM Debug Preview (SAMhera)",
    "SAMheraAddFramePrompt":  "Add Frame Prompt [SAMhera]",
    "VLMFacePartsBBox":       "VLM -> Face Parts BBox [SAMhera]",
    "VLMFacePrecisePoints":   "VLM -> Face Precise Points [SAMhera]",
    "VLMFaceRegion":          "VLM Face Region [SAMhera]",
    "SAMheraCropByBox":       "Crop by Box [SAMhera]",
    "SAMheraPasteBackMask":   "Paste Back Mask [SAMhera]",
    "SAMheraAutoLayer":       "Auto Layer Detect [SAMhera]",
    "SAMheraLayerPropagate":  "Layer Propagate [SAMhera]",
    "SAMheraReferenceMatch":  "Reference Match [SAMhera]",
    "SAMheraLayerSelector":   "Layer Selector [SAMhera]",
    "VLMPromptEditor":        "VLM Prompt Editor [SAMhera]",
    "SAMheraAutoCrop":              "Auto Crop [SAMhera]",
    "SAMheraAddFramePromptBundle":  "Add Frame Prompt Bundle [SAMhera]",
}
