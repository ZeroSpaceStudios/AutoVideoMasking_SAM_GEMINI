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
    if any(v > 2.0 for v in [x1, y1, x2, y2]):
        return x1/W, y1/H, x2/W, y2/H
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
                abs_x = (x * crop_w + crop_x1) if x <= 1.5 else (x + crop_x1)
                abs_y = (y * crop_h + crop_y1) if y <= 1.5 else (y + crop_y1)
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

    RETURN_TYPES  = ("SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "STRING")
    RETURN_NAMES  = ("box_prompt", "boxes_prompt", "positive_points", "negative_points", "raw_vlm_response")
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

        x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
        cx = (x1n+x2n)/2; cy = (y1n+y2n)/2
        bw = x2n-x1n;     bh = y2n-y1n

        box_prompt   = {"box":   [cx, cy, bw, bh], "label": is_positive}
        boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [is_positive]}

        def to_norm(pts, label_val):
            result, lbls = [], []
            for pt in pts:
                nx = max(0.0, min(1.0, pt[0]/W if pt[0] > 1.5 else pt[0]))
                ny = max(0.0, min(1.0, pt[1]/H if pt[1] > 1.5 else pt[1]))
                result.append([nx, ny]); lbls.append(label_val)
            return {"points": result, "labels": lbls}

        positive_points = to_norm(pos_raw, 1)
        negative_points = to_norm(neg_raw, 0)

        print(f"[VLMtoBBoxAndPoints] box:[{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}] pos:{len(positive_points['points'])} neg:{len(negative_points['points'])}")
        return (box_prompt, boxes_prompt, positive_points, negative_points, raw)


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

    def draw(self, image, boxes_prompt=None, positive_points=None,
             negative_points=None, line_width=3, point_radius=8, show_labels=True):
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
                "boxes_prompt": ("SAM3_BOXES_PROMPT",),
            },
            "optional": {
                "padding":   ("INT", {"default": 16, "min": 0, "max": 128}),
                "box_index": ("INT", {"default": 0,  "min": 0, "max": 4}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "CROP_META")
    RETURN_NAMES  = ("cropped_image", "crop_meta")
    FUNCTION      = "run"
    CATEGORY      = "SAMhera/Face"

    def run(self, image, boxes_prompt, padding=16, box_index=0):
        import torch
        B, H, W, C = image.shape
        boxes = boxes_prompt.get("boxes", [])

        if not boxes or box_index >= len(boxes):
            return (image, {"x1": 0, "y1": 0, "x2": W, "y2": H, "orig_w": W, "orig_h": H})

        cx, cy, bw, bh = boxes[box_index]
        x1 = max(0, int((cx-bw/2)*W) - padding)
        y1 = max(0, int((cy-bh/2)*H) - padding)
        x2 = min(W, int((cx+bw/2)*W) + padding)
        y2 = min(H, int((cy+bh/2)*H) + padding)

        print(f"[SAMheraCropByBox] [{x1},{y1},{x2},{y2}] from {W}x{H}")
        return (image[:, y1:y2, x1:x2, :], {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "orig_w": W, "orig_h": H})


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

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        N, crop_h, crop_w = masks.shape
        exp_h, exp_w = y2-y1, x2-x1

        if crop_h != exp_h or crop_w != exp_w:
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
    "SAMheraCropByBox":       SAMheraCropByBox,
    "SAMheraPasteBackMask":   SAMheraPasteBackMask,
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
    "SAMheraCropByBox":       "Crop by Box [SAMhera]",
    "SAMheraPasteBackMask":   "Paste Back Mask [SAMhera]",
}
