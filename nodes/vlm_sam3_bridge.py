"""
VLM -> SAM3 Bridge Node
Calls a VLM (Gemini / OpenAI) to auto-generate bbox or point prompts,
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
import base64
import io
import numpy as np
from PIL import Image


# =============================================================================
# SAMheraAPIKey — output api_key + provider as a single SAMHERA_API type
# =============================================================================

class SAMheraAPIKey:
    """Enter API key and provider once, connect to all SAMhera nodes via the api slot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":  ("STRING",   {"default": "", "multiline": False}),
                "provider": (["gemini", "openai"], {"default": "gemini"}),
            }
        }

    RETURN_TYPES  = ("SAMHERA_API",)
    RETURN_NAMES  = ("api",)
    FUNCTION      = "run"
    CATEGORY      = "SAMhera"

    def run(self, api_key, provider):
        return ({"api_key": api_key, "provider": provider},)


# -- helpers ------------------------------------------------------------------

def _tensor_to_pil(image_tensor):
    arr = (image_tensor[0].numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    return json.loads(text)

def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _maybe_normalize_corners(x1, y1, x2, y2, W, H):
    if any(v > 2.0 for v in [x1, y1, x2, y2]):
        return x1/W, y1/H, x2/W, y2/H
    return x1, y1, x2, y2


# -- Gemini backend -----------------------------------------------------------

def _call_gemini(pil_img, prompt, api_key, model_name="gemini-3.1-pro-preview"):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    client = genai.Client(api_key=api_key)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ]
    )
    return response.text


# -- OpenAI backend -----------------------------------------------------------

def _call_openai(pil_img, prompt, api_key, model_name="gpt-4o"):
    try:
        import openai
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_pil_to_b64(pil_img)}"}},
            {"type": "text", "text": prompt},
        ]}],
        response_format={"type": "json_object"},
        max_tokens=512,
    )
    return resp.choices[0].message.content


def _call_vlm(pil_img, prompt, api_key, provider, model_name):
    if provider == "gemini":
        return _call_gemini(pil_img, prompt, api_key, model_name)
    elif provider == "openai":
        return _call_openai(pil_img, prompt, api_key, model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# Node 1 -- VLMtoBBox
# =============================================================================

class VLMtoBBox:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "provider": (["gemini", "openai"], {"default": "gemini"}),
                "model_name": (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gpt-4o", "gpt-4o-mini"], {"default": "gemini-2.5-pro"}),
                "target_description": ("STRING", {"default": "the main subject", "multiline": False}),
                "is_positive": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "api": ("SAMHERA_API",),
                "few_shot_examples": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": 'Optional few-shot context. Example:\n"Good output": {"bbox": [120, 80, 400, 350], "label": "cat"}\nTight box around the cat, NOT including the background.',
                }),
                "confidence_hint": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT", "STRING")
    RETURN_NAMES  = ("box_prompt", "boxes_prompt", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "SAMhera"

    def run(self, image, api_key, provider, model_name,
            target_description, is_positive, few_shot_examples="", confidence_hint=1.0, api=None):
        if api is not None:
            api_key = api["api_key"]; provider = api["provider"]

        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = "\n\nHere are examples of good outputs for reference:\n" + few_shot_examples.strip() + "\n\nNow apply the same quality to the new image."

        prompt = (
            f"Locate: {target_description}\n"
            f"Image dimensions: {W}x{H} pixels.\n"
            "Return ONLY valid JSON (no markdown) with this exact schema:\n"
            '{"bbox": [x1, y1, x2, y2], "label": "<short name>"}\n'
            "Use pixel coordinates. x1<x2, y1<y2. Make the box tight around the object."
            + few_shot_block
        )

        raw = _call_vlm(pil_img, prompt, api_key, provider, model_name)
        print(f"[VLMtoBBox] Raw response: {raw}")

        try:
            data = _parse_json(raw)
            x1, y1, x2, y2 = data["bbox"]
        except Exception as e:
            print(f"[VLMtoBBox] Parse error: {e} -- using full-image fallback")
            x1, y1, x2, y2 = 0, 0, W, H

        x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
        cx = (x1n + x2n) / 2
        cy = (y1n + y2n) / 2
        bw = x2n - x1n
        bh = y2n - y1n

        box_prompt   = {"box": [cx, cy, bw, bh], "label": is_positive}
        boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [is_positive]}

        print(f"[VLMtoBBox] box normalized (cx,cy,w,h): [{cx:.3f}, {cy:.3f}, {bw:.3f}, {bh:.3f}]")
        return (box_prompt, boxes_prompt, raw)


# =============================================================================
# Node 2 -- VLMtoPoints
# =============================================================================

class VLMtoPoints:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "provider": (["gemini", "openai"], {"default": "gemini"}),
                "model_name": (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gpt-4o", "gpt-4o-mini"], {"default": "gemini-2.5-pro"}),
                "target_description": ("STRING", {"default": "the main subject", "multiline": False}),
                "num_pos_points": ("INT", {"default": 6, "min": 1, "max": 12}),
                "num_neg_points": ("INT", {"default": 3, "min": 0, "max": 6}),
            },
            "optional": {
                "api": ("SAMHERA_API",),
                "bbox_context": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Connect boxes_prompt from VLM->BBox to constrain point search area"
                }),
                "few_shot_examples": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": 'Optional few-shot guidance. Example:\n"Good output":\n{"foreground": [[240,180],[300,200]], "background": [[10,10]]}\nPoints should be ON the object body, not edges.',
                }),
            }
        }

    RETURN_TYPES  = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "STRING")
    RETURN_NAMES  = ("positive_points", "negative_points", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "SAMhera"

    def run(self, image, api_key, provider, model_name,
            target_description, num_pos_points, num_neg_points,
            bbox_context=None, few_shot_examples="", api=None):

        if api is not None:
            api_key = api["api_key"]; provider = api["provider"]

        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = "\n\nAdditional guidance:\n" + few_shot_examples.strip()

        # Build prompt — if bbox_context given, image will be cropped so coords are relative to crop
        if bbox_context is not None and len(bbox_context.get("boxes", [])) > 0:
            size_note = "This image is already cropped tightly around the target object."
        else:
            size_note = f"Image: {W}x{H} pixels."

        prompt = (
            f"Segment: {target_description}\n"
            f"{size_note}\n"
            f"Place {num_pos_points} positive point(s) ON the {target_description} — "
            "spread across its full extent, deep inside each part, never on edges.\n"
            f"Place {num_neg_points} negative point(s) on anything that is NOT {target_description} — "
            "near its boundary, in visually similar regions.\n\n"
            "Return ONLY this JSON (no explanation, no markdown):\n"
            '{"positive": [[x, y], ...], "negative": [[x, y], ...]}'
            + few_shot_block
        )

        # Crop image to bbox so Gemini can only see (and click) inside the object region
        crop_x1, crop_y1, crop_w, crop_h = 0, 0, W, H
        send_img = pil_img

        if bbox_context is not None and len(bbox_context.get("boxes", [])) > 0:
            b = bbox_context["boxes"][0]
            cx_n, cy_n, bw_n, bh_n = b
            cx1 = max(0, int((cx_n - bw_n/2) * W) - 10)
            cy1 = max(0, int((cy_n - bh_n/2) * H) - 10)
            cx2 = min(W, int((cx_n + bw_n/2) * W) + 10)
            cy2 = min(H, int((cy_n + bh_n/2) * H) + 10)
            send_img = pil_img.crop((cx1, cy1, cx2, cy2))
            crop_x1, crop_y1 = cx1, cy1
            crop_w, crop_h = cx2 - cx1, cy2 - cy1
            print(f"[VLMtoPoints] Cropped to bbox: [{cx1},{cy1},{cx2},{cy2}], crop size: {send_img.size}")

        print(f"[VLMtoPoints] Sending image size: {send_img.size}")
        raw = _call_vlm(send_img, prompt, api_key, provider, model_name)
        print(f"[VLMtoPoints] Raw response: {raw}")

        try:
            data = _parse_json(raw)
            pos_raw = data.get("positive", [[crop_w//2, crop_h//2]])
            neg_raw = data.get("negative", [])
        except Exception as e:
            print(f"[VLMtoPoints] Parse error: {e} -- using center fallback")
            pos_raw = [[crop_w//2, crop_h//2]]
            neg_raw = []

        pos_raw = pos_raw[:num_pos_points]
        neg_raw = neg_raw[:num_neg_points]

        def to_norm_points(pts_raw, label_val):
            pts, lbls = [], []
            for pt in pts_raw:
                x, y = pt[0], pt[1]
                # pts are relative to cropped image (pixels) -> map back to full image normalized
                abs_x = (x * crop_w + crop_x1) if x <= 1.5 else (x + crop_x1)
                abs_y = (y * crop_h + crop_y1) if y <= 1.5 else (y + crop_y1)
                nx = max(0.0, min(1.0, abs_x / W))
                ny = max(0.0, min(1.0, abs_y / H))
                pts.append([nx, ny])
                lbls.append(label_val)
            return {"points": pts, "labels": lbls}

        positive_points = to_norm_points(pos_raw, 1)
        negative_points = to_norm_points(neg_raw, 0)

        print(f"[VLMtoPoints] pos ({len(positive_points['points'])}): {positive_points['points']}")
        print(f"[VLMtoPoints] neg ({len(negative_points['points'])}): {negative_points['points']}")
        return (positive_points, negative_points, raw)


# =============================================================================
# Node 3 -- VLMtoMultiBBox
# =============================================================================

class VLMtoMultiBBox:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "provider": (["gemini", "openai"], {"default": "gemini"}),
                "model_name": (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gpt-4o", "gpt-4o-mini"], {"default": "gemini-2.5-pro"}),
                "target_description": ("STRING", {"default": "all bags", "multiline": False}),
                "max_objects": ("INT", {"default": 3, "min": 1, "max": 5}),
            },
            "optional": {
                "api": ("SAMHERA_API",),
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

    def run(self, image, api_key, provider, model_name,
            target_description, max_objects, few_shot_examples="", api=None):
        if api is not None:
            api_key = api["api_key"]; provider = api["provider"]

        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = "\n\nReference examples:\n" + few_shot_examples.strip()

        prompt = (
            f"Detect: {target_description}\n"
            f"Image: {W}x{H} pixels. Find up to {max_objects} instances.\n"
            "Return ONLY valid JSON:\n"
            '{"objects": [{"bbox": [x1,y1,x2,y2], "label": "name"}, ...]}\n'
            "Pixel coordinates, tight boxes, sorted by confidence descending."
            + few_shot_block
        )

        raw = _call_vlm(pil_img, prompt, api_key, provider, model_name)
        print(f"[VLMtoMultiBBox] Raw: {raw}")

        try:
            data    = _parse_json(raw)
            objects = data.get("objects", [])[:max_objects]
        except Exception as e:
            print(f"[VLMtoMultiBBox] Parse error: {e}")
            objects = []

        def obj_to_boxes_prompt(obj):
            x1, y1, x2, y2 = obj["bbox"]
            x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
            cx = (x1n + x2n) / 2; cy = (y1n + y2n) / 2
            bw = x2n - x1n;       bh = y2n - y1n
            return {"boxes": [[cx, cy, bw, bh]], "labels": [True]}

        empty = {"boxes": [], "labels": []}
        box_outputs = [obj_to_boxes_prompt(obj) for obj in objects]
        while len(box_outputs) < 5:
            box_outputs.append(empty)

        all_boxes = {
            "boxes":  [b for bp in box_outputs for b in bp["boxes"]],
            "labels": [l for bp in box_outputs for l in bp["labels"]],
        }

        print(f"[VLMtoMultiBBox] Detected {len(objects)} objects")
        return (*box_outputs, all_boxes, raw)


# =============================================================================
# Node 4 -- VLMBBoxPreview
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
# Node 5 -- VLMDebugPreview
# =============================================================================

class VLMDebugPreview:
    """
    All-in-one debug overlay.
    - boxes_prompt  -> colored rectangles with index
    - positive_points -> green filled circles (fg)
    - negative_points -> red circles with X (bg)
    All inputs optional.
    """

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
                "line_width":   ("INT",     {"default": 3,    "min": 1, "max": 10}),
                "point_radius": ("INT",     {"default": 8,    "min": 2, "max": 30}),
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

        if boxes_prompt is not None:
            for i, box in enumerate(boxes_prompt.get("boxes", [])):
                cx, cy, bw, bh = box
                x1 = int((cx-bw/2)*W); y1 = int((cy-bh/2)*H)
                x2 = int((cx+bw/2)*W); y2 = int((cy+bh/2)*H)
                color = self.BBOX_COLORS[i % len(self.BBOX_COLORS)]
                draw.rectangle([x1,y1,x2,y2], outline=color, width=line_width)
                if show_labels:
                    draw.rectangle([x1, max(0,y1-18), x1+28, y1], fill=color)
                    draw.text((x1+3, max(0,y1-16)), f"#{i+1}", fill=(255,255,255))

        if positive_points is not None:
            for i, pt in enumerate(positive_points.get("points", [])):
                px = int(pt[0]*W); py = int(pt[1]*H)
                draw.ellipse([px-r-2,py-r-2,px+r+2,py+r+2], fill=(255,255,255))
                draw.ellipse([px-r,py-r,px+r,py+r], fill=(50,210,50))
                draw.ellipse([px-2,py-2,px+2,py+2], fill=(255,255,255))
                if show_labels:
                    draw.text((px+r+4, py-6), f"fg{i+1}", fill=(50,210,50))

        if negative_points is not None:
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
# Node 6 -- VLMImageTest
#   Asks VLM "what do you see?" to verify image is being received correctly
# =============================================================================

class VLMImageTest:
    """
    Debug node: verifies VLM is receiving the image correctly.
    Outputs api (SAMHERA_API) and model_name — connect to other SAMhera nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "api_key":  ("STRING", {"default": "", "multiline": False}),
                "provider": (["gemini", "openai"], {"default": "gemini"}),
                "model_name": (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gpt-4o", "gpt-4o-mini"], {"default": "gemini-2.5-pro"}),
            },
        }

    RETURN_TYPES  = ("STRING", "SAMHERA_API", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("description", "api", "api_key", "provider", "model_name")
    FUNCTION      = "run"
    CATEGORY      = "SAMhera"
    OUTPUT_NODE   = True

    def run(self, image, api_key="", provider="gemini", model_name="gemini-2.5-pro"):
        pil_img = _tensor_to_pil(image)
        print(f"[VLMImageTest] Image size: {pil_img.size}, mode: {pil_img.mode}")

        prompt = (
            "Describe exactly what you see in this image in detail. "
            "List every object you can identify and their approximate positions "
            "(e.g. top-left, center, bottom-right). "
            "Be specific about colors, sizes, and locations."
        )

        raw = _call_vlm(pil_img, prompt, api_key, provider, model_name)
        print(f"[VLMImageTest] Response: {raw}")
        return (raw, {"api_key": api_key, "provider": provider}, api_key, provider, model_name)


# =============================================================================
# Registration
# =============================================================================


# =============================================================================
# VLMtoBBoxAndPoints — single API call: bbox + points in one shot
# =============================================================================

class VLMtoBBoxAndPoints:
    """
    Single VLM call returning bbox AND points together.
    Consistent coordinates — same reasoning context for both.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_description": ("STRING", {"default": "the main subject", "multiline": False}),
                "num_pos_points": ("INT", {"default": 6, "min": 1, "max": 12}),
                "num_neg_points": ("INT", {"default": 3, "min": 0, "max": 6}),
                "is_positive": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "api": ("SAMHERA_API",),
                "few_shot_examples": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES  = ("SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "STRING")
    RETURN_NAMES  = ("box_prompt", "boxes_prompt", "positive_points", "negative_points", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "SAMhera"

    def run(self, image, target_description, num_pos_points, num_neg_points,
            is_positive, api=None, few_shot_examples=""):

        api_key    = api["api_key"]                           if api else ""
        provider   = api["provider"]                          if api else "gemini"
        model_name = api.get("model_name", "gemini-2.5-pro")  if api else "gemini-2.5-pro"

        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = "\n\nReference examples:\n" + few_shot_examples.strip()

        prompt = (
            f"Segment: {target_description}\n"
            f"Image: {W}x{H} pixels.\n\n"
            "1. Draw a tight bounding box around the target.\n"
            f"2. Place {num_pos_points} positive point(s) ON the {target_description} — "
            "spread across its full extent, deep inside, never on edges.\n"
            f"3. Place {num_neg_points} negative point(s) on anything that is NOT {target_description} — "
            "near its boundary, visually similar regions.\n\n"
            "Return ONLY this JSON (no explanation, no markdown):\n"
            '{"bbox": [x1, y1, x2, y2], "positive": [[x, y], ...], "negative": [[x, y], ...]}' 
            + few_shot_block
        )

        raw = _call_vlm(pil_img, prompt, api_key, provider, model_name)
        print(f"[VLMtoBBoxAndPoints] Raw: {raw}")

        try:
            data = _parse_json(raw)
            x1, y1, x2, y2 = data["bbox"]
            pos_raw = data.get("positive", [[W//2, H//2]])
            neg_raw = data.get("negative", [])
        except Exception as e:
            print(f"[VLMtoBBoxAndPoints] Parse error: {e} — using fallbacks")
            x1, y1, x2, y2 = 0, 0, W, H
            pos_raw = [[W//2, H//2]]
            neg_raw = []

        pos_raw = pos_raw[:num_pos_points]
        neg_raw = neg_raw[:num_neg_points]

        # bbox → normalized center format
        x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
        cx = (x1n + x2n) / 2;  cy = (y1n + y2n) / 2
        bw = x2n - x1n;        bh = y2n - y1n

        box_prompt   = {"box":   [cx, cy, bw, bh], "label": is_positive}
        boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [is_positive]}

        # points → normalized
        def to_norm(pts, label_val):
            result, lbls = [], []
            for pt in pts:
                nx = max(0.0, min(1.0, pt[0] / W if pt[0] > 1.5 else pt[0]))
                ny = max(0.0, min(1.0, pt[1] / H if pt[1] > 1.5 else pt[1]))
                result.append([nx, ny]); lbls.append(label_val)
            return {"points": result, "labels": lbls}

        positive_points = to_norm(pos_raw, 1)
        negative_points = to_norm(neg_raw, 0)

        print(f"[VLMtoBBoxAndPoints] box: [{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}]")
        print(f"[VLMtoBBoxAndPoints] pos ({len(positive_points['points'])}): {positive_points['points']}")
        print(f"[VLMtoBBoxAndPoints] neg ({len(negative_points['points'])}): {negative_points['points']}")

        return (box_prompt, boxes_prompt, positive_points, negative_points, raw)



# =============================================================================
# SAMheraCropByBox
# =============================================================================

class SAMheraCropByBox:
    """
    Crops an IMAGE to the region defined by a SAM3_BOXES_PROMPT.
    Outputs CROP_META (original size + crop coords) for SAMheraPasteBackMask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "boxes_prompt": ("SAM3_BOXES_PROMPT",),
            },
            "optional": {
                "padding": ("INT", {
                    "default": 16, "min": 0, "max": 128,
                    "tooltip": "Extra pixels around the box before cropping."
                }),
                "box_index": ("INT", {
                    "default": 0, "min": 0, "max": 4,
                    "tooltip": "Which box to use if boxes_prompt contains multiple."
                }),
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
            print(f"[SAMheraCropByBox] No box at index {box_index}, returning full image")
            crop_meta = {"x1": 0, "y1": 0, "x2": W, "y2": H, "orig_w": W, "orig_h": H}
            return (image, crop_meta)

        cx, cy, bw, bh = boxes[box_index]
        x1 = max(0,   int((cx - bw / 2) * W) - padding)
        y1 = max(0,   int((cy - bh / 2) * H) - padding)
        x2 = min(W,   int((cx + bw / 2) * W) + padding)
        y2 = min(H,   int((cy + bh / 2) * H) + padding)

        print(f"[SAMheraCropByBox] [{x1},{y1},{x2},{y2}] from {W}x{H}")
        cropped = image[:, y1:y2, x1:x2, :]
        crop_meta = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "orig_w": W, "orig_h": H}
        return (cropped, crop_meta)


# =============================================================================
# SAMheraPasteBackMask
# =============================================================================

class SAMheraPasteBackMask:
    """
    Restores a cropped mask to original frame resolution using CROP_META.
    Masks from SAM3 Propagate on a cropped video -> paste back to full res.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks":     ("MASK",),
                "crop_meta": ("CROP_META",),
            },
            "optional": {
                "feather_px": ("INT", {
                    "default": 0, "min": 0, "max": 32,
                    "tooltip": "Blur radius at crop boundary. 0 = hard edge."
                }),
            }
        }

    RETURN_TYPES  = ("MASK",)
    RETURN_NAMES  = ("full_masks",)
    FUNCTION      = "run"
    CATEGORY      = "SAMhera/Face"

    def run(self, masks, crop_meta, feather_px=0):
        import torch
        import torch.nn.functional as F

        x1 = crop_meta["x1"]; y1 = crop_meta["y1"]
        x2 = crop_meta["x2"]; y2 = crop_meta["y2"]
        orig_w = crop_meta["orig_w"]; orig_h = crop_meta["orig_h"]

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        N, crop_h, crop_w = masks.shape
        expected_h = y2 - y1
        expected_w = x2 - x1

        if crop_h != expected_h or crop_w != expected_w:
            print(f"[SAMheraPasteBackMask] Resizing {crop_h}x{crop_w} -> {expected_h}x{expected_w}")
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(expected_h, expected_w),
                mode="bilinear", align_corners=False
            ).squeeze(1)

        full = torch.zeros((N, orig_h, orig_w), dtype=masks.dtype, device=masks.device)
        full[:, y1:y2, x1:x2] = masks

        if feather_px > 0:
            k = feather_px * 2 + 1
            full = F.avg_pool2d(
                full.unsqueeze(1).float(),
                kernel_size=k, stride=1, padding=feather_px
            ).squeeze(1)
            full = torch.clamp(full, 0.0, 1.0)

        print(f"[SAMheraPasteBackMask] {N} masks -> {orig_w}x{orig_h}, crop=[{x1},{y1},{x2},{y2}]")
        return (full,)


class SAMheraReload:
    """
    Reloads SAMhera node code instantly — no ComfyUI restart needed.
    Save your changes to vlm_sam3_bridge.py, then run this node.
    Note: existing node instances keep old behavior until you re-add them.
    New nodes added after reload will use updated code.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("status",)
    FUNCTION      = "run"
    CATEGORY      = "SAMhera"
    OUTPUT_NODE   = True

    def run(self, reload):
        if not reload:
            return ("Skipped.",)

        import importlib
        import sys

        reloaded, failed = [], []

        for mod_name in list(sys.modules.keys()):
            if "vlm_sam3_bridge" in mod_name:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reloaded.append(mod_name)
                except Exception as e:
                    failed.append(f"{mod_name}: {e}")

        if reloaded:
            try:
                mod = sys.modules[reloaded[0]]
                # Update global ComfyUI node registry
                import nodes as comfy_nodes
                comfy_nodes.NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
                comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
                status = f"✓ Reloaded. Re-add nodes to use updated code."
            except Exception as e:
                status = f"✓ Module reloaded but registry update failed: {e}"
        else:
            status = "Module not in sys.modules — full restart required (first time only)."

        if failed:
            status += f"\n✗ Failed: {', '.join(failed)}"

        print(f"[SAMhera] Reload: {status}")
        return (status,)




# =============================================================================
# SAMheraAddFramePrompt
# Adds point or box prompts to an existing SAM3_VIDEO_STATE at a target frame.
# Chain after SAM3VideoSegmentation to re-anchor tracking mid/end frames,
# reducing propagation drift on long or fast-moving videos.
#
# Chain pattern:
#   SAM3VideoSegmentation (frame 0)
#     -> SAMheraAddFramePrompt (frame N//2)
#     -> SAMheraAddFramePrompt (frame N-1)
#     -> SAM3Propagate
# =============================================================================

class SAMheraAddFramePrompt:
    """
    [SAMhera] Add point or box prompts to an existing SAM3_VIDEO_STATE at a specific frame.

    Use after SAM3VideoSegmentation to re-anchor tracking at mid/end frames,
    reducing propagation drift on long or fast-moving videos.

    Connect VLMtoBBoxAndPoints (or VLMtoPoints) output for each anchor frame,
    then chain the video_state output into SAM3Propagate.

    Requires ComfyUI-SAM3 installed (provides SAM3_VIDEO_STATE type).
    """

    PROMPT_MODES = ["point", "box"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state from SAM3VideoSegmentation or a previous SAMheraAddFramePrompt"
                }),
                "prompt_mode": (cls.PROMPT_MODES, {
                    "default": "point",
                    "tooltip": "point: normalized (x,y) SAM3_POINTS_PROMPT | box: SAM3_BOXES_PROMPT"
                }),
                "frame_idx": ("INT", {
                    "default": 15,
                    "min": 0,
                    "tooltip": "Frame to anchor. For a 30-frame clip: 14=mid, 29=end."
                }),
                "obj_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Object ID to re-anchor. Must match SAM3VideoSegmentation (default=1)."
                }),
            },
            "optional": {
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Foreground points at this frame. Connect from VLMtoBBoxAndPoints or VLMtoPoints."
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Background exclusion points at this frame."
                }),
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Bounding box around the target at this frame. Connect from VLMtoBBox or VLMtoBBoxAndPoints."
                }),
                "negative_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Bounding box region to exclude at this frame."
                }),
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
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        VideoPrompt = _mod.VideoPrompt

        print(f"[SAMheraAddFramePrompt] mode={prompt_mode} frame={frame_idx} obj_id={obj_id}")
        print(f"[SAMheraAddFramePrompt] Prompts before: {len(video_state.prompts)}")

        if prompt_mode == "point":
            all_points, all_labels = [], []

            if positive_points and positive_points.get("points"):
                for pt in positive_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])])
                    all_labels.append(1)

            if negative_points and negative_points.get("points"):
                for pt in negative_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])])
                    all_labels.append(0)

            if not all_points:
                print("[SAMheraAddFramePrompt] Warning: no points provided, returning state unchanged")
                return (video_state,)

            prompt = VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels)
            video_state = video_state.with_prompt(prompt)
            pos_n = all_labels.count(1)
            neg_n = all_labels.count(0)
            print(f"[SAMheraAddFramePrompt] Added {len(all_points)} points ({pos_n} pos, {neg_n} neg) at frame {frame_idx}")

        elif prompt_mode == "box":
            added = False

            if positive_boxes and positive_boxes.get("boxes"):
                cx, cy, w, h = positive_boxes["boxes"][0]
                x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=True)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAMheraAddFramePrompt] Added positive box [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] at frame {frame_idx}")
                added = True

            if negative_boxes and negative_boxes.get("boxes"):
                cx, cy, w, h = negative_boxes["boxes"][0]
                x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=False)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAMheraAddFramePrompt] Added negative box at frame {frame_idx}")
                added = True

            if not added:
                print("[SAMheraAddFramePrompt] Warning: no boxes provided, returning state unchanged")
                return (video_state,)

        print(f"[SAMheraAddFramePrompt] Prompts after: {len(video_state.prompts)}")
        return (video_state,)



FACE_PARTS = ["hair", "face", "neck", "face_neck", "clothing"]

FACE_PART_PROMPTS = {
    "hair":      "The person's hair only — scalp to hairline tips, including all strands. Exclude forehead.",
    "face":      "The person's face skin only — forehead, cheeks, nose, lips, chin. Exclude hair, ears, neck.",
    "neck":      "The person's neck only — below chin to top of collar/shoulders. Exclude face and clothing.",
    "face_neck": "The person's face AND neck combined — from forehead to collar. Exclude hair.",
    "clothing":  "The person's clothing/outfit — shirt, dress, jacket etc. Exclude skin, hair.",
}

class VLMFacePartsBBox:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),
                "api":        ("SAMHERA_API",),
                "model_name": (["gemini-2.5-pro", "gemini-2.5-flash", "gpt-4o"], {"default": "gemini-2.5-pro"}),
                "person_box": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Bounding box of the full person — use VLMtoBBox first. Crops image before sending to VLM for accuracy."
                }),
            },
            "optional": {
                "score_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "VLM confidence below this value returns empty box for that part."}),
                "padding_px": ("INT", {"default": 8, "min": 0, "max": 40,
                    "tooltip": "Pixel padding added around each returned box."}),
            }
        }

    RETURN_TYPES  = (
        "SAM3_BOXES_PROMPT",  # hair
        "SAM3_BOXES_PROMPT",  # face
        "SAM3_BOXES_PROMPT",  # neck
        "SAM3_BOXES_PROMPT",  # face_neck
        "SAM3_BOXES_PROMPT",  # clothing
        "STRING",             # raw_vlm_response
    )
    RETURN_NAMES  = ("hair", "face", "neck", "face_neck", "clothing", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "SAMhera/Face"

    def run(self, image, api, model_name, person_box,
            score_threshold=0.5, padding_px=8):


        api_key  = api["api_key"]
        provider = api["provider"]

        pil_full = _tensor_to_pil(image)
        W, H = pil_full.size

        # Crop to person bounding box for accuracy
        pil_img, crop_x1, crop_y1, crop_w, crop_h = pil_full, 0, 0, W, H
        if person_box and person_box.get("boxes"):
            cx_n, cy_n, bw_n, bh_n = person_box["boxes"][0]
            cx1 = max(0, int((cx_n - bw_n/2) * W) - 20)
            cy1 = max(0, int((cy_n - bh_n/2) * H) - 20)
            cx2 = min(W, int((cx_n + bw_n/2) * W) + 20)
            cy2 = min(H, int((cy_n + bh_n/2) * H) + 20)
            pil_img = pil_full.crop((cx1, cy1, cx2, cy2))
            crop_x1, crop_y1 = cx1, cy1
            crop_w, crop_h = cx2 - cx1, cy2 - cy1

        cW, cH = pil_img.size

        # Build single-call multi-part prompt
        parts_desc = "\n".join(
            f'  "{k}": {v}' for k, v in FACE_PART_PROMPTS.items()
        )
        prompt = (
            f"Image size: {cW}x{cH} pixels (cropped to person region).\n\n"
            "Locate each of the following regions on the person and return a TIGHT bounding box.\n"
            "Use pixel coordinates relative to THIS cropped image.\n\n"
            "Regions:\n" + parts_desc + "\n\n"
            "Return ONLY valid JSON, no markdown:\n"
            "{\n"
            '  "hair":      {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0},\n'
            '  "face":      {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0},\n'
            '  "neck":      {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0},\n'
            '  "face_neck": {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0},\n'
            '  "clothing":  {"bbox": [x1,y1,x2,y2], "confidence": 0.0-1.0}\n'
            "}\n"
            "Rules:\n"
            "- x1<x2, y1<y2, tight around region only\n"
            "- confidence=0.0 if region is not visible or occluded\n"
            "- face and hair must NOT overlap\n"
            "- neck box must be BELOW the chin line"
        )

        raw = _call_vlm(pil_img, prompt, api_key, provider, model_name)
        print(f"[VLMFacePartsBBox] Raw: {raw}")

        try:
            data = _parse_json(raw)
        except Exception as e:
            print(f"[VLMFacePartsBBox] Parse error: {e}")
            data = {}

        empty = {"boxes": [], "labels": []}

        def _to_boxes_prompt(part_key):
            entry = data.get(part_key, {})
            if not entry or not entry.get("bbox"):
                return empty
            conf = float(entry.get("confidence", 1.0))
            if conf < score_threshold:
                print(f"[VLMFacePartsBBox] {part_key} confidence {conf:.2f} < threshold, skipping")
                return empty
            x1, y1, x2, y2 = entry["bbox"]
            # Add padding
            x1 = max(0, x1 - padding_px); y1 = max(0, y1 - padding_px)
            x2 = min(cW, x2 + padding_px); y2 = min(cH, y2 + padding_px)
            # Map back to full image normalized
            ax1 = (x1 + crop_x1) / W; ay1 = (y1 + crop_y1) / H
            ax2 = (x2 + crop_x1) / W; ay2 = (y2 + crop_y1) / H
            cx = (ax1 + ax2) / 2; cy = (ay1 + ay2) / 2
            bw = ax2 - ax1;        bh = ay2 - ay1
            print(f"[VLMFacePartsBBox] {part_key}: conf={conf:.2f} box=[{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}]")
            return {"boxes": [[cx, cy, bw, bh]], "labels": [True]}

        return (
            _to_boxes_prompt("hair"),
            _to_boxes_prompt("face"),
            _to_boxes_prompt("neck"),
            _to_boxes_prompt("face_neck"),
            _to_boxes_prompt("clothing"),
            raw,
        )

# =============================================================================
# Registration — all classes defined above, single dict at bottom
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SAMheraAPIKey":          SAMheraAPIKey,
    "VLMtoBBoxAndPoints":     VLMtoBBoxAndPoints,
    "VLMtoBBox":              VLMtoBBox,
    "VLMtoPoints":            VLMtoPoints,
    "VLMtoMultiBBox":         VLMtoMultiBBox,
    "VLMBBoxPreview":         VLMBBoxPreview,
    "VLMDebugPreview":        VLMDebugPreview,
    "VLMImageTest":           VLMImageTest,
    "SAMheraReload":          SAMheraReload,
    "SAMheraAddFramePrompt":  SAMheraAddFramePrompt,
    "VLMFacePartsBBox":       VLMFacePartsBBox,
    "SAMheraCropByBox":       SAMheraCropByBox,
    "SAMheraPasteBackMask":   SAMheraPasteBackMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMheraAPIKey":          "SAMhera API Key",
    "VLMtoBBoxAndPoints":     "VLM -> BBox + Points (SAMhera)",
    "VLMtoBBox":              "VLM -> BBox (SAMhera)",
    "VLMtoPoints":            "VLM -> Points (SAMhera)",
    "VLMtoMultiBBox":         "VLM -> Multi-BBox (SAMhera)",
    "VLMBBoxPreview":         "VLM BBox Preview (SAMhera)",
    "VLMDebugPreview":        "VLM Debug Preview (SAMhera)",
    "VLMImageTest":           "VLM Image Test (SAMhera)",
    "SAMheraReload":          "SAMhera Reload",
    "SAMheraAddFramePrompt":  "Add Frame Prompt [SAMhera]",
    "VLMFacePartsBBox":       "VLM -> Face Parts BBox [SAMhera]",
    "SAMheraCropByBox":       "Crop by Box [SAMhera]",
    "SAMheraPasteBackMask":   "Paste Back Mask [SAMhera]",
}
