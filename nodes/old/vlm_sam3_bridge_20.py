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
    """Enter API key, provider, and model once — connect to all SAMhera nodes via the api slot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":    ("STRING", {"default": "", "multiline": False}),
                "provider":   (["gemini", "openai"], {"default": "gemini"}),
                "model_name": (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gpt-4o", "gpt-4o-mini"], {"default": "gemini-2.5-pro"}),
            }
        }

    RETURN_TYPES  = ("SAMHERA_API",)
    RETURN_NAMES  = ("api",)
    FUNCTION      = "run"
    CATEGORY      = "SAMhera"

    def run(self, api_key, provider, model_name):
        return ({"api_key": api_key, "provider": provider, "model_name": model_name},)


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
            if api.get("model_name"): model_name = api["model_name"]

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

    def run(self, image, target_description, num_pos_points, num_neg_points,
            bbox_context=None, few_shot_examples="", api=None):

        api_key = api["api_key"] if api else ""
        provider = api["provider"] if api else "gemini"
        model_name = api.get("model_name", "gemini-2.5-pro") if api else "gemini-2.5-pro"

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
            if api.get("model_name"): model_name = api["model_name"]

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

    RETURN_TYPES  = ("SAMHERA_API",)
    RETURN_NAMES  = ("api",)
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
        return ({"api_key": api_key, "provider": provider, "model_name": model_name},)


# =============================================================================
# Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SAMheraAPIKey":   SAMheraAPIKey,
    "VLMtoBBox":       VLMtoBBox,
    "VLMtoPoints":     VLMtoPoints,
    "VLMtoMultiBBox":  VLMtoMultiBBox,
    "VLMBBoxPreview":  VLMBBoxPreview,
    "VLMDebugPreview": VLMDebugPreview,
    "VLMImageTest":    VLMImageTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMheraAPIKey":   "SAMhera API Key",
    "VLMtoBBox":       "VLM -> BBox (SAMhera)",
    "VLMtoPoints":     "VLM -> Points (SAMhera)",
    "VLMtoMultiBBox":  "VLM -> Multi-BBox (SAMhera)",
    "VLMBBoxPreview":  "VLM BBox Preview (SAMhera)",
    "VLMDebugPreview": "VLM Debug Preview (SAMhera)",
    "VLMImageTest":    "VLM Image Test (SAMhera)",
}


# =============================================================================
# Node 7 -- SAMheraReload
#   Hot-reloads vlm_sam3_bridge.py without restarting ComfyUI
# =============================================================================

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


NODE_CLASS_MAPPINGS["SAMheraReload"] = SAMheraReload
NODE_DISPLAY_NAME_MAPPINGS["SAMheraReload"] = "SAMhera Reload"
