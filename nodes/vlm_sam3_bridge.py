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


# -- helpers ------------------------------------------------------------------

def _tensor_to_pil(image_tensor):
    """ComfyUI IMAGE tensor [B,H,W,C] -> PIL."""
    arr = (image_tensor[0].numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON robustly."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    return json.loads(text)


def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _maybe_normalize_corners(x1, y1, x2, y2, W, H):
    """Auto-detect pixel vs normalized coords and return normalized corners."""
    if any(v > 2.0 for v in [x1, y1, x2, y2]):   # pixel
        return x1/W, y1/H, x2/W, y2/H
    return x1, y1, x2, y2                          # already normalized


# -- Gemini backend -----------------------------------------------------------

def _call_gemini(pil_img: Image.Image, prompt: str, api_key: str,
                 model_name: str = "gemini-2.5-flash") -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ]
    )
    return response.text


# -- OpenAI backend -----------------------------------------------------------

def _call_openai(pil_img: Image.Image, prompt: str, api_key: str,
                 model_name: str = "gpt-4o") -> str:
    try:
        import openai
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")
    client = openai.OpenAI(api_key=api_key)
    b64 = _pil_to_b64(pil_img)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
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
    """
    Uses a VLM (Gemini / OpenAI) to detect a bounding box and outputs it
    as SAM3_BOX_PROMPT / SAM3_BOXES_PROMPT so it wires natively into
    SAM3Segmentation (box input) or SAM3Grounding (positive_boxes input).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "provider": (["gemini", "openai"], {"default": "gemini"}),
                "model_name": ("STRING", {"default": "gemini-2.5-flash"}),
                "target_description": ("STRING", {
                    "default": "the main subject",
                    "multiline": False,
                }),
                "is_positive": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "few_shot_examples": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": (
                        'Optional few-shot context. Example:\n'
                        '"Good output": {"bbox": [120, 80, 400, 350], "label": "cat"}\n'
                        'Tight box around the cat, NOT including the background.'
                    ),
                }),
                "confidence_hint": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
            }
        }

    RETURN_TYPES  = ("SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT", "STRING")
    RETURN_NAMES  = ("box_prompt", "boxes_prompt", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "SAM3/VLM"

    def run(self, image, api_key, provider, model_name,
            target_description, is_positive,
            few_shot_examples="", confidence_hint=1.0):

        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = (
                "\n\nHere are examples of good outputs for reference:\n"
                + few_shot_examples.strip()
                + "\n\nNow apply the same quality to the new image."
            )

        prompt = (
            f"Locate: {target_description}\n"
            f"Image dimensions: {W}x{H} pixels.\n"
            "Return ONLY valid JSON (no markdown) with this exact schema:\n"
            '{"bbox": [x1, y1, x2, y2], "label": "<short name>"}\n'
            "Use pixel coordinates. x1<x2, y1<y2. "
            "Make the box tight around the object."
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
    """
    Uses a VLM to generate foreground + background point prompts.
    Outputs SAM3_POINTS_PROMPT that wires directly into SAM3Segmentation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "provider": (["gemini", "openai"], {"default": "gemini"}),
                "model_name": ("STRING", {"default": "gemini-2.5-flash"}),
                "target_description": ("STRING", {
                    "default": "the main subject",
                    "multiline": False,
                }),
                "num_fg_points": ("INT", {"default": 6, "min": 1, "max": 12}),
                "num_bg_points": ("INT", {"default": 3, "min": 0, "max": 6}),
            },
            "optional": {
                "bbox_context": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Connect boxes_prompt from VLM->BBox to constrain point search area"
                }),
                "few_shot_examples": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": (
                        'Optional few-shot guidance. Example:\n'
                        '"Good output":\n'
                        '{"foreground": [[240,180],[300,200]], "background": [[10,10]]}\n'
                        'Points should be ON the object body, not edges.'
                    ),
                }),
            }
        }

    RETURN_TYPES  = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "STRING")
    RETURN_NAMES  = ("positive_points", "negative_points", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "SAM3/VLM"

    def run(self, image, api_key, provider, model_name,
            target_description, num_fg_points, num_bg_points,
            bbox_context=None, few_shot_examples=""):

        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = "\n\nAdditional guidance:\n" + few_shot_examples.strip()

        # Build bbox constraint string if provided
        bbox_block = ""
        if bbox_context is not None and len(bbox_context.get("boxes", [])) > 0:
            b = bbox_context["boxes"][0]
            cx, cy, bw, bh = b
            bx1 = int((cx - bw/2) * W); by1 = int((cy - bh/2) * H)
            bx2 = int((cx + bw/2) * W); by2 = int((cy + bh/2) * H)
            bbox_block = (
                f"\nThe object is already located within this bounding box: "
                f"[{bx1}, {by1}, {bx2}, {by2}] (x1,y1,x2,y2 in pixels). "
                f"All foreground points MUST be inside this box. "
                f"Background points should be just outside this box boundary.\n"
            )

        reasoning_prompt = (
            f"You are a precise visual analyst helping a segmentation model.\n"
            f"Task: segment the following from this image: {target_description}\n"
            f"Image size: {W}x{H} pixels.{bbox_block}\n"
            "STEP 1 - Look at the image and identify the target object.\n"
            f"Mentally divide it into {num_fg_points} distinct, spatially separated regions "
            "(e.g. for a person: head, shoulders, chest, arms, waist, legs — "
            "for a car: hood, roof, doors, trunk — "
            "for a building: upper/middle/lower sections — "
            "adapt to whatever the object actually is).\n\n"
            "STEP 2 - For each region, find the pixel coordinate at its CENTER "
            "(well inside the region, far from any edge or boundary).\n\n"
            f"STEP 3 - Identify {num_bg_points} background region(s) that are:\n"
            "  - Just OUTSIDE the object boundary (within 30-60px)\n"
            "  - Visually similar to the object (same lighting, color, texture)\n"
            "  - The hardest areas for a segmentation model to distinguish\n\n"
            "Rules:\n"
            "  - Foreground points must be SPREAD across the full spatial extent of the object\n"
            "  - Never place a point on an edge, boundary, or shadow\n"
            "  - Never cluster multiple points in the same region\n"
            "  - All coordinates must be within image bounds\n\n"
            "Return ONLY this JSON (no explanation, no markdown):\n"
            '{\n  "reasoning": [{"region": "<name>", "center": [x, y]}, ...],\n'
            '  "foreground": [[x, y], ...],\n'
            '  "background": [[x, y], ...]\n}'
            + few_shot_block
        )

        raw = _call_vlm(pil_img, reasoning_prompt, api_key, provider, model_name)
        print(f"[VLMtoPoints] Raw response: {raw}")

        try:
            data = _parse_json(raw)
            fg_raw = data.get("foreground", [[W//2, H//2]])
            bg_raw = data.get("background", [])
            reasoning = data.get("reasoning", [])
            if reasoning:
                print(f"[VLMtoPoints] Parts: {[r.get('part') for r in reasoning]}")
        except Exception as e:
            print(f"[VLMtoPoints] Parse error: {e} -- using center fallback")
            fg_raw = [[W//2, H//2]]
            bg_raw = []

        if len(fg_raw) > num_fg_points:
            fg_raw = fg_raw[:num_fg_points]
        if len(bg_raw) > num_bg_points:
            bg_raw = bg_raw[:num_bg_points]

        def to_norm_points(pts_raw, label_val):
            pts, lbls = [], []
            for pt in pts_raw:
                x, y = pt[0], pt[1]
                nx = max(0.0, min(1.0, x / W if x > 1.5 else x))
                ny = max(0.0, min(1.0, y / H if y > 1.5 else y))
                pts.append([nx, ny])
                lbls.append(label_val)
            return {"points": pts, "labels": lbls}

        positive_points = to_norm_points(fg_raw, 1)
        negative_points = to_norm_points(bg_raw, 0)

        print(f"[VLMtoPoints] fg ({len(positive_points['points'])}): {positive_points['points']}")
        print(f"[VLMtoPoints] bg ({len(negative_points['points'])}): {negative_points['points']}")
        return (positive_points, negative_points, raw)


# =============================================================================
# Node 3 -- VLMtoMultiBBox
#   box_1~5 are SAM3_BOXES_PROMPT so each wires directly into SAM3Segmentation -> box
# =============================================================================

class VLMtoMultiBBox:
    """
    Detects multiple objects via VLM.
    box_1~5: each is SAM3_BOXES_PROMPT -> connect directly to SAM3 Point Segmentation -> box
    all_boxes: all boxes combined -> connect to SAM3 Text Segmentation -> positive_boxes
    Use VLM BBox Preview to visualize before segmenting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "provider": (["gemini", "openai"], {"default": "gemini"}),
                "model_name": ("STRING", {"default": "gemini-2.5-flash"}),
                "target_description": ("STRING", {
                    "default": "all bags",
                    "multiline": False,
                }),
                "max_objects": ("INT", {"default": 3, "min": 1, "max": 5}),
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
    RETURN_NAMES  = (
        "box_1", "box_2", "box_3", "box_4", "box_5",
        "all_boxes", "raw_vlm_response"
    )
    FUNCTION      = "run"
    CATEGORY      = "SAM3/VLM"

    def run(self, image, api_key, provider, model_name,
            target_description, max_objects, few_shot_examples=""):

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
            cx = (x1n + x2n) / 2
            cy = (y1n + y2n) / 2
            bw = x2n - x1n
            bh = y2n - y1n
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
#   Draws SAM3_BOXES_PROMPT onto the image for visual verification
# =============================================================================

class VLMBBoxPreview:
    """
    Draws bounding boxes from SAM3_BOXES_PROMPT onto the image.
    Connect all_boxes (from VLMtoMultiBBox) or boxes_prompt (from VLMtoBBox).
    Use this to verify VLM detections before running SAM3.
    """

    COLORS = [
        (255, 80,  80),
        (80,  220, 80),
        (80,  120, 255),
        (255, 200, 50),
        (200, 80,  255),
    ]

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
    CATEGORY      = "SAM3/VLM"

    def draw(self, image, boxes_prompt, line_width=3, show_index=True):
        import torch
        from PIL import ImageDraw

        pil_img = _tensor_to_pil(image).copy()
        W, H = pil_img.size
        draw = ImageDraw.Draw(pil_img)

        boxes = boxes_prompt.get("boxes", [])

        for i, box in enumerate(boxes):
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)

            color = self.COLORS[i % len(self.COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            if show_index:
                label_text = f"#{i+1}"
                draw.rectangle([x1, y1 - 18, x1 + 28, y1], fill=color)
                draw.text((x1 + 3, y1 - 16), label_text, fill=(255, 255, 255))

        arr = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return (tensor,)


# =============================================================================
# Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "VLMtoBBox":       VLMtoBBox,
    "VLMtoPoints":     VLMtoPoints,
    "VLMtoMultiBBox":  VLMtoMultiBBox,
    "VLMBBoxPreview":  VLMBBoxPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMtoBBox":       "VLM -> BBox  (SAM3)",
    "VLMtoPoints":     "VLM -> Points (SAM3)",
    "VLMtoMultiBBox":  "VLM -> Multi-BBox (SAM3)",
    "VLMBBoxPreview":  "VLM BBox Preview",
}


# =============================================================================
# Node 5 -- VLMDebugPreview
# =============================================================================

class VLMDebugPreview:
    """
    All-in-one debug overlay. Connect any combination of:
    - boxes_prompt  -> colored rectangles with index
    - positive_points -> green filled circles (fg)
    - negative_points -> red circles with X (bg)
    """

    BBOX_COLORS = [
        (255, 80,  80),
        (80,  220, 80),
        (80,  120, 255),
        (255, 200, 50),
        (200, 80,  255),
    ]

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
    CATEGORY      = "SAM3/VLM"

    def draw(self, image, boxes_prompt=None, positive_points=None,
             negative_points=None, line_width=3, point_radius=8, show_labels=True):
        import torch
        from PIL import ImageDraw

        pil_img = _tensor_to_pil(image).copy()
        W, H = pil_img.size
        draw = ImageDraw.Draw(pil_img)

        # Draw boxes
        if boxes_prompt is not None:
            for i, box in enumerate(boxes_prompt.get("boxes", [])):
                cx, cy, bw, bh = box
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)
                color = self.BBOX_COLORS[i % len(self.BBOX_COLORS)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                if show_labels:
                    draw.rectangle([x1, max(0, y1 - 18), x1 + 28, y1], fill=color)
                    draw.text((x1 + 3, max(0, y1 - 16)), f"#{i+1}", fill=(255, 255, 255))

        # Draw positive points (green circle)
        if positive_points is not None:
            for i, pt in enumerate(positive_points.get("points", [])):
                px = int(pt[0] * W)
                py = int(pt[1] * H)
                r = point_radius
                draw.ellipse([px-r-2, py-r-2, px+r+2, py+r+2], fill=(255, 255, 255))
                draw.ellipse([px-r, py-r, px+r, py+r], fill=(50, 210, 50))
                draw.ellipse([px-2, py-2, px+2, py+2], fill=(255, 255, 255))
                if show_labels:
                    draw.text((px + r + 4, py - 6), f"fg{i+1}", fill=(50, 210, 50))

        # Draw negative points (red circle with X)
        if negative_points is not None:
            for i, pt in enumerate(negative_points.get("points", [])):
                px = int(pt[0] * W)
                py = int(pt[1] * H)
                r = point_radius
                draw.ellipse([px-r-2, py-r-2, px+r+2, py+r+2], fill=(255, 255, 255))
                draw.ellipse([px-r, py-r, px+r, py+r], fill=(210, 50, 50))
                draw.line([px-r//2, py-r//2, px+r//2, py+r//2], fill=(255,255,255), width=2)
                draw.line([px+r//2, py-r//2, px-r//2, py+r//2], fill=(255,255,255), width=2)
                if show_labels:
                    draw.text((px + r + 4, py - 6), f"bg{i+1}", fill=(210, 50, 50))

        arr = np.array(pil_img).astype(np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)


NODE_CLASS_MAPPINGS["VLMDebugPreview"] = VLMDebugPreview
NODE_DISPLAY_NAME_MAPPINGS["VLMDebugPreview"] = "VLM Debug Preview"


# =============================================================================
# Node 6 -- VLMBBoxAndPoints
#   Single API call that returns BOTH bbox AND points together
#   Eliminates mismatch between separate BBox and Points nodes
# =============================================================================

class VLMBBoxAndPoints:
    """
    Single VLM call that returns bbox + foreground points + background points.
    Use this instead of separate VLM->BBox and VLM->Points nodes to ensure
    the box and points are consistent with each other.

    Outputs:
      boxes_prompt    -> SAM3 Point Segmentation (box input)
      positive_points -> SAM3 Point Segmentation (positive_points input)
      negative_points -> SAM3 Point Segmentation (negative_points input)
      debug_preview   -> Preview Image (shows box + all points overlaid)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "provider": (["gemini", "openai"], {"default": "gemini"}),
                "model_name": ("STRING", {"default": "gemini-2.5-flash"}),
                "target_description": ("STRING", {
                    "default": "the main subject",
                    "multiline": False,
                }),
                "num_fg_points": ("INT", {"default": 6, "min": 1, "max": 12}),
                "num_bg_points": ("INT", {"default": 3, "min": 0, "max": 6}),
            },
            "optional": {
                "few_shot_examples": ("STRING", {"default": "", "multiline": True}),
                "line_width":   ("INT",     {"default": 3, "min": 1, "max": 10}),
                "point_radius": ("INT",     {"default": 8, "min": 2, "max": 30}),
            }
        }

    RETURN_TYPES  = ("SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "IMAGE", "STRING")
    RETURN_NAMES  = ("boxes_prompt", "positive_points", "negative_points", "debug_preview", "raw_vlm_response")
    FUNCTION      = "run"
    CATEGORY      = "SAM3/VLM"

    # Reuse colors from VLMDebugPreview
    BBOX_COLOR  = (255, 80, 80)
    FG_COLOR    = (50, 210, 50)
    BG_COLOR    = (210, 50, 50)

    def run(self, image, api_key, provider, model_name,
            target_description, num_fg_points, num_bg_points,
            few_shot_examples="", line_width=3, point_radius=8):

        import torch
        from PIL import ImageDraw

        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = "\n\nAdditional guidance:\n" + few_shot_examples.strip()

        prompt = (
            f"You are a precise visual analyst helping a segmentation model.\n"
            f"Task: segment the following from this image: {target_description}\n"
            f"Image size: {W}x{H} pixels.\n\n"
            "Do ALL of the following in ONE response:\n\n"
            "1. BOUNDING BOX: Draw a tight box around the entire target object.\n\n"
            f"2. FOREGROUND POINTS ({num_fg_points} points):\n"
            f"   - Mentally divide the object into {num_fg_points} spatially separated regions\n"
            "   - Place one point at the CENTER of each region (deep inside, never on edges)\n"
            "   - Spread points across the full extent: top/bottom/left/right/center\n"
            "   - Adapt to whatever the object is (person: head/torso/limbs, "
            "car: hood/roof/doors, animal: head/body/legs, etc.)\n\n"
            f"3. BACKGROUND POINTS ({num_bg_points} points):\n"
            "   - Place JUST OUTSIDE the object boundary (within 30-60px)\n"
            "   - Choose the visually hardest regions (same color/texture as object)\n\n"
            "Rules:\n"
            "  - All coordinates in pixels, within image bounds\n"
            "  - Foreground points must NOT cluster - each in a different region\n"
            "  - Never place any point on an edge, shadow, or occlusion boundary\n\n"
            "Return ONLY this JSON (no explanation, no markdown):\n"
            '{\n'
            '  "bbox": [x1, y1, x2, y2],\n'
            '  "reasoning": [{"region": "description", "center": [x, y]}, ...],\n'
            '  "foreground": [[x, y], ...],\n'
            '  "background": [[x, y], ...]\n'
            '}'
            + few_shot_block
        )

        raw = _call_vlm(pil_img, prompt, api_key, provider, model_name)
        print(f"[VLMBBoxAndPoints] Raw: {raw}")

        # Parse
        try:
            data = _parse_json(raw)
            x1, y1, x2, y2 = data["bbox"]
            fg_raw  = data.get("foreground", [[W//2, H//2]])
            bg_raw  = data.get("background", [])
            reasoning = data.get("reasoning", [])
            if reasoning:
                print(f"[VLMBBoxAndPoints] Regions: {[r.get('region') for r in reasoning]}")
        except Exception as e:
            print(f"[VLMBBoxAndPoints] Parse error: {e} -- using fallbacks")
            x1, y1, x2, y2 = 0, 0, W, H
            fg_raw = [[W//2, H//2]]
            bg_raw = []

        # Trim to requested counts
        fg_raw = fg_raw[:num_fg_points]
        bg_raw = bg_raw[:num_bg_points]

        # Build SAM3 outputs
        x1n, y1n, x2n, y2n = _maybe_normalize_corners(x1, y1, x2, y2, W, H)
        cx = (x1n + x2n) / 2
        cy = (y1n + y2n) / 2
        bw = x2n - x1n
        bh = y2n - y1n
        boxes_prompt = {"boxes": [[cx, cy, bw, bh]], "labels": [True]}

        def to_norm(pts, label):
            result = {"points": [], "labels": []}
            for pt in pts:
                nx = max(0.0, min(1.0, pt[0] / W if pt[0] > 1.5 else pt[0]))
                ny = max(0.0, min(1.0, pt[1] / H if pt[1] > 1.5 else pt[1]))
                result["points"].append([nx, ny])
                result["labels"].append(label)
            return result

        positive_points = to_norm(fg_raw, 1)
        negative_points = to_norm(bg_raw, 0)

        print(f"[VLMBBoxAndPoints] bbox px: [{x1},{y1},{x2},{y2}]")
        print(f"[VLMBBoxAndPoints] fg ({len(fg_raw)}): {positive_points['points']}")
        print(f"[VLMBBoxAndPoints] bg ({len(bg_raw)}): {negative_points['points']}")

        # Build debug preview
        preview = pil_img.copy()
        draw = ImageDraw.Draw(preview)

        # Box
        bx1 = int(x1n * W); by1 = int(y1n * H)
        bx2 = int(x2n * W); by2 = int(y2n * H)
        draw.rectangle([bx1, by1, bx2, by2], outline=self.BBOX_COLOR, width=line_width)
        draw.rectangle([bx1, max(0, by1-18), bx1+40, by1], fill=self.BBOX_COLOR)
        draw.text((bx1+3, max(0, by1-16)), "bbox", fill=(255,255,255))

        # Foreground points
        r = point_radius
        for i, pt in enumerate(positive_points["points"]):
            px = int(pt[0] * W); py = int(pt[1] * H)
            draw.ellipse([px-r-2, py-r-2, px+r+2, py+r+2], fill=(255,255,255))
            draw.ellipse([px-r, py-r, px+r, py+r], fill=self.FG_COLOR)
            draw.ellipse([px-2, py-2, px+2, py+2], fill=(255,255,255))
            draw.text((px+r+4, py-6), f"fg{i+1}", fill=self.FG_COLOR)

        # Background points
        for i, pt in enumerate(negative_points["points"]):
            px = int(pt[0] * W); py = int(pt[1] * H)
            draw.ellipse([px-r-2, py-r-2, px+r+2, py+r+2], fill=(255,255,255))
            draw.ellipse([px-r, py-r, px+r, py+r], fill=self.BG_COLOR)
            draw.line([px-r//2, py-r//2, px+r//2, py+r//2], fill=(255,255,255), width=2)
            draw.line([px+r//2, py-r//2, px-r//2, py+r//2], fill=(255,255,255), width=2)
            draw.text((px+r+4, py-6), f"bg{i+1}", fill=self.BG_COLOR)

        arr = np.array(preview).astype(np.float32) / 255.0
        preview_tensor = torch.from_numpy(arr).unsqueeze(0)

        return (boxes_prompt, positive_points, negative_points, preview_tensor, raw)


NODE_CLASS_MAPPINGS["VLMBBoxAndPoints"] = VLMBBoxAndPoints
NODE_DISPLAY_NAME_MAPPINGS["VLMBBoxAndPoints"] = "VLM -> BBox + Points (SAM3)"
