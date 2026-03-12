"""
VLM -> SAM3 Bridge Node
Calls a VLM (Gemini / OpenAI) to auto-generate bbox or point prompts,
then outputs native SAM3_BOX_PROMPT / SAM3_POINTS_PROMPT types that wire
directly into SAM3Segmentation or SAM3Grounding.

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
                 model_name: str = "gemini-3.1-pro-preview") -> str:
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
                "model_name": ("STRING", {"default": "gemini-3.1-pro-preview"}),
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
                "model_name": ("STRING", {"default": "gemini-3.1-pro-preview"}),
                "target_description": ("STRING", {
                    "default": "the main subject",
                    "multiline": False,
                }),
                "num_fg_points": ("INT", {"default": 3, "min": 1, "max": 8}),
                "num_bg_points": ("INT", {"default": 1, "min": 0, "max": 4}),
            },
            "optional": {
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
            few_shot_examples=""):

        pil_img = _tensor_to_pil(image)
        W, H = pil_img.size

        few_shot_block = ""
        if few_shot_examples.strip():
            few_shot_block = (
                "\n\nReference examples:\n"
                + few_shot_examples.strip()
                + "\n\nApply the same quality to this image."
            )

        prompt = (
            f"Locate: {target_description}\n"
            f"Image dimensions: {W}x{H} pixels.\n"
            f"Return ONLY valid JSON with exactly {num_fg_points} foreground point(s) "
            f"ON the object and {num_bg_points} background point(s) AWAY from it.\n"
            "Schema:\n"
            '{"foreground": [[x,y], ...], "background": [[x,y], ...]}\n'
            "Use pixel coordinates. Points should be well inside the object, not on edges."
            + few_shot_block
        )

        raw = _call_vlm(pil_img, prompt, api_key, provider, model_name)
        print(f"[VLMtoPoints] Raw response: {raw}")

        try:
            data   = _parse_json(raw)
            fg_raw = data.get("foreground", [[W//2, H//2]])
            bg_raw = data.get("background", [])
        except Exception as e:
            print(f"[VLMtoPoints] Parse error: {e} -- using center fallback")
            fg_raw = [[W//2, H//2]]
            bg_raw = []

        def to_norm_points(pts_raw, label_val):
            pts, lbls = [], []
            for pt in pts_raw:
                x, y = pt[0], pt[1]
                nx = x / W if x > 1.5 else x
                ny = y / H if y > 1.5 else y
                pts.append([nx, ny])
                lbls.append(label_val)
            return {"points": pts, "labels": lbls}

        positive_points = to_norm_points(fg_raw, 1)
        negative_points = to_norm_points(bg_raw, 0)

        print(f"[VLMtoPoints] fg points: {positive_points['points']}")
        print(f"[VLMtoPoints] bg points: {negative_points['points']}")
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
                "model_name": ("STRING", {"default": "gemini-3.1-pro-preview"}),
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
