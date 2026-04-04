"""
AVM Gemini Prompt Templates

All prompt construction lives here. Node classes import and call these
functions — keeping prompt engineering separate from node logic.
"""

# ── VLMImageTest ──────────────────────────────────────────────────────────────

DESCRIBE_IMAGE = (
    "Describe exactly what you see in this image. "
    "List every object, their positions and colors."
)

# ── VLMtoBBox ────────────────────────────────────────────────────────────────

def bbox_prompt(target_description, W, H, few_shot_block=""):
    return (
        f"Locate: {target_description}\n"
        f"Image dimensions: {W}x{H} pixels.\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"bbox": [x1, y1, x2, y2], "label": "<short name>"}\n'
        "Pixel coordinates, tight box, x1<x2, y1<y2."
        + few_shot_block
    )

# ── VLMtoPoints ──────────────────────────────────────────────────────────────

def points_prompt(target_description, size_note, num_pos_points, num_neg_points, few_shot_block=""):
    return (
        f"Segment: {target_description}\n{size_note}\n"
        f"Place {num_pos_points} positive point(s) ON the {target_description}"
        " — spread across, deep inside, never on edges.\n"
        f"Place {num_neg_points} negative point(s) on anything NOT {target_description}"
        " — near boundary.\n"
        "Return ONLY JSON:\n"
        '{"positive": [[x, y], ...], "negative": [[x, y], ...]}'
        + few_shot_block
    )

# ── VLMtoMultiBBox ───────────────────────────────────────────────────────────

def multi_bbox_prompt(target_description, W, H, max_objects, few_shot_block=""):
    return (
        f"Detect: {target_description}\n"
        f"Image: {W}x{H} px. Find up to {max_objects} instances.\n"
        "Return ONLY JSON:\n"
        '{"objects": [{"bbox": [x1,y1,x2,y2], "label": "name"}, ...]}\n'
        "Pixel coords, tight boxes, sorted by confidence."
        + few_shot_block
    )

# ── VLMtoBBoxAndPoints / VLMPromptEditor ─────────────────────────────────────

def bbox_and_points_prompt(target_description, W, H, num_pos_points, num_neg_points, few_shot_block=""):
    return (
        f"Segment: {target_description}\nImage: {W}x{H} pixels.\n\n"
        "1. Tight bounding box around the target.\n"
        f"2. {num_pos_points} positive point(s) ON the {target_description}"
        " — spread across, deep inside, never on edges.\n"
        f"3. {num_neg_points} negative point(s) on anything NOT {target_description}"
        " — near boundary.\n\n"
        "Return ONLY JSON:\n"
        '{"bbox": [x1, y1, x2, y2], "positive": [[x, y], ...], "negative": [[x, y], ...]}'
        + few_shot_block
    )

# ── VLMFacePartsBBox ──────────────────────────────────────────────────────────

def face_parts_bbox_prompt(cW, cH, parts_desc):
    return (
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

# ── VLMFacePrecisePoints ──────────────────────────────────────────────────────

def face_precise_points_prompt(cW, cH, cfg, num_fg_points, num_bg_points, modifier_block=""):
    return (
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

# ── VLMFaceRegion ─────────────────────────────────────────────────────────────

def face_region_stage1_prompt(sW, sH, region, face_rules):
    return (
        f"Image: {sW}x{sH} px.\n"
        f"TARGET: {region}\n\n"
        + face_rules +
        "\nReturn ONLY JSON (pixel coords):\n"
        '{"bbox": [x1, y1, x2, y2]}\n'
        "Tight box. x1<x2 y1<y2."
    )

def face_region_stage2_prompt(cW, cH, region, face_rules, num_fg_points, num_bg_points):
    return (
        f"Image: {cW}x{cH} px — cropped tightly to: {region}.\n"
        f"TARGET: {region}\n\n"
        + face_rules +
        f"\nPlace {num_fg_points} FOREGROUND points spread across the entire target.\n"
        f"Place {num_bg_points} BACKGROUND points just outside the target boundary.\n\n"
        "Return ONLY JSON (pixel coords in this cropped image):\n"
        '{"foreground": [[x, y], ...], "background": [[x, y], ...]}'
    )

# ── AVMAutoLayer / AVMMultiFrameAutoLayer ─────────────────────────────────────

def layer_discovery_prompt(guidance_line):
    return (
        (f"{guidance_line}\n\n" if guidance_line else "")
        + "Look at the image and list every distinct visual layer or region you can clearly see. "
        "Give each a SHORT, SPECIFIC label (e.g. 'black turtleneck', 'curly brown hair', 'gold hoop earrings'). "
        "Max 8 layers. Skip anything not clearly visible.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"layers": ["label1", "label2", ...]}'
    )

def layer_localize_prompt(W, H, num_pos_points, num_neg_points, labels_json):
    return (
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

# ── VLMReferenceMatch ─────────────────────────────────────────────────────────

def reference_match_prompt(subject_description, W, H):
    return (
        f"LEFT image: reference showing {subject_description}.\n"
        f"RIGHT image: target frame, {W}x{H} pixels.\n\n"
        f"Find {subject_description} from the LEFT image in the RIGHT image. "
        "Return a tight bounding box in the RIGHT image coordinate space.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}\n'
        "Pixel coordinates of the RIGHT image only. x1<x2, y1<y2. "
        'If the subject is not found, return {"bbox": null, "confidence": 0.0}.'
    )

# ── VLMAutoCrop ───────────────────────────────────────────────────────────────

def autocrop_discovery_prompt(hint_line, max_regions):
    return (
        f"{hint_line}"
        "Look at this image and list every distinct visual region you can clearly see. "
        "Give each a SHORT, SPECIFIC label (e.g. 'red jacket', 'person face', 'wooden table'). "
        f"Return at most {max_regions} regions. Skip anything not clearly visible.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"regions": ["label1", "label2", ...]}'
    )

def autocrop_localize_prompt(W, H, labels_json):
    return (
        f"Image: {W}x{H} pixels.\n\n"
        f"Return a tight bounding box for each of these regions:\n{labels_json}\n\n"
        "Pixel coordinates, x1<x2, y1<y2. Skip any region not visible. "
        "Confidence 0.0-1.0. Omit entries below 0.3.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"regions": [\n'
        '  {"label": "<exact label from list>", "bbox": [x1, y1, x2, y2], "confidence": 0.9},\n'
        "  ...\n]}"
    )
