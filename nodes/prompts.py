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
        "Coordinates: normalized integer 0-1000 scale. Tight box, x1<x2, y1<y2."
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
        "Return ONLY JSON (no markdown):\n"
        '{"objects": [{"bbox": [x1,y1,x2,y2], "label": "name"}, ...]}\n'
        "Coordinates: normalized integer 0-1000 scale. Tight boxes, sorted by confidence."
        + few_shot_block
    )

# ── VLMtoBBoxAndPoints / VLMPromptEditor ─────────────────────────────────────

def bbox_and_points_prompt(target_description, W, H, num_pos_points, num_neg_points, few_shot_block=""):
    return (
        f"Task: Spatial localization for computer vision segmentation.\n"
        f"Target: {target_description}\n"
        f"Image size: {W}x{H} pixels.\n\n"
        "Coordinate convention:\n"
        "- Normalized integer scale 0 to 1000 (NOT pixel values).\n"
        "- Bbox format: [x1, y1, x2, y2] with x1<x2, y1<y2.\n"
        "- Point format: [x, y].\n\n"
        "Bounding box rule:\n"
        "- Return the tightest rectangle enclosing ONLY the visible pixels of the target.\n"
        "- Do NOT include surrounding context (floor, sky, walls, sidewalk, nearby\n"
        "  objects, shadows) unless those pixels are part of the target itself.\n"
        "- If the target is a sub-part (e.g. 'face skin'), box only that sub-part.\n\n"
        f"Positive points ({num_pos_points}):\n"
        "- Identify distinct, physically-separated sub-parts OF THE TARGET and place\n"
        "  one point at the dead-center of each.\n"
        "- Anchor each point to a specific named feature — do NOT spread points\n"
        "  geometrically across the bounding box.\n"
        "- Every positive point MUST land on target pixels. Never place a positive\n"
        "  point on background, floor, walls, or nearby objects even if they fall\n"
        "  inside the bounding box.\n\n"
        f"Negative points ({num_neg_points}):\n"
        "- Identify specific background objects or non-target regions near the\n"
        "  target boundary and place one point on each.\n\n"
        "Reasoning: before emitting coordinates, internally note where the target's\n"
        "top, bottom, left, and right edges actually fall, and which sub-features you\n"
        "will anchor positive points to.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{\n'
        '  "boundary_description": "brief note on target edges and anchor sub-parts",\n'
        '  "bbox": [x1, y1, x2, y2],\n'
        '  "positive": [[x, y], ...],\n'
        '  "negative": [[x, y], ...]\n'
        '}'
        + few_shot_block
    )

# ── VLMFacePartsBBox ──────────────────────────────────────────────────────────

def face_parts_bbox_prompt(cW, cH, parts_desc):
    return (
        f"Image: {cW}x{cH} px (cropped to person).\n"
        "Return tight bounding boxes for each region. Coordinates must use a\n"
        "normalized integer 0-1000 scale relative to this cropped image.\n\n"
        "Regions:\n" + parts_desc + "\n\n"
        "Return ONLY JSON (no markdown):\n{\n"
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
        "Return ONLY JSON (coordinates: normalized integer 0-1000 scale relative to this crop):\n"
        '{"bbox": [x1, y1, x2, y2], "foreground": [[x, y], ...], "background": [[x, y], ...]}\n'
        "Rules: x1<x2 y1<y2, spread points — do NOT cluster them."
    )

# ── VLMFaceRegion ─────────────────────────────────────────────────────────────

def face_region_stage1_prompt(sW, sH, region, face_rules):
    return (
        f"Task: Localize one target region.\n"
        f"Image size: {sW}x{sH} px.\n"
        f"TARGET: {region}\n\n"
        + face_rules +
        "\nCoordinate convention:\n"
        "- Normalized integer scale 0 to 1000 (NOT pixel values).\n"
        "- Bbox format: [x1, y1, x2, y2] with x1<x2, y1<y2.\n\n"
        "Rules:\n"
        "- Return the tightest rectangle enclosing ONLY the visible pixels of the target.\n"
        "- Do NOT include background, nearby objects, or unrelated anatomy.\n"
        "- Before emitting coordinates, internally note where the target's top,\n"
        "  bottom, left, and right edges actually fall in the image.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{\n'
        '  "boundary_description": "brief note on where the target edges fall",\n'
        '  "bbox": [x1, y1, x2, y2]\n'
        '}'
    )

def face_region_stage2_prompt(cW, cH, region, face_rules, num_fg_points, num_bg_points):
    return (
        f"Task: Place segmentation prompt points in this cropped image.\n"
        f"Image size: {cW}x{cH} px (cropped tightly to: {region}).\n"
        f"TARGET: {region}\n\n"
        + face_rules +
        "\nCoordinate convention:\n"
        "- Normalized integer scale 0 to 1000, relative to THIS cropped image.\n"
        "- Point format: [x, y].\n\n"
        f"Foreground points ({num_fg_points}):\n"
        "- Identify distinct, physically-separated sub-parts OF THE TARGET\n"
        "  (e.g. 'center of left eye', 'tip of nose', 'center of chin', 'right cheek')\n"
        "  and place one point at the dead-center of each.\n"
        "- Anchor each point to a specific named feature — do NOT spread points\n"
        "  geometrically across the crop rectangle.\n"
        "- Every foreground point MUST land on target pixels. Never place one on\n"
        "  background or on excluded anatomy.\n\n"
        f"Background points ({num_bg_points}):\n"
        "- Identify specific non-target regions near the target boundary\n"
        "  (e.g. 'hair strands above forehead', 'collar below chin') and place\n"
        "  one point on each.\n\n"
        "Reasoning: before emitting coordinates, note which named sub-features you\n"
        "will anchor foreground points to, and which adjacent regions you will use\n"
        "for background points.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{\n'
        '  "anchor_plan": "brief note on which sub-features foreground points anchor to",\n'
        '  "foreground": [[x, y], ...],\n'
        '  "background": [[x, y], ...]\n'
        '}'
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
        f"Image: {W}x{H} pixels. Coordinates use a normalized integer 0-1000 scale.\n\n"
        f"For each region in the list below, return:\n"
        f"  • A tight bounding box (x1,y1,x2,y2, 0-1000 scale, x1<x2, y1<y2)\n"
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
        "Coordinates: normalized integer 0-1000 scale for the RIGHT image only. "
        "x1<x2, y1<y2. "
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
        "Coordinates: normalized integer 0-1000 scale. x1<x2, y1<y2. "
        "Skip any region not visible. Confidence 0.0-1.0. Omit entries below 0.3.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"regions": [\n'
        '  {"label": "<exact label from list>", "bbox": [x1, y1, x2, y2], "confidence": 0.9},\n'
        "  ...\n]}"
    )
