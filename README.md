# AutoVideoMasking (AVM)

VLM-powered automatic prompt generation for SAM3 in ComfyUI.
Uses Gemini to detect bounding boxes and points — no manual drawing required.

---

## Setup

### API Key

AVM resolves your API key in this order:

1. **Environment variable** — `GEMINI_API_KEY`
2. **.env file** — copy `.env.example` to `.env` in the AVM folder and fill in your key
3. **Node UI input** — enter directly in the **AVM API Config** node (least secure, avoid in shared workflows)

Place the **AVM API Config** node once on your canvas, set your model, and wire the `api` output to all other AVM nodes.

| Provider | Get Key | Recommended Model |
|---|---|---|
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `gemini-3.1-pro-preview` or `gemini-3-flash-preview` |

---

## Nodes

### VLM Detection
| Node | Output | Description |
|---|---|---|
| `AVM VLM → BBox` | box_prompt, boxes_prompt | Single object bounding box |
| `AVM VLM → Points` | positive_points, negative_points | Foreground + background points |
| `AVM VLM → Multi BBox` | box_1…box_5, all_boxes | Up to 5 objects |
| `AVM VLM → BBox + Points` | box, points | Single API call for both |
| `AVM Prompt Editor` | box, points, prompt_used | Like above + editable/overrideable prompt |

### Face (AVM/Face)
| Node | Output | Description |
|---|---|---|
| `AVM VLM → Face Parts BBox` | hair, face, neck, face_neck, clothing | Region bboxes for face parts |
| `AVM VLM → Face Points` | box_prompt, positive_points, negative_points | Precise points for a face part |
| `AVM Face Region` | cropped_image, crop_meta | Crops image to face region |

### Preview
| Node | Description |
|---|---|
| `AVM BBox Preview` | Draws detected boxes on image |
| `AVM Debug Preview` | Draws boxes + points overlaid on image |

### Crop / Paste
| Node | Description |
|---|---|
| `AVM Crop by Box` | Crops image to a detected box (with padding + resize) |
| `AVM Paste Back Mask` | Pastes a cropped mask back into the full-size image |
| `AVM Auto Crop` | Auto-crops based on VLM detection |

### Video / Layers
| Node | Description |
|---|---|
| `AVM Add Frame Prompt` | Adds VLM prompts to a SAM3 video state at a specific frame |
| `AVM Frame Prompt Bundle` | Bundles multiple frame prompts into one |
| `AVM Unpack Bundle` | Unpacks a bundle back into individual prompts |
| `AVM Auto Layer Detect` | Detects and layers objects across a video |
| `AVM Layer Propagate` | Propagates a layer set through video |
| `AVM Multi-Frame Layer Detect` | Multi-frame version of Auto Layer Detect |
| `AVM Multi-Frame Layer Propagate` | Multi-frame propagation |
| `AVM Reference Match` | Matches objects across frames using a reference |
| `AVM Layer Selector` | Selects a specific layer from a layer set |

### Utility
| Node | Description |
|---|---|
| `AVM API Config` | Sets API key + model name — wire `api` output to all nodes |
| `AVM VLM Test` | Verifies the VLM is receiving images correctly |
| `AVM Reload` | Hot-reloads node code without restarting ComfyUI |

---

## Tips

- Wire **AVM API Config** once and reuse — avoids entering credentials per node
- Use `bbox_context` on `AVM VLM → Points` to keep boxes and points consistent
- `num_fg_points` 6–8 and `num_bg_points` 3–4 work well for most subjects
- **AVM Prompt Editor** lets you inspect and override the exact prompt sent to the VLM
- **AVM Crop by Box** → segment → **AVM Paste Back Mask** for high-res face/object workflows

---

## License

MIT
