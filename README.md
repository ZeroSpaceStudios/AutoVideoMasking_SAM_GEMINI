# SAMhera

VLM-powered automatic prompt generation for SAM3 in ComfyUI.
Uses Gemini or GPT-4o to detect bounding boxes and points — no manual drawing required.

---

## Installation

**Via ComfyUI Manager:** search `SAMhera` and install.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SAMhera/SAMhera
.venv/Scripts/python.exe -m pip install google-genai openai
```

**Requirements:** [ComfyUI-SAM3](https://github.com/SAM3/ComfyUI-SAM3) must be installed.

---

## Setup

Use `SAMheraAPIKey` to set your API key and model once, then wire the `api` output to all nodes.

| Provider | Key | Models |
|---|---|---|
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `gemini-2.5-flash`, `gemini-2.5-pro` (default) |
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `gpt-4o` |

---

## Nodes

### VLM Detection
| Node | Output | Description |
|---|---|---|
| `VLM -> BBox (SAM3)` | box_prompt, boxes_prompt | Single object bbox |
| `VLM -> Points (SAM3)` | positive_points, negative_points | Foreground + background points |
| `VLM -> Multi-BBox (SAM3)` | box_1…box_5, all_boxes | Up to 5 objects |
| `VLM -> BBox + Points (SAM3)` | box, points | Single API call for both |
| `VLM Prompt Editor` | box, points, prompt_used | Like above + editable/overrideable prompt |

### Face (SAMhera/Face)
| Node | Output | Description |
|---|---|---|
| `VLM Face Parts BBox` | hair, face, neck, face_neck, clothing | Region bboxes for face parts |
| `VLM Face Precise Points` | box_prompt, positive_points, negative_points | Precise points for a face part |
| `VLM Face Region` | cropped_image, crop_meta | Crops image to face region |

### Preview
| Node | Description |
|---|---|
| `VLM BBox Preview` | Draws detected boxes on image |
| `VLM Debug Preview` | Draws boxes + points overlaid on image |

### Crop / Paste
| Node | Description |
|---|---|
| `SAMheraCropByBox` | Crops image to a detected box (with padding + resize) |
| `SAMheraPasteBackMask` | Pastes a cropped mask back into the full-size image |
| `SAMheraAutoCrop` | Auto-crops based on VLM detection |

### Video / Layers
| Node | Description |
|---|---|
| `SAMheraAddFramePrompt` | Adds VLM prompts to a SAM3 video state at a specific frame |
| `SAMheraAddFramePromptBundle` | Bundles multiple frame prompts into one |
| `SAMheraUnpackBundle` | Unpacks a bundle back into individual prompts |
| `SAMheraAutoLayer` | Detects and layers objects across a video |
| `SAMheraLayerPropagate` | Propagates a layer set through video |
| `SAMheraMultiFrameAutoLayer` | Multi-frame version of AutoLayer |
| `SAMheraMultiFrameLayerPropagate` | Multi-frame propagation |
| `SAMheraReferenceMatch` | Matches objects across frames using a reference |
| `SAMheraLayerSelector` | Selects a specific layer from a layer set |

### Utility
| Node | Description |
|---|---|
| `SAMheraAPIKey` | Stores API key + model name, wire to all nodes |
| `VLM Image Test` | Verifies the VLM is receiving images correctly |

---

## Tips

- Wire `SAMheraAPIKey` once and reuse — avoids entering credentials per node
- Use `bbox_context` on `VLM -> Points` to keep boxes and points consistent
- `num_fg_points` 6–8 and `num_bg_points` 3–4 work well for most subjects
- `VLM Prompt Editor` lets you inspect and override the exact prompt sent to the VLM
- `SAMheraCropByBox` → segment → `SAMheraPasteBackMask` for high-res face/object workflows

---

## License

MIT
