# SAMhera

**Author: SAMhera**

VLM-powered automatic prompt generation for SAM3 in ComfyUI.

Automatically generates bounding boxes and point prompts using Vision Language Models (Gemini, GPT-4o), eliminating the need to manually draw boxes or click points.

---

## Installation

### Via ComfyUI Manager
Search for `SAMhera` and install.

### Manual
```powershell
cd C:\ComfyUI\custom_nodes
git clone https://github.com/SAMhera/SAMhera
.venv\Scripts\python.exe -m pip install google-genai openai
```

---

## Requirements

- [ComfyUI-SAM3](https://github.com/SAM3/ComfyUI-SAM3) must be installed
- Gemini API key: https://aistudio.google.com/apikey
- Or OpenAI API key: https://platform.openai.com/api-keys

---

## Nodes

| Node | Description |
|---|---|
| `VLM -> BBox (SAM3)` | Detects a single object bbox via VLM |
| `VLM -> Points (SAM3)` | Generates foreground + background points via VLM |
| `VLM -> Multi-BBox (SAM3)` | Detects multiple objects (up to 5) |
| `VLM -> BBox + Points (SAM3)` | Single API call for bbox and points together |
| `VLM BBox Preview` | Draws detected boxes on image |
| `VLM Debug Preview` | Draws boxes + points overlaid on image |

---

## Workflow

### Basic (BBox only)
```
[Load Image] -> [VLM -> BBox (SAM3)]
                    boxes_prompt -> [SAM3 Point Segmentation] -> box
                                   sam3_model <- [Load SAM3 Model]
                                   image <- [Load Image]
```

### Full (BBox + Points, synchronized)
```
[Load Image] -> [VLM -> BBox (SAM3)]
                    boxes_prompt ──> box            [SAM3 Point Segmentation]
                    boxes_prompt ──> bbox_context   [VLM -> Points (SAM3)]
                                         positive_points -> positive_points
                                         negative_points -> negative_points
```

### Debug
```
[VLM -> BBox]   boxes_prompt    ──┐
[VLM -> Points] positive_points ──┼──> [VLM Debug Preview] -> [Preview Image]
[VLM -> Points] negative_points ──┘
```

---

## Tips

- Use `bbox_context` input on `VLM -> Points` to keep boxes and points consistent
- `num_fg_points`: 6-8 recommended for best SAM3 accuracy
- `num_bg_points`: 3-4 recommended
- VLM reasons about body parts automatically — works for people, animals, cars, buildings, etc.
- Use `few_shot_examples` field to guide VLM for specific use cases

---

## License

MIT
