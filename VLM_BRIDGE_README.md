# VLM → SAM3 Bridge

**Author: SAMhera**

Auto-generate bounding boxes and point prompts for SAM3 using a Vision Language Model (Gemini / OpenAI), eliminating the need to manually draw boxes or click points.

## Nodes

| Node | Description |
|---|---|
| `VLM -> BBox (SAM3)` | Single object bbox detection |
| `VLM -> Points (SAM3)` | Foreground + background point generation |
| `VLM -> Multi-BBox (SAM3)` | Multi-object detection (up to 5) |
| `VLM -> BBox + Points (SAM3)` | Single call for both bbox and points |
| `VLM BBox Preview` | Visualize detected boxes on image |
| `VLM Debug Preview` | Visualize boxes + points overlaid on image |

## Workflow

```
[Load Image] -> [VLM -> BBox (SAM3)]
                    boxes_prompt  -> [SAM3 Point Segmentation] -> box
                    boxes_prompt  -> [VLM -> Points (SAM3)] -> bbox_context
                                         positive_points -> [SAM3 Point Segmentation]
                                         negative_points -> [SAM3 Point Segmentation]
```

## Setup

```powershell
.venv\Scripts\python.exe -m pip install google-genai openai
```

Set your API key in the node's `api_key` field.
- Gemini: https://aistudio.google.com/apikey
- OpenAI: https://platform.openai.com/api-keys

## Recommended Models

- `gemini-2.5-flash` — fast, free tier
- `gemini-2.5-pro` — more accurate
- `gpt-4o` — OpenAI alternative
