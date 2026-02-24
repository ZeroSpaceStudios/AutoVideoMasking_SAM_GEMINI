"""
SAMhera - VLM-powered auto prompting for SAM3

Automatically generates bounding boxes and point prompts for SAM3
using Vision Language Models (Gemini, OpenAI GPT-4o).

Author: SAMhera
Version: 1.0.0
License: MIT
"""

import os
import sys
import traceback

__version__ = "1.0.0"
__author__ = "SAMhera"

INIT_SUCCESS = False

print(f"[SAMhera] v{__version__} initializing...")

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    INIT_SUCCESS = True
    print(f"[SAMhera] Loaded nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
except Exception as e:
    print(f"[SAMhera] ERROR: Failed to load nodes: {e}")
    print(traceback.format_exc())
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
