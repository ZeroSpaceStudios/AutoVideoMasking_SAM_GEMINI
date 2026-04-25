#!/usr/bin/env bash
# ComfyUI-AutoVideoMasking — lean Vast.ai provisioning script
# Installs only what the AVM workflow needs. No LoRA, no checkpoints, no extras.
#
# Usage on Vast.ai:
#   export PROVISIONING_SCRIPT=https://raw.githubusercontent.com/ZeroSpaceStudios/ComfyUI-AutoVideoMasking/main/provisioning_avm.sh
# Or run manually after instance starts:
#   curl -fsSL <url> | bash

set -euo pipefail

COMFYUI_DIR="${WORKSPACE:-/workspace}/ComfyUI"
CUSTOM_NODES="$COMFYUI_DIR/custom_nodes"
MODELS="$COMFYUI_DIR/models"

log() { echo "[AVM] $*"; }

# ── 0. Wait for ComfyUI to be present ────────────────────────────────────────
if [ ! -d "$COMFYUI_DIR" ]; then
    log "ERROR: $COMFYUI_DIR not found. Start from a vastai/comfy image or pre-install ComfyUI."
    exit 1
fi
log "ComfyUI found at $COMFYUI_DIR"

# ── 1. System packages ────────────────────────────────────────────────────────
log "Installing system packages..."
apt-get update -qq && apt-get install -y -qq ffmpeg git curl > /dev/null

# ── 2. Python packages ────────────────────────────────────────────────────────
log "Installing Python packages..."
pip install -q --upgrade \
    "google-genai>=1.0.0" \
    "huggingface_hub" \
    "Pillow" \
    "numpy" \
    "imageio[ffmpeg]" \
    "av"

# ── 3. Custom nodes ───────────────────────────────────────────────────────────
clone_or_pull() {
    local repo="$1" dest="$2"
    if [ -d "$dest/.git" ]; then
        log "Updating $(basename "$dest")..."
        git -C "$dest" pull --ff-only --quiet
    else
        log "Cloning $(basename "$dest")..."
        git clone --depth 1 --quiet "$repo" "$dest"
    fi
}

# SAM3 (video segmentation backbone)
clone_or_pull \
    "https://github.com/PozzettiAndrea/ComfyUI-SAM3.git" \
    "$CUSTOM_NODES/ComfyUI-SAM3"

# Video Helper Suite (VHS_LoadVideo node)
clone_or_pull \
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" \
    "$CUSTOM_NODES/ComfyUI-VideoHelperSuite"

# ComfyUI-AutoVideoMasking (this repo)
clone_or_pull \
    "https://github.com/ZeroSpaceStudios/ComfyUI-AutoVideoMasking.git" \
    "$CUSTOM_NODES/ComfyUI-AutoVideoMasking"

# Install each node's Python requirements
for node_dir in \
    "$CUSTOM_NODES/ComfyUI-SAM3" \
    "$CUSTOM_NODES/ComfyUI-VideoHelperSuite" \
    "$CUSTOM_NODES/ComfyUI-AutoVideoMasking"; do
    req="$node_dir/requirements.txt"
    if [ -f "$req" ]; then
        log "Installing requirements for $(basename "$node_dir")..."
        pip install -q -r "$req" || true
    fi
done

# ── 4. SAM3 model ─────────────────────────────────────────────────────────────
# LoadSAM3Model auto-downloads sam3.safetensors from apozz/sam3-safetensors
# on first run. Pre-download here so the first workflow run doesn't stall.
SAM3_MODEL_DIR="$MODELS/sam3"
SAM3_MODEL="$SAM3_MODEL_DIR/sam3.safetensors"
mkdir -p "$SAM3_MODEL_DIR"

if [ ! -f "$SAM3_MODEL" ]; then
    log "Downloading sam3.safetensors (~3 GB)..."
    python3 - <<'EOF'
from huggingface_hub import hf_hub_download
import os
dest = os.environ.get("SAM3_MODEL_DIR", "/workspace/ComfyUI/models/sam3")
hf_hub_download(
    repo_id="apozz/sam3-safetensors",
    filename="sam3.safetensors",
    local_dir=dest,
)
print(f"[AVM] sam3.safetensors saved to {dest}")
EOF
else
    log "sam3.safetensors already present, skipping download."
fi

# ── 5. Done ───────────────────────────────────────────────────────────────────
log "Provisioning complete."
log "Set your Gemini API key in the AVMAPIConfig node before running."
log "Workflow: $CUSTOM_NODES/ComfyUI-AutoVideoMasking/workflow/AVM_LayerSelectPreview.json"
