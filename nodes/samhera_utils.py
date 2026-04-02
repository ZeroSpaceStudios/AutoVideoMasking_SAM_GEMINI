"""
AVM Utilities
Non-SAM helper nodes for the AVM package.

Author: Hera Kang
"""


# =============================================================================
# AVMReload — hot-reload vlm_sam3_bridge without restarting ComfyUI
# =============================================================================

class AVMReload:
    """
    Reloads AVM node code instantly — no ComfyUI restart needed.
    Save changes to vlm_sam3_bridge.py, then run this node.
    Re-add nodes to canvas after reload to use updated code.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("status",)
    FUNCTION      = "run"
    CATEGORY      = "AVM/Utils"
    OUTPUT_NODE   = True

    def run(self, reload):
        if not reload:
            return ("Skipped.",)

        import importlib
        import sys

        reloaded, failed = [], []

        for mod_name in list(sys.modules.keys()):
            if "vlm_sam3_bridge" in mod_name:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reloaded.append(mod_name)
                except Exception as e:
                    failed.append(f"{mod_name}: {e}")

        if reloaded:
            try:
                mod = sys.modules[reloaded[0]]
                import nodes as comfy_nodes
                comfy_nodes.NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
                comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
                status = "Reloaded. Re-add nodes to use updated code."
            except Exception as e:
                status = f"Module reloaded but registry update failed: {e}"
        else:
            status = "Module not found — full restart required (first time only)."

        if failed:
            status += f"\nFailed: {', '.join(failed)}"

        print(f"[AVM] Reload: {status}")
        return (status,)


# =============================================================================
# Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "AVMReload": AVMReload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AVMReload": "AVM Reload",
}
