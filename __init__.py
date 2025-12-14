"""
ComfyUI-StarVector
Custom nodes for generating SVG files using StarVector models
https://huggingface.co/starvector/starvector-1b-im2svg
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
