import json
import os

import torch

import comfy.utils
import folder_paths
from comfy.cli_args import args

from .nodes_registry import comfy_node


@comfy_node(name="LTXVideoSaveLatent")
class LTXVideoSaveLatent:
    """Save a latent to a .latent file.

    File format is identical to ComfyUI's built-in SaveLatent.
    Returns filename and path in the API output, similar to SaveImage.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename_prefix": (
                    "STRING",
                    {"default": "latents/ComfyUI"},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"

    OUTPUT_NODE = True

    CATEGORY = "latent"

    DESCRIPTION = (
        "Save a latent to a .latent file. Format is identical to ComfyUI's "
        "built-in SaveLatent. Returns filename and path in the API output."
    )

    def save(self, samples, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )

        # Build metadata
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = None
        if not args.disable_metadata:
            metadata = {"prompt": prompt_info}
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        file = f"{filename}_{counter:05}_.latent"
        filepath = os.path.join(full_output_folder, file)

        # Save in the same format as ComfyUI's SaveLatent
        output = {}
        output["latent_tensor"] = samples["samples"].contiguous()
        output["latent_format_version_0"] = torch.tensor([])

        comfy.utils.save_torch_file(output, filepath, metadata=metadata)

        results = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output",
            }
        ]

        return {"ui": {"latents": results}}
