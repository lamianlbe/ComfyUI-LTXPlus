import hashlib
import os

import folder_paths
import safetensors.torch

from .nodes_registry import comfy_node


def _list_latent_files():
    """List all .latent files in the input directory."""
    input_dir = folder_paths.get_input_directory()
    return sorted(
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".latent")
    )


@comfy_node(name="LTXVideoLoadLatent")
class LTXVideoLoadLatent:
    """Load a latent from a .latent file.

    Supports a 'none' option that outputs None, allowing downstream nodes
    to treat the input as optional without errors.
    """

    @classmethod
    def INPUT_TYPES(s):
        files = _list_latent_files()
        return {
            "required": {
                "latent": (["none"] + files,),
            },
        }

    CATEGORY = "latent"

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load"

    DESCRIPTION = (
        "Load a latent from a .latent file. Select 'none' to output nothing, "
        "which allows downstream nodes to skip this input gracefully."
    )

    def load(self, latent):
        if latent == "none":
            return (None,)

        latent_path = folder_paths.get_annotated_filepath(latent)
        data = safetensors.torch.load_file(latent_path, device="cpu")

        multiplier = 1.0
        if "latent_format_version_0" not in data:
            multiplier = 1.0 / 0.18215

        samples = {"samples": data["latent_tensor"].float() * multiplier}
        return (samples,)

    @classmethod
    def IS_CHANGED(s, latent):
        if latent == "none":
            return "none"
        image_path = folder_paths.get_annotated_filepath(latent)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        if latent == "none":
            return True
        if not folder_paths.exists_annotated_filepath(latent):
            return f"Invalid latent file: {latent}"
        return True
