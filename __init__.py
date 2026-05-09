from .add_elements import LTXVideoAddElements, LTXVideoTrimLatent
from . import load_media  # noqa: F401 — registers LTXVideoLoadMedia node + API routes
from . import load_latent  # noqa: F401 — registers LTXVideoLoadLatent node
from . import save_latent  # noqa: F401 — registers LTXVideoSaveLatent node
from . import ltxv_generate  # noqa
from . import ltxv_iclora_guider  # noqa
from . import switch  # noqa — registers LTXPlusSwitch10
from . import batch_add_guide  # noqa — registers LTXPlusBatchAddGuide
from .nodes_registry import NODE_CLASS_MAPPINGS as RUNTIME_NODE_CLASS_MAPPINGS
from .nodes_registry import (
    NODE_DISPLAY_NAME_MAPPINGS as RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, camel_case_to_spaces

NODE_CLASS_MAPPINGS = {
    "LTXVideoAddElements": LTXVideoAddElements,
    "LTXVideoTrimLatent": LTXVideoTrimLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    name: f"{NODES_DISPLAY_NAME_PREFIX} {camel_case_to_spaces(name)}"
    for name in NODE_CLASS_MAPPINGS.keys()
}

# Update with runtime mappings
NODE_CLASS_MAPPINGS.update(RUNTIME_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(RUNTIME_NODE_DISPLAY_NAME_MAPPINGS)

# LTX Plus Generate + IC-LoRA Guider nodes
NODE_CLASS_MAPPINGS.update(ltxv_generate.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ltxv_generate.NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ltxv_iclora_guider.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ltxv_iclora_guider.NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
