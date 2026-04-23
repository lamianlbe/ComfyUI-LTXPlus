"""Fixed-arity Switch with select-on-execution semantics.

Motivation
----------
ImpactPack's `GeneralSwitch` ("Switch any") supports two selection modes:

- `select_on_prompt`: the selection is resolved before the workflow runs
  (legacy behavior for older ComfyUI).
- `select_on_execution`: the selection is resolved at runtime via ComfyUI's
  lazy-input mechanism, which means only the selected upstream branch is
  actually executed.

The execution-time mode is the useful one for branching workflows, but
Impact's implementation combines it with a *dynamic input arity* feature
(inputs grow as you connect more sources, implemented via an
`inspect.stack()` hack + an `AllContainer` dict that claims to contain
every key). That dynamic-inputs path is fragile when the node is run via
the API (`/prompt` endpoint) rather than through the UI — the frontend
never gets to materialize the extra slots, validation can short-circuit
in the wrong way, and `extra_pnginfo` is absent so the label lookup
silently degrades.

This node sidesteps the whole dynamic-arity story: it declares a fixed
set of 10 lazy inputs (`input1` .. `input10`) with hard-coded
`select_on_execution` semantics. Everything else — lazy evaluation via
`check_lazy_status`, `RETURN_TYPES = (any, STRING, INT)`, label lookup
from `extra_pnginfo` — mirrors `GeneralSwitch` so drop-in replacement
is straightforward.
"""

import logging

from .nodes_registry import comfy_node

logger = logging.getLogger(__name__)


class AnyType(str):
    """Type sentinel that matches every ComfyUI type during link validation.

    ComfyUI compares link types via `expected_type != actual_type`. Overriding
    `__ne__` to always return False makes this string compare-equal to any
    other type string, which is how ImpactPack and rgthree implement their
    "any" types.
    """

    def __ne__(self, other):
        return False


any_typ = AnyType("*")


NUM_INPUTS = 10


@comfy_node(name="LTXPlusSwitch10")
class LTXPlusSwitch10:
    """Select 1-of-10 inputs at execution time, evaluating only the chosen one.

    Equivalent to ImpactPack's `GeneralSwitch` with `sel_mode =
    select_on_execution`, but with a fixed 10-slot arity so it doesn't need
    the dynamic-input machinery that misbehaves in API mode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, NUM_INPUTS + 1):
            optional[f"input{i}"] = (
                any_typ,
                {
                    "lazy": True,
                    "tooltip": f"Slot {i}. Only the slot matching `select` "
                               f"is evaluated upstream.",
                },
            )
        return {
            "required": {
                "select": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": NUM_INPUTS,
                        "step": 1,
                        "tooltip": f"Which input (1..{NUM_INPUTS}) to forward.",
                    },
                ),
                # Declared for API-payload compatibility with ImpactSwitch
                # (`GeneralSwitch`) so existing API calls that include
                # `sel_mode` pass validation. The value is ignored: this
                # node is hard-wired to select_on_execution.
                "sel_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "select_on_prompt",
                        "label_off": "select_on_execution",
                        "forceInput": False,
                        "tooltip": "Kept for compatibility with ImpactSwitch's API "
                                   "payload. This node always behaves as "
                                   "select_on_execution regardless of the value.",
                    },
                ),
            },
            "optional": optional,
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (any_typ, "STRING", "INT")
    RETURN_NAMES = ("selected_value", "selected_label", "selected_index")
    OUTPUT_TOOLTIPS = (
        "The value of the input slot matching `select`.",
        "The slot's UI label (falls back to 'inputN' if not labeled or in API mode).",
        "The select value, passed through.",
    )
    FUNCTION = "doit"
    CATEGORY = "ltx-plus"

    # ------------------------------------------------------------------
    # Lazy evaluation: tell ComfyUI to only resolve the selected slot.
    # Without this method, all connected upstream branches would be
    # executed even though we only forward one of them.
    # ------------------------------------------------------------------
    def check_lazy_status(self, *args, **kwargs):
        selected_index = int(kwargs["select"])
        input_name = f"input{selected_index}"
        logger.info(f"LTXPlusSwitch10 selected: {input_name}")
        if input_name in kwargs:
            return [input_name]
        return []

    @staticmethod
    def doit(*args, **kwargs):
        selected_index = int(kwargs["select"])
        input_name = f"input{selected_index}"

        # Default label if we can't look up the UI-side slot label
        # (either because we're running in API mode so extra_pnginfo
        # is absent, or because the workflow JSON doesn't carry a
        # custom label for this slot).
        selected_label = input_name

        node_id = kwargs.get("unique_id")
        extra_pnginfo = kwargs.get("extra_pnginfo")

        if extra_pnginfo is not None and node_id is not None:
            try:
                nodelist = extra_pnginfo["workflow"]["nodes"]
                for node in nodelist:
                    if str(node.get("id")) == str(node_id):
                        for slot in node.get("inputs", []) or []:
                            if slot.get("name") == input_name and "label" in slot:
                                selected_label = slot["label"]
                                break
                        break
            except (KeyError, TypeError, AttributeError):
                # Malformed or missing workflow metadata — keep the
                # default label. Not an error; API-mode callers won't
                # usually care about the label anyway.
                pass
        else:
            # API mode: quiet debug log, no noisy warning.
            logger.debug(
                "LTXPlusSwitch10: running without extra_pnginfo "
                "(likely API mode); label will default to slot name."
            )

        if input_name in kwargs:
            return kwargs[input_name], selected_label, selected_index

        # The selected slot isn't connected — nothing to forward.
        logger.info(
            f"LTXPlusSwitch10: select={selected_index} but {input_name} "
            f"is not connected; returning None."
        )
        return None, selected_label, selected_index
