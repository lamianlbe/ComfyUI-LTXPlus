import re
from functools import wraps
from typing import Callable, Optional, Type

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODES_DISPLAY_NAME_PREFIX = "🅛🅣🅧"
EXPERIMENTAL_DISPLAY_NAME_PREFIX = "(Experimental)"
DEPRECATED_DISPLAY_NAME_PREFIX = "(Deprecated)"
DEFAULT_CATEGORY_NAME = "Lightricks"


def register_node(node_class: Type, name: str, description: str) -> None:
    if not isinstance(node_class, type):
        raise ValueError("`node_class` must be a class")
    if not isinstance(name, str):
        raise ValueError("`name` must be a string")
    if not isinstance(description, str):
        raise ValueError("`description` must be a string")

    NODE_CLASS_MAPPINGS[name] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[name] = description


def comfy_node(
    node_class: Optional[Type] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    experimental: bool = False,
    deprecated: bool = False,
    skip: bool = False,
) -> Callable:
    def decorator(node_class: Type) -> Type:
        if skip:
            return node_class

        if not isinstance(node_class, type):
            raise ValueError("`node_class` must be a class")

        nonlocal name, description
        if name is None:
            name = node_class.__name__
            if name is not None and name.endswith("Node"):
                name = name[:-4]

        description = _format_description(description, name, experimental, deprecated)
        register_node(node_class, name, description)
        return node_class

    if node_class is None:
        return decorator
    else:
        return decorator(node_class)


def _format_description(
    description: str, class_name: str, experimental: bool, deprecated: bool
) -> str:
    if description is None:
        description = camel_case_to_spaces(class_name)

    prefix_len = len(NODES_DISPLAY_NAME_PREFIX)
    if description.startswith(NODES_DISPLAY_NAME_PREFIX):
        description = description[prefix_len:].lstrip()

    if deprecated:
        description = f"{DEPRECATED_DISPLAY_NAME_PREFIX} {description}"
    elif experimental:
        description = f"{EXPERIMENTAL_DISPLAY_NAME_PREFIX} {description}"

    description = f"{NODES_DISPLAY_NAME_PREFIX} {description}"
    return description


def camel_case_to_spaces(text: str) -> str:
    spaced_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    spaced_text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced_text)
    spaced_text = re.sub(r"(?<=[A-Z])(?=[A-Z][A-Z][a-z])", " ", spaced_text)
    return spaced_text
