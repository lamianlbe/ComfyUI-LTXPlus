import comfy
import comfy_extras.nodes_lt as nodes_lt
import node_helpers

from .nodes_registry import comfy_node


def _append_guide_attention_entry(conditioning, pre_filter_count, latent_shape):
    """Append a guide attention entry to conditioning metadata.

    Equivalent to ComfyUI-LTXVideo's iclora_attention.append_guide_attention_entry
    with default attention_strength=1.0 and no attention_mask.
    """
    # Read existing entries
    entries = []
    for t in conditioning:
        existing = t[1].get("guide_attention_entries", None)
        if existing is not None:
            entries = [*existing]
            break

    entries.append(
        {
            "pre_filter_count": pre_filter_count,
            "strength": 1.0,
            "pixel_mask": None,
            "latent_shape": latent_shape,
        }
    )
    return node_helpers.conditioning_set_values(
        conditioning, {"guide_attention_entries": entries}
    )


@comfy_node(name="LTXVideoAddElements")
class LTXVideoAddElements:
    """Adds reference images (elements and first frame) to a video latent as conditioning guides.

    Elements are added before the first frame, all with negative latent_idx values.
    The last added guide always gets latent_idx=-1.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "img_compression": (
                    "INT",
                    {
                        "default": 18,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "CRF value for image preprocessing. Higher values mean more compression.",
                    },
                ),
                "first_frame_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Conditioning strength for the first frame guide.",
                    },
                ),
                "element_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Conditioning strength for all element guides.",
                    },
                ),
            },
            "optional": {
                "first_frame": (
                    "IMAGE",
                    {"tooltip": "First frame reference image."},
                ),
                "element1": (
                    "IMAGE",
                    {"tooltip": "Element 1 reference image."},
                ),
                "element2": (
                    "IMAGE",
                    {"tooltip": "Element 2 reference image."},
                ),
                "element3": (
                    "IMAGE",
                    {"tooltip": "Element 3 reference image."},
                ),
                "element4": (
                    "IMAGE",
                    {"tooltip": "Element 4 reference image."},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent_amount")

    CATEGORY = "conditioning/video_models"
    FUNCTION = "generate"

    DESCRIPTION = (
        "Adds reference images (elements and first frame) as conditioning guides to a video latent. "
        "Elements are placed before the first frame. All guides use negative latent indices, "
        "with the last guide at index -1. Use trim_latent_amount with LTXVideoTrimLatent "
        "to remove reference keyframes after sampling."
    )

    def generate(
        self,
        vae,
        latent,
        positive,
        negative,
        img_compression,
        first_frame_strength,
        element_strength,
        first_frame=None,
        element1=None,
        element2=None,
        element3=None,
        element4=None,
    ):
        # Collect connected guides: (image, strength) in order
        # Elements come before first_frame
        guides = []
        for elem in [element1, element2, element3, element4]:
            if elem is not None:
                guides.append((elem, element_strength))
        if first_frame is not None:
            guides.append((first_frame, first_frame_strength))

        trim_amount = len(guides)

        if trim_amount == 0:
            return (positive, negative, latent, 0)

        scale_factors = vae.downscale_index_formula
        _, width_scale_factor, height_scale_factor = scale_factors
        latent_samples = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)
        _, _, latent_length, latent_height, latent_width = latent_samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor

        for i, (image, strength) in enumerate(guides):
            # latent_idx: last guide is -1, going backwards
            # e.g. 3 guides: -3, -2, -1
            latent_idx = -(trim_amount - i)

            # Convert latent_idx to frame_idx (latent_idx <= 0)
            frame_idx = latent_idx * scale_factors[0]

            # Resize image to match latent dimensions
            image = (
                comfy.utils.common_upscale(
                    image.movedim(-1, 1), width, height, "lanczos", crop="disabled"
                )
                .movedim(1, -1)
                .clamp(0, 1)
            )

            # Preprocess with CRF (same as LTXVPreprocess)
            image = nodes_lt.LTXVPreprocess().execute(image, img_compression)[0]

            # Encode image to latent space
            _, guide_latent = nodes_lt.LTXVAddGuide.encode(
                vae, latent_width, latent_height, image, scale_factors
            )

            # Compute frame index
            frame_idx, _ = nodes_lt.LTXVAddGuide.get_latent_index(
                positive, latent_length, len(image), frame_idx, scale_factors
            )

            # Append keyframe to conditioning
            positive, negative, latent_samples, noise_mask = (
                nodes_lt.LTXVAddGuide.append_keyframe(
                    positive,
                    negative,
                    frame_idx,
                    latent_samples,
                    noise_mask,
                    guide_latent,
                    strength,
                    scale_factors,
                )
            )

            # Track guide in iclora attention entries
            pre_filter_count = (
                guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
            )
            guide_latent_shape = list(guide_latent.shape[2:])
            positive = _append_guide_attention_entry(
                positive, pre_filter_count, guide_latent_shape
            )
            negative = _append_guide_attention_entry(
                negative, pre_filter_count, guide_latent_shape
            )

        out_latent = {"samples": latent_samples, "noise_mask": noise_mask}
        return (positive, negative, out_latent, trim_amount)


@comfy_node(name="LTXVideoTrimLatent")
class LTXVideoTrimLatent:
    """Removes reference image keyframes from the end of a video latent.

    After sampling with reference images added via LTXVideoAddElements,
    use this node to strip the reference keyframes from the output latent.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "trim_amount": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Number of keyframes to remove from the end of the latent. "
                        "Connect this to the trim_latent_amount output of LTXVideoAddElements.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    CATEGORY = "latent/video"
    FUNCTION = "trim"

    DESCRIPTION = (
        "Removes the last trim_amount keyframes from a video latent along the time dimension. "
        "Use this after sampling to strip reference image keyframes added by LTXVideoAddElements."
    )

    def trim(self, latent, trim_amount):
        if trim_amount <= 0:
            return (latent,)

        s = latent.copy()
        samples = s["samples"]
        num_frames = samples.shape[2]

        if trim_amount >= num_frames:
            raise ValueError(
                f"trim_amount ({trim_amount}) must be less than the number of "
                f"latent frames ({num_frames})."
            )

        s["samples"] = samples[:, :, :-trim_amount, :, :]

        if "noise_mask" in s and s["noise_mask"] is not None:
            s["noise_mask"] = s["noise_mask"][:, :, :-trim_amount, :, :]

        return (s,)
