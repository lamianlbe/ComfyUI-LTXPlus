"""
LTX Plus Batch Img To Video Condition Only — inplace batch keyframe injection.

Functionally a batched extension of comfy_extras' `LTXVImgToVideoInplace`.
Per-keyframe behavior is identical:

  1. Resize the image to the latent's pixel dims (latent_dim × scale_factor)
  2. VAE-encode → `t` of shape (B, C, t_frames, H_lat, W_lat)
  3. Write `samples[:, :, lat_idx : lat_idx + t.shape[2]] = t`
  4. Mark those time slots in noise_mask with `(1 - strength)` to preserve
     them during sampling

Differences from the upstream single-keyframe node:

  - Accepts an IMAGE batch + a comma-separated frame_indices string
  - Loops `min(len(images), len(indices))` times
  - Each iteration resolves its own pixel frame_idx → latent_idx using the
    LTXV temporal stride convention (frame_idx must be 0 or 8k+1 in pixel
    space; in-between values are floored to the nearest valid latent slot)
  - Negative indices resolve relative to the latent's frame count (e.g.
    `-1` = last latent frame)
  - Clones the input latent's samples before writing so the caller's
    tensor isn't mutated (upstream node skips this — a footgun we
    deliberately fix here)

Inputs:
  - vae           : VAE
  - latent        : LATENT
  - images        : IMAGE batch (one image per keyframe)
  - frame_indices : STRING, comma-separated pixel frame indices
  - strength      : FLOAT (default 0.7)

Output:
  - latent : LATENT with samples written + noise_mask marking protected slots

This node does NOT touch positive/negative conditioning, matching
`LTXVImgToVideoInplace`'s scope.
"""

import logging

import torch

import comfy.utils

from .nodes_registry import comfy_node

logger = logging.getLogger(__name__)


def _parse_frame_indices(s):
    """Parse a comma-separated string of integers, ignoring blanks/bad tokens."""
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            logger.warning(
                f"LTXPlusBatchImgToVideoConditionOnly: "
                f"ignoring non-integer token {tok!r}"
            )
    return out


@comfy_node(name="LTXPlusBatchImgToVideoConditionOnly")
class LTXPlusBatchImgToVideoConditionOnly:
    """Batch keyframe injection in inplace mode (LTXVImgToVideoInplace semantics)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":    ("VAE",),
                "latent": ("LATENT",),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Image batch. Each item is one keyframe; "
                                   "pair with a frame_indices entry.",
                    },
                ),
                "frame_indices": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": "Comma-separated pixel frame indices "
                                   "(e.g. '0, 8, 16, -1'). Must be 0 or "
                                   "(8k+1) — non-aligned values are floored "
                                   "to the nearest valid latent slot. "
                                   "Negative values count from the end. "
                                   "The loop runs min(len(images), len(indices)) "
                                   "times — extras on either side are dropped.",
                    },
                ),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Strength applied uniformly to every "
                                   "keyframe. Noise mask at each keyframe's "
                                   "time slot is set to (1 - strength). "
                                   "Higher strength = more faithful to the "
                                   "input image.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "ltx-plus"
    DESCRIPTION = (
        "Batched inplace keyframe injection: writes N reference images into "
        "specified latent frame slots and builds a noise_mask that preserves "
        "them during sampling. Per-keyframe behavior matches "
        "LTXVImgToVideoInplace exactly."
    )

    def execute(self, vae, latent, images, frame_indices, strength=0.7):
        indices = _parse_frame_indices(frame_indices)
        n_images = int(images.shape[0]) if images is not None else 0
        n = min(n_images, len(indices))

        # Pass-through if nothing to do.
        if n == 0:
            logger.info(
                f"LTXPlusBatchImgToVideoConditionOnly: nothing to inject "
                f"(images={n_images}, indices={len(indices)}) — pass-through."
            )
            return (latent,)

        if n_images != len(indices):
            logger.info(
                f"LTXPlusBatchImgToVideoConditionOnly: image count "
                f"({n_images}) and index count ({len(indices)}) differ — "
                f"using min ({n}); extras are dropped."
            )

        scale_factors = vae.downscale_index_formula
        time_sf, height_sf, width_sf = scale_factors

        # Clone so we don't mutate the caller's latent tensor. (Upstream
        # LTXVImgToVideoInplace mutates in place — a footgun when the same
        # latent is consumed by multiple downstream nodes. We avoid that.)
        samples = latent["samples"].clone()
        batch_size, _, latent_frames, latent_height, latent_width = samples.shape
        pixel_width = latent_width * width_sf
        pixel_height = latent_height * height_sf

        # Build a fresh time-only noise_mask matching
        # LTXVImgToVideoInplace's shape convention (B, 1, T, 1, 1). All
        # ones initially → every slot is freely generated; each keyframe
        # iteration marks its own slot with (1 - strength).
        noise_mask = torch.ones(
            (batch_size, 1, latent_frames, 1, 1),
            dtype=torch.float32,
            device=samples.device,
        )

        for fi in range(n):
            single_image = images[fi:fi + 1]
            frame_idx_in = indices[fi]
            original_frame_idx = frame_idx_in  # for logging

            # Negative indices count from the end. We compute relative to
            # the total pixel-frame count of this latent: (T-1)*8 + 1.
            total_pixel_frames = (latent_frames - 1) * time_sf + 1
            if frame_idx_in < 0:
                frame_idx_in = max(0, total_pixel_frames + frame_idx_in)

            # Encode the image at the latent's pixel resolution. Match
            # upstream's resize logic: skip the bilinear pass when dims
            # already align, to avoid an unnecessary copy.
            if (single_image.shape[1] != pixel_height
                    or single_image.shape[2] != pixel_width):
                pixels = comfy.utils.common_upscale(
                    single_image.movedim(-1, 1),
                    pixel_width, pixel_height,
                    "bilinear", "center",
                ).movedim(1, -1)
            else:
                pixels = single_image
            encode_pixels = pixels[:, :, :, :3]
            t = vae.encode(encode_pixels)

            # Pixel frame_idx → latent frame_idx. LTXV's temporal stride
            # is 8: pixel frame 0 maps to latent 0, pixel frames 1..8 map
            # to latent 1, pixel frames 9..16 map to latent 2, and so on.
            if frame_idx_in == 0:
                lat_idx = 0
            else:
                lat_idx = (frame_idx_in - 1) // time_sf + 1

            # Clamp so the write window fits inside the latent. If the
            # encoded keyframe would overflow we shift it back; the
            # alternative (raise) feels worse for a batch operation that
            # may have many keyframes.
            if lat_idx + t.shape[2] > latent_frames:
                lat_idx = max(0, latent_frames - t.shape[2])

            # Inplace write into samples + noise_mask. The shape contract
            # is identical to LTXVImgToVideoInplace; we just shifted the
            # write window from `:t.shape[2]` to `lat_idx:lat_idx+t.shape[2]`.
            samples[:, :, lat_idx:lat_idx + t.shape[2]] = t
            noise_mask[:, :, lat_idx:lat_idx + t.shape[2]] = 1.0 - strength

            logger.info(
                f"LTXPlusBatchImgToVideoConditionOnly[{fi}]: "
                f"input pixel_idx={original_frame_idx} → resolved "
                f"pixel_idx={frame_idx_in} → latent_idx={lat_idx}, "
                f"guide_frames={t.shape[2]}, strength={strength}"
            )

        return ({"samples": samples, "noise_mask": noise_mask},)
