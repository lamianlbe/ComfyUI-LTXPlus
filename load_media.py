import hashlib
import io
import logging
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers
from server import PromptServer
from aiohttp import web

from .nodes_registry import comfy_node

logger = logging.getLogger(__name__)

# Video extensions handled by imageio/ffmpeg
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def _is_video_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    return ext in VIDEO_EXTENSIONS


def _resolve_frame_id(frame_id, total_frames):
    """Resolve frame_id to a valid 0-based index.

    Positive/zero: index from start. Negative: index from end (-1 = last).
    Clamp to [0, total_frames - 1].
    """
    if total_frames <= 0:
        return 0
    if frame_id < 0:
        idx = total_frames + frame_id
    else:
        idx = frame_id
    return max(0, min(idx, total_frames - 1))


def _pil_frame_to_tensor(frame):
    """Convert a single PIL frame to (image_tensor, mask_tensor)."""
    frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

    if frame.mode == "I":
        frame = frame.point(lambda i: i * (1 / 255))

    rgb = frame.convert("RGB")
    image = np.array(rgb).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image)[None,]

    if "A" in frame.getbands():
        mask = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
        mask_tensor = 1.0 - torch.from_numpy(mask)
    elif frame.mode == "P" and "transparency" in frame.info:
        mask = (
            np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
        )
        mask_tensor = 1.0 - torch.from_numpy(mask)
    else:
        mask_tensor = torch.zeros(
            (rgb.size[1], rgb.size[0]), dtype=torch.float32, device="cpu"
        )

    return image_tensor, mask_tensor.unsqueeze(0)


def _load_image_frame(filepath, frame_id):
    """Load a specific frame from an image file (static, APNG, GIF, WebP).

    Returns (image_tensor, mask_tensor) or (None, None) on failure.
    """
    img = node_helpers.pillow(Image.open, filepath)
    n_frames = getattr(img, "n_frames", 1)

    if n_frames <= 1:
        # Static image — ignore frame_id
        return _pil_frame_to_tensor(img)

    idx = _resolve_frame_id(frame_id, n_frames)
    img.seek(idx)
    return _pil_frame_to_tensor(img.copy())


def _load_video_frame(filepath, frame_id):
    """Load a specific frame from a video file using imageio/ffmpeg.

    Returns (image_tensor, mask_tensor) or (None, None) on failure.
    """
    import imageio.v3 as iio

    # Get frame count by reading properties
    props = iio.improps(filepath, plugin="pyav")
    if props.n_images is not None and props.n_images > 0:
        n_frames = props.n_images
    else:
        # Fallback: count frames (slower)
        n_frames = 0
        for _ in iio.imiter(filepath, plugin="pyav"):
            n_frames += 1

    idx = _resolve_frame_id(frame_id, n_frames)
    frame_np = iio.imread(filepath, index=idx, plugin="pyav")

    # frame_np shape: (H, W, C) with C=3 (RGB) or C=4 (RGBA)
    image = frame_np.astype(np.float32) / 255.0

    if image.ndim == 2:
        # Grayscale
        image = np.stack([image, image, image], axis=-1)

    if image.shape[-1] == 4:
        # Has alpha
        mask = 1.0 - image[:, :, 3]
        image = image[:, :, :3]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
    else:
        image = image[:, :, :3]
        mask_tensor = torch.zeros(
            (1, image.shape[0], image.shape[1]), dtype=torch.float32, device="cpu"
        )

    image_tensor = torch.from_numpy(image)[None,]
    return image_tensor, mask_tensor


def _load_frame(filepath, frame_id):
    """Load a frame from any supported media file.

    Returns (image_tensor, mask_tensor) or (None, None) on failure.
    """
    if _is_video_file(filepath):
        return _load_video_frame(filepath, frame_id)
    else:
        return _load_image_frame(filepath, frame_id)


def _get_frame_count(filepath):
    """Get total frame count for a media file."""
    try:
        if _is_video_file(filepath):
            import imageio.v3 as iio

            props = iio.improps(filepath, plugin="pyav")
            if props.n_images is not None and props.n_images > 0:
                return props.n_images
            # Fallback
            count = 0
            for _ in iio.imiter(filepath, plugin="pyav"):
                count += 1
            return count
        else:
            img = Image.open(filepath)
            return getattr(img, "n_frames", 1)
    except Exception:
        return 1


# ---------- Custom API route for frame preview ----------

@PromptServer.instance.routes.get("/ltxplus/preview_frame")
async def preview_frame(request):
    """Serve a specific frame as JPEG for frontend preview."""
    filename = request.rel_url.query.get("filename", "")
    frame_id = int(request.rel_url.query.get("frame_id", "0"))
    file_type = request.rel_url.query.get("type", "input")

    if not filename or "/" in filename or ".." in filename:
        return web.Response(status=400, text="Invalid filename")

    if file_type == "output":
        base_dir = folder_paths.get_output_directory()
    elif file_type == "temp":
        base_dir = folder_paths.get_temp_directory()
    else:
        base_dir = folder_paths.get_input_directory()

    filepath = os.path.join(base_dir, filename)
    if not os.path.isfile(filepath):
        return web.Response(status=404, text="File not found")

    try:
        if _is_video_file(filepath):
            import imageio.v3 as iio

            props = iio.improps(filepath, plugin="pyav")
            n_frames = props.n_images if (props.n_images and props.n_images > 0) else 1
            idx = _resolve_frame_id(frame_id, n_frames)
            frame_np = iio.imread(filepath, index=idx, plugin="pyav")
            pil_img = Image.fromarray(frame_np)
        else:
            img = Image.open(filepath)
            n_frames = getattr(img, "n_frames", 1)
            idx = _resolve_frame_id(frame_id, n_frames)
            if n_frames > 1:
                img.seek(idx)
            pil_img = img.copy()

        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        buf.seek(0)

        return web.Response(
            body=buf.read(),
            content_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache",
                "Content-Disposition": f'inline; filename="preview.jpg"',
            },
        )
    except Exception as e:
        logger.warning(f"Failed to generate preview frame: {e}")
        return web.Response(status=500, text=str(e))


@PromptServer.instance.routes.get("/ltxplus/frame_count")
async def frame_count(request):
    """Return the total frame count for a media file."""
    filename = request.rel_url.query.get("filename", "")
    file_type = request.rel_url.query.get("type", "input")

    if not filename or "/" in filename or ".." in filename:
        return web.Response(status=400, text="Invalid filename")

    if file_type == "output":
        base_dir = folder_paths.get_output_directory()
    elif file_type == "temp":
        base_dir = folder_paths.get_temp_directory()
    else:
        base_dir = folder_paths.get_input_directory()

    filepath = os.path.join(base_dir, filename)
    if not os.path.isfile(filepath):
        return web.json_response({"frame_count": 0})

    count = _get_frame_count(filepath)
    return web.json_response({"frame_count": count})


# ---------- Node definition ----------

def _list_media_files():
    """List all image and video files in the input directory."""
    input_dir = folder_paths.get_input_directory()
    files = []
    for f in os.listdir(input_dir):
        full = os.path.join(input_dir, f)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext in {
            ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
            ".gif", ".apng",
        } or ext in VIDEO_EXTENSIONS:
            files.append(f)
    return sorted(files)


@comfy_node(name="LTXVideoLoadMedia")
class LTXVideoLoadMedia:
    """Load a single frame from an image, animated image (APNG/GIF/WebP), or video file.

    Supports frame selection via frame_id. Returns None when bypass is enabled
    or the file cannot be loaded.
    """

    @classmethod
    def INPUT_TYPES(s):
        files = _list_media_files()
        return {
            "required": {
                "media": (files if files else ["none"], {"image_upload": True}),
                "frame_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": -99999,
                        "max": 99999,
                        "step": 1,
                        "tooltip": (
                            "Frame index to extract. 0 = first frame, "
                            "negative values count from the end (-1 = last frame). "
                            "Clamped to valid range."
                        ),
                    },
                ),
                "bypass": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, skip loading and output None.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_media"
    CATEGORY = "image"

    DESCRIPTION = (
        "Load a single frame from a static image, animated image "
        "(APNG/GIF/WebP), or video file. Supports frame_id for frame selection."
    )

    def load_media(self, media, frame_id, bypass):
        if bypass:
            return (None, None)

        try:
            filepath = folder_paths.get_annotated_filepath(media)
            if not os.path.isfile(filepath):
                logger.warning(f"LTXVideoLoadMedia: file not found: {filepath}")
                return (None, None)

            image, mask = _load_frame(filepath, frame_id)
            return (image, mask)
        except Exception as e:
            logger.warning(f"LTXVideoLoadMedia: failed to load media: {e}")
            return (None, None)

    @classmethod
    def IS_CHANGED(s, media, frame_id, bypass):
        if bypass:
            return "bypass"
        try:
            image_path = folder_paths.get_annotated_filepath(media)
            m = hashlib.sha256()
            with open(image_path, "rb") as f:
                m.update(f.read())
            m.update(str(frame_id).encode())
            return m.digest().hex()
        except Exception:
            return "error"

    @classmethod
    def VALIDATE_INPUTS(s, media, frame_id, bypass):
        if bypass:
            return True
        if media == "none":
            return True
        if not folder_paths.exists_annotated_filepath(media):
            return f"Invalid media file: {media}"
        return True
