"""Small helpers to turn CT slices + masks into VLM-ready inputs.

The Judging Panel sends each candidate mask to one or more VLMs. The
paper describes the input as "a lung CT image and its corresponding
mask" — we support both variants:

  * ``overlay``: one composite PNG with the mask drawn in red on top of
    the CT slice. Works reliably with any VLM that accepts a single
    image, including those that only handle one ``image_url`` per turn.
  * ``pair``: two separate images in one multimodal message (image +
    mask). Some models use the alignment between them as an explicit
    cross-reference cue.
"""
from __future__ import annotations

import base64
import io
from typing import Literal

import numpy as np

InputMode = Literal["overlay", "pair"]


# ── CT slice → PNG bytes ─────────────────────────────────────────────

def _normalize_slice(slice_image: np.ndarray) -> np.ndarray:
    """Return a uint8 (H, W, 3) RGB array suitable for PNG encoding.

    Accepts float arrays in any range (HU, [0,1], [0,255], …) and
    rescales to 0–255 via min-max. Already-uint8 inputs are passed
    through. Single-channel inputs are stacked to 3 channels.
    """
    arr = np.asarray(slice_image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D slice, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-6:
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def _encode_png(rgb: np.ndarray) -> bytes:
    """Encode a uint8 (H, W, 3) RGB array as PNG bytes."""
    from PIL import Image  # local import: PIL is a heavy optional dep

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ── Mask compositing ─────────────────────────────────────────────────

def render_mask_image(mask: np.ndarray) -> bytes:
    """Encode a binary mask as a pure black/white PNG (255 inside, 0 out)."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    rgb = np.stack([m, m, m], axis=-1)
    return _encode_png(rgb)


def render_overlay(
    slice_image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> bytes:
    """Draw ``mask`` in red (by default) on top of ``slice_image``.

    Both inputs must share H×W. Returns PNG bytes.
    """
    base = _normalize_slice(slice_image).astype(np.float32)
    m = (np.asarray(mask) > 0)
    if m.shape != base.shape[:2]:
        raise ValueError(
            f"mask shape {m.shape} does not match slice {base.shape[:2]}"
        )
    overlay = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    blended = np.where(
        m[..., None],
        (1 - alpha) * base + alpha * overlay,
        base,
    )
    return _encode_png(blended.clip(0, 255).astype(np.uint8))


# ── VLM multimodal content builders ──────────────────────────────────

def build_vlm_content(
    slice_image: np.ndarray,
    mask: np.ndarray,
    prompt_text: str,
    mode: InputMode = "overlay",
) -> list[dict]:
    """Return an OpenAI-compatible ``content`` list for one user turn.

    The returned list is suitable as the value of
    ``Message.user(content=...)`` in EasyAgent.
    """
    if mode == "overlay":
        overlay_png = render_overlay(slice_image, mask)
        return [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": _to_data_url(overlay_png)},
            },
        ]
    if mode == "pair":
        image_png = _encode_png(_normalize_slice(slice_image))
        mask_png = render_mask_image(mask)
        return [
            {"type": "text", "text": prompt_text},
            {
                "type": "text",
                "text": "[Image 1 — CT slice]",
            },
            {
                "type": "image_url",
                "image_url": {"url": _to_data_url(image_png)},
            },
            {
                "type": "text",
                "text": "[Image 2 — candidate mask, same coordinate frame]",
            },
            {
                "type": "image_url",
                "image_url": {"url": _to_data_url(mask_png)},
            },
        ]
    raise ValueError(f"unknown input mode: {mode!r}")
