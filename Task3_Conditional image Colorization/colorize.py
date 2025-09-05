import math
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


ColorRGB = Tuple[int, int, int]


def ensure_gray(image_bgr: np.ndarray) -> np.ndarray:
    """
    Ensure the input image is single-channel grayscale uint8.
    Accepts BGR (3-ch) or GRAY (1-ch). Returns GRAY (H, W) uint8.
    """
    if image_bgr is None:
        raise ValueError("Input image is None")
    if len(image_bgr.shape) == 2:
        gray = image_bgr
    elif image_bgr.shape[2] == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image shape for grayscale conversion")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def rgb_to_lab_chroma(rgb: ColorRGB) -> Tuple[float, float]:
    """
    Convert an RGB color (0-255) to Lab a,b components using OpenCV's D65 standard.
    Returns (a, b) as floats in OpenCV Lab space (0-255 scaled, but we'll treat as float for blending).
    """
    r, g, b = rgb
    color = np.array([[[b, g, r]]], dtype=np.uint8)  # OpenCV uses BGR
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = float(lab[0, 0, 1])
    b_val = float(lab[0, 0, 2])
    # Convert to centered a,b by subtracting 128 to allow signed blending, then caller can re-add 128.
    return a - 128.0, b_val - 128.0


def _feather_mask(mask: np.ndarray, feather_radius_px: int) -> np.ndarray:
    """
    Feather a binary/float mask using Gaussian blur with the given radius in pixels.
    Returns a float mask in [0,1].
    """
    if feather_radius_px <= 0:
        return np.clip(mask.astype(np.float32), 0.0, 1.0)
    # Kernel size must be odd and positive; approximate from radius
    k = int(max(3, feather_radius_px * 2 + 1))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigmaX=feather_radius_px, borderType=cv2.BORDER_REPLICATE)
    return np.clip(blurred, 0.0, 1.0)


def make_mask_from_canvas_rgba(canvas_rgba: np.ndarray) -> np.ndarray:
    """
    Extract a selection mask from a drawable canvas RGBA image (H, W, 4) in uint8.
    Any non-zero alpha is considered selected. Returns float mask in [0,1] with shape (H, W).
    """
    if canvas_rgba is None or canvas_rgba.size == 0:
        raise ValueError("Canvas RGBA is empty")
    if canvas_rgba.ndim != 3 or canvas_rgba.shape[2] != 4:
        raise ValueError("Canvas must be RGBA with 4 channels")
    alpha = canvas_rgba[:, :, 3].astype(np.float32) / 255.0
    # Some canvases draw antialiased strokes. Normalize to [0,1]
    return np.clip(alpha, 0.0, 1.0)


def magic_wand_mask(gray: np.ndarray, seed_xy: Tuple[int, int], tolerance: int) -> np.ndarray:
    """
    Compute a magic-wand style mask using OpenCV flood fill on the grayscale image.
    - gray: (H, W) uint8
    - seed_xy: (x, y) int coords in image space
    - tolerance: 0-255 inclusive
    Returns float mask in [0,1] with shape (H, W).
    """
    h, w = gray.shape
    x, y = seed_xy
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Seed point out of bounds")

    # OpenCV floodFill requires a mask that is 2 pixels larger than the image.
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    gray_copy = gray.copy()

    # Flood fill from seed using specified tolerance.
    # loDiff, upDiff apply to grayscale intensity differences.
    _, _, _, _ = cv2.floodFill(
        gray_copy,
        flood_mask,
        seedPoint=(x, y),
        newVal=255,
        loDiff=tolerance,
        upDiff=tolerance,
        flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )

    # flood_mask now has 1 where filled region is, within the inner (h,w) area.
    region = flood_mask[1:-1, 1:-1]
    return (region.astype(np.float32) / 1.0)  # region is 0 or 1 already


def compose_rules(
    gray: np.ndarray,
    rules: List[Dict],
    image_size: Optional[Tuple[int, int]] = None,
    mode: str = 'priority',  # 'priority' or 'blend'
) -> np.ndarray:
    """
    Apply a list of color rules to a grayscale image while preserving luminance.
    Each rule dict should contain:
      - 'mask': (H,W) float32 in [0,1]
      - 'color': (r,g,b) ints 0-255
      - 'intensity': float in [0,1]
      - 'feather': int pixels
      - 'enabled': bool
    Returns BGR uint8 image.
    """
    gray = ensure_gray(gray)
    h, w = gray.shape
    # Start with Lab image with L from gray, a,b = 0 (i.e., 128 after offset)
    L = gray.astype(np.float32)
    a_channel = np.zeros_like(L, dtype=np.float32)
    b_channel = np.zeros_like(L, dtype=np.float32)

    if mode == 'blend':
        accum_a = np.zeros_like(L, dtype=np.float32)
        accum_b = np.zeros_like(L, dtype=np.float32)
        accum_w = np.zeros_like(L, dtype=np.float32)

        for rule in rules:
            if not rule.get('enabled', True):
                continue
            mask = rule['mask']
            if mask is None:
                continue
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
            feather_px = int(max(0, rule.get('feather', 0)))
            if feather_px > 0:
                mask = _feather_mask(mask, feather_px)
            intensity = float(np.clip(rule.get('intensity', 1.0), 0.0, 1.0))
            weight = mask * intensity
            rgb = rule['color']
            target_a_c, target_b_c = rgb_to_lab_chroma((int(rgb[0]), int(rgb[1]), int(rgb[2])))
            accum_a += weight * target_a_c
            accum_b += weight * target_b_c
            accum_w += weight

        # Avoid division by zero; where sum weights == 0, keep zero chroma
        nonzero = accum_w > 1e-6
        a_channel[nonzero] = accum_a[nonzero] / accum_w[nonzero]
        b_channel[nonzero] = accum_b[nonzero] / accum_w[nonzero]
        # where zero, stays at 0 (neutral chroma)
    else:
        # priority compositing (later rules override earlier where they have weight)
        for rule in rules:
            if not rule.get('enabled', True):
                continue
            mask = rule['mask']
            if mask is None:
                continue
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
            feather_px = int(max(0, rule.get('feather', 0)))
            if feather_px > 0:
                mask = _feather_mask(mask, feather_px)
            intensity = float(np.clip(rule.get('intensity', 1.0), 0.0, 1.0))
            weight = mask * intensity
            rgb = rule['color']
            target_a_c, target_b_c = rgb_to_lab_chroma((int(rgb[0]), int(rgb[1]), int(rgb[2])))
            a_channel = a_channel * (1.0 - weight) + (target_a_c * weight)
            b_channel = b_channel * (1.0 - weight) + (target_b_c * weight)

    # Reconstruct Lab image; OpenCV expects L in [0,255], a,b in [0,255] with 128 center
    lab = np.stack([
        np.clip(L, 0, 255),
        np.clip(a_channel + 128.0, 0, 255),
        np.clip(b_channel + 128.0, 0, 255),
    ], axis=2).astype(np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def normalize_and_fit_image(image_bgr: np.ndarray, max_width: int = 1024, max_height: int = 1024) -> np.ndarray:
    """
    Resize image to fit within max dimensions while keeping aspect ratio.
    Returns BGR image.
    """
    if image_bgr is None:
        raise ValueError("Image is None")
    h, w = image_bgr.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_bgr


def hex_to_rgb(hex_color: str) -> ColorRGB:
    """
    Convert '#RRGGBB' or 'RRGGBB' to (R,G,B).
    """
    s = hex_color.lstrip('#')
    if len(s) != 6:
        raise ValueError("Color hex must be 6 characters")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def rgb_to_hex(rgb: ColorRGB) -> str:
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])


def to_display_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR to RGB for display (Streamlit shows RGB). Keeps dtype and shape.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def from_uploaded_file_to_bgr(file_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded file bytes using OpenCV imdecode. Returns BGR image (H,W,3) uint8.
    If input is grayscale, converts to BGR for uniform processing/display.
    """
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode image")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def save_image_bytes(image_bgr: np.ndarray, ext: str = '.png') -> bytes:
    """
    Encode image to bytes with given extension.
    """
    success, buf = cv2.imencode(ext, image_bgr)
    if not success:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()


