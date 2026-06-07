"""
dicom_loader.py
---------------
Load DICOM (.dcm) files and convert them to PIL RGB images
with proper window/level normalisation.

Requires: pydicom
"""
import io
import numpy as np
from PIL import Image


def _apply_windowing(pixel_array: np.ndarray,
                     window_center: float,
                     window_width: float) -> np.ndarray:
    """Apply radiological window/level to a raw pixel array."""
    lo = window_center - window_width / 2.0
    hi = window_center + window_width / 2.0
    clipped = np.clip(pixel_array.astype(np.float32), lo, hi)
    normalised = (clipped - lo) / (hi - lo) * 255.0
    return normalised.astype(np.uint8)


def _auto_window(pixel_array: np.ndarray) -> np.ndarray:
    """Fallback: auto window using percentile stretch."""
    p2, p98 = np.percentile(pixel_array, 2), np.percentile(pixel_array, 98)
    stretched = np.clip(pixel_array.astype(np.float32), p2, p98)
    if p98 > p2:
        stretched = (stretched - p2) / (p98 - p2) * 255.0
    return stretched.astype(np.uint8)


def load_dicom_from_bytes(dcm_bytes: bytes) -> Image.Image:
    """
    Load a DICOM file from raw bytes and return a PIL RGB Image.

    Applies window/level from the DICOM metadata if available,
    otherwise falls back to percentile-based auto-windowing.

    Returns:
        PIL.Image in 'RGB' mode, ready for model preprocessing.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        dcm = pydicom.dcmread(io.BytesIO(dcm_bytes), force=True)
    except InvalidDicomError as e:
        raise ValueError(f"Invalid DICOM file: {e}")

    if not hasattr(dcm, "pixel_array"):
        raise ValueError("DICOM file does not contain pixel data.")

    pixels = dcm.pixel_array.astype(np.float32)

    # Handle multi-frame DICOM (take the middle frame)
    if pixels.ndim == 3:
        mid = pixels.shape[0] // 2
        pixels = pixels[mid]

    # Apply rescale slope / intercept if present
    slope = float(getattr(dcm, "RescaleSlope", 1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    pixels = pixels * slope + intercept

    # Apply window/level if available
    wc = getattr(dcm, "WindowCenter", None)
    ww = getattr(dcm, "WindowWidth", None)

    if wc is not None and ww is not None:
        wc = float(wc[0]) if hasattr(wc, "__iter__") and not isinstance(wc, str) else float(wc)
        ww = float(ww[0]) if hasattr(ww, "__iter__") and not isinstance(ww, str) else float(ww)
        gray = _apply_windowing(pixels, wc, ww)
    else:
        gray = _auto_window(pixels)

    # Convert grayscale to RGB PIL Image
    pil_gray = Image.fromarray(gray, mode="L")
    pil_rgb = pil_gray.convert("RGB")
    return pil_rgb


def load_dicom_from_path(path: str) -> Image.Image:
    """Load a DICOM file from a filesystem path."""
    with open(path, "rb") as f:
        return load_dicom_from_bytes(f.read())
