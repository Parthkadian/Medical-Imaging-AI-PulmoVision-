import numpy as np
from PIL import Image, ImageFilter


def create_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGB")
    mask = mask.convert("L").resize(base.size)

    mask_arr = np.array(mask)

    # FIX: do NOT crash
    if mask_arr.max() == 0:
        return base  # fallback instead of error

    base_arr = np.array(base).astype(np.uint8)
    mask_arr = mask_arr.astype(np.uint8)

    edge_img = mask.filter(ImageFilter.FIND_EDGES).convert("L")
    edge_arr = np.array(edge_img).astype(np.uint8)
    edge_arr = np.where(edge_arr > 20, 255, 0).astype(np.uint8)

    overlay_arr = base_arr.copy()

    mask_region = mask_arr > 0
    teal = np.array([35, 214, 196], dtype=np.uint8)
    overlay_arr[mask_region] = (
        0.72 * overlay_arr[mask_region] + 0.28 * teal
    ).astype(np.uint8)

    edge_region = edge_arr > 0
    cyan = np.array([115, 250, 255], dtype=np.uint8)
    overlay_arr[edge_region] = cyan

    return Image.fromarray(overlay_arr, mode="RGB")

def compute_mask_coverage(mask: Image.Image) -> float:
    arr = np.array(mask.convert("L"))
    return round(float((arr > 0).mean() * 100.0), 2)