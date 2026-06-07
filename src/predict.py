import os
import logging
import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    label as ndimage_label,
)

logger = logging.getLogger("pulmovision.predict")

MODEL_PATH = "models/best_attention_unet.h5"
_model = None


def model_exists():
    return os.path.exists(MODEL_PATH)


def load_model_once():
    global _model
    if _model is None:
        import tensorflow as tf
        from src.metrics import (
            dice_coefficient,
            iou_score,
            precision_metric,
            recall_metric,
        )
        from src.losses import bce_dice_loss

        logger.info("Loading Attention U-Net model from %s...", MODEL_PATH)
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "dice_coefficient": dice_coefficient,
                "iou_score": iou_score,
                "precision_metric": precision_metric,
                "recall_metric": recall_metric,
                "bce_dice_loss": bce_dice_loss,
            },
            compile=False,
        )
        logger.info("Model loaded successfully.")

    return _model


def simple_fallback_segmentation(image):
    gray = image.convert("L")
    arr = np.array(gray).astype(np.uint8)
    threshold = np.mean(arr)
    mask = np.where(arr > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def keep_largest_components(binary_mask: np.ndarray, num_components: int = 2) -> np.ndarray:
    """
    Keep the N largest connected components using scipy (fast).
    Replaces the old pure-Python BFS implementation.
    """
    labeled, num_features = ndimage_label(binary_mask > 0)
    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    # Compute component sizes and sort descending
    sizes = np.array([(labeled == i).sum() for i in range(1, num_features + 1)])
    top_indices = np.argsort(sizes)[::-1][:num_components] + 1  # 1-indexed labels

    cleaned = np.zeros_like(binary_mask, dtype=np.uint8)
    for idx in top_indices:
        cleaned[labeled == idx] = 255
    return cleaned


def remove_small_components(binary_mask: np.ndarray, area_threshold: int = 1500) -> np.ndarray:
    """
    Remove connected components smaller than area_threshold pixels (fast scipy version).
    """
    labeled, num_features = ndimage_label(binary_mask > 0)
    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    cleaned = np.zeros_like(binary_mask, dtype=np.uint8)
    for i in range(1, num_features + 1):
        region = labeled == i
        if region.sum() >= area_threshold:
            cleaned[region] = 255
    return cleaned


def clear_border_regions(binary_mask: np.ndarray) -> np.ndarray:
    """
    Remove connected white regions touching image borders using scipy.
    Eliminates background blobs and top-edge artefacts.
    """
    from scipy.ndimage import binary_dilation

    arr = (binary_mask > 0).astype(bool)
    h, w = arr.shape

    # Seed from the border
    seed = np.zeros_like(arr)
    seed[0, :] = arr[0, :]
    seed[-1, :] = arr[-1, :]
    seed[:, 0] = arr[:, 0]
    seed[:, -1] = arr[:, -1]

    # Flood-fill from border using scipy dilation
    border_connected = np.zeros_like(arr)
    border_connected[seed] = True

    # Iteratively grow the border seed through connected foreground pixels
    prev = None
    while not np.array_equal(border_connected, prev):
        prev = border_connected.copy()
        dilated = binary_dilation(border_connected)
        border_connected = dilated & arr

    interior = arr & ~border_connected
    return (interior.astype(np.uint8) * 255)


def enforce_lung_side_constraints(binary_mask):
    """
    Keep the strongest component on the left half and the strongest on the right half.
    This stabilizes output when the model leaks into the centre/background.
    """
    h, w = binary_mask.shape
    mid = w // 2

    left_half = binary_mask[:, :mid].copy()
    right_half = binary_mask[:, mid:].copy()

    left_half = keep_largest_components(left_half, num_components=1)
    right_half = keep_largest_components(right_half, num_components=1)

    combined = np.zeros_like(binary_mask, dtype=np.uint8)
    combined[:, :mid] = left_half
    combined[:, mid:] = right_half

    return combined


def postprocess_mask(pred_mask):
    """
    Stable postprocessing for weak model outputs.
    Improvements:
    - normalization to visible range
    - percentile thresholding
    - border artifact removal
    - light morphology
    - hole filling
    - left/right lung stabilization
    - small noise removal
    """
    pred_mask = np.array(pred_mask, dtype=np.float32)
    pred_mask = np.squeeze(pred_mask)

    if pred_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape: {pred_mask.shape}")

    logger.debug(
        "Raw prediction stats → shape=%s min=%.4f max=%.4f mean=%.4f",
        pred_mask.shape, float(pred_mask.min()),
        float(pred_mask.max()), float(pred_mask.mean()),
    )

    # Scale probabilities if model outputs 0-1
    if pred_mask.max() <= 1.0:
        pred_mask = pred_mask * 255.0

    # Normalize to 0-255
    min_val = float(pred_mask.min())
    max_val = float(pred_mask.max())
    pred_mask = (pred_mask - min_val) / (max_val - min_val + 1e-6)
    pred_mask = (pred_mask * 255.0).astype(np.uint8)

    # Choose a stricter threshold than before
    candidate_thresholds = [
        np.percentile(pred_mask, 80),
        np.percentile(pred_mask, 75),
        np.percentile(pred_mask, 70),
        np.mean(pred_mask),
    ]

    best_binary = None
    best_score = -1.0

    for t in candidate_thresholds:
        binary = np.where(pred_mask >= t, 255, 0).astype(np.uint8)

        # remove a thin frame explicitly
        binary[:6, :] = 0
        binary[-6:, :] = 0
        binary[:, :6] = 0
        binary[:, -6:] = 0

        # remove border-connected blobs
        binary = clear_border_regions(binary)

        # light morphology only
        binary_bool = binary > 0
        binary_bool = binary_closing(binary_bool, iterations=1)
        binary_bool = binary_fill_holes(binary_bool)
        binary = (binary_bool.astype(np.uint8) * 255)

        # keep most likely lung regions
        binary = keep_largest_components(binary, num_components=3)
        binary = enforce_lung_side_constraints(binary)
        binary = remove_small_components(binary, area_threshold=1500)

        coverage = float((binary > 0).mean())

        # prefer plausible lung coverage
        if 0.05 <= coverage <= 0.45:
            score = 1.0 - abs(coverage - 0.22)
        else:
            score = -abs(coverage - 0.22)

        if score > best_score:
            best_score = score
            best_binary = binary

    if best_binary is None:
        best_binary = np.zeros_like(pred_mask, dtype=np.uint8)

    coverage = float((best_binary > 0).mean() * 100.0)
    logger.info("Final postprocessed mask coverage: %.2f%%", coverage)

    return Image.fromarray(best_binary, mode="L")


def predict_mask_with_model(model_input):
    model = load_model_once()
    raw_prediction = model.predict(model_input, verbose=0)
    return postprocess_mask(raw_prediction)


def predict_mask(image, model_input=None):
    if model_exists() and model_input is not None:
        try:
            mask = predict_mask_with_model(model_input)
            return mask, "Attention U-Net inference active"
        except Exception as e:
            print("Model inference failed, using fallback:", str(e))
            fallback_mask = simple_fallback_segmentation(image)
            return fallback_mask, f"Fallback used due to error: {str(e)}"

    return simple_fallback_segmentation(image), "Fallback inference active (no trained model connected)"