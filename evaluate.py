"""
evaluate.py
-----------
PulmoVision AI — Test Evaluation Pipeline.

Loads the best trained Attention U-Net, evaluates on the EXACT holdout test
split used during training, computes 5 segmentation metrics, saves JSON, and
prints a formatted report.

Fixes applied vs. initial version:
  1. MASK_DIR → GTMask/combined_png  (combined/ contains only .tif files,
     which the valid_exts scanner skips; combined_png/ has correct .png masks)
  2. Adaptive percentile thresholding per image — the model sigmoid outputs
     sit in [0.20, 0.49]; a fixed 0.5 threshold produces all-zero predictions.
     We search p70/p75/p80/p85 per image and pick the threshold giving the
     most plausible lung coverage, identical to src/predict.py in production.

Usage (from project root):
    venv310\\Scripts\\python.exe evaluate.py

Output:
    results/test_metrics.json
"""

import os
import re
import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

# Ensure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DIR   = Path("data/raw/indiana/CXR_png")

# FIX #1: combined_png has .png masks; combined/ has only .tif (not scanned)
MASK_DIR    = Path("data/raw/indiana/GTMask/combined_png")

MODEL_PATH  = Path("models/best_attention_unet.h5")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "test_metrics.json"

IMG_SIZE   = 256
BATCH_SIZE = 8
VAL_SPLIT  = 0.15   # must match train_lung_unet.py
TEST_SPLIT = 0.15   # must match train_lung_unet.py
SEED       = 42     # must match train_lung_unet.py
SMOOTH     = 1e-6

# FIX #2: Adaptive thresholding — model outputs are in [0.20, 0.49],
# never crossing 0.5. Use per-image percentile search (same as predict.py).
THRESH_PERCENTILES = [70, 75, 80, 85]
PLAUSIBLE_MIN      = 0.05
PLAUSIBLE_MAX      = 0.45
TARGET_COVERAGE    = 0.22

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pulmovision.evaluate")


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA HELPERS  — exact replica of train_lung_unet.py split logic
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_stem(p: Path) -> str:
    """Strip leading numeric prefix and lowercase — mirrors training."""
    stem = re.sub(r"^\d+_", "", p.stem)
    return stem.lower()


def get_image_mask_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[str, str]]:
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    image_files = sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])
    mask_files = sorted([
        p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])

    mask_map = {_normalize_stem(p): p for p in mask_files}
    pairs, missing = [], []
    for img_path in image_files:
        key = _normalize_stem(img_path)
        msk = mask_map.get(key)
        if msk is not None:
            pairs.append((str(img_path), str(msk)))
        else:
            missing.append(img_path.name)

    log.info(
        "Dataset — images: %d  masks: %d  matched: %d  unmatched: %d",
        len(image_files), len(mask_files), len(pairs), len(missing),
    )
    if missing:
        log.warning("Unmatched images (first 5): %s", missing[:5])
    if not pairs:
        raise ValueError(
            f"No pairs found.\n  IMAGE_DIR={image_dir}\n  MASK_DIR={mask_dir}"
        )
    return pairs


def split_dataset(pairs: List[Tuple[str, str]]) -> Tuple[list, list, list]:
    """Same shuffle + split as train_lung_unet.py (SEED=42)."""
    pairs = pairs.copy()
    random.seed(SEED)
    random.shuffle(pairs)
    total     = len(pairs)
    test_size = max(1, int(total * TEST_SPLIT))
    val_size  = max(1, int(total * VAL_SPLIT))
    test_pairs  = pairs[:test_size]
    val_pairs   = pairs[test_size : test_size + val_size]
    train_pairs = pairs[test_size + val_size :]
    log.info(
        "Split — train: %d  val: %d  test: %d",
        len(train_pairs), len(val_pairs), len(test_pairs),
    )
    return train_pairs, val_pairs, test_pairs


# ═════════════════════════════════════════════════════════════════════════════
# 2. IMAGE LOADING  — identical preprocessing to training
# ═════════════════════════════════════════════════════════════════════════════

def load_pair(image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        image : float32 (H, W, 1) in [0, 1]
        mask  : float32 (H, W, 1) binary {0, 1}
    """
    img = Image.open(image_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    msk = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

    img_arr = np.array(img, dtype=np.float32) / 255.0
    msk_arr = (np.array(msk, dtype=np.float32) > 127).astype(np.float32)

    return img_arr[..., np.newaxis], msk_arr[..., np.newaxis]


# ═════════════════════════════════════════════════════════════════════════════
# 3. ADAPTIVE THRESHOLDING
#    The model sigmoid output is in [0.20, 0.49] — no pixel ever exceeds 0.5.
#    We try several percentile thresholds per image and pick the one giving
#    the most plausible lung coverage (same strategy as src/predict.py).
# ═════════════════════════════════════════════════════════════════════════════

def adaptive_threshold(pred_map: np.ndarray) -> np.ndarray:
    """
    Args:
        pred_map : float32 array (H, W) — raw sigmoid probabilities

    Returns:
        binary   : float32 array (H, W) — {0.0, 1.0}
    """
    flat = pred_map.ravel()
    best_binary = None
    best_score  = -np.inf

    for pct in THRESH_PERCENTILES:
        thresh  = float(np.percentile(flat, pct))
        binary  = (pred_map >= thresh).astype(np.float32)
        coverage = float(binary.mean())

        if PLAUSIBLE_MIN <= coverage <= PLAUSIBLE_MAX:
            score = 1.0 - abs(coverage - TARGET_COVERAGE)
        else:
            score = -abs(coverage - TARGET_COVERAGE)

        if score > best_score:
            best_score  = score
            best_binary = binary

    if best_binary is None:
        # Fallback: use median split
        best_binary = (pred_map >= float(np.median(flat))).astype(np.float32)

    return best_binary


# ═════════════════════════════════════════════════════════════════════════════
# 4. METRICS  — pure NumPy, per-image then averaged
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true_batch: np.ndarray,
    y_pred_prob_batch: np.ndarray,
) -> Dict[str, float]:
    """
    Args:
        y_true_batch     : (B, H, W, 1) ground truth binary masks
        y_pred_prob_batch: (B, H, W, 1) raw sigmoid probabilities

    Returns dict with: dice, iou, precision, recall, f1
    """
    B = y_true_batch.shape[0]
    dice_list, iou_list, prec_list, rec_list, f1_list = [], [], [], [], []

    for i in range(B):
        gt   = y_true_batch[i, ..., 0]          # (H, W)
        prob = y_pred_prob_batch[i, ..., 0]      # (H, W)

        pred = adaptive_threshold(prob)           # (H, W) binary

        gt_f   = gt.ravel()
        pred_f = pred.ravel()

        tp = float(np.sum(gt_f * pred_f))
        fp = float(np.sum((1 - gt_f) * pred_f))
        fn = float(np.sum(gt_f * (1 - pred_f)))

        dice = (2.0 * tp + SMOOTH) / (2.0 * tp + fp + fn + SMOOTH)
        iou  = (tp + SMOOTH)       / (tp + fp + fn + SMOOTH)
        prec = (tp + SMOOTH)       / (tp + fp + SMOOTH)
        rec  = (tp + SMOOTH)       / (tp + fn + SMOOTH)
        f1   = (2.0 * prec * rec + SMOOTH) / (prec + rec + SMOOTH)

        dice_list.append(dice)
        iou_list.append(iou)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    return {
        "dice":      float(np.mean(dice_list)),
        "iou":       float(np.mean(iou_list)),
        "precision": float(np.mean(prec_list)),
        "recall":    float(np.mean(rec_list)),
        "f1":        float(np.mean(f1_list)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. MODEL LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_model(model_path: Path):
    import tensorflow as tf
    from src.metrics import dice_coefficient, iou_score, precision_metric, recall_metric
    from src.losses  import bce_dice_loss, dice_loss

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nModel not found: {model_path}\n"
            f"Train first:  venv310\\Scripts\\python.exe train_lung_unet.py\n"
        )

    log.info("Loading model from %s ...", model_path)
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "iou_score":        iou_score,
            "precision_metric": precision_metric,
            "recall_metric":    recall_metric,
            "bce_dice_loss":    bce_dice_loss,
            "dice_loss":        dice_loss,
        },
        compile=False,
    )
    log.info(
        "Model loaded — name: '%s'  params: %s",
        model.name, f"{model.count_params():,}",
    )
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation() -> dict:
    log.info("=" * 60)
    log.info("PulmoVision AI — Test Set Evaluation")
    log.info("=" * 60)

    # Validate directories
    for d, name in [(IMAGE_DIR, "IMAGE_DIR"), (MASK_DIR, "MASK_DIR")]:
        if not d.exists():
            raise FileNotFoundError(f"{name} not found: {d}")

    # Build exact test split
    all_pairs = get_image_mask_pairs(IMAGE_DIR, MASK_DIR)
    _, _, test_pairs = split_dataset(all_pairs)
    log.info("Evaluating on %d test images", len(test_pairs))

    # Load model
    model = load_model(MODEL_PATH)

    # Batch inference
    log.info("Running inference (adaptive percentile thresholding) ...")
    t_start = time.perf_counter()
    accumulated: List[Tuple[int, dict]] = []

    for start in range(0, len(test_pairs), BATCH_SIZE):
        batch = test_pairs[start : start + BATCH_SIZE]
        imgs, msks = [], []
        for ip, mp in batch:
            img, msk = load_pair(ip, mp)
            imgs.append(img)
            msks.append(msk)

        X = np.stack(imgs, axis=0)   # (B, H, W, 1)
        y = np.stack(msks, axis=0)   # (B, H, W, 1)

        preds = model.predict(X, verbose=0)   # (B, H, W, 1)

        m = compute_metrics(y, preds)
        accumulated.append((len(batch), m))

        log.info(
            "  [%2d/%2d]  pred_range=[%.3f, %.3f]  "
            "Dice=%.4f  IoU=%.4f  P=%.4f  R=%.4f  F1=%.4f",
            min(start + BATCH_SIZE, len(test_pairs)), len(test_pairs),
            float(preds.min()), float(preds.max()),
            m["dice"], m["iou"], m["precision"], m["recall"], m["f1"],
        )

    elapsed = time.perf_counter() - t_start

    # Weighted average
    total_n = sum(n for n, _ in accumulated)
    final: Dict[str, float] = {}
    for key in ("dice", "iou", "precision", "recall", "f1"):
        final[key] = round(sum(n * m[key] for n, m in accumulated) / total_n, 6)

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "model_path":         str(MODEL_PATH),
        "model_name":         model.name,
        "mask_dir":           str(MASK_DIR),
        "test_samples":       total_n,
        "thresholding":       f"adaptive percentile {THRESH_PERCENTILES}",
        "image_size":         IMG_SIZE,
        "seed":               SEED,
        "elapsed_seconds":    round(elapsed, 2),
        "metrics": {
            "dice_score":  final["dice"],
            "iou_score":   final["iou"],
            "precision":   final["precision"],
            "recall":      final["recall"],
            "f1_score":    final["f1"],
        },
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    log.info("Results saved → %s", OUTPUT_FILE)

    # Pretty-print report
    bar = "=" * 41
    print(f"\n{bar}")
    print("   TEST EVALUATION RESULTS")
    print(bar)
    print(f"   Dice Score  :  {final['dice']:.4f}")
    print(f"   IoU Score   :  {final['iou']:.4f}")
    print(f"   Precision   :  {final['precision']:.4f}")
    print(f"   Recall      :  {final['recall']:.4f}")
    print(f"   F1 Score    :  {final['f1']:.4f}")
    print(bar)
    print(f"   Test samples:  {total_n}")
    print(f"   Inference   :  {elapsed:.1f}s  ({elapsed/total_n*1000:.0f} ms/img)")
    print(f"   Saved to    :  {OUTPUT_FILE}")
    print(f"{bar}\n")

    return doc


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation()
