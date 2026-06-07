import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.model import dice_coefficient, iou_score, bce_dice_loss


IMAGE_DIR = "data/raw/images"
MASK_DIR = "data/raw/masks"
MODEL_PATH = "models/lung_unet_model.h5"
IMAGE_SIZE = (256, 256)
SEED = 42


def load_dataset_paths(image_dir, mask_dir):
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    image_paths = []
    mask_paths = []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if os.path.exists(mask_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)

    return image_paths, mask_paths


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    return image


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMAGE_SIZE)
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")

    image_paths, mask_paths = load_dataset_paths(IMAGE_DIR, MASK_DIR)

    if len(image_paths) == 0:
        raise ValueError("No image-mask pairs found for evaluation.")

    _, test_img, _, test_mask = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=SEED
    )

    X_test = np.array([read_image(p) for p in test_img])
    y_test = np.array([read_mask(p) for p in test_mask])

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "iou_score": iou_score,
            "bce_dice_loss": bce_dice_loss
        }
    )

    results = model.evaluate(X_test, y_test, verbose=1)

    metric_names = model.metrics_names
    evaluation = dict(zip(metric_names, results))

    print("Evaluation Results:")
    for key, value in evaluation.items():
        print(f"{key}: {value:.4f}")

    return evaluation


if __name__ == "__main__":
    evaluate_model()