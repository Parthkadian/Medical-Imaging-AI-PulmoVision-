import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf


def list_image_mask_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str]]:
    image_files = sorted(
        [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    pairs = []
    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)

        base_name, _ = os.path.splitext(image_name)
        possible_masks = [
            f"{base_name}.png",
            f"{base_name}.jpg",
            f"{base_name}.jpeg",
            f"{base_name}_mask.png",
            f"{base_name}_mask.jpg",
            f"{base_name}_mask.jpeg",
        ]

        mask_path = None
        for mask_name in possible_masks:
            candidate = os.path.join(masks_dir, mask_name)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is not None:
            pairs.append((image_path, mask_path))

    return pairs


def train_val_split(
    pairs: List[Tuple[str, str]],
    val_ratio: float = 0.15,
    seed: int = 42
):
    random.seed(seed)
    pairs = pairs.copy()
    random.shuffle(pairs)

    split_idx = int(len(pairs) * (1 - val_ratio))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    return train_pairs, val_pairs


def load_image_mask(
    image_path: str,
    mask_path: str,
    image_size: Tuple[int, int] = (256, 256)
):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)

    image = image.astype(np.float32) / 255.0
    mask = (mask > 127).astype(np.float32)

    image = np.expand_dims(image, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask


def augment_image_mask(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)

    return image, mask


def build_dataset(
    pairs: List[Tuple[str, str]],
    batch_size: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    training: bool = True
):
    images = []
    masks = []

    for image_path, mask_path in pairs:
        image, mask = load_image_mask(image_path, mask_path, image_size=image_size)
        images.append(image)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((images, masks))

    if training:
        dataset = dataset.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)
        dataset = dataset.map(augment_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset