import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

IMAGE_DIR = Path("data/raw/indiana/CXR_png")
MASK_DIR = Path("data/raw/indiana/GTMask/combined")
MODEL_SAVE_PATH = Path("models/lung_unet_model.h5")

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def weighted_bce_loss(y_true, y_pred, pos_weight=3.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    bce = -(pos_weight * y_true * tf.math.log(y_pred) +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)


def combined_loss(y_true, y_pred):
    return 0.5 * weighted_bce_loss(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)


def normalize_stem(path_obj: Path) -> str:
    stem = path_obj.stem
    stem = re.sub(r"^\d+_", "", stem)
    return stem.lower()


def get_image_mask_pairs(image_dir: Path, mask_dir: Path):
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise ValueError(f"Mask directory not found: {mask_dir}")

    image_files = sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])
    mask_files = sorted([
        p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])

    mask_map = {normalize_stem(p): p for p in mask_files}

    pairs = []
    missing_masks = []

    for img_path in image_files:
        key = normalize_stem(img_path)
        mask_path = mask_map.get(key)
        if mask_path is not None:
            pairs.append((str(img_path), str(mask_path)))
        else:
            missing_masks.append(img_path.name)

    print(f"[INFO] Image files found: {len(image_files)}")
    print(f"[INFO] Mask files found: {len(mask_files)}")
    print(f"[INFO] Matched pairs found: {len(pairs)}")

    if missing_masks:
        print(f"[WARNING] {len(missing_masks)} images do not have matching masks.")
        print("Examples:", missing_masks[:10])

    if not pairs:
        raise ValueError("No valid image-mask pairs found.")

    return pairs


def split_dataset(pairs, val_split=0.15, test_split=0.15):
    pairs = pairs.copy()
    random.shuffle(pairs)
    total = len(pairs)
    test_size = max(1, int(total * test_split))
    val_size = max(1, int(total * val_split))
    test_pairs = pairs[:test_size]
    val_pairs = pairs[test_size:test_size + val_size]
    train_pairs = pairs[test_size + val_size:]
    return train_pairs, val_pairs, test_pairs


def load_image_mask(image_path, mask_path):
    image = Image.open(image_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    mask = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image, dtype=np.float32) / 255.0
    mask = np.array(mask, dtype=np.float32)
    mask = (mask > 127).astype(np.float32)

    image = np.expand_dims(image, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask


def tf_load_image_mask(image_path, mask_path):
    def _loader(x, y):
        x = x.decode() if isinstance(x, bytes) else x
        y = y.decode() if isinstance(y, bytes) else y
        return load_image_mask(x, y)

    image, mask = tf.numpy_function(
        func=_loader,
        inp=[image_path, mask_path],
        Tout=[tf.float32, tf.float32]
    )

    image.set_shape((IMG_SIZE, IMG_SIZE, 1))
    mask.set_shape((IMG_SIZE, IMG_SIZE, 1))
    return image, mask


def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        mask = tf.image.rot90(mask, k)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.08)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask


def build_dataset(pairs, training=False):
    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(tf_load_image_mask, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(buffer_size=max(len(pairs), 1), seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def conv_block(x, filters, dropout_rate=0.0):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def encoder_block(x, filters, dropout_rate=0.0):
    c = conv_block(x, filters, dropout_rate)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p


def decoder_block(x, skip, filters, dropout_rate=0.0):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, dropout_rate)
    return x


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 16, 0.05)
    s2, p2 = encoder_block(p1, 32, 0.05)
    s3, p3 = encoder_block(p2, 64, 0.10)
    s4, p4 = encoder_block(p3, 128, 0.10)

    b1 = conv_block(p4, 256, 0.20)

    d1 = decoder_block(b1, s4, 128, 0.10)
    d2 = decoder_block(d1, s3, 64, 0.10)
    d3 = decoder_block(d2, s2, 32, 0.05)
    d4 = decoder_block(d3, s1, 16, 0.05)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)
    return models.Model(inputs, outputs, name="u_net_lung_segmentation")


def main():
    print("[INFO] Collecting image-mask pairs...")
    pairs = get_image_mask_pairs(IMAGE_DIR, MASK_DIR)

    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT
    )

    print(f"[INFO] Train pairs: {len(train_pairs)}")
    print(f"[INFO] Val pairs:   {len(val_pairs)}")
    print(f"[INFO] Test pairs:  {len(test_pairs)}")

    train_ds = build_dataset(train_pairs, training=True)
    val_ds = build_dataset(val_pairs, training=False)
    test_ds = build_dataset(test_pairs, training=False)

    print("[INFO] Building model...")
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coefficient, iou_score]
    )

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    cbs = [
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.CSVLogger("training_log.csv")
    ]

    print("[INFO] Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1
    )

    print("[INFO] Evaluating test set...")
    results = model.evaluate(test_ds, verbose=1)

    print("\n[INFO] Test Results:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    print(f"\n[INFO] Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()