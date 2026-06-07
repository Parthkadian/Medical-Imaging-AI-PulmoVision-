import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ======================
# DATA LOADER
# ======================
def load_data(path):
    images = sorted(glob(os.path.join(path, "images", "*")))
    masks = sorted(glob(os.path.join(path, "masks", "*")))
    return images, masks


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype("float32") / 255.0
    mask = (mask > 0.5).astype("float32")
    mask = np.expand_dims(mask, axis=-1)
    return mask


def augment_pair(image, mask):
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    if np.random.rand() < 0.15:
        image = np.flipud(image)
        mask = np.flipud(mask)

    if np.random.rand() < 0.3:
        k = np.random.choice([1, 2, 3])
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)

    if np.random.rand() < 0.4:
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-0.08, 0.08)
        image = np.clip(image * alpha + beta, 0.0, 1.0)

    if np.random.rand() < 0.2:
        noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    return image.astype(np.float32), mask.astype(np.float32)


def data_generator(images, masks, batch_size, augment=False):
    while True:
        indices = np.arange(len(images))
        np.random.shuffle(indices)

        shuffled_images = [images[i] for i in indices]
        shuffled_masks = [masks[i] for i in indices]

        for i in range(0, len(shuffled_images), batch_size):
            batch_imgs = shuffled_images[i:i + batch_size]
            batch_masks = shuffled_masks[i:i + batch_size]

            x_batch = []
            y_batch = []

            for img_path, mask_path in zip(batch_imgs, batch_masks):
                x = read_image(img_path)
                y = read_mask(mask_path)

                if augment:
                    x, y = augment_pair(x, y)

                x_batch.append(x)
                y_batch.append(y)

            yield np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)


# ======================
# LOSS + METRICS
# ======================
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def iou_score(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    return (true_positives + 1e-6) / (predicted_positives + 1e-6)


def recall_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    possible_positives = tf.reduce_sum(y_true)
    return (true_positives + 1e-6) / (possible_positives + 1e-6)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + (1.0 - dice_coefficient(y_true, y_pred))


# ======================
# MODEL BLOCKS
# ======================
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


def attention_block(g, x, filters):
    g1 = layers.Conv2D(filters, 1, padding="same")(g)
    x1 = layers.Conv2D(filters, 1, padding="same")(x)

    psi = layers.Add()([g1, x1])
    psi = layers.Activation("relu")(psi)
    psi = layers.Conv2D(1, 1, padding="same")(psi)
    psi = layers.Activation("sigmoid")(psi)

    return layers.Multiply()([x, psi])


# ======================
# MODEL
# ======================
def build_model():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1), name="input_gray")

    x = layers.Concatenate()([inputs, inputs, inputs])

    encoder = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    # Fine-tune later layers, freeze some early layers
    for layer in encoder.layers[:100]:
        layer.trainable = False
    for layer in encoder.layers[100:]:
        layer.trainable = True

    s1 = encoder.get_layer("block2a_expand_activation").output   # 128x128
    s2 = encoder.get_layer("block3a_expand_activation").output   # 64x64
    s3 = encoder.get_layer("block4a_expand_activation").output   # 32x32
    s4 = encoder.get_layer("block6a_expand_activation").output   # 16x16
    b1 = encoder.get_layer("top_activation").output              # 8x8

    d1 = layers.UpSampling2D((2, 2))(b1)                         # 16x16
    s4_att = attention_block(d1, s4, 256)
    d1 = layers.Concatenate()([d1, s4_att])
    d1 = conv_block(d1, 256, dropout_rate=0.2)

    d2 = layers.UpSampling2D((2, 2))(d1)                         # 32x32
    s3_att = attention_block(d2, s3, 128)
    d2 = layers.Concatenate()([d2, s3_att])
    d2 = conv_block(d2, 128, dropout_rate=0.15)

    d3 = layers.UpSampling2D((2, 2))(d2)                         # 64x64
    s2_att = attention_block(d3, s2, 64)
    d3 = layers.Concatenate()([d3, s2_att])
    d3 = conv_block(d3, 64, dropout_rate=0.10)

    d4 = layers.UpSampling2D((2, 2))(d3)                         # 128x128
    s1_att = attention_block(d4, s1, 32)
    d4 = layers.Concatenate()([d4, s1_att])
    d4 = conv_block(d4, 32, dropout_rate=0.05)

    d5 = layers.UpSampling2D((2, 2))(d4)                         # 256x256
    d5 = conv_block(d5, 16, dropout_rate=0.05)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(d5)

    return Model(inputs, outputs, name="AttentionUNet_EfficientNetB0")


# ======================
# TRAIN
# ======================
def main():
    train_path = "data/train"
    val_path = "data/val"

    train_images, train_masks = load_data(train_path)
    val_images, val_masks = load_data(val_path)

    print(f"Train images: {len(train_images)}")
    print(f"Train masks: {len(train_masks)}")
    print(f"Val images: {len(val_images)}")
    print(f"Val masks: {len(val_masks)}")

    if len(train_images) == 0 or len(train_masks) == 0:
        raise ValueError("Training data not found. Check data/train/images and data/train/masks")

    if len(val_images) == 0 or len(val_masks) == 0:
        raise ValueError("Validation data not found. Check data/val/images and data/val/masks")

    train_gen = data_generator(train_images, train_masks, BATCH_SIZE, augment=True)
    val_gen = data_generator(val_images, val_masks, BATCH_SIZE, augment=False)

    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=bce_dice_loss,
        metrics=[dice_coefficient, iou_score, precision_metric, recall_metric]
    )

    os.makedirs("models", exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            "models/best_attention_unet.h5",
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger("models/training_log.csv", append=False)
    ]

    steps_per_epoch = max(1, len(train_images) // BATCH_SIZE)
    validation_steps = max(1, len(val_images) // BATCH_SIZE)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Training complete. Best model saved to models/best_attention_unet.h5")
    return history


if __name__ == "__main__":
    main()