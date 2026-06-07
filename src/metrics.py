import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def iou_score(y_true, y_pred, smooth: float = 1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_metric(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_bin = K.flatten(tf.cast(y_pred > threshold, tf.float32))
    true_positive = K.sum(y_true_f * y_pred_bin)
    predicted_positive = K.sum(y_pred_bin)
    return (true_positive + smooth) / (predicted_positive + smooth)


def recall_metric(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_bin = K.flatten(tf.cast(y_pred > threshold, tf.float32))
    true_positive = K.sum(y_true_f * y_pred_bin)
    actual_positive = K.sum(y_true_f)
    return (true_positive + smooth) / (actual_positive + smooth)


def f1_score_metric(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """
    F1 = 2 · Precision · Recall / (Precision + Recall)

    For binary segmentation this is mathematically equivalent to the Dice
    coefficient when both are computed from the same TP/FP/FN counts, but
    it is kept separate so each metric is independently auditable.
    """
    p = precision_metric(y_true, y_pred, threshold, smooth)
    r = recall_metric(y_true, y_pred, threshold, smooth)
    return (2.0 * p * r + smooth) / (p + r + smooth)