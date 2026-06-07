import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    BatchNormalization,
    Activation,
    Multiply,
    Add,
    Dropout,
)
from tensorflow.keras.models import Model


def conv_block(x, filters: int, dropout_rate: float = 0.3):
    """
    Two Conv-BN-ReLU layers with dropout for regularisation.
    """
    x = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dropout(dropout_rate)(x)

    x = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def attention_gate(skip, gating, inter_channels: int):
    """
    Attention gate to focus decoder on relevant encoder features.
    """
    theta_x = Conv2D(inter_channels, 1, padding="same")(skip)
    phi_g = Conv2D(inter_channels, 1, padding="same")(gating)

    add = Add()([theta_x, phi_g])
    add = Activation("relu")(add)

    psi = Conv2D(1, 1, padding="same")(add)
    psi = Activation("sigmoid")(psi)

    out = Multiply()([skip, psi])
    return out


def build_attention_unet(input_shape=(256, 256, 1), dropout_rate: float = 0.3):
    """
    Lightweight Attention U-Net for binary lung segmentation.
    Reduced filter sizes help with small datasets and lower overfitting.
    """
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32, dropout_rate)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64, dropout_rate)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128, dropout_rate)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256, dropout_rate)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    bn = conv_block(p4, 512, dropout_rate)

    # Decoder
    u5 = UpSampling2D((2, 2))(bn)
    a5 = attention_gate(c4, u5, 128)
    u5 = Concatenate()([u5, a5])
    c5 = conv_block(u5, 256, dropout_rate)

    u6 = UpSampling2D((2, 2))(c5)
    a6 = attention_gate(c3, u6, 64)
    u6 = Concatenate()([u6, a6])
    c6 = conv_block(u6, 128, dropout_rate)

    u7 = UpSampling2D((2, 2))(c6)
    a7 = attention_gate(c2, u7, 32)
    u7 = Concatenate()([u7, a7])
    c7 = conv_block(u7, 64, dropout_rate)

    u8 = UpSampling2D((2, 2))(c7)
    a8 = attention_gate(c1, u8, 16)
    u8 = Concatenate()([u8, a8])
    c8 = conv_block(u8, 32, dropout_rate)

    outputs = Conv2D(1, 1, activation="sigmoid")(c8)

    model = Model(inputs, outputs, name="Attention_UNet_Light")
    return model