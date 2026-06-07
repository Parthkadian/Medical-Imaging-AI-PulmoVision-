import numpy as np
from PIL import Image, ImageOps


def prepare_for_model(image: Image.Image, target_size=(256, 256)):
    """
    Returns:
        resized_image: PIL image resized for downstream display / API use
        model_input: NumPy array shaped (1, H, W, 1)
    """
    gray = ImageOps.grayscale(image)
    resized = gray.resize(target_size)

    arr = np.array(resized).astype("float32")
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    arr = arr / (np.max(np.abs(arr)) + 1e-6)
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)

    return resized.convert("RGB"), arr