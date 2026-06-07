import os
import shutil
import random

IMAGE_DIR = "data/raw/indiana/CXR_png"
MASK_DIR = "data/raw/indiana/GTMask/combined_png"

TRAIN_IMG = "data/train/images"
TRAIN_MASK = "data/train/masks"
VAL_IMG = "data/val/images"
VAL_MASK = "data/val/masks"

VAL_SPLIT = 0.15
SEED = 42

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_MASK, exist_ok=True)
os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_MASK, exist_ok=True)

random.seed(SEED)

image_files = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)

pairs = []

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, img_name)

    if os.path.exists(mask_path):
        pairs.append((img_name, img_path, mask_path))

print(f"Total valid pairs found: {len(pairs)}")

if len(pairs) == 0:
    raise ValueError("No matching image-mask pairs found. Check filenames and folders.")

random.shuffle(pairs)

split_idx = int(len(pairs) * (1 - VAL_SPLIT))
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]


def clear_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def copy_data(pairs_list, img_dest, mask_dest):
    for name, img_path, mask_path in pairs_list:
        shutil.copy2(img_path, os.path.join(img_dest, name))
        shutil.copy2(mask_path, os.path.join(mask_dest, name))


# Optional cleanup so rerun doesn't duplicate old files
clear_folder(TRAIN_IMG)
clear_folder(TRAIN_MASK)
clear_folder(VAL_IMG)
clear_folder(VAL_MASK)

copy_data(train_pairs, TRAIN_IMG, TRAIN_MASK)
copy_data(val_pairs, VAL_IMG, VAL_MASK)

print(f"Train samples: {len(train_pairs)}")
print(f"Validation samples: {len(val_pairs)}")
print("✅ Dataset preparation complete")