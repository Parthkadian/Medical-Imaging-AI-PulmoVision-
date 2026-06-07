import os
from pathlib import Path
import numpy as np
from PIL import Image

LEFT_DIR = Path("data/raw/indiana/GTMask/leftMask")
RIGHT_DIR = Path("data/raw/indiana/GTMask/rightMask")
OUTPUT_DIR = Path("data/raw/indiana/GTMask/combined")

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    left_files = sorted([
        p for p in LEFT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ])

    if not left_files:
        raise ValueError(f"No mask files found in {LEFT_DIR}")

    count = 0
    skipped = []

    for left_path in left_files:
        right_path = RIGHT_DIR / left_path.name

        if not right_path.exists():
            skipped.append(left_path.name)
            continue

        left = Image.open(left_path).convert("L")
        right = Image.open(right_path).convert("L")

        left_arr = np.array(left, dtype=np.uint8)
        right_arr = np.array(right, dtype=np.uint8)

        combined = np.maximum(left_arr, right_arr)
        combined = (combined > 0).astype(np.uint8) * 255

        out_path = OUTPUT_DIR / left_path.name
        Image.fromarray(combined).save(out_path)
        count += 1

    print(f"Combined masks saved: {count}")
    if skipped:
        print(f"Skipped {len(skipped)} files with missing right mask.")
        print("Examples:", skipped[:10])


if __name__ == "__main__":
    main()