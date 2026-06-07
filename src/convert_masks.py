import os
from PIL import Image

INPUT_DIR = "data/raw/indiana/GTMask/combined"
OUTPUT_DIR = "data/raw/indiana/GTMask/combined_png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".tif")]

print(f"Converting {len(files)} masks...")

for file in files:
    path = os.path.join(INPUT_DIR, file)

    img = Image.open(path).convert("L")

    new_name = file.replace(".tif", ".png")
    save_path = os.path.join(OUTPUT_DIR, new_name)

    img.save(save_path)

print("✅ Conversion done")