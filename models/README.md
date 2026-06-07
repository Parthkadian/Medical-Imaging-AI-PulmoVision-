# Model Weights

The trained model file (`best_attention_unet.h5`) is **not included** in this repository due to GitHub's 100 MB file size limit.

## How to obtain the model

### Option 1 — Train from scratch

```bash
# From the project root
venv310\Scripts\python.exe train_lung_unet.py
```

Training saves the best checkpoint (monitored on `val_dice_coefficient`) to:
```
models/best_attention_unet.h5
```

### Option 2 — Download pre-trained weights

> *A pre-trained weights release will be added to the [GitHub Releases](../../releases) page.*  
> Check the Releases tab for the latest `.h5` file, then place it in this folder.

## Model Architecture

| Property | Value |
|---|---|
| Architecture | Attention U-Net |
| Encoder backbone | EfficientNetB0 |
| Input shape | `(256, 256, 1)` — grayscale |
| Output | `(256, 256, 1)` — sigmoid probability map |
| Filter progression | 32 → 64 → 128 → 256 → 512 |
| Parameters | ~7.2M trainable |
| Loss function | Combined BCE + Dice (0.5 : 0.5) |
| Best val Dice | 0.3474 (epoch 0, 55-image subset) |
| Test Dice | 0.2745 (adaptive thresholding) |
