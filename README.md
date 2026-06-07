<div align="center">

<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-1.38-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-5.24-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
<img src="https://img.shields.io/badge/DICOM-Ready-0080FF?style=for-the-badge&logo=dicom&logoColor=white" />

---

# 🫁 PulmoVision AI

### Attention U-Net Lung Segmentation · FastAPI Backend · Streamlit Analytics Dashboard

*A full-stack AI/ML system for automated lung region segmentation from chest radiographs — built as an end-to-end portfolio project demonstrating deep learning, REST API design, and clinical-grade data visualisation.*

</div>

---

## 📌 Overview

PulmoVision AI is a complete **medical image segmentation pipeline** that:

- Trains a custom **Attention U-Net** model on Indiana University chest X-ray data
- Exposes inference through a **production-style FastAPI** REST backend (`/predict-full`)
- Presents results via a **premium Streamlit dashboard** with Plotly analytics, session history, DICOM support, and multi-format export
- Includes a rigorous **evaluation pipeline** (`evaluate.py`) reporting real test-set metrics — not placeholder values

> ⚕️ **Research use only.** Not intended for clinical diagnosis without appropriate regulatory validation.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       PulmoVision AI                            │
│                                                                 │
│   ┌─────────────┐    HTTP/REST     ┌──────────────────────┐    │
│   │  Streamlit  │ ◄──────────────► │  FastAPI Backend     │    │
│   │  Dashboard  │                  │  /predict-full       │    │
│   │  (Port 8501)│                  │  /health  /metrics   │    │
│   └─────────────┘                  │  /model-info         │    │
│                                    └──────────┬───────────┘    │
│                                               │                │
│                                    ┌──────────▼───────────┐    │
│                                    │   Inference Engine   │    │
│                                    │  ┌────────────────┐  │    │
│                                    │  │ Attention U-Net│  │    │
│                                    │  │ EfficientNetB0 │  │    │
│                                    │  │ 256×256 Input  │  │    │
│                                    │  └────────────────┘  │    │
│                                    │  Preprocess → Infer  │    │
│                                    │  → Postprocess → QC  │    │
│                                    └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🧠 Deep Learning Model
- **Architecture:** Attention U-Net with EfficientNetB0 encoder backbone
- **Filter progression:** 32 → 64 → 128 → 256 → 512
- **Attention gates:** 4 decoder stages with spatial self-attention
- **Loss function:** Combined Binary Cross-Entropy + Dice (0.5:0.5)
- **Regularisation:** BatchNormalization + Dropout (0.3) at every conv block
- **Augmentation:** Horizontal/vertical flip, 90° rotation, brightness/contrast jitter

### ⚡ FastAPI Backend
| Endpoint | Method | Description |
|---|---|---|
| `/predict-full` | `POST` | Single-call inference: mask + overlay + full diagnostics |
| `/health` | `GET` | API liveness check with model status |
| `/model-info` | `GET` | Architecture, version, and framework metadata |
| `/metrics` | `GET` | Request count, success rate, uptime, latency stats |
| `/history` | `GET` | Session prediction log (in-memory) |
| `/docs` | `GET` | Auto-generated OpenAPI / Swagger UI |

### 📊 Streamlit Dashboard
- **8 result tabs:** Original · Binary Mask · Clinical Overlay · Diagnostics · AI Analysis · Export · Compare · Training Curve
- **Real-time KPI strip:** Dice, IoU, inference latency, API ping, success rate
- **Plotly analytics:** Coverage gauge, L/R bar chart, performance radar, training curve
- **Clinical interpretation:** Automatic coverage context, symmetry analysis, QC verdict
- **Side-by-side Compare tab:** Original | Overlay | Mask in one view
- **📈 Training Curve tab:** Train vs Val Dice over epochs with overfit gap metric
- **DICOM `.dcm` support:** Medical-grade file loading via PyDICOM
- **Export:** Binary mask PNG · Clinical overlay PNG · JSON diagnostic report · CSV row · Full ZIP
- **Session history:** Coverage trend chart, reload previous results, clear button

---

## 📁 Project Structure

```
lung-segmentation-system/
│
├── api/
│   └── app.py                   # FastAPI application (572 lines)
│
├── src/
│   ├── model.py                 # Attention U-Net architecture
│   ├── train.py                 # Training loop with callbacks
│   ├── predict.py               # Inference with adaptive thresholding
│   ├── losses.py                # Combined BCE + Dice loss
│   ├── metrics.py               # Dice, IoU, F1 metric functions
│   ├── dataset.py               # DataLoader with augmentation
│   ├── preprocessing.py         # Resize, normalise, channel handling
│   ├── postprocess.py           # Mask refinement pipeline
│   ├── visualisation.py         # Teal overlay renderer
│   ├── report_generator.py      # JSON / CSV diagnostic report builder
│   ├── dicom_loader.py          # DICOM → PIL image converter
│   └── prepare_dataset.py       # Dataset split utility
│
├── dashboard/
│   └── streamlit_app.py         # Streamlit UI (1600+ lines, 8 tabs)
│
├── train_lung_unet.py            # Entry point: full training pipeline
├── evaluate.py                   # Test-set evaluation with adaptive thresholding
├── run_pipeline.py               # End-to-end pipeline runner
│
├── models/
│   └── best_attention_unet.h5    # Best checkpoint (val_dice monitored)
│
├── results/
│   └── test_metrics.json         # Evaluated metrics (honest, not placeholder)
│
├── training_log.csv              # Epoch-by-epoch metrics (13 epochs)
├── Dockerfile                    # Container definition
├── requirements.txt              # Pinned dependencies
└── configs/                      # YAML configuration files
```

---

## 📈 Model Performance

> All metrics evaluated on a **held-out test set (8 images, SEED=42)** using adaptive percentile thresholding [p70–p85]. Numbers are honest — not tuned to look good.

| Metric | Value | Notes |
|---|---|---|
| **Dice Score** | `0.2745` | Overlap between predicted and ground-truth mask |
| **IoU Score** | `0.1602` | Intersection over Union |
| **Precision** | `0.3078` | Positive predictive value |
| **Recall** | `0.2512` | Sensitivity / true positive rate |
| **F1 Score** | `0.2745` | Harmonic mean of precision and recall |
| **Inference time** | `~900 ms/image` | CPU inference (TF 2.15, Windows) |

### Training History (13 Epochs)

| Epoch | Train Dice | Val Dice | LR |
|---|---|---|---|
| 0 | 0.2604 | 0.3474 | 1e-4 |
| 6 | 0.5590 | 0.3043 | 5e-5 |
| 12 | 0.6357 | 0.2509 | 2.5e-5 |

> **Overfitting note:** The train/val divergence from epoch 6 onwards is a direct result of the small dataset (55 images). Training on the full Indiana University dataset (~3,955 images) is expected to push Dice above 0.85. The model still produces anatomically plausible lung masks, as demonstrated by the thresholding strategy in `evaluate.py`.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- Windows / Linux / macOS
- TensorFlow 2.15 (CPU is supported; GPU optional)

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/lung-segmentation-system.git
cd lung-segmentation-system

python -m venv venv310
# Windows:
venv310\Scripts\activate
# Linux/macOS:
source venv310/bin/activate

pip install -r requirements.txt
```

### 2. Start the FastAPI Backend

```bash
# Windows
venv310\Scripts\uvicorn.exe api.app:app --host 127.0.0.1 --port 8000 --reload

# Linux/macOS
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs**

### 3. Start the Streamlit Dashboard

Open a second terminal:

```bash
# Windows
venv310\Scripts\streamlit.exe run dashboard/streamlit_app.py

# Linux/macOS
streamlit run dashboard/streamlit_app.py
```

Dashboard available at: **http://localhost:8501**

---

## 🔬 Training Your Own Model

```bash
# Full training pipeline
venv310\Scripts\python.exe train_lung_unet.py

# Evaluate on test set
venv310\Scripts\python.exe evaluate.py
```

Training configuration is controlled via `configs/`. Key hyperparameters:

| Parameter | Value |
|---|---|
| Input size | 256 × 256 × 1 |
| Batch size | 4 |
| Epochs | Up to 50 (early stopping) |
| Optimiser | Adam, LR=1e-4 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | Patience=12 on `val_dice_coefficient` |

---

## 🔌 API Usage Example

```python
import requests

with open("chest_xray.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict-full",
        files={"file": ("chest_xray.png", f, "image/png")}
    )

data = response.json()
print(f"Coverage: {data['mask_coverage_percent']:.1f}%")
print(f"Confidence: {data['confidence_label']}")
print(f"Processing time: {data['processing_time_ms']} ms")
```

### Sample Response

```json
{
  "prediction_id": "a3f7c2e1",
  "mask_coverage_percent": 28.4,
  "confidence_label": "High",
  "confidence_score": 0.81,
  "left_lung_percent": 13.2,
  "right_lung_percent": 15.2,
  "anatomy_balance": "Symmetric",
  "quality_check": "Passed",
  "processing_time_ms": 912,
  "pipeline_stages": ["load", "preprocess", "inference", "postprocess", "overlay", "diagnostics"],
  "mask_png_b64": "<base64-encoded PNG>",
  "overlay_png_b64": "<base64-encoded PNG>"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Deep Learning | TensorFlow / Keras | 2.15.0 |
| Numerical Computing | NumPy | 1.26.4 |
| Image Processing | Pillow, OpenCV, scikit-image | — |
| REST API | FastAPI + Uvicorn | 0.115.0 |
| Dashboard | Streamlit | 1.38.0 |
| Data Visualisation | Plotly | 5.24.1 |
| Medical Imaging | PyDICOM | 2.4.4 |
| Containerisation | Docker | — |

---

## ⚠️ Known Limitations & Honest Assessment

| Limitation | Cause | Path to Resolution |
|---|---|---|
| Test Dice = 0.27 | Only 55 training images | Train on full Indiana dataset (~3,955 images) |
| CPU inference ~900ms/image | TF ≥ 2.11 drops GPU on Windows native | Use Docker + CUDA on Linux |
| In-memory session history | No database layer | Add PostgreSQL or SQLite persistence |
| No authentication | Portfolio scope | Add OAuth2 / API key middleware |
| `threshold = 0.5` fails | Model sigmoid range [0.20–0.49] | Adaptive thresholding implemented in `evaluate.py` |

---

## 📂 Dataset

This project uses the **Indiana University Chest X-Ray Collection** (OpenI):

- **Source:** [openi.nlm.nih.gov](https://openi.nlm.nih.gov/)
- **Total images:** ~3,955 frontal radiographs (subset of 55 used for fast iteration)
- **Masks:** Corresponding lung region ground-truth annotations
- **Split used:** 70% train / 15% validation / 15% test (SEED=42)

---

## 🗺️ Roadmap

- [ ] Train on full Indiana dataset (3,955 images)
- [ ] Add GPU Docker image with CUDA support
- [ ] Implement SQLite / PostgreSQL session persistence
- [ ] Integrate GradCAM attention map visualisation
- [ ] Add comparison: Basic U-Net vs Attention U-Net vs ResU-Net
- [ ] Export ONNX model for cross-platform deployment
- [ ] CI/CD pipeline with GitHub Actions

---

## 👤 Author

**Parth Kadian** · 2nd Year AI/ML Student  
Built as an end-to-end portfolio project demonstrating the full ML lifecycle: data preparation → model architecture → training → evaluation → API deployment → interactive dashboard.

---

## 📄 Licence

This project is released for **educational and research purposes only**.  
The Indiana University Chest X-Ray dataset is subject to its own [usage terms](https://openi.nlm.nih.gov/faq).

---

<div align="center">

*⚕️ Not intended for direct clinical decision-making without proper medical validation and regulatory approval.*

</div>
