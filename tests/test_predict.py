import sys
import os
from PIL import Image
import numpy as np

# Add project root to path so we can import src and api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.app import (
    app,
    postprocess_predicted_mask,
    safe_mask_coverage,
    derive_anatomy_balance,
    derive_quality_check,
    derive_confidence,
)
from src.preprocessing import prepare_for_model

client = TestClient(app)


def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["service"] == "lung-segmentation-api"
    assert "version" in json_data


def test_health_route():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "uptime_seconds" in json_data


def test_model_info_route():
    response = client.get("/model-info")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["model_name"] == "Attention U-Net (Lung)"
    assert "supported_formats" in json_data


def test_metrics_route():
    response = client.get("/metrics")
    assert response.status_code == 200
    json_data = response.json()
    assert "total_requests" in json_data
    assert "success_rate_percent" in json_data


def test_history_route():
    response = client.get("/history")
    assert response.status_code == 200
    json_data = response.json()
    assert "records" in json_data
    assert isinstance(json_data["records"], list)


def test_preprocessing():
    # Create a dummy image
    img = Image.new("RGB", (300, 300), color="white")
    resized, model_input = prepare_for_model(img, target_size=(256, 256))

    assert resized.size == (256, 256)
    assert model_input.shape == (1, 256, 256, 1)
    assert model_input.dtype == np.float32


def test_postprocess_mask():
    # Create a dummy raw mask array (values between 0 and 1)
    raw_mask = np.zeros((256, 256, 1), dtype=np.float32)
    raw_mask[100:150, 100:150, 0] = 0.8  # high confidence region
    raw_mask[50:100, 50:100, 0] = 0.3   # low confidence region

    post_mask = postprocess_predicted_mask(raw_mask)
    assert post_mask.size == (256, 256)
    assert post_mask.mode == "L"

    arr = np.array(post_mask)
    assert np.any(arr == 255)
    assert np.any(arr == 0)


def test_derive_confidence():
    assert derive_confidence(30.0) == (0.91, "High")
    assert derive_confidence(15.0) == (0.72, "Medium")
    assert derive_confidence(5.0) == (0.43, "Low")


def test_derive_quality_check():
    img = Image.new("L", (10, 10), color=0)
    assert derive_quality_check(img, 0.0) == "Failed"

    img_data = np.zeros((10, 10), dtype=np.uint8)
    img_data[2:8, 2:8] = 255
    img = Image.fromarray(img_data, mode="L")
    assert derive_quality_check(img, 36.0) == "Passed"


def test_derive_anatomy_balance():
    # Symmetric check
    img_sym = np.zeros((10, 10), dtype=np.uint8)
    img_sym[2:8, 1:4] = 255  # Left
    img_sym[2:8, 6:9] = 255  # Right
    img = Image.fromarray(img_sym, mode="L")
    assert derive_anatomy_balance(img) == "Symmetric"

    # Indeterminate check (empty image)
    img_empty = Image.new("L", (10, 10), color=0)
    assert derive_anatomy_balance(img_empty) == "Indeterminate"
