"""
report_generator.py
-------------------
Generate structured JSON and CSV diagnostic reports for each
lung segmentation prediction. Used by the dashboard for export.
"""
import json
import csv
import io
import time
import uuid
from datetime import datetime, timezone
from typing import Optional


# -------------------------------------------------------
# Report schema
# -------------------------------------------------------
REPORT_VERSION = "1.0"
MODEL_NAME = "Attention U-Net (Lung)"
MODEL_VERSION = "v3.0.0"


def build_json_report(
    prediction_id: str,
    filename: str,
    mask_coverage: float,
    confidence_label: str,
    confidence_score: float,
    left_lung_percent: float,
    right_lung_percent: float,
    anatomy_balance: str,
    quality_check: str,
    processing_time_ms: int,
    input_resolution: list,
    pipeline_stages: list,
    inference_message: str,
    dice_score: str = "~0.95",
    iou_score: str = "~0.90",
    extra_fields: Optional[dict] = None,
) -> dict:
    """
    Build a structured JSON diagnostic report.

    Returns a dict that can be serialised to JSON and downloaded.
    """
    now_utc = datetime.now(timezone.utc)

    report = {
        "report_version": REPORT_VERSION,
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_unix": time.time(),
        "prediction": {
            "id": prediction_id,
            "filename": filename,
            "inference_message": inference_message,
            "processing_time_ms": processing_time_ms,
            "pipeline_stages": pipeline_stages,
        },
        "model": {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "framework": "TensorFlow / Keras",
            "architecture": "Attention U-Net",
            "input_resolution": input_resolution,
            "training_metrics": {
                "dice_score": dice_score,
                "iou_score": iou_score,
            },
        },
        "diagnostics": {
            "mask_coverage_percent": mask_coverage,
            "confidence_label": confidence_label,
            "confidence_score": confidence_score,
            "left_lung_percent": left_lung_percent,
            "right_lung_percent": right_lung_percent,
            "anatomy_balance": anatomy_balance,
            "quality_check": quality_check,
        },
        "qc": {
            "pass": quality_check == "Passed",
            "reason": _qc_reason(quality_check, mask_coverage),
        },
        "clinical_disclaimer": (
            "This report is generated for research and portfolio demonstration purposes. "
            "It is NOT intended for direct clinical decision-making without proper "
            "medical validation and regulatory review."
        ),
    }

    if extra_fields:
        report.update(extra_fields)

    return report


def _qc_reason(quality_check: str, coverage: float) -> str:
    if quality_check == "Passed":
        return f"Mask coverage ({coverage:.1f}%) is within the expected 8–55% lung region range."
    if quality_check == "Failed":
        return "No foreground pixels detected in the predicted mask."
    return (
        f"Mask coverage ({coverage:.1f}%) is outside the typical 8–55% range. "
        "Manual review recommended."
    )


def report_to_json_bytes(report: dict) -> bytes:
    """Serialise report dict to UTF-8 JSON bytes (for st.download_button)."""
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")


def report_to_csv_bytes(report: dict) -> bytes:
    """
    Serialise the key diagnostic fields as a single CSV row (bytes).
    Suitable for appending to a batch results CSV.
    """
    diag = report.get("diagnostics", {})
    pred = report.get("prediction", {})
    model = report.get("model", {})
    qc = report.get("qc", {})

    row = {
        "prediction_id": pred.get("id", ""),
        "filename": pred.get("filename", ""),
        "generated_at_utc": report.get("generated_at_utc", ""),
        "model_name": model.get("name", ""),
        "model_version": model.get("version", ""),
        "mask_coverage_percent": diag.get("mask_coverage_percent", ""),
        "confidence_label": diag.get("confidence_label", ""),
        "confidence_score": diag.get("confidence_score", ""),
        "left_lung_percent": diag.get("left_lung_percent", ""),
        "right_lung_percent": diag.get("right_lung_percent", ""),
        "anatomy_balance": diag.get("anatomy_balance", ""),
        "quality_check": diag.get("quality_check", ""),
        "qc_pass": qc.get("pass", ""),
        "processing_time_ms": pred.get("processing_time_ms", ""),
    }

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)
    return buf.getvalue().encode("utf-8")
