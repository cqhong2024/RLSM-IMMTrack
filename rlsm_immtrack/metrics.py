"""Original pooled-frame OPE metrics; CPU-only NumPy implementation."""
from __future__ import annotations
import numpy as np

def iou_xywh(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_br = pred[:, :2] + np.maximum(pred[:, 2:], 0.0)
    gt_br = gt[:, :2] + np.maximum(gt[:, 2:], 0.0)
    tl = np.maximum(pred[:, :2], gt[:, :2])
    br = np.minimum(pred_br, gt_br)
    wh = np.maximum(br - tl, 0.0)
    intersection = wh[:, 0] * wh[:, 1]
    union = (
        np.maximum(pred[:, 2], 0.0) * np.maximum(pred[:, 3], 0.0)
        + np.maximum(gt[:, 2], 0.0) * np.maximum(gt[:, 3], 0.0)
        - intersection
    )
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def frame_statistics(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(gt).all(axis=1) & (gt[:, 2] > 0.0) & (gt[:, 3] > 0.0)
    if not valid.any():
        raise ValueError("No valid ground-truth boxes in the evaluated frames")
    pred = pred[valid]
    gt = gt[valid]
    pred_center = pred[:, :2] + 0.5 * pred[:, 2:]
    gt_center = gt[:, :2] + 0.5 * gt[:, 2:]
    delta = pred_center - gt_center
    center_error = np.linalg.norm(delta, axis=1)
    normalized_error = np.linalg.norm(
        delta / np.maximum(gt[:, 2:], np.finfo(np.float64).eps), axis=1
    )
    return {
        "iou": iou_xywh(pred, gt),
        "center_error": center_error,
        "normalized_error": normalized_error,
    }


def summarize_frames(stats: dict[str, np.ndarray]) -> dict[str, float]:
    success_thresholds = np.linspace(0.0, 1.0, 21)
    normalized_thresholds = np.linspace(0.0, 0.5, 51)
    success_curve = np.array([(stats["iou"] >= t).mean() for t in success_thresholds])
    normalized_curve = np.array(
        [(stats["normalized_error"] <= t).mean() for t in normalized_thresholds]
    )
    return {
        "success_auc": float(success_curve.mean()),
        "average_overlap": float(stats["iou"].mean()),
        "sr_0.50": float((stats["iou"] >= 0.50).mean()),
        "sr_0.75": float((stats["iou"] >= 0.75).mean()),
        "precision_10px": float((stats["center_error"] <= 10.0).mean()),
        "precision_20px": float((stats["center_error"] <= 20.0).mean()),
        "normalized_precision_auc": float(normalized_curve.mean()),
        "mean_iou": float(stats["iou"].mean()),
        "mean_center_error_px": float(stats["center_error"].mean()),
        "median_center_error_px": float(np.median(stats["center_error"])),
        "rmse_center_error_px": float(np.sqrt(np.mean(stats["center_error"] ** 2))),
        "iou_lt_0.1": float((stats["iou"] < 0.10).mean()),
    }
