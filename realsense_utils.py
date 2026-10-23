from __future__ import annotations

import math
from typing import Optional

import numpy as np

def get_depth_scale_from_pipeline(pipeline) -> float:
    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    return float(depth_sensor.get_depth_scale())

def median_depth_meters(
    depth_image: np.ndarray,
    depth_scale: float,
    x: float,
    y: float,
    w: float,
    h: float,
    frame_w: int,
    frame_h: int,
    valid_min_m: float = 0.15,
    valid_max_m: float = 10.0,
    min_valid_pixels: int = 5,
) -> float:
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_w, int(x + w))
    y2 = min(frame_h, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return float("nan")

    roi = depth_image[y1:y2, x1:x2].astype(np.float32) * depth_scale
    mask = (roi > valid_min_m) & (roi < valid_max_m)
    vals = roi[mask]
    if vals.size < min_valid_pixels:
        return float("nan")
    return float(np.median(vals))

def face_depth_valid_ratio(
    depth_image: np.ndarray,
    depth_scale: float,
    x: float,
    y: float,
    w: float,
    h: float,
    frame_w: int,
    frame_h: int,
    valid_min_m: float = 0.15,
    valid_max_m: float = 10.0,
) -> float:
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_w, int(x + w))
    y2 = min(frame_h, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = depth_image[y1:y2, x1:x2].astype(np.float32) * depth_scale
    total = roi.size
    if total <= 0:
        return 0.0
    mask = (roi > valid_min_m) & (roi < valid_max_m)
    return float(np.count_nonzero(mask) / total)

def depth_zone_label(
    depth_m: Optional[float],
    optimal_lo_m: float,
    optimal_hi_m: float,
) -> str:
    if depth_m is None:
        return "unknown"
    try:
        d = float(depth_m)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(d) or math.isinf(d):
        return "unknown"
    if d < optimal_lo_m:
        return "too_near"
    if d > optimal_hi_m:
        return "too_far"
    return "optimal"
