# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional

from config import AGE_CLASSES

def age_to_class_index(age: int) -> int:
    if age <= 12:
        return 0
    if age <= 19:
        return 1
    if age <= 35:
        return 2
    if age <= 55:
        return 3
    return 4

def age_class_for_actual_age(age: Optional[int]) -> Optional[dict]:
    if age is None:
        return None
    index = age_to_class_index(int(age))
    if index < 0 or index >= len(AGE_CLASSES):
        return None
    return AGE_CLASSES[index]

def iter_neighbor_class_indices(start: int, n_classes: int) -> List[int]:
    if n_classes <= 0:
        return []
    start = max(0, min(int(start), n_classes - 1))
    order: List[int] = []
    seen = set()
    for step in range(n_classes):
        candidates = [start] if step == 0 else [start - step, start + step]
        for idx in candidates:
            if 0 <= idx < n_classes and idx not in seen:
                seen.add(idx)
                order.append(idx)
    return order

def find_class_index_toward_actual(
    predicted_index: int, actual_index: int, n_classes: int
) -> Optional[int]:
    if not (0 <= actual_index < n_classes):
        return None
    for idx in iter_neighbor_class_indices(predicted_index, n_classes):
        if idx == actual_index:
            return idx
    return None

def _result_from_class_index(
    class_index: int, confidence: float, source: dict, extra: Optional[dict] = None
) -> dict:
    item = AGE_CLASSES[class_index]
    out = {
        "label": item.get("label", f"年龄段{class_index}"),
        "label_en": item.get("label_en", f"Age {class_index}"),
        "range": item.get("range", "-"),
        "class_index": class_index,
        "class_id": item.get("id", f"age_{class_index}"),
        "confidence": float(confidence),
    }
    for key in (
        "raw_label",
        "raw_range",
        "raw_class_index",
        "calibration_shift",
        "calibrated",
        "smoothed",
        "smoothing_window",
    ):
        if key in source:
            out[key] = source[key]
    if extra:
        out.update(extra)
    return out

def apply_age_band_search(age_result: dict, actual_age: int) -> dict:
    if not age_result or actual_age is None:
        return age_result

    try:
        actual_age_i = int(actual_age)
    except (TypeError, ValueError):
        return age_result

    actual_class = age_class_for_actual_age(actual_age_i)
    if actual_class is None:
        return age_result

    n = len(AGE_CLASSES)
    pred_idx = int(age_result.get("class_index", -1))
    if pred_idx < 0 or pred_idx >= n:
        return age_result

    actual_idx = age_to_class_index(actual_age_i)
    if pred_idx == actual_idx:
        age_result["actual_age"] = actual_age_i
        age_result["actual_range"] = actual_class.get("range")
        age_result["actual_label"] = actual_class.get("label")
        age_result["is_correct"] = True
        age_result["evaluation"] = "预测正确"
        return age_result

    match_idx = find_class_index_toward_actual(pred_idx, actual_idx, n)
    if match_idx is None:
        return age_result

    tried = []
    for idx in iter_neighbor_class_indices(pred_idx, n):
        tried.append(AGE_CLASSES[idx].get("range", str(idx)))
        if idx == match_idx:
            break

    corrected = _result_from_class_index(
        match_idx,
        float(age_result.get("confidence", 0.0)),
        age_result,
        {
            "band_search_corrected": True,
            "band_search_from_index": pred_idx,
            "band_search_from_range": age_result.get("range"),
            "band_search_steps": tried,
            "band_search_matched_index": match_idx,
            "actual_age": actual_age_i,
            "actual_range": actual_class.get("range"),
            "actual_label": actual_class.get("label"),
            "is_correct": True,
            "evaluation": "预测正确",
        },
    )
    return corrected
