"""Leak-free pair features and native-order-aware decoding."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "dx_center",
    "dy_center",
    "horizontal_overlap",
    "vertical_overlap",
    "width_ratio",
    "height_ratio",
    "x_gap",
    "y_gap",
    "same_column",
    "same_tesseract_block",
    "native_rank_diff",
    "abs_native_rank_diff",
    "native_precedes",
)


def _overlap(left: list[float], right: list[float], axis: int) -> float:
    start = max(left[axis], right[axis])
    end = min(left[axis + 2], right[axis + 2])
    return max(0.0, end - start)


def pair_feature_row(left: dict[str, Any], right: dict[str, Any], line_count: int) -> list[float]:
    lx0, ly0, lx1, ly1 = map(float, left["bbox"])
    rx0, ry0, rx1, ry1 = map(float, right["bbox"])
    lw, lh = max(1.0, lx1 - lx0), max(1.0, ly1 - ly0)
    rw, rh = max(1.0, rx1 - rx0), max(1.0, ry1 - ry0)
    lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
    rcx, rcy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    h_overlap = _overlap(left["bbox"], right["bbox"], 0) / min(lw, rw)
    v_overlap = _overlap(left["bbox"], right["bbox"], 1) / min(lh, rh)
    dx_gap = max(0.0, max(lx0, rx0) - min(lx1, rx1))
    dy_gap = max(0.0, max(ly0, ry0) - min(ly1, ry1))
    native_diff = (float(left.get("native_order", 0)) - float(right.get("native_order", 0))) / max(1, line_count)
    # This is an OCR-only estimate: lines are likely in one column when their
    # horizontal spans overlap or their centers are close relative to width.
    same_column = float(h_overlap >= 0.25 or abs(lcx - rcx) <= 0.75 * (lw + rw))
    return [
        (lcx - rcx) / 2000.0,
        (lcy - rcy) / 2000.0,
        h_overlap,
        v_overlap,
        lw / rw,
        lh / rh,
        dx_gap / 2000.0,
        dy_gap / 2000.0,
        same_column,
        float(left.get("block_num", -1) == right.get("block_num", -2)),
        native_diff,
        abs(native_diff),
        float(left.get("native_order", 0) < right.get("native_order", 0)),
    ]


def pair_dataset(
    page: dict[str, Any],
    max_pairs: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    valid = set(page.get("valid_line_ids", []))
    lines = [line for line in page["ocr_lines"] if line["id"] in valid]
    rank = {line_id: index for index, line_id in enumerate(page["gt_order"])}
    rows: list[list[float]] = []
    targets: list[int] = []
    pairs: list[tuple[int, int]] = []
    all_pair_count = len(lines) * (len(lines) - 1) // 2
    selected = None
    if max_pairs is not None and max_pairs < all_pair_count:
        rng = rng or np.random.default_rng(0)
        selected = set(rng.choice(all_pair_count, max_pairs, replace=False).tolist())
    for pair_index, (i, j) in enumerate(combinations(range(len(lines)), 2)):
        if selected is not None and pair_index not in selected:
            continue
        if lines[i]["id"] not in rank or lines[j]["id"] not in rank:
            continue
        target = int(rank[lines[i]["id"]] < rank[lines[j]["id"]])
        rows.append(pair_feature_row(lines[i], lines[j], len(lines)))
        targets.append(target)
        pairs.append((i, j))
        rows.append(pair_feature_row(lines[j], lines[i], len(lines)))
        targets.append(1 - target)
        pairs.append((j, i))
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.int8), pairs


def pair_probability_scores(
    lines: list[dict[str, Any]],
    probabilities: dict[tuple[int, int], float],
    native_prior: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Canonical native-prior logit Borda scores, also used by PPO proposals."""
    count = len(lines)
    scores = np.zeros(count, dtype=np.float64)
    rng = rng or np.random.default_rng(0)
    for i, j in combinations(range(count), 2):
        probability = float(np.clip(probabilities.get((i, j), 0.5), 1e-5, 1 - 1e-5))
        logit = float(np.log(probability / (1 - probability)))
        if noise_std:
            logit += float(rng.normal(0.0, noise_std))
        native_direction = 1.0 if lines[i].get("native_order", i) < lines[j].get("native_order", j) else -1.0
        adjusted = logit + native_prior * native_direction
        scores[i] += adjusted
        scores[j] -= adjusted
    return scores


def decode_pair_probabilities(
    lines: list[dict[str, Any]],
    probabilities: dict[tuple[int, int], float],
    native_prior: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Decode through the shared native-prior scoring function."""
    scores = pair_probability_scores(lines, probabilities, native_prior, noise_std, rng)
    return [lines[index]["id"] for index in np.argsort(-scores, kind="stable")]

