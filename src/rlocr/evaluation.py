from __future__ import annotations

from itertools import combinations
from typing import Any

from rlocr.ocr.alignment import kendall_tau


def xy_cut_order(lines: list[dict[str, Any]]) -> list[str]:
    """Return a deterministic recursive XY-cut order for OCR line boxes.

    The largest whitespace gap is used to split a region.  X gaps are useful
    for columns; Y gaps finish the top-to-bottom order inside a region.
    This is a non-learned diagnostic, not a replacement for the policy.
    """
    if not lines:
        return []

    heights = sorted(max(1.0, float(line["bbox"][3] - line["bbox"][1])) for line in lines)
    median_height = heights[len(heights) // 2]

    def best_gap(group: list[dict[str, Any]], axis: int) -> tuple[float, int] | None:
        ordered = sorted(group, key=lambda line: (line["bbox"][axis], line["bbox"][axis + 2]))
        edge = float(ordered[0]["bbox"][axis + 2])
        best: tuple[float, int] | None = None
        for index, line in enumerate(ordered[1:], 1):
            start = float(line["bbox"][axis])
            gap = start - edge
            if best is None or gap > best[0]:
                best = (gap, index)
            edge = max(edge, float(line["bbox"][axis + 2]))
        return best

    def split(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(group) <= 1:
            return group
        x_gap = best_gap(group, 0)
        y_gap = best_gap(group, 1)
        candidates: list[tuple[float, int, int]] = []
        # Column gutters can be narrower than a rendered OCR line height.
        # Keep the X threshold low enough to retain those columns, while the
        # Y threshold remains stricter to avoid treating line spacing as a
        # page-level split.
        if x_gap and x_gap[0] >= max(8.0, 0.4 * median_height):
            candidates.append((x_gap[0], 0, x_gap[1]))
        if y_gap and y_gap[0] >= max(6.0, 0.75 * median_height):
            candidates.append((y_gap[0], 1, y_gap[1]))
        if not candidates:
            return sorted(group, key=lambda line: (line["bbox"][1], line["bbox"][0], line.get("native_order", 0)))
        _, axis, index = max(candidates)
        ordered = sorted(group, key=lambda line: (line["bbox"][axis], line["bbox"][axis + 2]))
        return split(ordered[:index]) + split(ordered[index:])

    return [line["id"] for line in split(lines)]


def decode_pairwise_order(ids: list[str], precedence_pairs: set[tuple[str, str]]) -> list[str]:
    wins = {line_id: 0 for line_id in ids}
    for left, right in precedence_pairs:
        if left in wins and right in wins:
            wins[left] += 1
    return sorted(ids, key=lambda line_id: (-wins[line_id], ids.index(line_id)))


def corrected_order(lines: list[dict[str, Any]], actions: dict[str, int]) -> list[str]:
    """Construct an order from actions without using ground truth.

    Linked lines are spatially ordered column-first; retained, merged, and
    split lines preserve native OCR order. This makes the action policy's
    reading-order effect measurable without leaking labels into inference.
    """
    linked = [line for line in lines if actions.get(line["id"], 0) == 3]
    linked_ids = {line["id"] for line in linked}
    linked.sort(key=lambda line: (line["bbox"][0], line["bbox"][1], line["native_order"]))
    native = sorted(lines, key=lambda line: line["native_order"])
    return [line["id"] for line in linked] + [line["id"] for line in native if line["id"] not in linked_ids]


def pair_f1(predicted: set[tuple[str, str]], target: set[tuple[str, str]]) -> float:
    if not predicted and not target:
        return 1.0
    if not predicted or not target:
        return 0.0
    precision = len(predicted & target) / len(predicted)
    recall = len(predicted & target) / len(target)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def predicted_merge_pairs(lines: list[dict[str, Any]], actions: dict[str, int]) -> set[tuple[str, str]]:
    native = sorted(lines, key=lambda line: line["native_order"])
    return {(a["id"], b["id"]) for a, b in zip(native, native[1:]) if actions.get(a["id"], 0) == 1 or actions.get(b["id"], 0) == 1}


def evaluate_orders(lines: list[dict[str, Any]], actions: dict[str, int], gt_order: list[str]) -> dict[str, float]:
    corrected = corrected_order(lines, actions)
    native = [line["id"] for line in sorted(lines, key=lambda line: line["native_order"])]
    return {"corrected_kendall_tau": kendall_tau(corrected, gt_order), "native_kendall_tau": kendall_tau(native, gt_order)}
