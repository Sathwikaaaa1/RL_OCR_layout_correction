from __future__ import annotations

from itertools import combinations
from typing import Any

from rlocr.ocr.alignment import kendall_tau


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
