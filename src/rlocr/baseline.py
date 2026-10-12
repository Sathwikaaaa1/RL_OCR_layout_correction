from __future__ import annotations

from typing import Any


def heuristic_order(lines: list[dict[str, Any]], y_tolerance: int = 18) -> list[str]:
    """Order lines top-to-bottom, then left-to-right within a visual row."""
    return [line["id"] for line in sorted(lines, key=lambda item: (round(item["bbox"][1] / y_tolerance), item["bbox"][0]))]


def order_accuracy(predicted: list[str], target: list[str]) -> float:
    target_pos = {value: index for index, value in enumerate(target)}
    common = [value for value in predicted if value in target_pos]
    if not common:
        return 0.0
    return sum(a == b for a, b in zip(common, sorted(common, key=target_pos.get))) / len(common)
