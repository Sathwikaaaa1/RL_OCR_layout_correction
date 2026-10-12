from __future__ import annotations

from collections import Counter
from typing import Any


def class_distribution(page: dict[str, Any]) -> Counter[str]:
    labels = {line["id"]: "retain" for line in page["ocr_lines"] if line["id"] in set(page.get("valid_line_ids", []))}
    for left, right in page.get("merge_pairs", []):
        labels[left] = labels[right] = "merge"
    for line_id in page.get("split_needed", []):
        if line_id in labels:
            labels[line_id] = "split"
    return Counter(labels.values())
