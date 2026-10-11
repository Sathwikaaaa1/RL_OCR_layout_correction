from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

from rlocr.common import intersection_over_line, iou


def align_page(ocr_lines: list[dict[str, Any]], gt_blocks: list[dict[str, Any]], reading_order: list[str]) -> dict[str, Any]:
    scores: dict[str, list[tuple[str, float]]] = {}
    for line in ocr_lines:
        candidates = []
        for block in gt_blocks:
            overlap = intersection_over_line(line["bbox"], block["bbox"])
            similarity = SequenceMatcher(None, line["text"].lower(), block["text"].lower()).ratio()
            candidates.append((block["id"], overlap, similarity))
        candidates.sort(key=lambda value: (value[1], value[2]), reverse=True)
        scores[line["id"]] = [(block_id, overlap * 0.8 + similarity * 0.2) for block_id, overlap, similarity in candidates]
    mapping = {line_id: (values[0][0] if values and values[0][1] >= 0.10 else None) for line_id, values in scores.items()}
    merge_pairs = [[a["id"], b["id"]] for a, b in combinations(ocr_lines, 2) if mapping.get(a["id"]) and mapping.get(a["id"]) == mapping.get(b["id"])]
    split_needed = []
    for line in ocr_lines:
        covered = [block["id"] for block in gt_blocks if intersection_over_line(line["bbox"], block["bbox"]) >= 0.15]
        if len(covered) > 1:
            split_needed.append(line["id"])
    block_rank = {block_id: index for index, block_id in enumerate(reading_order)}
    gt_order = sorted([line["id"] for line in ocr_lines if mapping[line["id"]]], key=lambda line_id: block_rank.get(mapping[line_id], 10**9))
    tesseract_order = [line["id"] for line in sorted(ocr_lines, key=lambda item: item["native_order"])]
    return {"ocr_lines": ocr_lines, "gt_blocks": gt_blocks, "ocr_to_gt": mapping, "merge_pairs": merge_pairs, "split_needed": split_needed, "gt_order": gt_order, "tesseract_order": tesseract_order, "stats": {"ocr_line_count": len(ocr_lines), "merge_pair_count": len(merge_pairs), "split_line_count": len(split_needed)}}


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    common = [value for value in order_a if value in set(order_b)]
    positions = {value: i for i, value in enumerate(order_b)}
    pairs = list(combinations(common, 2))
    if not pairs: return 1.0
    concordant = sum((positions[a] - positions[b]) * (i - j) > 0 for (i, a), (j, b) in combinations(enumerate(common), 2))
    return (2 * concordant - len(pairs)) / len(pairs)
