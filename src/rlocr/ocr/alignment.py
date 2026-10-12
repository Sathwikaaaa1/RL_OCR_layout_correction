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
    valid_lines = [line for line in ocr_lines if mapping.get(line["id"])]
    block_by_id = {block["id"]: block for block in gt_blocks}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for line in valid_lines:
        block = block_by_id[mapping[line["id"]]]; grouped.setdefault((block["id"], int(block.get("column_index", 0))), []).append(line)
    merge_pairs = []
    for group in grouped.values():
        group.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
        merge_pairs.extend([[left["id"], right["id"]] for left, right in zip(group, group[1:])])
    split_needed = []
    for line in ocr_lines:
        covered = [block["id"] for block in gt_blocks if intersection_over_line(line["bbox"], block["bbox"]) >= 0.15]
        if len(covered) > 1:
            split_needed.append(line["id"])
    block_rank = {block_id: index for index, block_id in enumerate(reading_order)}
    gt_order = [line["id"] for line in sorted(valid_lines, key=lambda line: (block_rank.get(mapping[line["id"]], 10**9), line["bbox"][1], line["bbox"][0]))]
    tesseract_order = [line["id"] for line in sorted(valid_lines, key=lambda item: item["native_order"])]
    order_rank = {line_id: index for index, line_id in enumerate(gt_order)}
    order_pairs = [[left, right] for left, right in combinations(gt_order, 2) if order_rank[left] < order_rank[right]]
    return {"ocr_lines": ocr_lines, "gt_blocks": gt_blocks, "ocr_to_gt": mapping, "valid_line_ids": [line["id"] for line in valid_lines], "dropped_line_count": len(ocr_lines) - len(valid_lines), "merge_pairs": merge_pairs, "split_needed": split_needed, "order_pairs": order_pairs, "gt_order": gt_order, "tesseract_order": tesseract_order, "stats": {"ocr_line_count": len(ocr_lines), "valid_line_count": len(valid_lines), "dropped_line_count": len(ocr_lines) - len(valid_lines), "merge_pair_count": len(merge_pairs), "split_line_count": len(split_needed)}}


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    common = [value for value in order_a if value in set(order_b)]
    positions = {value: i for i, value in enumerate(order_b)}
    pairs = list(combinations(common, 2))
    if not pairs: return 1.0
    concordant = sum((positions[a] - positions[b]) * (i - j) > 0 for (i, a), (j, b) in combinations(enumerate(common), 2))
    return (2 * concordant - len(pairs)) / len(pairs)
