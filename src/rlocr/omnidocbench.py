"""Small, deterministic OmniDocBench subsets and conservative text alignment.

This is an application reading-order check, not the official benchmark metric.
Ground truth is used for subset eligibility and scoring, never model features.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
import random
from typing import Any

TEXT_CATEGORIES = {"text_block", "title", "list_group", "reference",
                   "figure_caption", "table_caption", "page_footnote"}
MULTICOLUMN = {"double_column", "three_column", "1andmore_column"}


def image_filename(value: str) -> str:
    """Resolve annotation basenames safely inside the repository images folder."""
    path = PurePosixPath(value.replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"Unsafe image path: {value}")
    return str(path if path.parts[0] == "images" else PurePosixPath("images") / path.name)


def text_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for block in page["layout_dets"]
            if block.get("category_type") in TEXT_CATEGORIES
            and not block.get("ignore", False) and block.get("text", "").strip()
            and isinstance(block.get("order"), (int, float))]


def select_pages(pages: list[dict], sizes: dict[str, int], seed: int = 1337,
                 limit: int = 150, budget: int = 200_000_000,
                 annotation_bytes: int = 0) -> list[dict]:
    """Seeded shuffle; eligibility and byte budget are fixed before OCR/model use."""
    candidates = []
    for index, page in enumerate(pages):
        info = page["page_info"]
        attributes = info["page_attribute"]
        sidebar = any("sidebar" in str(block.get("category_type", "")).lower()
                      for block in page["layout_dets"]) or "sidebar" in str(attributes).lower()
        if (attributes.get("language") != "english"
                or not (attributes.get("layout") in MULTICOLUMN or sidebar)
                or len(text_blocks(page)) < 3):
            continue
        filename = image_filename(info["image_path"])
        if filename not in sizes:
            raise ValueError(f"Image not present in repository: {filename}")
        candidates.append({"annotation_index": index, "image_path": filename,
                           "page_id": f"omni_{index:04d}", "attributes": attributes,
                           "text_blocks": len(text_blocks(page)), "sidebar_label": sidebar,
                           "bytes": sizes[filename]})
    random.Random(seed).shuffle(candidates)
    selected, used, seen = [], annotation_bytes, set()
    for candidate in candidates:
        if len(selected) >= limit:
            break
        if candidate["image_path"] in seen or used + candidate["bytes"] > budget:
            continue
        selected.append(candidate)
        seen.add(candidate["image_path"])
        used += candidate["bytes"]
    return selected


def directory_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def align_text(lines: list[dict], page: dict, image_size: tuple[int, int]) -> dict:
    """Require >=50% line overlap; exclude cross-block and tied-order ambiguity.

    Text similarity breaks geometric ties only. Never align disjoint text by
    text similarity alone. Orders within a paragraph are derived top-to-bottom.
    """
    info = page["page_info"]
    sx, sy = image_size[0] / info["width"], image_size[1] / info["height"]
    blocks = []
    for block in text_blocks(page):
        poly = block["poly"]
        box = [min(poly[::2])*sx, min(poly[1::2])*sy,
               max(poly[::2])*sx, max(poly[1::2])*sy]
        blocks.append({**block, "bbox": box})
    order_counts = {}
    for block in blocks:
        order_counts[block["order"]] = order_counts.get(block["order"], 0) + 1
    matches, ambiguous = {}, []
    for line in lines:
        x0, y0, x1, y1 = line["bbox"]
        area = max(1, (x1-x0)*(y1-y0))
        candidates = []
        for index, block in enumerate(blocks):
            bx0, by0, bx1, by1 = block["bbox"]
            overlap = max(0, min(x1,bx1)-max(x0,bx0))*max(0,min(y1,by1)-max(y0,by0))/area
            if overlap > 0:
                candidates.append((overlap, SequenceMatcher(None, line["text"], block["text"]).ratio(), index))
        candidates.sort(reverse=True)
        if sum(item[0] >= .15 for item in candidates) > 1:
            ambiguous.append(line["id"])
            continue
        if not candidates or candidates[0][0] < .5:
            continue
        best = candidates[0][2]
        if order_counts[blocks[best]["order"]] > 1:
            ambiguous.append(line["id"])
            continue
        matches[line["id"]] = best
    valid = [line for line in lines if line["id"] in matches]
    truth = [line["id"] for line in sorted(valid, key=lambda line:
             (blocks[matches[line["id"]]]["order"], line["bbox"][1], line["bbox"][0]))]
    return {"gt_order": truth, "mapping": {key: blocks[value]["anno_id"] for key,value in matches.items()},
            "matched_blocks": len(set(matches.values())), "annotated_text_blocks": len(blocks),
            "ambiguous_lines": ambiguous, "scored_lines": len(valid), "ocr_lines": len(lines)}
