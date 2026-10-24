"""Outcome-blind sidebar geometry fixed by preregistration f9949a4."""
from __future__ import annotations

from collections import Counter
import random
from typing import Any

from rlocr.omnidocbench import image_filename, text_blocks

DOCUMENT_TYPES = ("magazine", "newspaper", "colorful_textbook", "research_report")
LAYOUT_TYPES = ("single_column", "double_column", "three_column", "1andmore_column", "other_layout")
SELECTION_SEED = 20261024
MINIMUM_POOL = 80
TARGET = 100


def sidebar_witness(page: dict[str, Any]) -> dict | None:
    """Return the first annotated pair satisfying the frozen geometric rule.

    Reads no OCR, probabilities or predictions. Polygon bounds are normalized
    by annotated page width/height; character counts exclude whitespace.
    """
    width, height = page["page_info"]["width"], page["page_info"]["height"]
    if width <= 0 or height <= 0:
        return None
    blocks = []
    for block in page["layout_dets"]:
        if (block.get("category_type") != "text_block" or block.get("ignore",False)
                or not isinstance(block.get("order"),(int,float))):
            continue
        poly = block["poly"]
        box = (min(poly[::2])/width,min(poly[1::2])/height,
               max(poly[::2])/width,max(poly[1::2])/height)
        blocks.append((block,box,len("".join(block.get("text","").split()))))
    for side,side_box,side_chars in blocks:
        sx0,sy0,sx1,sy1 = side_box
        sw,sh,center = sx1-sx0,sy1-sy0,(sx0+sx1)/2
        if not (side_chars >= 40 and 0 < sw <= .35 and sh >= .02 and (center <= .25 or center >= .75)):
            continue
        for main,main_box,main_chars in blocks:
            if main is side:
                continue
            mx0,my0,mx1,my1 = main_box
            mw,mh = mx1-mx0,my1-my0
            if main_chars < 100 or mw < .40 or sw > .70*mw or mh <= 0:
                continue
            if not (sx1 <= mx0 or mx1 <= sx0):
                continue
            gap = max(mx0-sx1,sx0-mx1)
            vertical_overlap = max(0,min(sy1,my1)-max(sy0,my0))
            if gap <= .15 and vertical_overlap >= .50*min(sh,mh):
                return {"sidebar_anno_id":side["anno_id"],"main_anno_id":main["anno_id"],
                        "sidebar_bbox_normalized":list(side_box),"main_bbox_normalized":list(main_box),
                        "sidebar_nonspace_characters":side_chars,"main_nonspace_characters":main_chars,
                        "horizontal_gap":gap,"vertical_overlap_fraction_of_shorter":vertical_overlap/min(sh,mh)}
    return None


def select_sidebar_pages(pages: list[dict], previous_images: set[str]) -> dict:
    counts = Counter()
    candidates = []
    seen = set()
    for index,page in enumerate(pages):
        counts["all_annotations"] += 1
        info = page["page_info"]
        filename = image_filename(info["image_path"])
        if filename in previous_images:
            counts["excluded_previous_images"] += 1
            continue
        counts["unseen"] += 1
        attributes = info["page_attribute"]
        if attributes.get("language") != "english":
            continue
        counts["unseen_english"] += 1
        if attributes.get("data_source") not in DOCUMENT_TYPES:
            continue
        counts["allowed_document_type"] += 1
        if attributes.get("layout") not in LAYOUT_TYPES:
            continue
        counts["allowed_layout"] += 1
        if len(text_blocks(page)) < 3:
            continue
        counts["at_least_three_ordered_text_blocks"] += 1
        witness = sidebar_witness(page)
        if witness is None or filename in seen:
            continue
        seen.add(filename)
        counts["qualifying_unique_pages"] += 1
        candidates.append({"page_id":f"omni_sidebar_{index:04d}","annotation_index":index,
                           "image_path":filename,"attributes":attributes,"sidebar_proxy":witness})
    qualifying_count = len(candidates)
    selected = []
    if qualifying_count >= MINIMUM_POOL:
        shuffled = list(candidates)
        random.Random(SELECTION_SEED).shuffle(shuffled)
        selected = shuffled[:TARGET]
    return {"stage_counts":dict(counts),"qualifying_count":qualifying_count,
            "qualifying_document_counts":{name:sum(row["attributes"]["data_source"] == name for row in candidates) for name in DOCUMENT_TYPES},
            "qualifying_layout_counts":{name:sum(row["attributes"]["layout"] == name for row in candidates) for name in LAYOUT_TYPES},
            "qualifying_pages":candidates,"selected_pages":selected,"seed":SELECTION_SEED,
            "target":TARGET,"minimum_pool":MINIMUM_POOL,
            "status":"ready" if qualifying_count >= MINIMUM_POOL else "stopped_insufficient_qualifying_pages"}
