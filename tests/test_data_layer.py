from __future__ import annotations

import json
from pathlib import Path

from rlocr.common import load_config
from rlocr.ocr.alignment import align_page
from rlocr.synth.generator import generate_pages


def test_generator_determinism_and_bbox_validity(tmp_path: Path) -> None:
    config = load_config(Path("config/default.yaml")); config["paths"] = {"pdf": "pdf", "images_tmp": "images", "gt": "gt", "ocr": "ocr", "aligned": "aligned", "viz": "viz", "splits": "splits.json"}; config["num_pages"] = 1
    generate_pages(config, tmp_path, 1, True); first = tmp_path / "gt" / "page_0000.json"
    original = json.loads(first.read_text())
    for block in original["blocks"]:
        x0, y0, x1, y1 = block["bbox"]; assert 0 <= x0 <= x1 <= original["width"]; assert 0 <= y0 <= y1 <= original["height"]


def test_alignment_toy_case() -> None:
    blocks = [{"id": "a", "bbox": [0, 0, 100, 50], "text": "hello world"}, {"id": "b", "bbox": [0, 50, 100, 100], "text": "second line"}]
    lines = [{"id": "l0", "bbox": [2, 2, 95, 45], "text": "hello world", "native_order": 0}, {"id": "l1", "bbox": [2, 52, 95, 95], "text": "second line", "native_order": 1}]
    result = align_page(lines, blocks, ["a", "b"]); assert result["ocr_to_gt"] == {"l0": "a", "l1": "b"}; assert not result["split_needed"]


def test_json_schema_loading() -> None:
    config = load_config(Path("config/default.yaml")); assert config["dpi"] == 150; assert set(config["split_ratios"]) == {"train", "val", "test"}; assert sum(config["split_ratios"].values()) == 1
