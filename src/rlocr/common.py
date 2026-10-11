from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    tesseract = os.environ.get("TESSERACT_CMD")
    if tesseract:
        config.setdefault("ocr", {})["tesseract_cmd"] = tesseract
    return config


def paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    values = config.get("paths", {})
    return {key: root / Path(value) for key, value in values.items() if key != "splits"} | {
        "splits": root / Path(values.get("splits", "data/synthetic/splits.json"))
    }


def ensure_dirs(config: dict[str, Any], root: Path) -> dict[str, Path]:
    result = paths(config, root)
    for key, value in result.items():
        (value.parent if value.suffix else value).mkdir(parents=True, exist_ok=True)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pt_bbox_to_px(bbox: tuple[float, float, float, float], height_pt: float, dpi: int) -> list[int]:
    scale = dpi / 72.0
    x0, y0, x1, y1 = bbox
    return [round(x0 * scale), round((height_pt - y1) * scale), round(x1 * scale), round((height_pt - y0) * scale)]


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def intersection_over_line(line: list[float], block: list[float]) -> float:
    x0, y0 = max(line[0], block[0]), max(line[1], block[1])
    x1, y1 = min(line[2], block[2]), min(line[3], block[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(0.0, line[2] - line[0]) * max(0.0, line[3] - line[1])
    return inter / area if area else 0.0
