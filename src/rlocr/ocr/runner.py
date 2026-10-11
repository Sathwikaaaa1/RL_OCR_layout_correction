from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pytesseract
from PIL import Image


class OCRRunner(Protocol):
    def run(self, image_path: Path) -> list[dict[str, Any]]: ...


class TesseractRunner:
    def __init__(self, language: str = "eng", command: str | None = None) -> None:
        if command:
            pytesseract.pytesseract.tesseract_cmd = command
        self.language = language

    def run(self, image_path: Path) -> list[dict[str, Any]]:
        with Image.open(image_path) as image:
            data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)
        grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
        for i, raw in enumerate(data["text"]):
            text = str(raw).strip()
            if not text:
                continue
            key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
            entry = grouped.setdefault(key, {"block_num": key[0], "par_num": key[1], "line_num": key[2], "words": [], "confidences": [], "x0": 10**9, "y0": 10**9, "x1": 0, "y1": 0})
            x, y, w, h = (int(data[k][i]) for k in ("left", "top", "width", "height"))
            entry["words"].append(text); entry["confidences"].append(float(data["conf"][i]))
            entry["x0"], entry["y0"] = min(entry["x0"], x), min(entry["y0"], y)
            entry["x1"], entry["y1"] = max(entry["x1"], x + w), max(entry["y1"], y + h)
        lines = []
        for native_order, entry in enumerate(grouped.values()):
            lines.append({"id": f"line_{native_order:04d}", "bbox": [entry["x0"], entry["y0"], entry["x1"], entry["y1"]], "text": " ".join(entry["words"]), "confidence": sum(entry["confidences"]) / len(entry["confidences"]), "block_num": entry["block_num"], "par_num": entry["par_num"], "line_num": entry["line_num"], "native_order": native_order})
        return lines
