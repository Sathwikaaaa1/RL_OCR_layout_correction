"""Resumable, process-parallel Tesseract cache creation."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


def default_worker_count() -> int:
    """Leave one logical CPU free, with a conservative cap for Tesseract."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _ocr_one(image: str, target: str, language: str, command: str | None) -> tuple[str, float]:
    from rlocr.ocr.runner import TesseractRunner

    image_path = Path(image)
    target_path = Path(target)
    started = time.perf_counter()
    lines = TesseractRunner(language, command).run(image_path)
    elapsed = time.perf_counter() - started
    payload = {"page_id": image_path.stem, "engine": "tesseract", "lines": lines, "seconds": elapsed}
    temporary = target_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target_path)
    return image_path.stem, elapsed


def cache_tesseract_pages(
    image_directory: Path,
    ocr_directory: Path,
    language: str,
    command: str | None,
    page_ids: Iterable[str] | None = None,
    workers: int | None = None,
    keep_images: bool = False,
) -> dict[str, Any]:
    """OCR missing pages in parallel and leave completed JSON files resumable."""
    ocr_directory.mkdir(parents=True, exist_ok=True)
    wanted = set(page_ids) if page_ids is not None else None
    images = [path for path in sorted(image_directory.glob("*.png")) if wanted is None or path.stem in wanted]
    cached = 0
    pending: list[tuple[Path, Path]] = []
    for image in images:
        target = ocr_directory / f"{image.stem}.json"
        if target.exists():
            cached += 1
            if not keep_images:
                image.unlink()
        else:
            pending.append((image, target))

    elapsed_sum = 0.0
    completed = 0
    worker_count = max(1, workers or default_worker_count())
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_ocr_one, str(image), str(target), language, command): image for image, target in pending}
        for future in as_completed(futures):
            image = futures[future]
            _, elapsed = future.result()
            elapsed_sum += elapsed
            completed += 1
            if not keep_images and image.exists():
                image.unlink()
            if completed % 25 == 0 or completed == len(pending):
                print(f"ocr completed={completed}/{len(pending)} cached={cached}", flush=True)
    return {"completed": completed, "cached": cached, "workers": worker_count, "worker_seconds": elapsed_sum}
