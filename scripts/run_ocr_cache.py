from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import ensure_dirs, load_config
from rlocr.ocr.cache import cache_tesseract_pages, default_worker_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resumable Tesseract JSON cache in parallel.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--keep-images", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    output = ensure_dirs(config, root)
    command = config.get("ocr", {}).get("tesseract_cmd") or os.environ.get("TESSERACT_CMD") or str(Path("F:/Tesseract-OCR/tesseract.exe"))
    result = cache_tesseract_pages(
        output["images_tmp"], output["ocr"], config.get("ocr", {}).get("language", "eng"),
        command, workers=args.workers, keep_images=args.keep_images,
    )
    print(result)


if __name__ == "__main__":
    main()
