from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import ensure_dirs, load_config, read_json, write_json
from rlocr.ocr.runner import TesseractRunner

parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/default.yaml"); parser.add_argument("--keep-images", action="store_true")
args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; config = load_config(root / args.config); out = ensure_dirs(config, root)
command = config.get("ocr", {}).get("tesseract_cmd") or os.environ.get("TESSERACT_CMD") or str(Path("F:/Tesseract-OCR/tesseract.exe"))
runner = TesseractRunner(config.get("ocr", {}).get("language", "eng"), command)
for image in sorted(out["images_tmp"].glob("*.png")):
    target = out["ocr"] / f"{image.stem}.json"
    if target.exists(): continue
    started = time.perf_counter(); lines = runner.run(image); write_json(target, {"page_id": image.stem, "engine": "tesseract", "lines": lines, "seconds": time.perf_counter() - started})
    if not args.keep_images: image.unlink()
