from __future__ import annotations
import argparse, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import ensure_dirs, load_config, read_json, write_json
from rlocr.ocr.alignment import align_page, kendall_tau
from rlocr.ocr.runner import TesseractRunner
from rlocr.synth.generator import generate_pages

parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/default.yaml"); parser.add_argument("--num-pages", type=int); parser.add_argument("--keep-images", action="store_true")
args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; config = load_config(root / args.config); out = ensure_dirs(config, root); count = args.num_pages or int(config.get("num_pages", 600))
t0 = time.perf_counter(); ids = generate_pages(config, root, count, True); generation_seconds = time.perf_counter() - t0
command = config.get("ocr", {}).get("tesseract_cmd") or str(Path("F:/Tesseract-OCR/tesseract.exe")); runner = TesseractRunner(config.get("ocr", {}).get("language", "eng"), command)
ocr_seconds = 0.0; alignment_seconds = 0.0; taus = []; template_stats = {}
for page_id in ids:
    image = out["images_tmp"] / f"{page_id}.png"; ocr_path = out["ocr"] / f"{page_id}.json"; gt = read_json(out["gt"] / f"{page_id}.json")
    if not ocr_path.exists():
        started = time.perf_counter(); write_json(ocr_path, {"page_id": page_id, "engine": "tesseract", "lines": runner.run(image)}); ocr_seconds += time.perf_counter() - started
    ocr = read_json(ocr_path); started = time.perf_counter(); aligned = align_page(ocr["lines"], gt["blocks"], gt["reading_order"]); write_json(out["aligned"] / f"{page_id}.json", aligned); alignment_seconds += time.perf_counter() - started
    taus.append(kendall_tau(aligned["gt_order"], aligned["tesseract_order"])); stats = template_stats.setdefault(gt["template"], {"pages": 0, "merge_pairs": 0, "split_lines": 0}); stats["pages"] += 1; stats["merge_pairs"] += aligned["stats"]["merge_pair_count"]; stats["split_lines"] += aligned["stats"]["split_line_count"]
    if not args.keep_images and image.exists(): image.unlink()
indices = list(range(len(ids))); seed = int(config.get("seed", 1337)); import random; random.Random(seed).shuffle(indices); n = len(ids); ratios = config.get("split_ratios", {}); train_end = int(n * float(ratios.get("train", 0.70))); val_end = train_end + int(n * float(ratios.get("val", 0.15))); write_json(out["splits"], {"seed": seed, "train": [ids[i] for i in indices[:train_end]], "val": [ids[i] for i in indices[train_end:val_end]], "test": [ids[i] for i in indices[val_end:]]})
write_json(root / "runs" / "metrics.json", {"pages": len(ids), "seconds_per_page": {"generation": generation_seconds / max(1, len(ids)), "ocr": ocr_seconds / max(1, len(ids)), "alignment": alignment_seconds / max(1, len(ids))}, "mean_kendall_tau": sum(taus) / max(1, len(taus)), "native_order_disagreement_pct": 100 * sum(t < 0.999 for t in taus) / max(1, len(taus)), "by_template": template_stats})
print(f"generation={generation_seconds/len(ids):.3f}s/page ocr={ocr_seconds/len(ids):.3f}s/page alignment={alignment_seconds/len(ids):.3f}s/page")
print(f"mean_kendall_tau={sum(taus)/max(1,len(taus)):.3f} disagreement_pct={100*sum(t < .999 for t in taus)/max(1,len(taus)):.1f}")
