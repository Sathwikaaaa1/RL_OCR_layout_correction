from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import ensure_dirs, load_config, read_json, write_json
from rlocr.ocr.alignment import align_page, kendall_tau
from rlocr.ocr.cache import cache_tesseract_pages, default_worker_count
from rlocr.synth.generator import generate_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, OCR, align, and split a resumable corpus.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--num-pages", type=int)
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--keep-images", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    output = ensure_dirs(config, root)
    count = args.num_pages or int(config.get("num_pages", 600))

    started = time.perf_counter()
    page_ids = generate_pages(config, root, count, True)
    generation_seconds = time.perf_counter() - started
    command = config.get("ocr", {}).get("tesseract_cmd") or os.environ.get("TESSERACT_CMD") or str(Path("F:/Tesseract-OCR/tesseract.exe"))
    ocr_started = time.perf_counter()
    ocr_result = cache_tesseract_pages(
        output["images_tmp"], output["ocr"], config.get("ocr", {}).get("language", "eng"), command,
        page_ids, args.workers, args.keep_images,
    )
    ocr_wall_seconds = time.perf_counter() - ocr_started

    alignment_seconds = 0.0
    taus: list[float] = []
    template_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"pages": 0, "merge_pairs": 0, "split_lines": 0, "taus": []})
    for index, page_id in enumerate(page_ids, start=1):
        aligned_path = output["aligned"] / f"{page_id}.json"
        gt = read_json(output["gt"] / f"{page_id}.json")
        if aligned_path.exists():
            aligned = read_json(aligned_path)
        else:
            ocr = read_json(output["ocr"] / f"{page_id}.json")
            align_started = time.perf_counter()
            aligned = align_page(ocr["lines"], gt["blocks"], gt["reading_order"])
            write_json(aligned_path, aligned)
            alignment_seconds += time.perf_counter() - align_started
        value = kendall_tau(aligned["gt_order"], aligned["tesseract_order"])
        taus.append(value)
        stats = template_stats[gt["template"]]
        stats["pages"] += 1
        stats["merge_pairs"] += aligned["stats"]["merge_pair_count"]
        stats["split_lines"] += aligned["stats"]["split_line_count"]
        stats["taus"].append(value)
        if index % 100 == 0:
            print(f"alignment reviewed={index}/{len(page_ids)}", flush=True)

    indices = list(range(len(page_ids)))
    seed = int(config.get("seed", 1337))
    random.Random(seed).shuffle(indices)
    ratios = config.get("split_ratios", {})
    train_end = int(len(page_ids) * float(ratios.get("train", 0.70)))
    val_end = train_end + int(len(page_ids) * float(ratios.get("val", 0.15)))
    write_json(output["splits"], {"seed": seed, "train": [page_ids[i] for i in indices[:train_end]], "val": [page_ids[i] for i in indices[train_end:val_end]], "test": [page_ids[i] for i in indices[val_end:]]})
    by_template = {
        template: {"pages": values["pages"], "native_tau": sum(values["taus"]) / max(1, len(values["taus"])), "merge_pairs": values["merge_pairs"], "split_lines": values["split_lines"]}
        for template, values in sorted(template_stats.items())
    }
    metrics = {
        "pages": len(page_ids), "seed": seed, "workers": ocr_result["workers"],
        "seconds_per_page": {
            "generation_wall": generation_seconds / max(1, len(page_ids)),
            "ocr_wall": ocr_wall_seconds / max(1, len(page_ids)),
            "alignment_uncached": alignment_seconds / max(1, len(page_ids)),
        },
        "ocr": ocr_result,
        "mean_kendall_tau": sum(taus) / max(1, len(taus)),
        "native_order_disagreement_pct": 100 * sum(value < 0.999 for value in taus) / max(1, len(taus)),
        "by_template": by_template,
    }
    data_name = Path(config.get("paths", {}).get("data_root", "synthetic")).name
    metrics_path = root / "runs" / f"{data_name}_metrics.json"
    write_json(metrics_path, metrics)
    print(f"metrics={metrics_path}")
    print(f"generation={metrics['seconds_per_page']['generation_wall']:.3f}s/page ocr_wall={metrics['seconds_per_page']['ocr_wall']:.3f}s/page alignment={metrics['seconds_per_page']['alignment_uncached']:.3f}s/page")
    print(f"mean_kendall_tau={metrics['mean_kendall_tau']:.6f} disagreement_pct={metrics['native_order_disagreement_pct']:.2f}")
    for template, values in by_template.items():
        print(f"template={template} pages={values['pages']} native_tau={values['native_tau']:.6f}")


if __name__ == "__main__":
    main()
