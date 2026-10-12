from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.ocr.alignment import kendall_tau

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned"); parser.add_argument("--gt", default="data/synthetic_hard_large/gt"); args = parser.parse_args()
groups = defaultdict(list)
for path in sorted(Path(args.aligned).glob("*.json")):
    page = json.loads(path.read_text(encoding="utf-8")); gt = json.loads((Path(args.gt) / f"{path.stem}.json").read_text(encoding="utf-8")); groups[gt["template"]].append(kendall_tau(page["tesseract_order"], page["gt_order"]))
for template, values in sorted(groups.items()):
    print(f"{template}: pages={len(values)} disagreement_pct={100*sum(value < .999 for value in values)/len(values):.2f} mean_kendall_tau={sum(values)/len(values):.4f}")
