from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.evaluation import xy_cut_order
from rlocr.ocr.alignment import kendall_tau

parser = argparse.ArgumentParser()
parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned")
parser.add_argument("--gt", default="data/synthetic_hard_large/gt")
args = parser.parse_args()

groups: dict[str, list[float]] = defaultdict(list)
for path in sorted(Path(args.aligned).glob("*.json")):
    aligned = json.loads(path.read_text(encoding="utf-8"))
    gt = json.loads((Path(args.gt) / path.name).read_text(encoding="utf-8"))
    valid = set(aligned.get("valid_line_ids", []))
    lines = [line for line in aligned["ocr_lines"] if line["id"] in valid]
    groups[gt["template"]].append(kendall_tau(xy_cut_order(lines), aligned["gt_order"]))

all_values = [value for values in groups.values() for value in values]
print(f"overall: pages={len(all_values)} disagreement_pct={100 * sum(v < .999 for v in all_values) / max(1, len(all_values)):.2f} mean_kendall_tau={sum(all_values) / max(1, len(all_values)):.4f}")
for template, values in sorted(groups.items()):
    print(f"{template}: pages={len(values)} disagreement_pct={100 * sum(v < .999 for v in values) / len(values):.2f} mean_kendall_tau={sum(values) / len(values):.4f}")
