from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.baseline import heuristic_order, order_accuracy

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic/aligned"); args = parser.parse_args()
scores = []
for path in sorted(Path(args.aligned).glob("*.json")):
    page = json.loads(path.read_text(encoding="utf-8")); predicted = heuristic_order(page["ocr_lines"]); scores.append(order_accuracy(predicted, page["gt_order"]))
print(f"pages={len(scores)} mean_order_accuracy={sum(scores)/max(1,len(scores)):.4f}")
