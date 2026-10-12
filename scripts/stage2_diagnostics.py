from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
from scipy.stats import kendalltau
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.ocr.alignment import kendall_tau
from rlocr.evaluation import decode_pairwise_order

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned"); args = parser.parse_args(); oracle_failures = 0; custom_values = []; scipy_values = []; dropped = 0
for path in sorted(Path(args.aligned).glob("*.json")):
    page = json.loads(path.read_text(encoding="utf-8")); ids = page.get("valid_line_ids", []); predicted = decode_pairwise_order(ids, {tuple(pair) for pair in page.get("order_pairs", [])}); target = page["gt_order"]; custom = kendall_tau(predicted, target); rank_pred = [predicted.index(value) for value in target if value in predicted]; rank_target = list(range(len(rank_pred))); scipy_value = float(kendalltau(rank_pred, rank_target).statistic) if len(rank_pred) > 1 else 1.0; custom_values.append(custom); scipy_values.append(scipy_value); dropped += int(page.get("dropped_line_count", 0)); oracle_failures += custom < 0.999999
print(f"pages={len(custom_values)} oracle_failures={oracle_failures} oracle_mean_tau={sum(custom_values)/max(1,len(custom_values)):.6f} scipy_mean_tau={sum(scipy_values)/max(1,len(scipy_values)):.6f} dropped_lines={dropped}")
