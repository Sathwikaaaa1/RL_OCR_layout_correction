from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.env import OCRCorrectionEnv
from rlocr.gated_decoder import gated_order, native_order, paired_significance, raw_gat_order
from rlocr.model import PairwiseOrderModel
from rlocr.ocr.alignment import kendall_tau
from rlocr.ppo import spatial_edges

PAIR_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.01)
PAGE_THRESHOLDS = (0, 1, 2, 3, 5, 8, 10, 15, 20)


def load_record(path: Path, gt_root: Path, max_lines: int) -> dict[str, Any]:
    page = json.loads(path.read_text(encoding="utf-8"))
    page["template"] = json.loads((gt_root / path.name).read_text(encoding="utf-8"))["template"]
    valid = set(page.get("valid_line_ids", []))
    if valid:
        page["ocr_lines"] = [line for line in page["ocr_lines"] if line["id"] in valid]
    env = OCRCorrectionEnv(page, max_lines=max_lines)
    observation, _ = env.reset()
    n = len(env.lines)
    return {"page": page, "lines": env.lines, "features": torch.from_numpy(observation[:n]).float(), "edges": spatial_edges(observation[:n], k=4), "path": str(path)}


def infer(model: PairwiseOrderModel, record: dict[str, Any]) -> dict[tuple[int, int], float]:
    with torch.no_grad():
        logits = model(record["features"], record["edges"])
    return {(i, j): float(torch.sigmoid(logits[i, j]).item()) for i, j in combinations(range(len(record["lines"])), 2)}


def prepare(directory: Path, gt_root: Path, ids: list[str], max_lines: int, model: PairwiseOrderModel) -> list[dict[str, Any]]:
    wanted = set(ids)
    records = []
    for path in sorted(directory.glob("*.json")):
        if path.stem in wanted:
            record = load_record(path, gt_root, max_lines)
            record["probabilities"] = infer(model, record)
            records.append(record)
    return records


def tau(order: list[str], record: dict[str, Any], clean: bool = False) -> float:
    gt = record["page"]["gt_order"]
    if clean:
        excluded = set(record["page"].get("split_needed", []))
        order = [x for x in order if x not in excluded]
        gt = [x for x in gt if x not in excluded]
    return kendall_tau(order, gt)


def evaluate(records: list[dict[str, Any]], pair_threshold: float, page_threshold: int) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        native = native_order(record["lines"])
        raw = raw_gat_order(record["lines"], record["probabilities"])
        gated, disagreements, changed = gated_order(record["lines"], record["probabilities"], pair_threshold, page_threshold)
        gt = record["page"]["gt_order"]
        clean = set(line["id"] for line in record["lines"] if line["id"] not in set(record["page"].get("split_needed", [])))
        rows.append({"page_id": record["page"].get("page_id", Path(record["path"]).stem), "template": record["page"]["template"], "native": {"tau": tau(native, record), "unambiguous_tau": tau(native, record, True)}, "raw_gat": {"tau": tau(raw, record), "unambiguous_tau": tau(raw, record, True)}, "gated_gat": {"tau": tau(gated, record), "unambiguous_tau": tau(gated, record, True)}, "disagreements": disagreements, "changed": changed, "native_correct": tau(native, record) >= 0.999999})
    return rows


def mean(rows: list[dict[str, Any]], method: str, field: str = "tau") -> float:
    return float(np.mean([row[method][field] for row in rows])) if rows else 0.0


def summarize(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    correct = [row for row in rows if row["native_correct"]]
    wrong = [row for row in rows if not row["native_correct"]]
    return {"overall_tau": mean(rows, method), "native_correct_tau": mean(correct, method), "native_wrong_tau": mean(wrong, method), "per_template_tau": {template: mean([row for row in rows if row["template"] == template], method) for template in sorted({row["template"] for row in rows})}, "unambiguous_tau": mean(rows, method, "unambiguous_tau"), "pages_changed": sum(row["changed"] for row in rows) if method == "gated_gat" else None}


def table_markdown(table: dict[str, Any]) -> list[str]:
    lines = ["| Method | Overall τ | Native-correct τ | Native-wrong τ | Unambiguous τ | Pages changed | Per-template τ |", "|---|---:|---:|---:|---:|---:|---|"]
    for method, value in table.items():
        templates = "; ".join(f"{key}: {score:.4f}" for key, score in value["per_template_tau"].items())
        changed = value["pages_changed"] if value["pages_changed"] is not None else "-"
        lines.append(f"| {method} | {value['overall_tau']:.4f} | {value['native_correct_tau']:.4f} | {value['native_wrong_tau']:.4f} | {value['unambiguous_tau']:.4f} | {changed} | {templates} |")
    return lines


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="runs/native_aware_gat_full.pt")
parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned")
parser.add_argument("--gt", default="data/synthetic_hard_large/gt")
parser.add_argument("--splits", default="data/synthetic_hard_large/splits.json")
parser.add_argument("--output-dir", default="runs/gated_decoder")
parser.add_argument("--max-lines", type=int, default=96)
parser.add_argument("--threads", type=int, default=1)
args = parser.parse_args()

torch.set_num_threads(args.threads)
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model = PairwiseOrderModel(hidden_dim=int(checkpoint["args"].get("hidden_dim", 32)))
model.load_state_dict(checkpoint["model"])
model.eval()
split_data = json.loads(Path(args.splits).read_text(encoding="utf-8"))
records = {split: prepare(Path(args.aligned), Path(args.gt), split_data[split], args.max_lines, model) for split in ("val", "test")}

best = None
safe_best = None
for pair_threshold in PAIR_THRESHOLDS:
    for page_threshold in PAGE_THRESHOLDS:
        rows = evaluate(records["val"], pair_threshold, page_threshold)
        candidate = (mean(rows, "gated_gat"), -sum(row["changed"] for row in rows), -pair_threshold, -page_threshold)
        if best is None or candidate > best[0]:
            best = (candidate, pair_threshold, page_threshold, rows)
        native_correct_safe = all(row["gated_gat"]["tau"] >= row["native"]["tau"] - 1e-12 for row in rows if row["native_correct"])
        if native_correct_safe and (safe_best is None or candidate > safe_best[0]):
            safe_best = (candidate, pair_threshold, page_threshold, rows)
selected = safe_best or best
_, pair_threshold, page_threshold, validation_rows = selected
thresholds = {"pair_threshold": pair_threshold, "page_threshold": page_threshold, "validation_mean_tau": mean(validation_rows, "gated_gat"), "validation_pages_changed": sum(row["changed"] for row in validation_rows)}
thresholds["validation_native_correct_safe"] = safe_best is not None

test_rows = evaluate(records["test"], pair_threshold, page_threshold)
for row in test_rows:
    row["oracle_gate"] = row["native"] if row["native_correct"] else row["raw_gat"]
table = {method: summarize(test_rows, method) for method in ("native", "raw_gat", "gated_gat", "oracle_gate")}
native_values = [row["native"]["tau"] for row in test_rows]
significance = {method: paired_significance([row[method]["tau"] for row in test_rows], native_values, 10_000, 90210) for method in ("raw_gat", "gated_gat", "oracle_gate")}
report = {"checkpoint": args.checkpoint, "thresholds": thresholds, "validation": {"pages": len(validation_rows), "mean_tau": mean(validation_rows, "gated_gat")}, "test": {"pages": len(test_rows), "table": table, "significance": significance, "rows": test_rows}}
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
(out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
markdown = ["# Native-retaining GAT decoder", "", f"Checkpoint: `{args.checkpoint}`", f"Validation thresholds: `T_pair={pair_threshold}`, `T_page={page_threshold}`", "", *table_markdown(table), "", "## Paired significance versus native", "", "| Method | Mean difference | 95% bootstrap CI | Sign-flip p-value | Better | Worse | Same |", "|---|---:|---|---:|---:|---:|---:|"]
for method, value in significance.items():
    diff = value["difference"]
    markdown.append(f"| {method} | {diff['mean']:.6f} | [{diff['low']:.6f}, {diff['high']:.6f}] | {value['permutation_p_value']:.4f} | {value['better_pages']} | {value['worse_pages']} | {value['same_pages']} |")
(out / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
print(json.dumps({"thresholds": thresholds, "table": table, "significance": significance}, indent=2))
