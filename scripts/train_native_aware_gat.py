from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.evaluation import kendall_tau, xy_cut_order
from rlocr.env import OCRCorrectionEnv
from rlocr.model import PairwiseOrderModel
from rlocr.order_features import decode_pair_probabilities
from rlocr.ppo import spatial_edges


PRIOR_CANDIDATES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NOISE_LEVELS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)


def load_page(path: Path, max_lines: int, gt_directory: Path | None = None) -> dict[str, Any]:
    page = json.loads(path.read_text(encoding="utf-8"))
    if gt_directory is not None:
        gt_path = gt_directory / path.name
        if gt_path.exists():
            page["template"] = json.loads(gt_path.read_text(encoding="utf-8")).get("template", "unknown")
    valid_ids = set(page.get("valid_line_ids", []))
    if valid_ids:
        page["ocr_lines"] = [line for line in page["ocr_lines"] if line["id"] in valid_ids]
    env = OCRCorrectionEnv(page, max_lines=max_lines)
    observation, _ = env.reset()
    n = len(env.lines)
    features = torch.from_numpy(observation[:n]).float()
    edges = spatial_edges(observation[:n], k=4)
    rank = {value: index for index, value in enumerate(page["gt_order"])}
    target = torch.zeros((n, n), dtype=torch.float32)
    mask = torch.ones((n, n), dtype=torch.bool)
    for i, left in enumerate(env.lines):
        for j, right in enumerate(env.lines):
            if i == j:
                mask[i, j] = False
            else:
                target[i, j] = float(rank[left["id"]] < rank[right["id"]])
    native_rank = {line["id"]: index for index, line in enumerate(sorted(env.lines, key=lambda item: item["native_order"]))}
    native_failure_mask = torch.zeros((n, n), dtype=torch.bool)
    for i, left in enumerate(env.lines):
        for j, right in enumerate(env.lines):
            if i != j:
                native_failure_mask[i, j] = bool(target[i, j]) != (native_rank[left["id"]] < native_rank[right["id"]])
    return {"path": str(path), "page": page, "lines": env.lines, "features": features, "edges": edges, "target": target, "mask": mask, "native_failure_mask": native_failure_mask}


def load_records(directory: Path, ids: list[str], max_lines: int, limit: int = 0, gt_directory: Path | None = None) -> list[dict[str, Any]]:
    wanted = set(ids)
    records = []
    for path in sorted(directory.glob("*.json")):
        if path.stem in wanted:
            records.append(load_page(path, max_lines, gt_directory))
            if limit and len(records) >= limit:
                break
    return records


def native_order(record: dict[str, Any]) -> list[str]:
    return [line["id"] for line in sorted(record["lines"], key=lambda item: item["native_order"])]


def decode_logits(record: dict[str, Any], logits: torch.Tensor, prior: float, noise: float = 0.0, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    probabilities = {(i, j): float(torch.sigmoid(logits[i, j]).item()) for i in range(len(record["lines"])) for j in range(i + 1, len(record["lines"]))}
    return decode_pair_probabilities(record["lines"], probabilities, native_prior=prior, noise_std=noise, rng=rng)


def pair_accuracy(record: dict[str, Any], logits: torch.Tensor) -> float:
    mask = record["mask"]
    return float(((logits > 0) == record["target"])[mask].float().mean().item())


def page_metrics(record: dict[str, Any], logits: torch.Tensor, prior: float, noise: float = 0.0, seed: int = 0) -> dict[str, Any]:
    page = record["page"]
    gt = page["gt_order"]
    native = native_order(record)
    decoded = decode_logits(record, logits, prior, noise, seed)
    split = set(page.get("split_needed", []))
    unambiguous = {line["id"] for line in record["lines"] if line["id"] not in split}
    native_clean = [value for value in native if value in unambiguous]
    decoded_clean = [value for value in decoded if value in unambiguous]
    gt_clean = [value for value in gt if value in unambiguous]
    native_rank = {value: index for index, value in enumerate(native)}
    native_pairwise_accuracy = sum(
        int((native_rank[left["id"]] < native_rank[right["id"]]) == bool(record["target"][i, j]))
        for i, left in enumerate(record["lines"])
        for j, right in enumerate(record["lines"])
        if i != j
    ) / max(1, int(record["mask"].sum().item()))
    return {
        "page_id": page.get("page_id", Path(record["path"]).stem),
        "template": page.get("template", "unknown"),
        "native_tau": kendall_tau(native, gt),
        "model_tau": kendall_tau(decoded, gt),
        "native_unambiguous_tau": kendall_tau(native_clean, gt_clean),
        "model_unambiguous_tau": kendall_tau(decoded_clean, gt_clean),
        "pairwise_accuracy": pair_accuracy(record, logits),
        "native_pairwise_accuracy": native_pairwise_accuracy,
        "valid_pair_count": int(record["mask"].sum().item()),
        "native_correct": kendall_tau(native, gt) >= 0.999999,
    }


def evaluate(model: PairwiseOrderModel, records: list[dict[str, Any]], prior: float, noise: float = 0.0, seed: int = 0) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    with torch.no_grad():
        for index, record in enumerate(records):
            logits = model(record["features"], record["edges"])
            rows.append(page_metrics(record, logits, prior, noise, seed + index))
    return rows


def tune_prior(model: PairwiseOrderModel, records: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for prior in PRIOR_CANDIDATES:
        rows = evaluate(model, records, prior)
        scores[str(prior)] = float(np.mean([row["model_tau"] for row in rows])) if rows else 0.0
    best_prior = max(PRIOR_CANDIDATES, key=lambda value: (scores[str(value)], value))
    return best_prior, scores


def bootstrap(values: list[float], seed: int = 90210, samples: int = 1000) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [array[rng.integers(0, len(array), len(array))].mean() for _ in range(samples)]
    return {"mean": float(array.mean()), "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975))}


def summarize(rows: list[dict[str, Any]], seed: int, bootstrap_samples: int) -> dict[str, Any]:
    metrics = ("native_tau", "model_tau", "native_unambiguous_tau", "model_unambiguous_tau", "native_pairwise_accuracy", "pairwise_accuracy")
    summary: dict[str, Any] = {"pages": len(rows), "overall": {metric: bootstrap([row[metric] for row in rows], seed, bootstrap_samples) for metric in metrics}}
    summary["by_template"] = {}
    summary["by_native_status"] = {}
    for key, groups in (("by_template", {name: [row for row in rows if row["template"] == name] for name in sorted({row["template"] for row in rows})}), ("by_native_status", {"native_correct": [row for row in rows if row["native_correct"]], "native_wrong": [row for row in rows if not row["native_correct"]]})):
        for group_name, group in groups.items():
            summary[key][group_name] = {metric: bootstrap([row[metric] for row in group], seed, bootstrap_samples) for metric in metrics}
    return summary


def corr(xs: list[float], ys: list[float]) -> float:
    return float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 and np.std(xs) and np.std(ys) else 0.0


parser = argparse.ArgumentParser(description="Train and evaluate the native-aware pairwise GAT.")
parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned")
parser.add_argument("--gt", default="data/synthetic_hard_large/gt")
parser.add_argument("--splits", default="data/synthetic_hard_large/splits.json")
parser.add_argument("--max-lines", type=int, default=96)
parser.add_argument("--max-epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--hidden-dim", type=int, default=32)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--train-pages", type=int, default=0)
parser.add_argument("--val-pages", type=int, default=0)
parser.add_argument("--test-pages", type=int, default=0)
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--failure-oversample", type=float, default=0.0, help="Extra training repetitions for native-wrong pages.")
parser.add_argument("--failure-pair-weight", type=float, default=0.0, help="Additional loss weight for pairs where native order is wrong.")
parser.add_argument("--checkpoint", default="runs/native_aware_gat.pt")
parser.add_argument("--output", default="runs/native_aware_gat_report.json")
args = parser.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(90210)
np.random.seed(90210)
split_data = json.loads(Path(args.splits).read_text(encoding="utf-8"))
directory = Path(args.aligned)
gt_directory = Path(args.gt)
train_records = load_records(directory, split_data["train"], args.max_lines, args.train_pages, gt_directory)
val_records = load_records(directory, split_data["val"], args.max_lines, args.val_pages, gt_directory)
test_records = load_records(directory, split_data["test"], args.max_lines, args.test_pages, gt_directory)
model = PairwiseOrderModel(hidden_dim=args.hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
best_state = copy.deepcopy(model.state_dict())
best_tau = -1.0
best_epoch = 0
stale = 0
history = []

for epoch in range(1, args.max_epochs + 1):
    model.train()
    order = list(range(len(train_records)))
    rng = np.random.default_rng(90210 + epoch)
    rng.shuffle(order)
    losses = []
    for index in order:
        record = train_records[index]
        repeats = 1 + int(args.failure_oversample) if not page_metrics(record, torch.zeros_like(record["target"]), 0)["native_correct"] else 1
        for _ in range(repeats):
            logits = model(record["features"], record["edges"])
            valid_logits = logits[record["mask"]]
            valid_targets = record["target"][record["mask"]]
            weights = torch.ones_like(valid_targets)
            if args.failure_pair_weight:
                weights = weights + args.failure_pair_weight * record["native_failure_mask"][record["mask"]].float()
            loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    prior, prior_scores = tune_prior(model, val_records)
    val_rows = evaluate(model, val_records, prior)
    val_tau = float(np.mean([row["model_tau"] for row in val_rows])) if val_rows else 0.0
    history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_tau": val_tau, "validation_prior": prior, "validation_prior_scores": prior_scores})
    print(f"epoch={epoch} train_loss={np.mean(losses):.5f} val_tau={val_tau:.5f} prior={prior}")
    if val_tau > best_tau + 1e-6:
        best_tau = val_tau
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        stale = 0
    else:
        stale += 1
        if stale >= args.patience:
            break

model.load_state_dict(best_state)
best_prior, validation_prior_scores = tune_prior(model, val_records)
test_rows = evaluate(model, test_records, best_prior)
native_rows = [{**row, "model_tau": row["native_tau"], "model_unambiguous_tau": row["native_unambiguous_tau"], "pairwise_accuracy": row["native_pairwise_accuracy"]} for row in test_rows]
noise_probe = {str(noise): float(np.mean([row["model_tau"] for row in evaluate(model, test_records, best_prior, noise, 1000)])) for noise in NOISE_LEVELS}
pair_tau_correlation = corr([row["pairwise_accuracy"] for row in test_rows], [row["model_tau"] for row in test_rows])
worst = sorted(test_rows, key=lambda row: row["model_tau"])[:5]
result = {
    "train_pages": len(train_records), "validation_pages": len(val_records), "test_pages": len(test_records),
    "best_epoch": best_epoch, "best_validation_tau": best_tau, "validation_prior": best_prior,
    "validation_prior_scores": validation_prior_scores, "history": history,
    "test": summarize(test_rows, 90210, args.bootstrap), "test_native_reference": summarize(native_rows, 90210, args.bootstrap),
    "pairwise_tau_correlation": pair_tau_correlation, "noise_probe_tau": noise_probe,
    "worst_pages": worst, "checkpoint": args.checkpoint,
}
Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict(), "args": vars(args), "best_epoch": best_epoch, "validation_prior": best_prior}, args.checkpoint)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"train_pages": len(train_records), "best_epoch": best_epoch, "validation_tau": best_tau, "validation_prior": best_prior, "test": result["test"]["overall"], "pairwise_tau_correlation": pair_tau_correlation, "worst_pages": worst}, indent=2))
