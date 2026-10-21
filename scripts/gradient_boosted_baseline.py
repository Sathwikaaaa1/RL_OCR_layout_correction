from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.evaluation import kendall_tau, xy_cut_order
from rlocr.order_features import FEATURE_NAMES, decode_pair_probabilities, pair_dataset


def bootstrap(values: list[float], seed: int, samples: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    means = [float(array[rng.integers(0, len(array), len(array))].mean()) for _ in range(samples)]
    return {"mean": float(array.mean()), "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975))}


def page_lines(page: dict[str, Any]) -> list[dict[str, Any]]:
    valid = set(page.get("valid_line_ids", []))
    return [line for line in page["ocr_lines"] if line["id"] in valid]


def evaluate_page(
    page: dict[str, Any],
    model: HistGradientBoostingClassifier,
    native_prior: float,
    noise_std: float = 0.0,
    seed: int = 0,
    predicted: np.ndarray | None = None,
) -> dict[str, Any]:
    lines = page_lines(page)
    features, pair_targets_array, pairs = pair_dataset(page)
    predicted = model.predict_proba(features)[:, 1] if predicted is None and len(features) else (predicted if predicted is not None else np.empty(0))
    probabilities = {pair: float(probability) for pair, probability in zip(pairs, predicted)}
    decoded = decode_pair_probabilities(lines, probabilities, native_prior, noise_std, np.random.default_rng(seed))
    native = [line["id"] for line in sorted(lines, key=lambda item: item["native_order"])]
    xy = xy_cut_order(lines)
    gt = page["gt_order"]
    split = set(page.get("split_needed", []))
    unambiguous = [line["id"] for line in lines if line["id"] not in split]
    pair_target = {pair: int(target) for pair, target in zip(pairs, pair_targets_array)}
    pair_correct = sum(int((probability >= 0.5) == bool(pair_target[pair])) for pair, probability in probabilities.items())
    pair_total = len(probabilities)
    clean_pairs = [pair for pair in pairs if lines[pair[0]]["id"] not in split and lines[pair[1]]["id"] not in split]
    clean_correct = sum(int((probabilities[pair] >= 0.5) == bool(pair_target[pair])) for pair in clean_pairs)
    return {
        "template": page.get("template", "unknown"),
        "native_tau": kendall_tau(native, gt),
        "xy_cut_tau": kendall_tau(xy, gt),
        "model_tau": kendall_tau(decoded, gt),
        "native_unambiguous_tau": kendall_tau([value for value in native if value in unambiguous], [value for value in gt if value in unambiguous]),
        "model_unambiguous_tau": kendall_tau([value for value in decoded if value in unambiguous], [value for value in gt if value in unambiguous]),
        "pairwise_accuracy": pair_correct / max(1, pair_total),
        "unambiguous_pairwise_accuracy": clean_correct / max(1, len(clean_pairs)),
        "pair_count": pair_total,
        "unambiguous_pair_count": len(clean_pairs),
    }


def load_pages(directory: Path, ids: set[str], limit: int | None = None) -> list[dict[str, Any]]:
    pages = []
    for path in sorted(directory.glob("*.json")):
        if path.stem in ids:
            pages.append(json.loads(path.read_text(encoding="utf-8")))
            if limit and len(pages) >= limit:
                break
    return pages


def collect_training(pages: list[dict[str, Any]], pairs_per_page: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets = [], []
    rng = np.random.default_rng(seed)
    for page in pages:
        page_features, page_targets_array, _ = pair_dataset(page, max_pairs=pairs_per_page, rng=rng)
        if len(page_features):
            # pair_dataset contains both directions for evaluation. One
            # canonical direction is sufficient for fitting and halves memory.
            page_features = page_features[::2]
            page_targets_array = page_targets_array[::2]
            features.append(page_features)
            targets.append(page_targets_array)
    train_features = np.concatenate(features)
    train_targets = np.concatenate(targets)
    counts = np.bincount(train_targets, minlength=2).astype(np.float64)
    weights = np.asarray([1.0 / counts[int(target)] for target in train_targets], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return train_features, train_targets, weights


def summarize(rows: list[dict[str, Any]], seed: int, bootstrap_samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pages": len(rows), "overall": {}}
    for metric in ("native_tau", "xy_cut_tau", "model_tau", "native_unambiguous_tau", "model_unambiguous_tau", "pairwise_accuracy", "unambiguous_pairwise_accuracy"):
        result["overall"][metric] = bootstrap([float(row[metric]) for row in rows], seed, bootstrap_samples)
    result["by_template"] = {}
    for template in sorted({row["template"] for row in rows}):
        group = [row for row in rows if row["template"] == template]
        result["by_template"][template] = {metric: bootstrap([float(row[metric]) for row in group], seed, bootstrap_samples) for metric in ("native_tau", "xy_cut_tau", "model_tau", "native_unambiguous_tau", "model_unambiguous_tau")}
    return result


parser = argparse.ArgumentParser(description="Full-split native-aware HistGradientBoosting pairwise baseline.")
parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned")
parser.add_argument("--splits", default="data/synthetic_hard_large/splits.json")
parser.add_argument("--split", choices=("train", "val", "test"), default="test")
parser.add_argument("--train-pages", type=int, default=0)
parser.add_argument("--eval-pages", type=int, default=0)
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--seed", type=int, default=90210)
parser.add_argument("--max-iter", type=int, default=80)
parser.add_argument("--pairs-per-page", type=int, default=256)
parser.add_argument("--output", default="runs/gradient_boosted_baseline.json")
args = parser.parse_args()

split_data = json.loads(Path(args.splits).read_text(encoding="utf-8"))
directory = Path(args.aligned)
train_pages = load_pages(directory, set(split_data["train"]), args.train_pages or None)
eval_pages = load_pages(directory, set(split_data[args.split]), args.eval_pages or None)
train_features, train_targets, train_weights = collect_training(train_pages, args.pairs_per_page, args.seed)
model = HistGradientBoostingClassifier(max_iter=args.max_iter, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=args.seed)
model.fit(train_features, train_targets, sample_weight=train_weights)

validation_pages = load_pages(directory, set(split_data["val"]), args.eval_pages or None)
validation_predictions = [model.predict_proba(pair_dataset(page)[0])[:, 1] if len(pair_dataset(page)[0]) else np.empty(0) for page in validation_pages]
prior_candidates = (0.0, 0.25, 0.5, 1.0, 2.0)
prior_scores = []
for prior in prior_candidates:
    rows = [evaluate_page(page, model, prior, predicted=prediction) for page, prediction in zip(validation_pages, validation_predictions)]
    prior_scores.append((float(np.mean([row["model_tau"] for row in rows])), prior))
best_tau, best_prior = max(prior_scores)
eval_predictions = [model.predict_proba(pair_dataset(page)[0])[:, 1] if len(pair_dataset(page)[0]) else np.empty(0) for page in eval_pages]
rows = [evaluate_page(page, model, best_prior, seed=args.seed + index, predicted=prediction) for index, (page, prediction) in enumerate(zip(eval_pages, eval_predictions))]
noise_probe = {}
for noise in (0.0, 0.05, 0.1, 0.25, 0.5):
    noise_probe[str(noise)] = float(np.mean([evaluate_page(page, model, best_prior, noise, args.seed + index, prediction)["model_tau"] for index, (page, prediction) in enumerate(zip(eval_pages, eval_predictions))]))
result = {"train_pages": len(train_pages), "train_pairs": int(len(train_targets)), "eval_split": args.split, "native_prior": best_prior, "validation_prior_scores": {str(prior): tau for tau, prior in prior_scores}, "feature_names": list(FEATURE_NAMES), "metrics": summarize(rows, args.seed, args.bootstrap), "noise_probe": noise_probe}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"train_pages": result["train_pages"], "train_pairs": result["train_pairs"], "eval_split": args.split, "native_prior": best_prior, "metrics": result["metrics"]["overall"], "output": args.output}, indent=2))
