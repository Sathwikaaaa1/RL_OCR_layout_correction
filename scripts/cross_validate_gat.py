from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json
from rlocr.cross_validation import Fold, grouped_folds
from rlocr.gat_data import load_gat_record
from rlocr.gat_inference import GATInference, gat_inference_from_logits, infer_gat
from rlocr.gated_decoder import gated_order, paired_significance
from rlocr.model import PairwiseOrderModel
from rlocr.ocr.alignment import kendall_tau


PRIOR_CANDIDATES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PAIR_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.01)
PAGE_THRESHOLDS = (0, 1, 2, 3, 5, 8, 10, 15, 20)


def discover_pages(corpora: list[tuple[str, Path, Path]]) -> tuple[list[tuple[str, str]], dict[str, tuple[Path, Path]]]:
    pages: list[tuple[str, str]] = []
    paths: dict[str, tuple[Path, Path]] = {}
    for corpus, aligned_root, gt_root in corpora:
        for aligned_path in sorted(aligned_root.glob("*.json")):
            gt_path = gt_root / aligned_path.name
            if not gt_path.exists():
                continue
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            key = f"{corpus}/{aligned_path.stem}"
            pages.append((key, gt.get("template", "unknown")))
            paths[key] = (aligned_path, gt_path)
    return pages, paths


def model_logits(model: PairwiseOrderModel, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for record in records:
            result[record["key"]] = model(record["features"], record["edges"])
    return result


def tune_prior(records: list[dict[str, Any]], logits: dict[str, torch.Tensor]) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for prior in PRIOR_CANDIDATES:
        values = []
        for record in records:
            order = gat_inference_from_logits(logits[record["key"]], record["lines"], prior).order
            values.append(kendall_tau(order, record["page"]["gt_order"]))
        scores[str(prior)] = float(np.mean(values)) if values else 0.0
    best = max(PRIOR_CANDIDATES, key=lambda value: (scores[str(value)], value))
    return best, scores


def prepare_inference(model: PairwiseOrderModel, records: list[dict[str, Any]], prior: float) -> dict[str, GATInference]:
    return {
        record["key"]: infer_gat(model, record["features"], record["edges"], record["lines"], prior)
        for record in records
    }


def tune_gate(records: list[dict[str, Any]], inference: dict[str, GATInference]) -> tuple[float, int, float, int]:
    best: tuple[tuple[float, int, float, int], float, int, int] | None = None
    for pair_threshold in PAIR_THRESHOLDS:
        for page_threshold in PAGE_THRESHOLDS:
            taus = []
            changed_pages = 0
            for record in records:
                order, _, changed = gated_order(record["lines"], inference[record["key"]].probabilities, pair_threshold, page_threshold)
                taus.append(kendall_tau(order, record["page"]["gt_order"]))
                changed_pages += int(changed)
            mean_tau = float(np.mean(taus)) if taus else 0.0
            score = (mean_tau, -changed_pages, -pair_threshold, -page_threshold)
            if best is None or score > best[0]:
                best = (score, pair_threshold, page_threshold, changed_pages)
    assert best is not None
    return best[1], best[2], best[0][0], best[3]


def evaluate_test(
    records: list[dict[str, Any]],
    inference: dict[str, GATInference],
    pair_threshold: float,
    page_threshold: int,
    fold_index: int,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        native = record["native_order"]
        raw = inference[record["key"]].order
        gated, disagreements, gated_changed = gated_order(record["lines"], inference[record["key"]].probabilities, pair_threshold, page_threshold)
        gt = record["page"]["gt_order"]
        native_tau = kendall_tau(native, gt)
        raw_tau = kendall_tau(raw, gt)
        gated_tau = kendall_tau(gated, gt)
        best_order, best_tau = (raw, raw_tau) if raw_tau > native_tau else (native, native_tau)
        rows.append({
            "key": record["key"], "page_id": record["page"]["page_id"], "template": record["page"]["template"], "fold": fold_index,
            "native_wrong": native_tau < 0.999999,
            "native": {"tau": native_tau, "changed": False},
            "raw_gat": {"tau": raw_tau, "changed": raw != native},
            "gated_gat": {"tau": gated_tau, "changed": gated_changed},
            "best_of_oracle": {"tau": best_tau, "changed": best_order != native},
            "confident_disagreements": disagreements,
        })
    return rows


def mean_tau(rows: list[dict[str, Any]], method: str) -> float:
    return float(np.mean([row[method]["tau"] for row in rows])) if rows else 0.0


def method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    wrong = [row for row in rows if row["native_wrong"]]
    templates = sorted({row["template"] for row in rows})
    return {
        "overall_tau": mean_tau(rows, method),
        "native_wrong_tau": mean_tau(wrong, method),
        "pages_changed": sum(bool(row[method]["changed"]) for row in rows),
        "per_template_tau": {template: mean_tau([row for row in rows if row["template"] == template], method) for template in templates},
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Five-fold native-aware GAT cross-validation", "",
        f"Pages: `{report['pages']}`; folds: `{report['folds']}`; seed: `{report['seed']}`", "",
        "| Method | Overall tau | Native-wrong tau | Pages changed | Per-template tau |",
        "|---|---:|---:|---:|---|",
    ]
    for method, values in report["table"].items():
        per_template = "; ".join(f"{name}: {tau:.4f}" for name, tau in values["per_template_tau"].items())
        lines.append(f"| {method} | {values['overall_tau']:.6f} | {values['native_wrong_tau']:.6f} | {values['pages_changed']} | {per_template} |")
    lines += ["", "## Paired inference versus native", "", "| Method | Mean difference | 95% bootstrap CI | Sign-flip p-value | Better | Worse | Same |", "|---|---:|---|---:|---:|---:|---:|"]
    for method, values in report["significance"].items():
        difference = values["difference"]
        lines.append(f"| {method} | {difference['mean']:.6f} | [{difference['low']:.6f}, {difference['high']:.6f}] | {values['permutation_p_value']:.6f} | {values['better_pages']} | {values['worse_pages']} | {values['same_pages']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_fold(
    fold_index: int,
    fold: Fold,
    records: dict[str, dict[str, Any]],
    output: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prediction_path = output / f"fold_{fold_index}_predictions.json"
    if prediction_path.exists():
        return json.loads(prediction_path.read_text(encoding="utf-8"))
    train_records = [records[key] for key in fold.train]
    validation_records = [records[key] for key in fold.validation]
    test_records = [records[key] for key in fold.test]
    torch.manual_seed(args.seed + fold_index)
    model = PairwiseOrderModel(hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    state_path = output / f"fold_{fold_index}_state.pt"
    epoch_start, best_tau, best_epoch, stale = 1, -1.0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        best_state = saved["best_state"]
        best_tau, best_epoch, stale = saved["best_tau"], saved["best_epoch"], saved["stale"]
        history = saved["history"]
        epoch_start = saved["epoch"] + 1

    for epoch in range(epoch_start, args.max_epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + fold_index * 100 + epoch).permutation(len(train_records))
        losses = []
        for record_index in order:
            record = train_records[int(record_index)]
            logits = model(record["features"], record["edges"])
            loss = F.binary_cross_entropy_with_logits(logits[record["mask"]], record["target"][record["mask"]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_logits = model_logits(model, validation_records)
        prior, prior_scores = tune_prior(validation_records, validation_logits)
        validation_tau = prior_scores[str(prior)]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_tau": validation_tau, "native_prior": prior, "prior_scores": prior_scores})
        if validation_tau > best_tau + 1e-8:
            best_tau, best_epoch, stale = validation_tau, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_state": best_state, "best_tau": best_tau, "best_epoch": best_epoch, "stale": stale, "history": history}, state_path)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"fold": fold_index, **history[-1]}) + "\n")
        print(f"fold={fold_index} epoch={epoch} loss={np.mean(losses):.6f} val_tau={validation_tau:.6f} prior={prior}", flush=True)
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    validation_logits = model_logits(model, validation_records)
    prior, prior_scores = tune_prior(validation_records, validation_logits)
    validation_inference = prepare_inference(model, validation_records, prior)
    pair_threshold, page_threshold, gate_tau, gate_changed = tune_gate(validation_records, validation_inference)
    test_inference = prepare_inference(model, test_records, prior)
    rows = evaluate_test(test_records, test_inference, pair_threshold, page_threshold, fold_index)
    checkpoint = {
        "model": model.state_dict(), "fold": fold_index, "best_epoch": best_epoch,
        "validation_tau": best_tau, "native_prior": prior, "prior_scores": prior_scores,
        "pair_threshold": pair_threshold, "page_threshold": page_threshold,
        "validation_gate_tau": gate_tau, "validation_gate_pages_changed": gate_changed,
        "train_keys": list(fold.train), "validation_keys": list(fold.validation), "test_keys": list(fold.test),
        "args": vars(args), "history": history,
    }
    torch.save(checkpoint, output / f"fold_{fold_index}.pt")
    write_json(prediction_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-fold native-aware GAT evaluation over both hard corpora.")
    parser.add_argument("--hard-aligned", default="data/synthetic_hard_large/aligned")
    parser.add_argument("--hard-gt", default="data/synthetic_hard_large/gt")
    parser.add_argument("--sidebar-aligned", default="data/synthetic_sidebar_large/aligned")
    parser.add_argument("--sidebar-gt", default="data/synthetic_sidebar_large/gt")
    parser.add_argument("--output-dir", default="runs/gat_cross_validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-lines", type=int, default=96)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--only-fold", type=int, default=-1, help="Run one fold only; useful for timing and independent resume.")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pages, paths = discover_pages([
        ("hard800", Path(args.hard_aligned), Path(args.hard_gt)),
        ("sidebar1500", Path(args.sidebar_aligned), Path(args.sidebar_gt)),
    ])
    folds = grouped_folds(pages, args.folds, args.validation_fraction, args.seed)
    write_json(output / "folds.json", [{"train": list(fold.train), "validation": list(fold.validation), "test": list(fold.test)} for fold in folds])
    records: dict[str, dict[str, Any]] = {}
    for index, (key, _) in enumerate(pages, start=1):
        aligned_path, gt_path = paths[key]
        records[key] = load_gat_record(aligned_path, gt_path, args.max_lines, key)
        if index % 250 == 0:
            print(f"loaded={index}/{len(pages)}", flush=True)

    if args.only_fold >= args.folds:
        raise ValueError("--only-fold must be smaller than --folds")
    selected_folds = range(args.folds) if args.only_fold < 0 else (args.only_fold,)
    rows: list[dict[str, Any]] = []
    for fold_index in selected_folds:
        fold = folds[fold_index]
        rows.extend(train_fold(fold_index, fold, records, output, args))
    if args.only_fold >= 0:
        partial = {
            "fold": args.only_fold,
            "pages": len(rows),
            "table": {method: method_summary(rows, method) for method in ("native", "raw_gat", "gated_gat", "best_of_oracle")},
        }
        write_json(output / f"fold_{args.only_fold}_report.json", partial)
        print(json.dumps(partial, indent=2))
        return
    expected = {key for key, _ in pages}
    observed = [row["key"] for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("out-of-fold predictions do not cover every page exactly once")
    methods = ("native", "raw_gat", "gated_gat", "best_of_oracle")
    table = {method: method_summary(rows, method) for method in methods}
    native_values = [row["native"]["tau"] for row in rows]
    significance = {
        method: paired_significance([row[method]["tau"] for row in rows], native_values, args.bootstrap, args.seed)
        for method in methods[1:]
    }
    report = {"pages": len(rows), "folds": args.folds, "seed": args.seed, "table": table, "significance": significance, "rows": rows}
    write_json(output / "report.json", report)
    write_markdown(output / "report.md", report)
    print(json.dumps({"pages": len(rows), "table": table, "significance": significance}, indent=2))


if __name__ == "__main__":
    main()
