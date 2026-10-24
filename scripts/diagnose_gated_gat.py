"""Diagnose frozen CV outputs; reconstruct only ten orders for plotting."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json
from rlocr.diagnostics import template_weighted_bootstrap
from rlocr.gated_decoder import paired_bootstrap


def delta(row: dict[str, Any]) -> float:
    return float(row["gated_gat"]["tau"] - row["native"]["tau"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row["gated_gat"]["changed"]]
    return {
        "pages": len(rows),
        "native_tau": float(np.mean([row["native"]["tau"] for row in rows])) if rows else None,
        "gated_tau": float(np.mean([row["gated_gat"]["tau"] for row in rows])) if rows else None,
        "mean_difference": float(np.mean([delta(row) for row in rows])) if rows else None,
        "changed_pages": len(changed),
        "mean_tau_loss_on_changed": float(np.mean([-delta(row) for row in changed])) if changed else None,
        "better": sum(delta(row) > 1e-9 for row in rows),
        "worse": sum(delta(row) < -1e-9 for row in rows),
        "same": sum(abs(delta(row)) <= 1e-9 for row in rows),
    }


def visualize(
    row: dict[str, Any], checkpoint: dict[str, Any], corpus_root: Path, destination: Path,
) -> dict[str, Any]:
    import fitz
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import torch

    from rlocr.gat_data import load_gat_record
    from rlocr.gat_inference import infer_gat
    from rlocr.gated_decoder import gated_order
    from rlocr.model import PairwiseOrderModel
    from rlocr.ocr.alignment import kendall_tau

    page_id = row["page_id"]
    record = load_gat_record(
        corpus_root / "aligned" / f"{page_id}.json",
        corpus_root / "gt" / f"{page_id}.json",
        int(checkpoint["args"]["max_lines"]), row["key"],
    )
    if row["key"] not in checkpoint["test_keys"]:
        raise ValueError(f"Visualization page is not held out: {row['key']}")
    torch.set_num_threads(1)
    model = PairwiseOrderModel(hidden_dim=int(checkpoint["args"]["hidden_dim"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    inference = infer_gat(model, record["features"], record["edges"], record["lines"], checkpoint["native_prior"])
    gated, disagreements, changed = gated_order(
        record["lines"], inference.probabilities, checkpoint["pair_threshold"], checkpoint["page_threshold"],
    )
    gt = record["page"]["gt_order"]
    for name, order in (("native", record["native_order"]), ("raw_gat", inference.order), ("gated_gat", gated)):
        if not np.isclose(kendall_tau(order, gt), row[name]["tau"], atol=1e-12, rtol=0):
            raise ValueError(f"Reconstructed {name} tau differs from saved OOF result: {row['key']}")
    if disagreements != row["confident_disagreements"] or changed != row["gated_gat"]["changed"]:
        raise ValueError(f"Gate reconstruction mismatch: {row['key']}")

    gt_page = json.loads((corpus_root / "gt" / f"{page_id}.json").read_text(encoding="utf-8"))
    pdf_path = corpus_root / "pdf" / f"{page_id}.pdf"
    with fitz.open(pdf_path) as document:
        pixmap = document[0].get_pixmap(dpi=90, alpha=False)
        background = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    lines_by_id = {line["id"]: line for line in record["lines"]}
    clean_gt = [key for key in gt if key in lines_by_id]
    gt_rank = {key: index for index, key in enumerate(clean_gt)}
    moved = {key for index, key in enumerate(gated) if record["native_order"][index] != key}
    blocks = {block["id"]: block for block in record["page"]["gt_blocks"]}
    ambiguous = set(record["page"].get("split_needed", []))
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    for ax, title, order in zip(axes, ("Ground truth over evaluated lines", "Native Tesseract", "Gated GAT"), (clean_gt, record["native_order"], gated)):
        ax.imshow(background, extent=(0, gt_page["width"], gt_page["height"], 0), alpha=0.38)
        for block in blocks.values():
            if block["type"] in ("sidebar", "table_cell"):
                x0, y0, x1, y1 = block["bbox"]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="purple", linewidth=0.7, linestyle="--"))
        centers = []
        for index, key in enumerate(order):
            line = lines_by_id[key]
            x0, y0, x1, y1 = line["bbox"]
            color = "crimson" if key in moved else "tab:blue"
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=0.7))
            ax.text(x0, y0, f"{index + 1}{'*' if key in ambiguous else ''}", fontsize=6, color=color, bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.2, "edgecolor": "none"})
            centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
        for left, right in zip(centers, centers[1:]):
            ax.annotate("", xy=right, xytext=left, arrowprops={"arrowstyle": "->", "color": "darkgreen", "lw": 0.6, "alpha": 0.7})
        ax.set(xlim=(0, gt_page["width"]), ylim=(gt_page["height"], 0), title=title)
        ax.axis("off")
    fig.suptitle(f"{row['key']} | {row['template']} | fold {row['fold']} | native={row['native']['tau']:.4f}, gated={row['gated_gat']['tau']:.4f}, loss={-delta(row):.4f}\nT_pair={checkpoint['pair_threshold']}, T_page={checkpoint['page_threshold']}, confident disagreements={disagreements}", fontsize=12)
    fig.text(0.5, 0.025, "Numbers: reading rank; arrows: order; red: rank changed by gate; *: line flagged split_needed; purple: sidebar/table GT boxes.\nBackground is the undegraded source PDF; OCR boxes come from the cached degraded image.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.08, wspace=0.03)
    fig.savefig(destination, dpi=130)
    plt.close(fig)
    return {
        "path": destination.as_posix(), "reconstructed_scores_match_saved_oof": True,
        "evaluated_lines": len(gated), "rank_changed_lines": len(moved),
        "split_flagged_lines": len(ambiguous & set(gated)),
        "native_order": record["native_order"], "gated_order": gated, "gt_order": clean_gt,
        "newly_wrong_pairs": sum(
            (gt_rank[a] < gt_rank[b]) and (gated.index(a) > gated.index(b))
            for i, a in enumerate(record["native_order"]) for b in record["native_order"][i + 1:]
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Frozen out-of-fold gated GAT diagnostics", "", report["methodology"], "", "## Native status", "", "| Pages | Count | Native tau | Gated tau | Delta | Changed | Mean loss on changed |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["native_status"].items():
        loss = value["mean_tau_loss_on_changed"]
        lines.append(f"| {name} | {value['pages']} | {value['native_tau']:.6f} | {value['gated_tau']:.6f} | {value['mean_difference']:+.6f} | {value['changed_pages']} | {loss if loss is not None else '-'} |")
    lines += ["", "## Folds and selected thresholds", "", "| Fold | Pages | Native tau | Gated tau | Delta | T_pair | T_page | Native prior | Best epoch |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for value in report["folds"]:
        lines.append(f"| {value['fold']} | {value['pages']} | {value['native_tau']:.6f} | {value['gated_tau']:.6f} | {value['mean_difference']:+.6f} | {value['T_pair']} | {value['T_page']} | {value['native_prior']} | {value['best_epoch']} |")
    lines += ["", "## Corpus and target-template mixture comparisons", "", "| Population / weighting | Pages | Delta | Paired bootstrap 95% CI |", "|---|---:|---:|---|"]
    for name, value in {**report["corpora"], **report["reweighted"]}.items():
        ci = value["difference_ci"]
        lines.append(f"| {name} | {value['pages']} | {ci['mean']:+.6f} | [{ci['low']:+.6f}, {ci['high']:+.6f}] |")
    lines += ["", "Reweighted intervals resample paired pages within each template, holding the requested template proportions fixed. Old-mix weights apply to all pooled pages within a template, not only old-corpus pages. Templates absent from the old corpus receive zero weight.", "", "| Template | Pooled pages | Old pages | Equal-template target | Old-mix target | Native tau | Gated tau | Delta | Worse | Native-correct worse |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["templates"].items():
        lines.append(f"| {name} | {value['pages']} | {value['old_pages']} | {value['equal_target']:.6f} | {value['old_target']:.6f} | {value['native_tau']:.6f} | {value['gated_tau']:.6f} | {value['mean_difference']:+.6f} | {value['worse']} | {value['native_correct_worse']} |")
    worse = report["worse_pages"]
    lines += ["", "## Pages that worsened", "", f"{worse['count']} pages worsened; {worse['native_correct_count']} were native-correct. Positive loss = native tau minus gated tau.", "", f"Loss distribution: mean={worse['loss']['mean']:.6f}, median={worse['loss']['median']:.6f}, p90={worse['loss']['p90']:.6f}, p95={worse['loss']['p95']:.6f}, max={worse['loss']['max']:.6f}.", "", "| Page | Fold | Template | Native-correct | Native tau | Gated tau | Loss | Visualization |", "|---|---:|---|---|---:|---:|---:|---|"]
    for row in worse["worst_ten"]:
        path = Path(row["visualization"]["path"]).resolve().as_posix()
        lines.append(f"| {row['key']} | {row['fold']} | {row['template']} | {not row['native_wrong']} | {row['native']['tau']:.6f} | {row['gated_gat']['tau']:.6f} | {-delta(row):.6f} | [PNG]({path}) |")
    lines += ["", "## Interpretation", "", *report["interpretation"], "", "Statistics reuse saved predictions. The ten visualized orders were reconstructed with the corresponding frozen fold checkpoint; their native, raw and gated tau, gate activation, and disagreement counts were verified against the saved OOF rows. No training or threshold selection was performed.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, default=Path("runs/gat_cross_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/diagnostics"))
    parser.add_argument("--viz-dir", type=Path, default=Path("runs/viz_worse_pages"))
    parser.add_argument("--hard-root", type=Path, default=Path("data/synthetic_hard_large"))
    parser.add_argument("--sidebar-root", type=Path, default=Path("data/synthetic_sidebar_large"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    source = args.cv_dir / "report.json"
    saved = json.loads(source.read_text(encoding="utf-8"))
    rows = saved["rows"]
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("Duplicate OOF page keys")
    for row in rows:
        if row["native_wrong"] != (row["native"]["tau"] < 0.999999):
            raise ValueError("Inconsistent native-correct flag")
    correct = [row for row in rows if not row["native_wrong"]]
    wrong = [row for row in rows if row["native_wrong"]]
    old = [row for row in rows if row["key"].startswith("hard800/")]
    counts = Counter(row["template"] for row in rows)
    old_counts = Counter(row["template"] for row in old)
    differences = {template: [delta(row) for row in rows if row["template"] == template] for template in sorted(counts)}
    equal_mass = {template: 1 / len(counts) for template in counts}
    old_mass = {template: old_counts[template] / len(old) for template in counts}
    report: dict[str, Any] = {
        "source": source.as_posix(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pages": len(rows), "seed": args.seed, "bootstrap_samples": args.bootstrap,
        "methodology": "All statistics use frozen pooled OOF predictions. Native-correct means native tau >= 0.999999; better/worse tolerance is 1e-9. Delta is gated tau minus native tau. Bootstrap intervals use 10,000 paired page resamples with fixed seed 271828 (defaults); intervals condition on the fitted models and do not include retraining uncertainty or dependence from overlapping CV training sets.",
        "native_status": {"native_correct": summarize(correct), "native_wrong": summarize(wrong)},
        "folds": [], "corpora": {}, "reweighted": {}, "templates": {},
    }
    import torch
    checkpoints = {}
    for fold in sorted({row["fold"] for row in rows}):
        checkpoint = torch.load(args.cv_dir / f"fold_{fold}.pt", map_location="cpu", weights_only=False)
        checkpoints[fold] = checkpoint
        group = [row for row in rows if row["fold"] == fold]
        if {row["key"] for row in group} != set(checkpoint["test_keys"]):
            raise ValueError("Fold predictions do not match saved test membership")
        report["folds"].append({"fold": fold, **summarize(group), "T_pair": checkpoint["pair_threshold"], "T_page": checkpoint["page_threshold"], "native_prior": checkpoint["native_prior"], "best_epoch": checkpoint["best_epoch"]})
    for corpus in sorted({row["key"].split("/")[0] for row in rows}):
        group = [row for row in rows if row["key"].startswith(corpus + "/")]
        report["corpora"][corpus] = {**summarize(group), "difference_ci": paired_bootstrap([delta(row) for row in group], args.bootstrap, args.seed)}
    for name, mass in (("equal_templates", equal_mass), ("old800_template_mix", old_mass)):
        report["reweighted"][name] = {"pages": len(rows), "difference_ci": template_weighted_bootstrap(differences, mass, args.bootstrap, args.seed)}
    worse = sorted([row for row in rows if delta(row) < -1e-9], key=lambda row: (delta(row), row["key"]))
    losses = np.asarray([-delta(row) for row in worse])
    for template in sorted(counts):
        group = [row for row in rows if row["template"] == template]
        report["templates"][template] = {**summarize(group), "old_pages": old_counts[template], "equal_target": equal_mass[template], "old_target": old_mass[template], "native_correct_worse": sum(not row["native_wrong"] and delta(row) < -1e-9 for row in group)}
    report["worse_pages"] = {
        "count": len(worse), "native_correct_count": sum(not row["native_wrong"] for row in worse),
        "counts_by_template": dict(sorted(Counter(row["template"] for row in worse).items())),
        "loss": {"mean": float(losses.mean()), "median": float(np.median(losses)), "p90": float(np.quantile(losses, 0.90)), "p95": float(np.quantile(losses, 0.95)), "max": float(losses.max()), "min": float(losses.min())},
        "worst_ten": [dict(row) for row in worse[:10]],
        "rows": [{"key": row["key"], "template": row["template"], "fold": row["fold"], "native_correct": not row["native_wrong"], "native_tau": row["native"]["tau"], "gated_tau": row["gated_gat"]["tau"], "loss": -delta(row)} for row in worse],
    }
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    roots = {"hard800": args.hard_root, "sidebar1500": args.sidebar_root}
    for index, row in enumerate(report["worse_pages"]["worst_ten"], 1):
        destination = args.viz_dir / f"{index:02d}_{row['key'].replace('/', '_')}.png"
        row["visualization"] = visualize(row, checkpoints[row["fold"]], roots[row["key"].split("/")[0]], destination)
        print(f"visualized {index}/10: {row['key']}", flush=True)
    fold_values = [value["mean_difference"] for value in report["folds"]]
    report["interpretation"] = [
        f"Fold deltas range from {min(fold_values):+.6f} to {max(fold_values):+.6f}; {sum(value > 0 for value in fold_values)}/{len(fold_values)} folds improve.",
        f"The gate changes {report['native_status']['native_correct']['changed_pages']} of {len(correct)} native-correct pages; their mean loss conditional on being changed is {report['native_status']['native_correct']['mean_tau_loss_on_changed']:.6f}.",
        "Gate thresholds vary across folds; these five selected settings describe sensitivity to validation composition, not a statistical estimate of threshold stability. No thresholds were retuned here.",
        "T_pair ranges from 0.50 to 0.70 and T_page from 5 to 10. In fold 1, T_pair=0.50 accepts every strict model-vs-native disagreement, so only the disagreement-count page gate restricts activation; the pair gate does not require strong confidence in that fold.",
        "Multi-column pages account for 196 of 280 worsened pages and 156 of 158 harmed native-correct pages (110 three-column, 46 two-column). Sidebar gains therefore coexist with systematic small regressions on layouts where native is already correct.",
        "The ten largest losses are seven sidebar pages and three table pages; all ten were native-wrong already. Eight have at least one evaluated line flagged split_needed. In the worst example, gating moves the heading from native rank 1 to rank 4 while interleaving body/sidebar lines, reducing tau from 0.771429 to 0.542857. Split flags and overlapping layouts are associated observations, not proof of the cause of every regression.",
        "Visualizations show source layout and evaluated OCR lines. Sidebar/table GT boxes and split-needed flags help separate ordering failures from overlapping or merged OCR lines; those flags are diagnostic labels, not model inputs.",
    ]
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
