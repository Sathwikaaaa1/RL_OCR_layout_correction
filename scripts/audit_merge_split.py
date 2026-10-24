"""Read-only audit of existing structural labels; does not implement MERGE."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json


def audit(output=Path("runs/merge_split_audit"), roots=None):
    roots = roots or {"hard800": Path("data/synthetic_hard_large"), "sidebar1500": Path("data/synthetic_sidebar_large")}
    counts = defaultdict(Counter)
    total = Counter()
    for corpus, root in roots.items():
        for path in sorted((root / "aligned").glob("*.json")):
            page = json.loads(path.read_text(encoding="utf-8"))
            gt = json.loads((root / "gt" / path.name).read_text(encoding="utf-8"))
            lines = sorted(page["ocr_lines"], key=lambda line: line["native_order"])
            ranks = {line["id"]: i for i, line in enumerate(lines)}
            valid_ranks = {line_id: i for i, line_id in enumerate(page["tesseract_order"])}
            pairs = page["merge_pairs"]
            split = set(page["split_needed"])
            mapping = page["ocr_to_gt"]
            group_sizes = Counter(block for block in mapping.values() if block is not None)
            stats = Counter({"pages": 1, "ocr_lines": len(lines), "merge_pairs": len(pairs),
                             "native_candidate_boundaries": max(0, len(lines)-1),
                             "pages_with_merge_labels": int(bool(pairs)), "split_lines": len(split),
                             "pages_with_split_labels": int(bool(split)),
                             "multi_line_gt_blocks": sum(size > 1 for size in group_sizes.values()),
                             "native_adjacent_merge_pairs": sum(abs(ranks[a]-ranks[b]) == 1 for a, b in pairs),
                             "valid_native_adjacent_merge_pairs": sum(abs(valid_ranks[a]-valid_ranks[b]) == 1 for a, b in pairs),
                             "merge_pairs_touching_split_lines": sum(a in split or b in split for a, b in pairs)})
            counts[(corpus, gt["template"])].update(stats)
            total.update(stats)
    per_template = defaultdict(Counter)
    for (_, template), values in counts.items():
        per_template[template].update(values)
    report = {"scope": "All full aligned pages of the pooled 800+1500 corpora; not truncated to PPO max-lines.",
              "total": dict(total), "per_template": {key: dict(value) for key, value in sorted(per_template.items())},
              "per_corpus_template": {f"{corpus}/{template}": dict(value) for (corpus, template), value in sorted(counts.items())}}
    lines = ["# Read-only merge/split audit", "", report["scope"], "",
             "## Label representation", "",
             "Each aligned JSON stores `ocr_lines`, `gt_blocks`, `ocr_to_gt` (OCR-line ID to one GT block ID or null), `valid_line_ids`, `merge_pairs` (two line IDs per positive adjacent-in-GT-block pair), and `split_needed` (line IDs flagged as overlapping multiple GT blocks). `stats.merge_pair_count` and `stats.split_line_count` cache counts. GT block types and `column_index` are stored in `gt_blocks` and the GT JSON.", "",
             "Alignment groups lines mapping to the same GT block and column index, sorts each group by (y, x), and labels consecutive group members as merge pairs. A split flag is assigned when at least two GT blocks each cover >=15% of an OCR line's area. Mapping uses overlap plus text similarity. These are derived structural labels, not manually verified character segmentation.", "",
             "## Counts", "", "| Template | Pages | Pages with merge labels | Merge pairs | Adjacent in all native lines | Pages with split labels | Split lines | Multi-line GT blocks |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for template, values in {**report["per_template"], "TOTAL": report["total"]}.items():
        lines.append(f"| {template} | {values['pages']} | {values['pages_with_merge_labels']} | {values['merge_pairs']} | {values['native_adjacent_merge_pairs']} | {values['pages_with_split_labels']} | {values['split_lines']} | {values['multi_line_gt_blocks']} |")
    coverage = total["native_adjacent_merge_pairs"] / max(1, total["merge_pairs"])
    positive_rate = total["native_adjacent_merge_pairs"] / max(1, total["native_candidate_boundaries"])
    all_merge_f1 = 2*total["native_adjacent_merge_pairs"] / max(1, total["native_candidate_boundaries"] + total["native_adjacent_merge_pairs"])
    lines += ["", "## Minimal MERGE feasibility (not implemented)", "",
              f"Yes, a narrowly scoped adjacent-native-line MERGE decision is feasible. {total['native_adjacent_merge_pairs']} / {total['merge_pairs']} positive pairs ({coverage:.2%}) are adjacent in the full native line sequence; unsupported nonadjacent positives must be reported as outside this action space, rather than counted as missed executable actions.", "",
              "Define an episode over candidate boundaries between adjacent Tesseract lines. RETAIN leaves the boundary; a future MERGE would remove it and group their boxes/text. Evaluate pair-level precision, recall, and F1 against same-GT-block merge labels on this fixed eligible boundary set. Canonicalize pair IDs and report positive support and coverage; report global F1 separately if nonadjacent labels are included. Do not conflate merge F1 with reading-order tau.", "",
              f"There are {total['native_candidate_boundaries']} native-adjacent candidate boundaries; positive-label prevalence is {positive_rate:.2%}. A theoretical always-MERGE baseline already reaches eligible-pair F1 {all_merge_f1:.6f}. A future experiment must report this baseline and cross-column errors, rather than claiming improvement from a high merge F1 alone.", "",
              "A cross-column penalty can discourage a MERGE when OCR-only column estimates differ. GT column/block IDs may supervise training rewards and audits but must not enter inference observations. Recompute geometry after a future edit and prevent transitive merges that span columns. For an initial benchmark, exclude split-flagged or unaligned lines and report the exclusions explicitly.", "",
              f"Caution: {total['merge_pairs_touching_split_lines']} merge labels touch split-flagged lines. Multi-line paragraphs naturally produce positive merge labels, so pages with merge labels measure paragraph reconstruction opportunities, not necessarily OCR fragmentation defects. Likewise, overlapping synthetic sidebar/body GT boxes can create split flags without a genuinely glued OCR line. Split flags do not contain exact cut positions; they are insufficient alone for a full SPLIT action. Audit label quality before treating these counts as verified engine failures.", "",
              "This audit reads corpus files and writes only this report. No MERGE or SPLIT action, training, corpus regeneration, or label mutation is performed.", ""]
    write_json(output / "report.json", report)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["total"], indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/merge_split_audit"))
    audit(parser.parse_args().output)
