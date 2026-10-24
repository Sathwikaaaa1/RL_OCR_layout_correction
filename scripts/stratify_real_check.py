"""Analyze saved OmniDocBench predictions, plus an optional uncapped report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rlocr.common import read_json, write_json
from rlocr.evaluation import xy_cut_order
from rlocr.gated_decoder import native_order
from rlocr.ocr.alignment import kendall_tau
from rlocr.real_stratified import METHODS, SEEDS, SEED, stratify, switched_summary


def scores(tau: dict) -> dict:
    return {**{name:tau[name] for name in METHODS[:4]},
            "Sequential PPO v2":float(np.mean([tau[f"PPO v2 {seed}"] for seed in SEEDS])),
            "Page-level PPO":float(np.mean([tau[f"Page PPO {seed}"] for seed in SEEDS]))}


def page_rows(report: dict, manifest: dict, root: Path, *, original: dict | None = None) -> list[dict]:
    lookup = {page["page_id"]:page for page in report["pages"]}
    exclusions = {page["page_id"]:page.get("reason", "insufficient reliable alignment") for page in report.get("excluded",[])}
    original_lookup = {page["page_id"]:page for page in original["pages"]} if original else None
    cap = report["provenance"]["max_lines"]
    rows = []
    for selected in manifest["pages"]:
        page_id = selected["page_id"]
        page = lookup.get(page_id)
        lines = read_json(root/"ocr"/f"{page_id}.json")["lines"]
        input_lines = lines[:cap] if cap else lines
        if page:
            agreement = kendall_tau(page["orders"]["Native"],page["orders"]["XY-cut"])
            truth = page["alignment"]["gt_order"]
            if original_lookup is not None:
                truth = original_lookup[page_id]["alignment"]["gt_order"] if page_id in original_lookup else []
            valid = set(truth)
            tau = {name:kendall_tau(order,truth) for name,order in page["orders"].items()} if truth else {}
            if original_lookup is None and any(abs(value-page["tau"][name])>1e-12 for name,value in tau.items()):
                raise ValueError(f"Saved orders and saved tau disagree on {page_id}")
        else:
            agreement = kendall_tau(native_order(input_lines),xy_cut_order(input_lines))
            truth, valid, tau = [],set(),{}
        values = scores(tau) if tau else dict.fromkeys(METHODS)
        row = {"page_id":page_id,"layout_type":selected["attributes"]["layout"],
               "number_of_lines":len(lines),"input_lines":len(input_lines),"scored_lines":len(truth),
               "over_96_lines":len(lines)>96,"cap_applied":bool(cap and len(lines)>cap),
               "status":"scored" if tau else "excluded",
               "exclusion_reason":"" if tau else exclusions.get(page_id,"not eligible on original capped scoring support"),
               "native_tau":values["Native"],
               "xy_cut_tau":values["XY-cut"],"raw_gat_tau":values["Raw GAT"],"gated_gat_tau":values["Gated GAT"],
               "sequential_ppo_v2_tau":values["Sequential PPO v2"],"page_level_ppo_tau":values["Page-level PPO"],
               "native_xy_tau":agreement,"native_xy_agree":agreement>=.98,
               "native_xy_orders_identical":agreement == 1.0,"scores":values,
               "seed_tau":tau,"switched":{},"scored_switched":{}}
        if tau:
            for name,order in page["orders"].items():
                row["switched"][name] = order != page["orders"]["Native"]
                row["scored_switched"][name] = [key for key in order if key in valid] != [key for key in page["orders"]["Native"] if key in valid]
        rows.append(row)
    return rows


def switch_table(rows):
    rows = [row for row in rows if row["status"] == "scored"]
    native = [row["native_tau"] for row in rows]
    output = {"Gated GAT":switched_summary(native,[row["gated_gat_tau"] for row in rows],
                                           [row["switched"]["Gated GAT"] for row in rows])}
    for seed in SEEDS:
        name = f"Page PPO {seed}"
        output[name] = switched_summary(native,[row["seed_tau"][name] for row in rows],[row["switched"][name] for row in rows])
    output["Page-level PPO seed-mean / any-seed switch"] = switched_summary(native,[row["page_level_ppo_tau"] for row in rows],
                                  [any(row["switched"][f"Page PPO {seed}"] for seed in SEEDS) for row in rows])
    for name,value in output.items():
        keys = [f"Page PPO {seed}" for seed in SEEDS] if name.startswith("Page-level") else [name]
        value["scored_order_switched_pages"] = sum(any(row["scored_switched"][key] for key in keys) for row in rows)
    return output


def error_rate(values):
    values = list(values)
    return {"pages":len(values),"native_wrong_pages":sum(value != 1.0 for value in values),
            "native_error_rate":sum(value != 1.0 for value in values)/len(values),"mean_native_tau":float(np.mean(values))}


def distribution_shift(versions):
    result = {name:error_rate(row["native_tau"] for row in rows if row["status"] == "scored") for name,rows in versions.items()}
    cv = read_json(Path("runs/gat_cross_validation/report.json"))["rows"]
    result["synthetic_oof_2300_cap96"] = error_rate(row["native"]["tau"] for row in cv)
    ppo_training = set(read_json(Path("runs/ppo_gate/splits.json"))["train"])
    training_rows = [row for row in cv if row["key"] in ppo_training]
    if len(training_rows) != len(ppo_training):
        raise ValueError("PPO training keys do not match the saved OOF corpus")
    result[f"synthetic_ppo_train{len(training_rows)}_cap96"] = error_rate(row["native"]["tau"] for row in training_rows)
    for corpus in sorted({row["key"].split("/")[0] for row in cv}):
        result[f"synthetic_oof_{corpus}_cap96"] = error_rate(row["native"]["tau"] for row in cv if row["key"].startswith(corpus+"/"))
    root = Path("data/synthetic_hard_large")
    splits = read_json(root/"splits.json")
    training_values, full_training_values = [],[]
    for page_id in splits["train"]:
        aligned = read_json(root/"aligned"/f"{page_id}.json")
        valid = set(aligned["valid_line_ids"])
        lines = [line for line in aligned["ocr_lines"] if line["id"] in valid]
        training_values.append(kendall_tau(native_order(lines[:96]),aligned["gt_order"]))
        full_training_values.append(kendall_tau(native_order(lines),aligned["gt_order"]))
    result["synthetic_gat_train560_cap96"] = error_rate(training_values)
    result["synthetic_gat_train560_full_lines"] = error_rate(full_training_values)
    return result


def plots(rows, output, prefix):
    rows = [row for row in rows if row["status"] == "scored"]
    layouts = sorted({row["layout_type"] for row in rows})
    fig,axis = plt.subplots(figsize=(12,5))
    width = .13
    for index,method in enumerate(METHODS):
        means = [np.mean([row["scores"][method] for row in rows if row["layout_type"] == layout]) for layout in layouts]
        axis.bar(np.arange(len(layouts))+(index-2.5)*width,means,width,label=method)
    axis.set_xticks(np.arange(len(layouts)),layouts)
    axis.set_ylabel("Mean page Kendall tau")
    axis.set_ylim(min(0,min(row["scores"][method] for row in rows for method in METHODS)),1.05)
    axis.legend(fontsize=8,ncol=3)
    fig.tight_layout(); fig.savefig(output/f"{prefix}_layout_tau.png",dpi=140); plt.close(fig)
    native = np.asarray([row["native_tau"] for row in rows])
    fig,axes = plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
    low = min(row["scores"][method] for row in rows for method in METHODS)-.03
    for axis,method in zip(axes.flat,METHODS):
        axis.scatter(native,[row["scores"][method] for row in rows],s=12,alpha=.6)
        axis.plot([low,1.02],[low,1.02],"k--",linewidth=.7)
        axis.set_title(method); axis.set_xlabel("Native tau"); axis.set_ylabel("Method tau")
    fig.tight_layout(); fig.savefig(output/f"{prefix}_scatter.png",dpi=140); plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(12,8))
    deltas = [[row["scores"][method]-row["native_tau"] for row in rows] for method in METHODS]
    bins = np.linspace(min(min(values) for values in deltas)-.005,max(max(values) for values in deltas)+.005,31)
    for axis,method,values in zip(axes.flat,METHODS,deltas):
        axis.hist(values,bins=bins); axis.axvline(0,color="black",linestyle="--",linewidth=.7)
        axis.set_title(method); axis.set_xlabel("Method tau - native tau"); axis.set_ylabel("Pages")
    fig.tight_layout(); fig.savefig(output/f"{prefix}_delta_histogram.png",dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("runs/omnidocbench/report.json"))
    parser.add_argument("--uncapped",type=Path,default=Path("runs/real_check/stratified/uncapped_inference/report.json"))
    parser.add_argument("--root",type=Path,default=Path("data/omnidocbench"))
    parser.add_argument("--output",type=Path,default=Path("runs/real_check/stratified"))
    args = parser.parse_args()
    started = time.perf_counter()
    original, uncapped = read_json(args.source),read_json(args.uncapped)
    for field in ("checkpoint_sha256","manifest_sha256","annotation_sha256","native_prior","thresholds","page_policies","sequential_policies"):
        if original["provenance"][field] != uncapped["provenance"][field]:
            raise ValueError(f"Frozen inference provenance differs: {field}")
    manifest = read_json(args.root/"selected_pages.json")
    versions = {"cap96":page_rows(original,manifest,args.root),"all_lines":page_rows(uncapped,manifest,args.root),
                "all_lines_original_support":page_rows(uncapped,manifest,args.root,original=original)}
    original_pages = {row["page_id"]:row for row in original["pages"]}
    for row in uncapped["pages"]:
        previous = original_pages.get(row["page_id"])
        if previous and row["total_ocr_lines"] <= 96 and previous["orders"] != row["orders"]:
            raise ValueError(f"Unchanged-size page predictions differ: {row['page_id']}")
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in versions.items():
        filename = "per_page" if name == "cap96" else f"per_page_{name}"
        write_json(args.output/f"{filename}.json",rows)
        columns = [key for key in rows[0] if key not in ("scores","seed_tau","switched","scored_switched")]
        with (args.output/f"{filename}.csv").open("w",encoding="utf-8",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        plots(rows,args.output,name)
    report = {"protocol":"No retraining, tuning or model selection. XY-cut is the control. Native-wrong strata are conditioned on a ground-truth outcome and are descriptive, not deployable gates. No learned model outputs define strata. PPO scores average three fixed seeds within each page before page bootstrap. 10,000 paired bootstrap resamples; seed 314159; unadjusted descriptive intervals, not independent repeated tests.",
              "cap_explanation":"Originally every method used the first 96 cached OCR lines, then only unambiguous aligned text lines within that prefix were scored; pages were not skipped merely for exceeding 96. One page lacked three mapped blocks. The limit is a pipeline/training-size constant, not an architectural maximum: GAT shared node/pair layers and PPO fixed-size observations accept variable node counts. Frozen inference was rerun on all cached lines (maximum 213), with identical parameters, thresholds and normalizers. No OCR rerun. Larger graphs change normalization, pair evidence, and context and are out-of-training-size. Original files are preserved. All-lines metrics expand scoring support; original-support metrics instead restrict all-lines orders to the original scored lines on the original eligible pages. Native-vs-XY agreement is measured over the version's full input order, including unmatched lines.",
              "switch_definitions":"Switched means full output differs from native, including unmatched lines. Also report scored-order changes. PPO has no seed-mean order: show each seed, plus an explicitly descriptive any-seed-switched union evaluated using seed-mean tau. Net improved-minus-worsened counts any tau change. Net fixed-minus-broken instead counts originally wrong pages restored to tau=1 minus originally correct pages harmed. Fractions use switched pages as denominator; no switches gives null, not zero.",
              "strata_definitions":{"layout":"annotation layout type","line_count":"total cached OCR lines <=96 vs >96, also for the all-lines run","native_outcome":"exact native tau == 1 versus <1 on each version's scoring support (GT-conditioned)","native_xy_agreement":"OCR-only native/XY-cut full input order tau >=0.98 versus lower"},
              "sources":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.source,args.uncapped,Path("runs/gat_cross_validation/report.json"),Path("data/synthetic_hard_large/splits.json"),Path("runs/ppo_gate/splits.json"))},
              "versions":{name:{"selected_pages":len(rows),"scored_pages":sum(row["status"]=="scored" for row in rows),"strata":stratify(rows),"switches":switch_table(rows)} for name,rows in versions.items()},
              "native_error_rates":distribution_shift(versions),"bootstrap_samples":10_000,"bootstrap_seed":SEED}
    report["seconds"] = time.perf_counter()-started
    lines = ["# Real-data stratified analysis","",report["protocol"],"","## Overall comparison","",
             "| Support | N | Native | XY-cut (control) | Raw GAT | Gated GAT | Sequential PPO v2 | Page-level PPO |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,value in report["versions"].items():
        overall = value["strata"]["overall"]["all"]["methods"]
        lines.append(f"| {name} | {value['scored_pages']} | " + " | ".join(f"{overall[method]['mean_tau']:.6f}" for method in METHODS) + " |")
    capped = report["versions"]["cap96"]["strata"]
    correct = capped["native_outcome"]["native-correct"]["methods"]
    uncapped = report["versions"]["all_lines"]["strata"]["overall"]["all"]["methods"]
    report["findings"] = [
        f"No learned method exceeds Native mean tau in either capped or all-lines scoring. Raising the cap is a diagnostic, not a selected model setting.",
        f"On {correct['Native']['pages']} originally native-correct pages, gated GAT worsens {correct['Gated GAT']['worse']} and seed-mean page PPO worsens {correct['Page-level PPO']['worse']}; their conditional mean tau is {correct['Gated GAT']['mean_tau']:.6f} and {correct['Page-level PPO']['mean_tau']:.6f}.",
        f"Gated GAT loses mean tau in both <=96 and >96 line-count strata: {capped['line_count']['<=96']['methods']['Gated GAT']['mean_difference']:+.6f} and {capped['line_count']['>96']['methods']['Gated GAT']['mean_difference']:+.6f}. Harm is not confined to truncated pages.",
        f"The XY-cut control reaches {uncapped['XY-cut']['mean_tau']:.6f} on all lines, versus Native {uncapped['Native']['mean_tau']:.6f}; its original-support result separates context from extra scored lines."
    ]
    write_json(args.output/"report.json",report)
    lines += ["","## Findings","",*[f"- {finding}" for finding in report["findings"]],"","## What was scored and why rerun?","",report["cap_explanation"],"",
             "Coverage remains conditional on OCR/GT alignment; these are not official OmniDocBench scores. Corpus comparisons are descriptive: real-data alignment and template mix differ from synthetic data; synthetic model input was GT-valid-line filtered, real input was not.","",
             "## Native error rates / distribution shift","", "| Corpus / scoring support | Scored pages | Native-wrong | Error rate | Native tau |", "|---|---:|---:|---:|---:|"]
    for name,value in report["native_error_rates"].items():
        lines.append(f"| {name} | {value['pages']} | {value['native_wrong_pages']} | {value['native_error_rate']:.2%} | {value['mean_native_tau']:.6f} |")
    for version,value in report["versions"].items():
        lines += ["",f"## {version}","",f"Selected {value['selected_pages']}, scored {value['scored_pages']}. XY-cut is the control.","",
                  "| Stratum | Method | N | Mean tau | Delta vs native [95% CI] | Better/worse/same | Worst loss |", "|---|---|---:|---:|---|---|---:|"]
        for dimension,groups in value["strata"].items():
            for label,group in groups.items():
                for method,stats in group["methods"].items():
                    if not stats["pages"]:
                        continue
                    low,high = stats["ci95"]
                    lines.append(f"| {dimension}: {label} | {method} | {stats['pages']} | {stats['mean_tau']:.6f} | {stats['mean_difference']:+.6f} [{low:+.6f}, {high:+.6f}] | {stats['better']}/{stats['worse']}/{stats['same']} | {stats['worst_single_page_loss']:.6f} |")
        lines += ["","### Switched pages","",report["switch_definitions"],"", "| Method | Switched | Improved/worse/same | Fractions improved/worse/same | Net improved-worsened | Fully fixed / native-correct broken | Net fixed-broken |", "|---|---:|---|---|---:|---|---:|"]
        for method,stats in value["switches"].items():
            fractions = "/".join("n/a" if stats[key] is None else f"{stats[key]:.1%}" for key in ("fraction_improved","fraction_worse","fraction_same"))
            lines.append(f"| {method} | {stats['pages_switched']} | {stats['improved']}/{stats['worse']}/{stats['same']} | {fractions} | {stats['net_improved_minus_worsened']} | {stats['fully_fixed']}/{stats['native_correct_broken']} | {stats['net_pages_fixed_minus_broken']} |")
        lines += ["",f"![Layout tau]({version}_layout_tau.png)","",f"![Native versus method]({version}_scatter.png)","",f"![Paired differences]({version}_delta_histogram.png)"]
    lines += ["","## Files and reproduction","","`per_page.csv` and `per_page.json` cover all 150 selected pages, with null scores and explicit status for exclusions. Separate all-lines and original-support CSV/JSON files preserve comparison scope. Line count is the pre-cap OCR count; input and scored line counts are also recorded. `native_xy_agree` is the >=0.98 stratum indicator; `native_xy_orders_identical` tests exact order equality. Worst loss is nonnegative max(native - method); the signed minimum is also in JSON. Frozen predictions are in `uncapped_inference/report.json`. Original capped predictions remain unchanged. The analysis verifies frozen provenance, saved-order versus saved-score consistency, and identical predictions on unchanged-size pages.","",
              "```powershell","python scripts/evaluate_omnidocbench.py --cached-ocr-only --max-lines 0 --output runs/real_check/stratified/uncapped_inference","python scripts/stratify_real_check.py","```",""]
    (args.output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({name:{method:stats["mean_tau"] for method,stats in value["strata"]["overall"]["all"]["methods"].items()} for name,value in report["versions"].items()},indent=2))
    print(f"Analysis completed in {report['seconds']:.1f}s",flush=True)


if __name__ == "__main__":
    main()
