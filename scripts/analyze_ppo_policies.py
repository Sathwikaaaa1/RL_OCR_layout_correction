"""Read-only synthetic template analysis and real all-lines paired comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rlocr.common import read_json, write_json
from rlocr.ppo_analysis import compare_pages, holm_adjust
from rlocr.real_stratified import SEEDS, switched_summary

FAMILIES = ("Sequential PPO v2","Page-level PPO")


def keyed(rows, expected):
    lookup = {row["key"]:row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != set(expected):
        raise ValueError("Prediction page keys must exactly match the saved PPO-test split")
    return [lookup[key] for key in expected]


def synthetic_pages(sequential, page_gate, splits):
    if sequential["provenance"] != page_gate["provenance"] or sequential["thresholds"] != page_gate["thresholds"]:
        raise ValueError("PPO reports must share cached GAT probabilities, splits and gate")
    test = splits["test"]
    if set(test)&(set(splits["train"])|set(splits["val"])):
        raise ValueError("Test split overlaps training or validation")
    base = sequential["seeds"][str(SEEDS[0])]["rows"]
    native_rows = keyed(base["native"],test)
    methods = {name:[row["tau"] for row in keyed(base[source],test)]
               for name,source in (("Native","native"),("Raw GAT","raw_gat"),("Gated GAT","gated_gat"))}
    switches = {}
    for seed in SEEDS:
        seq = sequential["seeds"][str(seed)]["rows"]
        pag = page_gate["seeds"][str(seed)]["rows"]
        for source,report in (("native",seq),("gated_gat",seq),("native",pag),("gated_gat",pag)):
            expected = methods["Native" if source == "native" else "Gated GAT"]
            if not np.allclose([row["tau"] for row in keyed(report[source],test)],expected,rtol=0,atol=1e-12):
                raise ValueError("Saved reports disagree on native or gated baseline")
        for family,rows in ((FAMILIES[0],seq["fine_tuned"]),(FAMILIES[1],pag["page_level_ppo"])):
            ordered = keyed(rows,test)
            if any(row["template"] != native["template"] for row,native in zip(ordered,native_rows)):
                raise ValueError("Template labels differ across saved reports")
            name = f"{family} / {seed}"
            methods[name] = [row["tau"] for row in ordered]
            switches[name] = [bool(row["changed"]) for row in ordered]
    for family in FAMILIES:
        methods[f"{family} / seed mean"] = np.mean([methods[f"{family} / {seed}"] for seed in SEEDS],axis=0).tolist()
    return {"keys":test,"templates":[row["template"] for row in native_rows],"methods":methods,"switches":switches}


def grouped_metrics(data):
    output = {}
    groups = {"all":list(range(len(data["keys"])))}
    groups.update({template:[i for i,value in enumerate(data["templates"]) if value == template]
                   for template in sorted(set(data["templates"]))})
    for group,indices in groups.items():
        native = np.asarray(data["methods"]["Native"])[indices]
        gate = np.asarray(data["methods"]["Gated GAT"])[indices]
        output[group] = {"pages":len(indices),"native_correct_pages":int((native==1).sum()),"methods":{}}
        for method,all_values in data["methods"].items():
            values = np.asarray(all_values)[indices]
            output[group]["methods"][method] = {"mean_tau":float(values.mean()),
                "native_correct_tau":float(values[native==1].mean()) if (native==1).any() else None,
                "versus_native":compare_pages(values,native),"versus_gated":compare_pages(values,gate)}
    # Each method/reference pair is one family spanning every template,
    # not all pages, seeds, controls and baselines in one giant family.
    templates = [name for name in output if name != "all"]
    for method in data["methods"]:
        for reference in ("versus_native","versus_gated"):
            adjusted = holm_adjust([output[t]["methods"][method][reference]["sign_flip_p"] for t in templates])
            for template,p_value in zip(templates,adjusted):
                output[template]["methods"][method][reference]["holm_template_p"] = p_value
    return output


def decisions(data):
    native = data["methods"]["Native"]
    result = {}
    for seed in SEEDS:
        name = f"Page-level PPO / {seed}"
        result[name] = switched_summary(native,data["methods"][name],data["switches"][name])
    name = "Page-level PPO / seed mean"
    union = np.any([data["switches"][f"Page-level PPO / {seed}"] for seed in SEEDS],axis=0)
    result["seed mean / any-seed switch (descriptive)"] = switched_summary(native,data["methods"][name],union)
    return result


def real_pages(report):
    rows = report["pages"]
    if report["provenance"]["max_lines"] != 0 or len(rows) != report["selected_pages"]:
        raise ValueError("Require the complete all-lines real-data report")
    methods = {name:[row["tau"][source] for row in rows] for name,source in (("Native","Native"),("Gated GAT","Gated GAT"))}
    switches = {}
    for family,prefix in ((FAMILIES[0],"PPO v2"),(FAMILIES[1],"Page PPO")):
        for seed in SEEDS:
            name = f"{family} / {seed}"
            methods[name] = [row["tau"][f"{prefix} {seed}"] for row in rows]
            switches[name] = [row["orders"][f"{prefix} {seed}"] != row["orders"]["Native"] for row in rows]
        methods[f"{family} / seed mean"] = np.mean([methods[f"{family} / {seed}"] for seed in SEEDS],axis=0).tolist()
    return {"keys":[row["page_id"] for row in rows],"templates":[row["layout"] for row in rows],"methods":methods,"switches":switches}


def format_ci(value):
    return f"{value['mean_difference']:+.6f} [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("runs/ppo_analysis"))
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=True)
    plan = {"primary":"PPO - native on synthetic PPO-test sidebar pages",
            "secondary":"PPO - gated GAT on all synthetic PPO-test pages",
            "policies":"Sequential PPO v2 validation-selected fine-tuned checkpoints and page-level PPO; each family uses mean tau across the same three fixed seeds within page. No seed selection.",
            "multiplicity":"Two co-primary policy-family p-values are Holm-adjusted together; likewise the two secondary policy-family p-values. Additionally, all template p-values are Holm-adjusted across the observed templates separately for each method/reference pair (including per-seed exploratory rows). Raw and adjusted values are retained. Real layout p-values get the same across-layout adjustment. This does not control one global family over every reported method, seed and comparison.",
            "exploratory":"All non-designated template cells, per-seed rows, controls, native-correct subsets, decision quality, and real-data overall/per-layout transfer comparisons are exploratory. Sidebar seed-mean cells carry both the primary-family and across-template adjustments.",
            "qualification":"Pre-specified for this retrospective analysis of already inspected datasets, not an independent prospective preregistration or fresh confirmatory holdout.",
            "resampling":"Paired page bootstrap 10,000 resamples, two-sided paired sign-flip p-values 10,000 samples, fixed seed 314159; each page is one observation, never three pseudo-independent seed observations."}
    write_json(args.output/"analysis_plan.json",plan)  # Before reading any outcome tables.
    source_paths = [Path("runs/ppo_gate_v2/report.json"),Path("runs/ppo_page_gate/report.json"),
                    Path("runs/ppo_gate/splits.json"),Path("runs/real_check/stratified/uncapped_inference/report.json")]
    sequential,page_gate,splits,real = [read_json(path) for path in source_paths]
    split_hash = hashlib.sha256(source_paths[2].read_bytes()).hexdigest()
    if sequential["provenance"]["split_sha256"] != split_hash:
        raise ValueError("Saved split hash differs from PPO report provenance")
    synthetic = synthetic_pages(sequential,page_gate,splits)
    real_data = real_pages(real)
    synthetic_metrics,real_metrics = grouped_metrics(synthetic),grouped_metrics(real_data)
    designated = {}
    for label,group,reference in (("primary","sidebar","versus_native"),("secondary","all","versus_gated")):
        designated[label] = {}
        for family in FAMILIES:
            method = f"{family} / seed mean"
            designated[label][family] = dict(synthetic_metrics[group]["methods"][method][reference])
        adjusted = holm_adjust([designated[label][family]["sign_flip_p"] for family in FAMILIES])
        for family,p_value in zip(FAMILIES,adjusted):
            designated[label][family]["holm_policy_family_p"] = p_value
    report = {"plan":plan,"designated_comparisons":designated,
              "synthetic":{"pages":len(synthetic["keys"]),"template_counts":dict(Counter(synthetic["templates"])),"groups":synthetic_metrics,
                           "page_level_decision_quality":decisions(synthetic)},
              "real_all_lines":{"pages":len(real_data["keys"]),"layout_counts":dict(Counter(real_data["templates"])),"groups":real_metrics,
                                "page_level_decision_quality":decisions(real_data)},
              "decision_definitions":"Switch = output order differs from native, not merely choosing REORDER when the gated order is unchanged. Synthetic changed flags cover all synthetic evaluation lines. Real switches compare full OCR output including unmatched lines. Fractions condition on switched pages. Net fixed-minus-broken = native-wrong pages restored to tau=1 minus native-correct pages harmed; also report net improved-minus-worsened for any tau change. A seed mean has no deployed order: union-of-seed switches is descriptive only; actual per-seed results are shown.",
              "limitations":"Synthetic PPO-test uses saved 345-page splits and OOF GAT predictions, but GAT fold training sets overlap; not a fully nested independent end-to-end evaluation. Real data use all 150 pages with alignment-conditioned scores and frozen full synthetic GAT inference, not OOF models. Confidence intervals condition on fitted policies. No learning, tuning, new OCR, or inference occurs in this analysis.",
              "sources":{path.as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
              "seed_values":list(SEEDS),"seconds":time.perf_counter()-started}
    write_json(args.output/"report.json",report)
    text = ["# PPO template and decision-quality analysis","","## Analysis plan","",*[
        f"- {key}: {value}" for key,value in plan.items()],"","## Primary and secondary comparisons","",
        "| Status | Policy (seed mean) | N | Difference [paired 95% CI] | Raw p | Holm across policies | Holm across templates |",
        "|---|---|---:|---|---:|---:|---:|"]
    for status,values in designated.items():
        for family,value in values.items():
            template_p = "n/a" if "holm_template_p" not in value else f"{value['holm_template_p']:.6f}"
            text.append(f"| {status} | {family} | {value['pages']} | {format_ci(value)} | {value['sign_flip_p']:.6f} | {value['holm_policy_family_p']:.6f} | {template_p} |")
    text += ["","## Synthetic template cells","","Sidebar seed-mean PPO-versus-native cells are designated primary above; all other template cells and all individual seeds are exploratory. Overall seed-mean PPO-versus-gated cells are secondary. Counts better/worse/same refer to native. Template Holm families are defined in the plan.","",
             "| Template | Method | N | Native-correct N | Mean tau | Native-correct tau | Better/worse/same | Delta vs native [95% CI] | Raw / Holm p vs native | Delta vs gate [95% CI] | Raw / Holm p vs gate |",
             "|---|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for group,cell in synthetic_metrics.items():
        for method,stats in cell["methods"].items():
            native,gate = stats["versus_native"],stats["versus_gated"]
            correct = "n/a" if stats["native_correct_tau"] is None else f"{stats['native_correct_tau']:.6f}"
            p_strings = [f"{value['sign_flip_p']:.6f} / " + (f"{value['holm_template_p']:.6f}" if "holm_template_p" in value else "n/a") for value in (native,gate)]
            text.append(f"| {group} | {method} | {cell['pages']} | {cell['native_correct_pages']} | {stats['mean_tau']:.6f} | {correct} | {native['better']}/{native['worse']}/{native['same']} | {format_ci(native)} | {p_strings[0]} | {format_ci(gate)} | {p_strings[1]} |")
    text += ["","## Real all-lines: PPO minus gated GAT (exploratory transfer)","",
             "Positive differences below do not imply improvement over native: native mean tau is "
             f"{real_metrics['all']['methods']['Native']['mean_tau']:.6f}, gated GAT {real_metrics['all']['methods']['Gated GAT']['mean_tau']:.6f}, "
             f"sequential PPO {real_metrics['all']['methods']['Sequential PPO v2 / seed mean']['mean_tau']:.6f}, "
             f"and page-level PPO {real_metrics['all']['methods']['Page-level PPO / seed mean']['mean_tau']:.6f}. Both PPO policies remain below native.","",
             "| Layout | Policy | N | Difference [paired 95% CI] | Raw p | Holm across layouts |", "|---|---|---:|---|---:|---:|"]
    for group,cell in real_metrics.items():
        for family in FAMILIES:
            value = cell["methods"][f"{family} / seed mean"]["versus_gated"]
            adjusted = f"{value['holm_template_p']:.6f}" if "holm_template_p" in value else "n/a"
            text.append(f"| {group} | {family} | {cell['pages']} | {format_ci(value)} | {value['sign_flip_p']:.6f} | {adjusted} |")
    text += ["","## Page-level PPO decision quality (exploratory)","",report["decision_definitions"],"",
             "| Dataset | Policy | Switched | Improved/worse/same | Fractions improved/worse/same | Fully fixed / broken | Net fixed-broken | Net improved-worsened |",
             "|---|---|---:|---|---|---|---:|---:|"]
    for dataset in ("synthetic","real_all_lines"):
        for policy,stats in report[dataset]["page_level_decision_quality"].items():
            fractions = "/".join("n/a" if stats[key] is None else f"{stats[key]:.1%}" for key in ("fraction_improved","fraction_worse","fraction_same"))
            text.append(f"| {dataset} | {policy} | {stats['pages_switched']} | {stats['improved']}/{stats['worse']}/{stats['same']} | {fractions} | {stats['fully_fixed']}/{stats['native_correct_broken']} | {stats['net_pages_fixed_minus_broken']} | {stats['net_improved_minus_worsened']} |")
    text += ["","## Limitations and reproduction","",report["limitations"],"",
             "`python scripts/analyze_ppo_policies.py` regenerates this report from saved predictions. `analysis_plan.json` is written before outcome tables are read. Complete numeric comparisons, source hashes and corrected p-values are in report.json.",""]
    (args.output/"report.md").write_text("\n".join(text),encoding="utf-8")
    print("Saved",args.output/"report.md",f"in {report['seconds']:.1f}s",flush=True)
    for status,values in designated.items():
        for method,value in values.items():
            print(status,method,format_ci(value),"Holm p",value["holm_policy_family_p"],flush=True)


if __name__ == "__main__":
    main()
