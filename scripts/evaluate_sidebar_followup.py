"""Authorized exploratory follow-up: three new candidates versus seen same-type pages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rlocr.common import read_json,write_json
from rlocr.omnidocbench import directory_bytes
from rlocr.ppo_analysis import compare_pages
from rlocr.real_stratified import SEEDS,switched_summary
from rlocr.sidebar_selection import sidebar_witness

ROOT = Path("data/omnidocbench_sidebar_followup")
OUTPUT = Path("runs/real_check_sidebar/followup")
REVISION = "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
ANNOTATIONS = Path("data/omnidocbench/OmniDocBench.json")


def prepare():
    from huggingface_hub import HfApi,hf_hub_download
    candidates = read_json(Path("runs/real_check_sidebar/selection.json"))["qualifying_pages"]
    if len(candidates) != 3:
        raise ValueError("This follow-up authorizes precisely the three saved qualifying candidates")
    old = read_json(Path("data/omnidocbench/selected_pages.json"))
    if {row["image_path"] for row in candidates}&{row["image_path"] for row in old["pages"]}:
        raise ValueError("New candidates overlap earlier evaluated pages")
    info = HfApi().dataset_info("opendatalab/OmniDocBench",revision=REVISION,files_metadata=True)
    sizes = {item.rfilename:item.size for item in info.siblings}
    required = sum(sizes[row["image_path"]] for row in candidates)
    if required+ANNOTATIONS.stat().st_size+10_000_000 >= 200_000_000:
        raise RuntimeError("Selected subset plus shared annotations and cache reserve exceeds 200 MB")
    ROOT.mkdir(parents=True,exist_ok=True)
    write_json(ROOT/"selected_pages.json",{"revision":REVISION,"pages":candidates,
               "protocol":"User explicitly authorized evaluation of the three candidates despite the earlier minimum-pool stop. Exploratory, not the preregistered final evaluation."})
    for row in candidates:
        path=ROOT/row["image_path"]
        if not path.exists() or path.stat().st_size != sizes[row["image_path"]]:
            if directory_bytes(ROOT)+sizes[row["image_path"]]+ANNOTATIONS.stat().st_size >= 200_000_000:
                raise RuntimeError("200 MB budget would be exceeded")
            hf_hub_download("opendatalab/OmniDocBench",row["image_path"],repo_type="dataset",revision=REVISION,local_dir=ROOT)
        print(row["page_id"],row["attributes"]["data_source"],row["image_path"],path.stat().st_size,flush=True)
    print(f"New subset plus shared annotations: {(directory_bytes(ROOT)+ANNOTATIONS.stat().st_size)/1e6:.2f} MB",flush=True)


def methods(tau):
    values={name:tau[name] for name in ("Native","XY-cut","Raw GAT","Gated GAT")}
    for family,prefix in (("Sequential PPO v2","PPO v2"),("Page-level PPO","Page PPO")):
        for seed in SEEDS:
            values[f"{family} / {seed}"]=tau[f"{prefix} {seed}"]
        values[family]=float(np.mean([tau[f"{prefix} {seed}"] for seed in SEEDS]))
    return values


def make_report():
    new_path=OUTPUT/"new_inference"/"report.json"
    seen_path=Path("runs/real_check/stratified/uncapped_inference/report.json")
    new,seen=read_json(new_path),read_json(seen_path)
    for field in ("checkpoint_sha256","native_prior","thresholds","page_policies","sequential_policies","annotation_sha256","max_lines"):
        if new["provenance"][field] != seen["provenance"][field]:
            raise ValueError(f"Frozen inference differs across new and seen sources: {field}")
    selection=read_json(ROOT/"selected_pages.json")
    types={row["attributes"]["data_source"] for row in selection["pages"]}
    annotations=read_json(ANNOTATIONS)
    old_manifest=read_json(Path("data/omnidocbench/selected_pages.json"))
    annotation_by_image={row["image_path"]:annotations[row["annotation_index"]] for row in [*selection["pages"],*old_manifest["pages"]]}
    rows=[]
    for cohort,source in (("unseen_three",new),("previously_seen_same_types",seen)):
        for page in source["pages"]:
            if cohort.startswith("previously") and page["document_type"] not in types:
                continue
            rows.append({"cohort":cohort,"page_id":page["page_id"],"document_type":page["document_type"],
                         "layout":page["layout"],"total_ocr_lines":page["total_ocr_lines"],
                         "scored_lines":page["alignment"]["scored_lines"],"matched_blocks":page["alignment"]["matched_blocks"],
                         "sidebar_proxy":sidebar_witness(annotation_by_image[page["image_path"]]) is not None,
                         "status":"scored","scores":methods(page["tau"]),"orders":page["orders"]})
        for page in source.get("excluded",[]):
            doc=page["attributes"]["data_source"]
            if cohort.startswith("previously") and doc not in types:
                continue
            rows.append({"cohort":cohort,"page_id":page["page_id"],"document_type":doc,
                         "layout":page["attributes"]["layout"],"status":"excluded","reason":page["reason"],"scores":{}})
    groups={}
    for cohort in ("unseen_three","previously_seen_same_types","combined_exploratory"):
        selected=[row for row in rows if cohort == "combined_exploratory" or row["cohort"] == cohort]
        for doc in ("all",*sorted(types)):
            subset=[row for row in selected if doc == "all" or row["document_type"] == doc]
            scored=[row for row in subset if row["status"] == "scored"]
            cell={"selected_pages":len(subset),"scored_pages":len(scored),"methods":{}}
            if scored:
                native=[row["scores"]["Native"] for row in scored]
                gate=[row["scores"]["Gated GAT"] for row in scored]
                for method in scored[0]["scores"]:
                    values=np.asarray([row["scores"][method] for row in scored])
                    correct=np.asarray(native)==1
                    stats={"mean_tau":float(values.mean()),"native_correct_tau":float(values[correct].mean()) if correct.any() else None,
                           "versus_native":compare_pages(values,native),"versus_gated":compare_pages(values,gate)}
                    if method in ("Sequential PPO v2","Page-level PPO"):
                        prefix="PPO v2" if method.startswith("Sequential") else "Page PPO"
                        switched=[any(row["orders"][f"{prefix} {seed}"] != row["orders"]["Native"] for seed in SEEDS) for row in scored]
                        stats["switch_semantics"]="Descriptive any-seed union; seed-mean tau has no single output order."
                    else:
                        key=method
                        if " / " in method:
                            family,seed=method.split(" / ")
                            key=f"{'PPO v2' if family.startswith('Sequential') else 'Page PPO'} {seed}"
                        switched=[row["orders"][key] != row["orders"]["Native"] for row in scored]
                    stats["decisions"]=switched_summary(native,values,switched)
                    cell["methods"][method]=stats
            groups[f"{cohort}/{doc}"]=cell
    protocol=("User-authorized exploratory follow-up after the preregistered minimum-pool stop, NOT the original preregistered final evaluation. All three qualifying unseen pages were included, without replacement. Seen pages are all previously evaluated magazine/newspaper pages, not necessarily sidebar-style; they were chosen by document type only, never tau. New and seen groups are separate; combined results are descriptive and not an independent new test. Frozen all-lines models, prior 4.0, original gate and normalizers, same three seeds, no training/tuning. Seed taus are averaged within page, not treated as independent pages. Paired bootstrap and two-sided sign flips use 10,000 samples, seed 314159; tiny N=3 uncertainty cannot establish generalization. Scores condition on reliable text alignment, not official OmniDocBench scores. Earlier 150-page results had no method beating native.")
    cohort_counts={cohort:{"pages":sum(row["cohort"] == cohort for row in rows),
                           "proxy_matching_pages":sum(row["cohort"] == cohort and row.get("sidebar_proxy",False) for row in rows)}
                   for cohort in ("unseen_three","previously_seen_same_types")}
    report={"protocol":protocol,"cohort_counts":cohort_counts,"groups":groups,"pages":[{key:value for key,value in row.items() if key != "orders"} for row in rows],
            "sources":{path.as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in (new_path,seen_path,ROOT/"selected_pages.json")}}
    write_json(OUTPUT/"report.json",report)
    flat=[{key:value for key,value in row.items() if key not in ("scores","orders")} | row["scores"] for row in rows]
    columns=list(dict.fromkeys(key for row in flat for key in row))
    with (OUTPUT/"per_page.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(flat)
    main_methods=("Native","XY-cut","Raw GAT","Gated GAT","Sequential PPO v2","Page-level PPO")
    text=["# Exploratory three-unseen-page follow-up","",protocol,"",
          f"New: {cohort_counts['unseen_three']['pages']} pages (2 magazine, 1 newspaper), all satisfying the frozen sidebar proxy. Seen: {cohort_counts['previously_seen_same_types']['pages']} pages (12 magazine, 22 newspaper); {cohort_counts['previously_seen_same_types']['proxy_matching_pages']} satisfy that proxy. Same document type does not mean same sidebar geometry.","",
          "## All-method comparison","","| Cohort / type | N scored / selected | Method | Mean tau | Delta vs native [95% CI] | p vs native | Better/worse/same | Delta vs gate [95% CI] |",
          "|---|---:|---|---:|---|---:|---|---|"]
    for name,cell in groups.items():
        for method in main_methods:
            if method not in cell["methods"]: continue
            stats=cell["methods"][method];native=stats["versus_native"];gate=stats["versus_gated"]
            def ci(value):return f"{value['mean_difference']:+.6f} [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}]"
            text.append(f"| {name} | {cell['scored_pages']}/{cell['selected_pages']} | {method} | {stats['mean_tau']:.6f} | {ci(native)} | {native['sign_flip_p']:.6f} | {native['better']}/{native['worse']}/{native['same']} | {ci(gate)} |")
    text += ["","## Individual unseen pages","","| Page | Type | Lines / scored | Native | XY-cut | Raw GAT | Gated GAT | Sequential PPO | Page PPO |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["cohort"] != "unseen_three":continue
        if row["status"] == "excluded":
            text += [f"| {row['page_id']} | {row['document_type']} | excluded: {row['reason']} | n/a | n/a | n/a | n/a | n/a | n/a |"]
        else:
            text.append(f"| {row['page_id']} | {row['document_type']} | {row['total_ocr_lines']}/{row['scored_lines']} | " + " | ".join(f"{row['scores'][method]:.6f}" for method in main_methods)+" |")
    text += ["","All per-seed scores, switch/fix/break counts, native-correct conditional tau and paired comparisons are in report.json; every page is in per_page.csv. Native-correct is ground-truth-conditioned. CIs are descriptive/unadjusted. Original preregistration and stop reports are unchanged.",""]
    (OUTPUT/"report.md").write_text("\n".join(text),encoding="utf-8")
    print(json.dumps({name:{method:cell['methods'][method]['mean_tau'] for method in main_methods if method in cell['methods']} for name,cell in groups.items() if name.endswith('/all')},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    if args.prepare:prepare()
    else:make_report()
