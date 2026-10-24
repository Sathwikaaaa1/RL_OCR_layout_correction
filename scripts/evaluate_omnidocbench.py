"""Inference-only real-page reading-order check; never trains or tunes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json
from rlocr.env import OCRCorrectionEnv
from rlocr.evaluation import xy_cut_order
from rlocr.gat_inference import infer_gat
from rlocr.gated_decoder import gated_order, native_order, paired_significance
from rlocr.model import PairwiseOrderModel
from rlocr.ocr.alignment import kendall_tau
from rlocr.ocr.runner import TesseractRunner
from rlocr.omnidocbench import align_text, directory_bytes
from rlocr.order_features import pair_probability_scores
from rlocr.ppo import spatial_edges
from rlocr.ppo_gate import OverrideActorCritic
from rlocr.ppo_page_gate import page_gate_features
from rlocr.ppo_gate_v2 import OverrideNativeV2Env, lookahead_features


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ocr_one(job):
    root, row, command = job
    os.environ["OMP_THREAD_LIMIT"] = "1"
    target = root / "ocr" / f"{row['page_id']}.json"
    if target.exists():
        return row["page_id"], 0.0
    started = time.perf_counter()
    lines = TesseractRunner(language="eng", command=command).run(root / row["image_path"])
    temporary = target.with_suffix(".tmp")
    write_json(temporary, {"image_path": row["image_path"], "lines": lines})
    temporary.replace(target)
    return row["page_id"], time.perf_counter()-started


def summarize(rows, method):
    native = np.asarray([row["tau"]["Native"] for row in rows])
    values = np.asarray([row["tau"][method] for row in rows])
    correct = np.isclose(native, 1.0, rtol=0, atol=1e-9)
    return {"mean_tau": float(values.mean()),
            "mean_block_tau": float(np.mean([row["block_tau"][method] for row in rows])),
            "native_correct_tau": float(values[correct].mean()) if correct.any() else None,
            "native_wrong_tau": float(values[~correct].mean()) if (~correct).any() else None,
            "pages_changed": sum(row["changed"][method] for row in rows),
            "per_layout_tau": {layout: float(np.mean([row["tau"][method] for row in rows if row["layout"] == layout])) for layout in sorted({row["layout"] for row in rows})},
            "paired_vs_native": paired_significance(values, native, 10_000, 314159)}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/omnidocbench"))
    parser.add_argument("--output", type=Path, default=Path("runs/omnidocbench"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/native_aware_gat_full.pt"))
    parser.add_argument("--thresholds", type=Path, default=Path("runs/ppo_gate/thresholds.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ocr-only", action="store_true", help="Cache OCR for currently downloaded images, then stop (no partial-set scores).")
    parser.add_argument("--tesseract-cmd", default=os.environ.get("TESSERACT_CMD") or shutil.which("tesseract"))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.tesseract_cmd or not Path(args.tesseract_cmd).is_file():
        parser.error("Install Tesseract with English data and set TESSERACT_CMD (or --tesseract-cmd) to its executable.")
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    manifest = read(args.root / "selected_pages.json")
    annotations = read(args.root / "OmniDocBench.json")
    missing = [row["image_path"] for row in manifest["pages"] if not (args.root / row["image_path"]).is_file()]
    if missing and not args.ocr_only:
        raise RuntimeError(f"{len(missing)} images missing; resume prepare_omnidocbench.py first.")
    print(f"CPU estimate: approximately 3-12 minutes for {len(manifest['pages'])} pages with {args.workers} OCR workers; cached OCR is skipped. No training or tuning.", flush=True)
    jobs = [(args.root,row,args.tesseract_cmd) for row in manifest["pages"] if (args.root/row["image_path"]).is_file()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (page_id,seconds) in enumerate(pool.map(ocr_one,jobs)):
            print(f"OCR {index+1}/{len(jobs)} {page_id}: {seconds:.2f}s", flush=True)
    if args.ocr_only:
        print(f"OCR cached for {len(jobs)} available pages; {len(missing)} images still missing. No partial-set evaluation.",flush=True)
        return
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PairwiseOrderModel(hidden_dim=checkpoint["args"]["hidden_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    limit = checkpoint["args"]["max_lines"]
    thresholds = read(args.thresholds)
    pair, page_threshold = thresholds["pair_threshold"], thresholds["page_threshold"]
    policies = {}
    sequential_policies = {}
    for seed in (1337,4242,90210):
        path = Path("runs/ppo_page_gate") / f"seed_{seed}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        policy = OverrideActorCritic(6)
        if saved["kind"] != "always_gated":
            policy.load_state_dict(saved["model"])
        policy.eval()
        policies[seed] = (policy,saved,path)
        path = Path("runs/ppo_gate_v2") / f"fine_tuned_seed_{seed}.pt"
        saved = torch.load(path,map_location="cpu",weights_only=False)
        sequential = OverrideActorCritic(17)
        sequential.load_state_dict(saved["model"])
        sequential.eval()
        sequential_policies[seed] = (sequential,path)
    rows, excluded = [], []
    for selected in manifest["pages"]:
        source = annotations[selected["annotation_index"]]
        all_lines = read(args.root / "ocr" / f"{selected['page_id']}.json")["lines"]
        # Keep the deployment cap and ALL OCR lines, not GT-filtered input.
        lines = all_lines[:limit]
        with Image.open(args.root / selected["image_path"]) as image:
            size = image.size
        alignment = align_text(lines, source, size)
        write_json(args.root / "aligned" / f"{selected['page_id']}.json", alignment)
        if alignment["scored_lines"] < 3 or alignment["matched_blocks"] < 3:
            excluded.append({**selected, "alignment": alignment, "reason": "fewer than 3 unambiguous aligned lines or text blocks within deployment cap"})
            continue
        env = OCRCorrectionEnv({"ocr_lines":lines}, max_lines=limit)
        features, _ = env.reset()
        features = features[:len(lines)]
        result = infer_gat(model, torch.from_numpy(features).float(), spatial_edges(features,k=4),lines,native_prior=4.0)
        native = native_order(lines)
        gated, _, _ = gated_order(lines,result.probabilities,pair,page_threshold)
        xy = xy_cut_order(lines)
        orders = {"Native":native, "XY-cut":xy, "Raw GAT":result.order,"Gated GAT":gated}
        index = {line["id"]: i for i,line in enumerate(lines)}
        rank = np.empty(len(lines),dtype=np.int64)
        rank[[index[key] for key in native]] = np.arange(len(lines))
        probabilities = np.full((len(lines),len(lines)),.5,dtype=np.float32)
        for (i,j),value in result.probabilities.items():
            probabilities[i,j],probabilities[j,i] = value,1-value
        page = {"native_rank":rank,"probabilities":probabilities,"gated":[index[key] for key in gated],
                "xy_tau":kendall_tau(native,xy),
                "widths":np.asarray([(line["bbox"][2]-line["bbox"][0])/size[0] for line in lines]),
                "x_centers":np.asarray([(line["bbox"][2]+line["bbox"][0])/(2*size[0]) for line in lines])}
        observation = page_gate_features(page,pair)
        actions = {}
        for seed,(policy,saved,_) in policies.items():
            normalization = saved["normalization"]
            normalized = ((observation-np.asarray(normalization["mean"]))/np.asarray(normalization["std"])).astype(np.float32)
            with torch.no_grad():
                action = 1 if saved["kind"] == "always_gated" else int(policy(torch.from_numpy(normalized)[None])[0].argmax(-1).item())
            name = f"Page PPO {seed}"
            actions[name] = action
            orders[name] = gated if action else native
        page.update({"native":[index[key] for key in native],"raw":[index[key] for key in result.order],
                     "scores":pair_probability_scores(lines,result.probabilities,native_prior=4.0),
                     # Dummy ranks satisfy the environment's reward API. Actual
                     # annotation ranks never enter policy rollout or features.
                     "gt_rank":rank.copy(),"native_tau":1.0})
        page["lookahead"] = lookahead_features(page,pair)
        for seed,(policy,_) in sequential_policies.items():
            rollout = OverrideNativeV2Env(page)
            state,info = rollout.reset(seed=1337)
            done = info["automatic_terminal"]
            while not done:
                with torch.no_grad():
                    action = int(policy(torch.from_numpy(state)[None])[0].argmax(-1).item())
                state,_,done,_,_ = rollout.step(action)
            name = f"PPO v2 {seed}"
            orders[name] = [lines[i]["id"] for i in rollout.order]
            actions[name] = {"reorders":rollout.overrides,"decisions":rollout.disagreements}
        truth = alignment["gt_order"]
        valid = set(truth)
        # Deduplicate aligned GT block IDs in first-occurrence order. This
        # complementary metric does not assume within-paragraph line ordering.
        block_orders = {name:list(dict.fromkeys(alignment["mapping"][key] for key in order if key in valid))
                        for name,order in orders.items()}
        block_truth = list(dict.fromkeys(alignment["mapping"][key] for key in truth))
        row = {"page_id":selected["page_id"],"image_path":selected["image_path"],
               "layout":selected["attributes"]["layout"],"document_type":selected["attributes"]["data_source"],
               "alignment":alignment,"total_ocr_lines":len(all_lines),"input_lines":len(lines),
               "truncated":len(all_lines)>limit,"tau":{name:kendall_tau(order,truth) for name,order in orders.items()},
               "block_tau":{name:kendall_tau(order,block_truth) for name,order in block_orders.items()},
               "changed":{name:[key for key in order if key in valid] != [key for key in native if key in valid] for name,order in orders.items()},
               "orders":orders,"actions":actions}
        rows.append(row)
        print(f"Inference {len(rows)}: {selected['page_id']} native={row['tau']['Native']:.4f}, gated={row['tau']['Gated GAT']:.4f}", flush=True)
    if not rows:
        raise RuntimeError("No pages have enough reliable alignment; do not report perfect tau on empty orders.")
    methods = list(rows[0]["tau"])
    summary = {name:summarize(rows,name) for name in methods}
    native_values = [row["tau"]["Native"] for row in rows]
    mean_seed = np.mean([[row["tau"][f"Page PPO {seed}"] for row in rows] for seed in policies],axis=0)
    protocol = ("Inference only; fixed full synthetic GAT checkpoint, native prior 4.0; gate thresholds and PPO checkpoints/normalizers selected on synthetic validation only. All 3 saved PPO seeds evaluated, no seed selected on real pages. Inputs use all OCR lines up to the saved 96-line deployment cap, without GT filtering. Metrics use only confidently aligned, unambiguous ordered text lines (>=50% line-area overlap; exclude >=15% overlap with multiple blocks and tied block ranks). At least 3 mapped text blocks and lines required per scored page. Within-block line order is derived top-to-bottom, not independently annotated. Complementary block tau deduplicates GT block IDs in first-occurrence order. This is NOT the official OmniDocBench score or an OCR recognition/merge/split metric; GT filtering of the scoring subset, truncation and alignment coverage limit interpretation. No explicit sidebar labels were found. Images and annotations are not committed.")
    provenance = {"dataset_revision":manifest["revision"],"manifest_sha256":hashlib.sha256((args.root/"selected_pages.json").read_bytes()).hexdigest(),
                  "checkpoint":str(args.checkpoint),"checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  "annotation_sha256":hashlib.sha256((args.root/"OmniDocBench.json").read_bytes()).hexdigest(),
                  "native_prior":4.0,"thresholds":thresholds,"max_lines":limit,
                  "page_policies":{str(seed):{"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"kind":saved["kind"]} for seed,(_,saved,path) in policies.items()}}
    provenance["sequential_policies"] = {str(seed):{"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for seed,(_,path) in sequential_policies.items()}
    report = {"protocol":protocol,"provenance":provenance,"selected_pages":len(manifest["pages"]),"scored_pages":len(rows),
              "native_correct_pages":sum(abs(row["tau"]["Native"]-1) <= 1e-9 for row in rows),
              "native_wrong_pages":sum(abs(row["tau"]["Native"]-1) > 1e-9 for row in rows),
              "excluded":excluded,"summary":summary,"pages":rows,
              "page_ppo_seed_mean_tau":float(mean_seed.mean()),
              "page_ppo_seed_std_tau":float(np.std([summary[f"Page PPO {seed}"]["mean_tau"] for seed in policies])),
              "page_ppo_seed_mean_vs_native":paired_significance(mean_seed,native_values,10_000,314159),
              "data_bytes":directory_bytes(args.root),"seconds":time.perf_counter()-started,
              "coverage":{"input_lines":sum(row["input_lines"] for row in rows),"scored_lines":sum(row["alignment"]["scored_lines"] for row in rows),
                          "matched_blocks":sum(row["alignment"]["matched_blocks"] for row in rows),"annotated_blocks":sum(row["alignment"]["annotated_text_blocks"] for row in rows),
                          "truncated_pages":sum(row["truncated"] for row in rows)}}
    report["seed_means"] = {}
    for family in ("Page PPO","PPO v2"):
        seed_values = [[row["tau"][f"{family} {seed}"] for row in rows] for seed in policies]
        means = np.mean(seed_values,axis=0)
        report["seed_means"][family] = {"mean_tau":float(means.mean()),"std_tau":float(np.std(np.mean(seed_values,axis=1))),
                                      "paired_vs_native":paired_significance(means,native_values,10_000,314159),
                                      "paired_vs_gated":paired_significance(means,[row["tau"]["Gated GAT"] for row in rows],10_000,314159)}
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"report.json",report)
    text = ["# OmniDocBench frozen-model check","",protocol,"",f"Selected: {report['selected_pages']}; scored: {len(rows)}; excluded: {len(excluded)}. Data: {report['data_bytes']/1e6:.2f} MB.","",
            "| Method | Line tau | Block tau | Native-correct | Native-wrong | Changed | Better/worse/same | Delta vs native [95% CI] | p |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for name,value in summary.items():
        sig=value["paired_vs_native"]
        delta=sig["difference"]
        correct = "n/a" if value["native_correct_tau"] is None else f"{value['native_correct_tau']:.6f}"
        wrong = "n/a" if value["native_wrong_tau"] is None else f"{value['native_wrong_tau']:.6f}"
        text.append(f"| {name} | {value['mean_tau']:.6f} | {value['mean_block_tau']:.6f} | {correct} | {wrong} | {value['pages_changed']} | {sig['better_pages']}/{sig['worse_pages']}/{sig['same_pages']} | {delta['mean']:+.6f} [{delta['low']:+.6f}, {delta['high']:+.6f}] | {sig['permutation_p_value']:.6f} |")
    text += ["","## Alignment coverage","",json.dumps(report["coverage"]),"","## Per-layout tau","", "| Method | Layout | Tau |", "|---|---|---:|"]
    for name,value in summary.items():
        for layout,tau in value["per_layout_tau"].items():
            text.append(f"| {name} | {layout} | {tau:.6f} |")
    text += ["","## Seed-mean paired comparisons","",
             "| Family | Mean tau +/- seed std | Delta vs native [95% CI] | Delta vs gated [95% CI] |", "|---|---:|---|---|"]
    for family,value in report["seed_means"].items():
        native_delta=value["paired_vs_native"]["difference"]
        gate_delta=value["paired_vs_gated"]["difference"]
        text.append(f"| {family} | {value['mean_tau']:.6f} +/- {value['std_tau']:.6f} | {native_delta['mean']:+.6f} [{native_delta['low']:+.6f}, {native_delta['high']:+.6f}] | {gate_delta['mean']:+.6f} [{gate_delta['low']:+.6f}, {gate_delta['high']:+.6f}] |")
    text += ["",
             "Paired page bootstrap and sign-flip tests: 10,000 resamples, seed 314159. Intervals condition on these fitted models and this selected/aligned subset. Full per-page coverage, exclusions and checkpoint hashes are in report.json.","",
             "Dataset/schema: https://huggingface.co/datasets/opendatalab/OmniDocBench",""]
    (args.output/"report.md").write_text("\n".join(text),encoding="utf-8")
    print(json.dumps({name:value["mean_tau"] for name,value in summary.items()},indent=2),flush=True)
    print(f"Elapsed {report['seconds']:.1f}s; data {report['data_bytes']/1e6:.2f} MB",flush=True)


if __name__ == "__main__":
    main()
