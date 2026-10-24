"""Apply committed sidebar preregistration; stop if the fixed pool is too small."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rlocr.common import read_json, write_json
from rlocr.omnidocbench import image_filename
from rlocr.sidebar_selection import select_sidebar_pages

PREREGISTRATION_COMMIT = "f9949a4"


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("data/omnidocbench"))
    parser.add_argument("--output",type=Path,default=Path("runs/real_check_sidebar"))
    args = parser.parse_args()
    prereg_path = Path("runs/real_check_sidebar/preregistration.md")
    committed = subprocess.run(["git","show",f"{PREREGISTRATION_COMMIT}:runs/real_check_sidebar/preregistration.md"],
                               check=True,capture_output=True).stdout
    if committed.replace(b"\r\n",b"\n") != prereg_path.read_bytes().replace(b"\r\n",b"\n"):
        raise RuntimeError("Preregistration differs from the committed rule; stop.")
    annotation_path,previous_path = args.root/"OmniDocBench.json",args.root/"selected_pages.json"
    pages,previous = read_json(annotation_path),read_json(previous_path)
    previous_images = {image_filename(row["image_path"]) for row in previous["pages"]}
    if len(previous_images) != 150:
        raise ValueError("Expected all 150 earlier images in the exclusion manifest")
    result = select_sidebar_pages(pages,previous_images)
    result.update({"preregistration_commit":PREREGISTRATION_COMMIT,"model_inference_run":False,
                   "annotation_sha256":hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                   "previous_manifest_sha256":hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                   "all_document_counts":dict(Counter(page["page_info"]["page_attribute"]["data_source"] for page in pages)),
                   "all_layout_counts":dict(Counter(page["page_info"]["page_attribute"]["layout"] for page in pages))})
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"selection.json",result)
    print(json.dumps({key:value for key,value in result.items() if key not in ("qualifying_pages","selected_pages")},indent=2),flush=True)
    print("Selected page list:",json.dumps([row["image_path"] for row in result["selected_pages"]],ensure_ascii=False),flush=True)
    if result["status"] != "ready":
        result["reason"] = (f"Only {result['qualifying_count']} unseen pages satisfy the frozen rule, below the minimum of 80. "
                            "Stopped before downloads, OCR or model inference. User decision is required before any rule change.")
        result["earlier_real_result"] = "On the earlier 150-page set no method beat native. This new set has NOT been evaluated."
        write_json(args.output/"report.json",result)
        text = ["# Final real-data sidebar check: selection stopped","",result["reason"],"",
                f"Preregistration committed and pushed before selection: `{PREREGISTRATION_COMMIT}`. Selection seed: {result['seed']}. Target: 100; minimum eligible pool: 80.","",
                result["earlier_real_result"],"","No rule was relaxed, no page was substituted, and no new-page model run was performed. This is a selection feasibility report, not a negative or positive model result.","",
                "## Eligibility funnel","", "| Stage | Pages |", "|---|---:|"]
        for stage,count in result["stage_counts"].items():
            text.append(f"| {stage} | {count} |")
        for title,key in (("All document types","all_document_counts"),("All layout types","all_layout_counts"),
                          ("Qualifying document types","qualifying_document_counts"),("Qualifying layout types","qualifying_layout_counts")):
            text += ["",f"## {title}","","| Attribute | Pages |","|---|---:|"]
            text += [f"| {name} | {count} |" for name,count in result[key].items()]
        text += ["","Qualifying annotation indices and geometric witnesses are saved in selection.json. No final page sample was selected. Reproduce counts with `python scripts/select_sidebar_real_check.py`. The next action requires explicit user direction; do not relax eligibility or evaluate this undersized pool automatically.",""]
        (args.output/"report.md").write_text("\n".join(text),encoding="utf-8")
        print(result["reason"],flush=True)


if __name__ == "__main__":
    main()
