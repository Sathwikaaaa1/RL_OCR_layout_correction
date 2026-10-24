"""Download only annotations and a reproducible, byte-bounded image allowlist."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from huggingface_hub import HfApi, hf_hub_download
from rlocr.common import write_json
from rlocr.omnidocbench import directory_bytes, select_pages

REPO = "opendatalab/OmniDocBench"
REVISION = "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
HARD_LIMIT = 300_000_000


def main():
    # Windows redirected output may default to cp1252; some image names are CJK.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/omnidocbench"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--budget-mb", type=float, default=200)
    parser.add_argument("--annotations-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 150:
        parser.error("Select between 1 and 150 pages, never the whole image folder.")
    if not 0 < args.budget_mb < 299:
        parser.error("Budget must be below 299 decimal MB (300 MB hard stop).")
    args.root.mkdir(parents=True, exist_ok=True)
    annotation = args.root / "OmniDocBench.json"
    info = None
    if not annotation.exists():
        info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
        size = next(item.size for item in info.siblings if item.rfilename == "OmniDocBench.json")
        if size is None or directory_bytes(args.root) + size + 100_000 >= HARD_LIMIT:
            raise RuntimeError("Annotation download could reach 300 MB; stopped before fetching.")
        hf_hub_download(REPO, "OmniDocBench.json", repo_type="dataset", revision=REVISION, local_dir=args.root)
    if directory_bytes(args.root) >= HARD_LIMIT:
        raise RuntimeError("300 MB hard limit reached; no further downloads.")
    pages = json.loads(annotation.read_text(encoding="utf-8"))
    print(f"Annotations: {len(pages)} pages, {annotation.stat().st_size/1e6:.2f} MB", flush=True)
    print("Page-info example:", json.dumps(pages[0]["page_info"], ensure_ascii=False), flush=True)
    if args.annotations_only:
        return
    manifest_path = args.root / "selected_pages.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["revision"], manifest["seed"], manifest["limit"], manifest["budget_mb"]) != (REVISION,args.seed,args.limit,args.budget_mb):
            raise ValueError("Existing selection has different settings; do not overwrite an evaluated subset.")
    else:
        info = info or HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
        sizes = {item.rfilename: item.size for item in info.siblings if item.size is not None}
        selected = select_pages(pages, sizes, args.seed, args.limit,
                                int(args.budget_mb*1e6)-1_000_000, annotation.stat().st_size)
        if not selected:
            raise RuntimeError("No eligible images fit the byte budget.")
        manifest = {"repo": REPO, "revision": REVISION, "seed": args.seed, "limit":args.limit,
                    "budget_mb":args.budget_mb, "pages": selected,
                    "selection": "English; double/three/1andmore-column or explicit sidebar; >=3 ordered nonignored text blocks; seeded shuffle; byte cap. No OCR or model scores used."}
        temporary = manifest_path.with_suffix(".tmp")
        write_json(temporary, manifest)
        temporary.replace(manifest_path)
    print("Selected layouts:", dict(Counter(row["attributes"]["layout"] for row in manifest["pages"])), flush=True)
    print("Selected documents:", dict(Counter(row["attributes"]["data_source"] for row in manifest["pages"])), flush=True)
    for index, row in enumerate(manifest["pages"]):
        target = args.root / row["image_path"]
        current = directory_bytes(args.root)
        remaining = 0 if target.exists() and target.stat().st_size == row["bytes"] else row["bytes"]
        if current + remaining + 100_000 >= HARD_LIMIT:
            raise RuntimeError("Next image would reach 300 MB; download stopped.")
        if remaining:
            hf_hub_download(REPO, row["image_path"], repo_type="dataset", revision=REVISION, local_dir=args.root)
        if target.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Unexpected download size: {target}")
        print(f"{index+1}/{len(manifest['pages'])}: {row['image_path']} ({directory_bytes(args.root)/1e6:.2f} MB total)", flush=True)
    print(f"Total on disk: {directory_bytes(args.root)/1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
