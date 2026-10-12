from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json
from rlocr.ocr.alignment import align_page

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned"); parser.add_argument("--gt", default="data/synthetic_hard_large/gt"); args = parser.parse_args()
for path in sorted(Path(args.aligned).glob("*.json")):
    aligned = json.loads(path.read_text(encoding="utf-8")); gt = json.loads((Path(args.gt) / path.name).read_text(encoding="utf-8")); updated = align_page(aligned["ocr_lines"], gt["blocks"], gt["reading_order"]); write_json(path, updated)
print(f"rebuilt={len(list(Path(args.aligned).glob('*.json')))}")
