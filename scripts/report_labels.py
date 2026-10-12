from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.labels import class_distribution

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned"); parser.add_argument("--gt", default="data/synthetic_hard_large/gt"); args = parser.parse_args(); groups = defaultdict(Counter); dropped = defaultdict(int)
for path in sorted(Path(args.aligned).glob("*.json")):
    page = json.loads(path.read_text(encoding="utf-8")); gt = json.loads((Path(args.gt) / path.name).read_text(encoding="utf-8")); counts = class_distribution(page); groups[gt["template"]].update(counts); dropped[gt["template"]] += page.get("dropped_line_count", 0)
for template, counts in sorted(groups.items()):
    total = sum(counts.values()); print(f"{template}: total={total} dropped={dropped[template]} " + " ".join(f"{label}={counts[label]}({100*counts[label]/max(1,total):.2f}%)" for label in ("retain", "merge", "split")))
