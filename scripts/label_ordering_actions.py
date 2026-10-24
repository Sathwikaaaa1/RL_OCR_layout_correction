"""Label binary ordering actions without changing IDs, metrics, or weights."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.common import write_json
from rlocr.order_actions import ACTION_NAMES


def label_existing_reports():
    replacements = {"follow native": "RETAIN", "follow GAT": "REORDER",
                    "always action 0": "always RETAIN", "always action 1": "always REORDER",
                    "action 0 returns native": "RETAIN (action 0) returns native",
                    "action 1 returns the full fixed gated decoder order": "REORDER (action 1) returns the full fixed gated decoder order"}
    def labels(value):
        if isinstance(value, str):
            for old, new in replacements.items():
                value = value.replace(old, new)
        elif isinstance(value, dict):
            return {key: labels(item) for key, item in value.items()}
        elif isinstance(value, list):
            return [labels(item) for item in value]
        return value
    for directory in ("ppo_gate", "ppo_gate_v2", "ppo_page_gate"):
        path = Path("runs") / directory / "report.json"
        if not path.exists():
            continue
        report = labels(json.loads(path.read_text(encoding="utf-8")))
        report["action_names"] = ACTION_NAMES
        write_json(path, report)
        markdown = path.with_suffix(".md")
        text = labels(markdown.read_text(encoding="utf-8"))
        note = "Action labels: **RETAIN = 0**, **REORDER = 1**. Numeric IDs and checkpoint tensors are unchanged."
        if note not in text:
            text = text.replace("\n\n", "\n\n" + note + "\n\n", 1)
        markdown.write_text(text, encoding="utf-8")
    print("Labeled RETAIN/REORDER in existing binary-action reports; metrics and checkpoint files preserved")


if __name__ == "__main__":
    label_existing_reports()
