from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch
from torch.distributions import Categorical
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.baseline import heuristic_order, order_accuracy
from rlocr.env import OCRCorrectionEnv
from rlocr.ppo import GATActorCritic, spatial_edges

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic/aligned"); parser.add_argument("--splits", default="data/synthetic/splits.json"); parser.add_argument("--split", default="test"); parser.add_argument("--checkpoint", required=True); parser.add_argument("--max-pages", type=int, default=50); parser.add_argument("--max-lines", type=int, default=128); parser.add_argument("--output", default="runs/evaluation.json"); args = parser.parse_args()
model = GATActorCritic(); model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)); model.eval(); rows = []
split_data = json.loads(Path(args.splits).read_text(encoding="utf-8")); selected = set(split_data.get(args.split, [])); all_paths = sorted(Path(args.aligned).glob("*.json")); paths = [path for path in all_paths if path.stem in selected][:args.max_pages] if selected else all_paths[:args.max_pages]
for path in paths:
    page = json.loads(path.read_text(encoding="utf-8")); env = OCRCorrectionEnv(page, max_lines=args.max_lines); observation, _ = env.reset(); edges = spatial_edges(observation); targets = []; predicted = []
    with torch.no_grad():
        while env.index < len(env.lines):
            logits, _ = model(torch.from_numpy(observation).float().unsqueeze(0), edges); node = env.index; action = int(Categorical(logits=logits[0, node]).probs.argmax()); targets.append(env.page["action_targets"].get(env.lines[node]["id"], 0)); predicted.append(action); observation, _, _, _, _ = env.step(action)
    rows.append({"page_id": path.stem, "policy_action_accuracy": sum(a == b for a, b in zip(predicted, targets)) / max(1, len(targets)), "baseline_order_accuracy": order_accuracy(heuristic_order(page["ocr_lines"]), page["gt_order"])})
result = {"pages": len(rows), "mean_policy_action_accuracy": sum(row["policy_action_accuracy"] for row in rows) / max(1, len(rows)), "mean_baseline_order_accuracy": sum(row["baseline_order_accuracy"] for row in rows) / max(1, len(rows)), "rows": rows}; Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
