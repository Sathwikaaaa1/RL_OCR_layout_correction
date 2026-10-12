from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.env import OCRCorrectionEnv
from rlocr.model import PairwiseOrderModel
from rlocr.ppo import spatial_edges
from rlocr.ocr.alignment import kendall_tau

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic_hard/aligned"); parser.add_argument("--splits", default="data/synthetic_hard/splits.json"); parser.add_argument("--train-pages", type=int, default=28); parser.add_argument("--test-pages", type=int, default=6); parser.add_argument("--epochs", type=int, default=4); parser.add_argument("--max-lines", type=int, default=64); parser.add_argument("--output", default="runs/pairwise_order.pt"); args = parser.parse_args()
split_data = json.loads(Path(args.splits).read_text(encoding="utf-8")); all_paths = sorted(Path(args.aligned).glob("*.json")); train_ids = set(split_data["train"]); test_ids = set(split_data["test"]); train_paths = [p for p in all_paths if p.stem in train_ids][:args.train_pages]; test_paths = [p for p in all_paths if p.stem in test_ids][:args.test_pages]; model = PairwiseOrderModel(); optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
def tensors(path: Path):
    page = json.loads(path.read_text(encoding="utf-8")); env = OCRCorrectionEnv(page, max_lines=args.max_lines); observation, _ = env.reset(); n = len(env.lines); features = torch.from_numpy(observation[:n]).float(); edges = spatial_edges(observation[:n], k=4); rank = {value: i for i, value in enumerate(page["gt_order"])}; target = torch.zeros((n, n)); mask = torch.ones((n, n), dtype=torch.bool); 
    for i, left in enumerate(env.lines):
        for j, right in enumerate(env.lines):
            if i == j: mask[i, j] = False
            else: target[i, j] = float(rank.get(left["id"], i) < rank.get(right["id"], j))
    return page, features, edges, target, mask
for _ in range(args.epochs):
    for path in train_paths:
        _, features, edges, target, mask = tensors(path); loss = F.binary_cross_entropy_with_logits(model(features, edges)[mask], target[mask]); optimizer.zero_grad(); loss.backward(); optimizer.step()
model.eval(); taus = []
with torch.no_grad():
    for path in test_paths:
        page, features, edges, _, _ = tensors(path); scores = torch.sigmoid(model(features, edges)); ranking = scores.fill_diagonal_(0).mean(1).argsort(descending=True).tolist(); predicted = [page["ocr_lines"][i]["id"] for i in ranking]; taus.append(kendall_tau(predicted, page["gt_order"]))
Path(args.output).parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), args.output); print(f"train_pages={len(train_paths)} test_pages={len(test_paths)} mean_pairwise_kendall={sum(taus)/max(1,len(taus)):.4f} output={args.output}")
