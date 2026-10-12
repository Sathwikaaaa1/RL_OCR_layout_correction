from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.env import OCRCorrectionEnv
from rlocr.model import PairwiseOrderModel
from rlocr.ocr.alignment import kendall_tau
from rlocr.ppo import spatial_edges
from rlocr.evaluation import decode_pairwise_order

parser = argparse.ArgumentParser(); parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned"); parser.add_argument("--pages", type=int, default=20); parser.add_argument("--epochs", type=int, default=20); args = parser.parse_args(); paths = sorted(Path(args.aligned).glob("*.json"))[:args.pages]; model = PairwiseOrderModel(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); records = []
for path in paths:
    page = json.loads(path.read_text(encoding="utf-8")); valid = set(page.get("valid_line_ids", [])); lines = [line for line in page["ocr_lines"] if line["id"] in valid]; observation = torch.from_numpy(__import__("numpy").array([[*([v / 2000 for v in [line["bbox"][0], line["bbox"][1], line["bbox"][2], line["bbox"][3]]]), min(len(line["text"]) / 200, 1), line.get("confidence", 0) / 100, line.get("block_num", 0) / 20, line.get("native_order", 0) / max(1, len(lines))] for line in lines], dtype="float32")); edges = spatial_edges(observation.numpy(), k=4); rank = {value: i for i, value in enumerate(page["gt_order"])}; target = torch.zeros((len(lines), len(lines))); mask = torch.ones_like(target, dtype=torch.bool)
    for i, left in enumerate(lines):
        for j, right in enumerate(lines):
            if i == j: mask[i, j] = False
            else: target[i, j] = float(rank[left["id"]] < rank[right["id"]])
    records.append((page, lines, observation, edges, target, mask))
for _ in range(args.epochs):
    for _, _, observation, edges, target, mask in records:
        loss = F.binary_cross_entropy_with_logits(model(observation, edges)[mask], target[mask]); optimizer.zero_grad(); loss.backward(); optimizer.step()
taus = []
with torch.no_grad():
    for page, lines, observation, edges, _, _ in records:
        scores = model(observation, edges); order = [lines[i]["id"] for i in scores.fill_diagonal_(0).mean(1).argsort(descending=True).tolist()]; taus.append(kendall_tau(order, page["gt_order"]))
print(f"pages={len(taus)} epochs={args.epochs} overfit_mean_tau={sum(taus)/max(1,len(taus)):.4f} min_tau={min(taus) if taus else 0:.4f}")
