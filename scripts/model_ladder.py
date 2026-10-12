from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlocr.ocr.alignment import kendall_tau


def pair_features(lines: list[dict]) -> torch.Tensor:
    rows = []
    scale = 2000.0
    for left in lines:
        for right in lines:
            lx0, ly0, lx1, ly1 = left["bbox"]
            rx0, ry0, rx1, ry1 = right["bbox"]
            lw, lh = lx1 - lx0, ly1 - ly0
            rw, rh = rx1 - rx0, ry1 - ry0
            lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
            rcx, rcy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
            rows.append([
                (lcx - rcx) / scale, (lcy - rcy) / scale,
                max(0, min(lx1, rx1) - max(lx0, rx0)) / max(1, min(lw, rw)),
                max(0, min(ly1, ry1) - max(ly0, ry0)) / max(1, min(lh, rh)),
                (lw - rw) / scale, (lh - rh) / scale,
                (left.get("native_order", 0) - right.get("native_order", 0)) / max(1, len(lines)),
                float(left.get("confidence", 0) - right.get("confidence", 0)) / 100,
            ])
    return torch.tensor(rows, dtype=torch.float32)


class PairMLP(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_records(directory: Path, count: int) -> list[tuple[dict, list[dict], torch.Tensor, torch.Tensor]]:
    records = []
    for path in sorted(directory.glob("*.json"))[:count]:
        page = json.loads(path.read_text(encoding="utf-8"))
        valid = set(page.get("valid_line_ids", []))
        lines = [line for line in page["ocr_lines"] if line["id"] in valid]
        rank = {value: i for i, value in enumerate(page["gt_order"])}
        target = torch.tensor([float(rank[left["id"]] < rank[right["id"]]) for left in lines for right in lines], dtype=torch.float32)
        records.append((page, lines, pair_features(lines), target))
    return records


def evaluate(model: nn.Module, records: list[tuple[dict, list[dict], torch.Tensor, torch.Tensor]]) -> tuple[float, float]:
    correct = total = 0
    taus = []
    with torch.no_grad():
        for page, lines, features, target in records:
            scores = model(features).reshape(len(lines), len(lines))
            mask = ~torch.eye(len(lines), dtype=torch.bool)
            correct += int(((scores > 0) == target.reshape(len(lines), len(lines)))[mask].sum())
            total += int(mask.sum())
            order = [lines[i]["id"] for i in scores.sum(1).argsort(descending=True).tolist()]
            taus.append(kendall_tau(order, page["gt_order"]))
    return correct / max(1, total), sum(taus) / max(1, len(taus))


parser = argparse.ArgumentParser()
parser.add_argument("--aligned", default="data/synthetic_hard_large/aligned")
parser.add_argument("--pages", type=int, default=20)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()
torch.manual_seed(7)
records = load_records(Path(args.aligned), args.pages)
model = PairMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
for _ in range(args.epochs):
    for _, _, features, target in records:
        logits = model(features)
        mask = torch.ones_like(target, dtype=torch.bool)
        # Diagonal labels are undefined and must not contribute to training.
        n = int(target.numel() ** 0.5)
        mask.reshape(n, n).fill_diagonal_(False)
        loss = F.binary_cross_entropy_with_logits(logits[mask], target[mask])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
accuracy, tau = evaluate(model, records)
print(f"pages={len(records)} epochs={args.epochs} pairwise_accuracy={accuracy:.4f} mean_tau={tau:.4f}")
