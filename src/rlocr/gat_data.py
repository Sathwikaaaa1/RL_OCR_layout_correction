"""Page loading shared by CPU GAT evaluation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from rlocr.env import OCRCorrectionEnv
from rlocr.ocr.alignment import kendall_tau
from rlocr.ppo import spatial_edges


def load_gat_record(aligned_path: Path, gt_path: Path, max_lines: int, key: str | None = None) -> dict[str, Any]:
    page = json.loads(aligned_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    page["template"] = ground_truth.get("template", "unknown")
    page["page_id"] = ground_truth.get("page_id", aligned_path.stem)
    valid_ids = set(page.get("valid_line_ids", []))
    if valid_ids:
        page["ocr_lines"] = [line for line in page["ocr_lines"] if line["id"] in valid_ids]
    env = OCRCorrectionEnv(page, max_lines=max_lines)
    observation, _ = env.reset()
    count = len(env.lines)
    features = torch.from_numpy(observation[:count]).float()
    edges = spatial_edges(observation[:count], k=4)
    rank = {line_id: index for index, line_id in enumerate(page["gt_order"])}
    target = torch.zeros((count, count), dtype=torch.float32)
    mask = torch.ones((count, count), dtype=torch.bool)
    for i, left in enumerate(env.lines):
        for j, right in enumerate(env.lines):
            if i == j:
                mask[i, j] = False
            else:
                target[i, j] = float(rank[left["id"]] < rank[right["id"]])
    native = [line["id"] for line in sorted(env.lines, key=lambda item: item.get("native_order", 0))]
    return {
        "key": key or aligned_path.stem,
        "page": page,
        "lines": env.lines,
        "features": features,
        "edges": edges,
        "target": target,
        "mask": mask,
        "native_order": native,
        "native_tau": kendall_tau(native, page["gt_order"]),
    }
