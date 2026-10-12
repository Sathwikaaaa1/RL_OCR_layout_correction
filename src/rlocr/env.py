from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OCRCorrectionEnv(gym.Env[np.ndarray, int]):
    """Small single-page environment for structural action prototyping.

    Actions apply to the current OCR line: 0=retain, 1=merge, 2=split,
    3=link. Rewards are label-based placeholders until learned structural
    rewards are added; the observation remains OCR-only.
    """

    metadata = {"render_modes": []}

    def __init__(self, page: dict[str, Any] | str | Path, max_lines: int = 256) -> None:
        super().__init__()
        self.page = self._load(page)
        self.lines = self.page["ocr_lines"][:max_lines]
        self.page["action_targets"] = self.page.get("action_targets") or self._derive_targets()
        self.max_lines = max_lines
        self.feature_dim = 8
        self.observation_space = spaces.Box(0.0, 1.0, shape=(max_lines, self.feature_dim), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.index = 0
        self.actions: list[int] = []
        self.merged: set[str] = set()
        self.split: set[str] = set()
        self.links: list[str] = []

    def _derive_targets(self) -> dict[str, int]:
        targets = {line["id"]: 0 for line in self.lines}
        for line_id in self.page.get("split_needed", []):
            if line_id in targets:
                targets[line_id] = 2
        for pair in self.page.get("merge_pairs", []):
            for line_id in pair:
                if line_id in targets and targets[line_id] == 0:
                    targets[line_id] = 1
        gt_order = self.page.get("gt_order", [])
        native_order = self.page.get("tesseract_order", [])
        if gt_order != native_order:
            for line_id in targets:
                if targets[line_id] == 0:
                    targets[line_id] = 3
        return targets

    @staticmethod
    def _load(page: dict[str, Any] | str | Path) -> dict[str, Any]:
        if isinstance(page, (str, Path)):
            return json.loads(Path(page).read_text(encoding="utf-8"))
        return page

    def _observation(self) -> np.ndarray:
        observation = np.zeros((self.max_lines, self.feature_dim), dtype=np.float32)
        for row, line in enumerate(self.lines):
            x0, y0, x1, y1 = line["bbox"]
            observation[row] = [x0 / 2000, y0 / 2000, x1 / 2000, y1 / 2000, min(len(line["text"]) / 200, 1), line.get("confidence", 0) / 100, line.get("block_num", 0) / 20, line.get("native_order", 0) / max(1, len(self.lines))]
        return np.clip(observation, 0, 1)

    @property
    def edge_index(self) -> np.ndarray:
        """Symmetric spatial kNN graph over the padded OCR-line observations."""
        points = np.zeros((self.max_lines, 2), dtype=np.float32)
        for row, line in enumerate(self.lines):
            points[row] = [(line["bbox"][0] + line["bbox"][2]) / 2, (line["bbox"][1] + line["bbox"][3]) / 2]
        edges: set[tuple[int, int]] = {(i, i) for i in range(self.max_lines)}
        k = min(4, max(1, len(self.lines) - 1))
        for i in range(len(self.lines)):
            distances = np.sum((points - points[i]) ** 2, axis=1); distances[i] = np.inf
            for j in np.argsort(distances)[:k]:
                if j < len(self.lines): edges.add((i, int(j))); edges.add((int(j), i))
        return np.asarray(list(edges), dtype=np.int64).T

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed); self.index = 0; self.actions = []; self.merged = set(); self.split = set(); self.links = []; return self._observation(), {"page_id": self.page.get("page_id")}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        line = self.lines[self.index]
        target = self.page.get("action_targets", {}).get(line["id"], 0)
        action = int(action)
        if action == 1: self.merged.add(line["id"])
        elif action == 2: self.split.add(line["id"])
        elif action == 3: self.links.append(line["id"])
        reward = 1.0 if action == int(target) else -1.0
        if action == 1 and line["id"] in self.split: reward -= 0.5
        if action == 2 and line["id"] in self.merged: reward -= 0.5
        self.actions.append(int(action)); self.index += 1
        terminated = self.index >= len(self.lines)
        return self._observation(), reward, terminated, False, {"line_id": line["id"], "target_action": target}
