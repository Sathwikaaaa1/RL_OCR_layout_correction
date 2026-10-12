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
        self.max_lines = max_lines
        self.feature_dim = 8
        self.observation_space = spaces.Box(0.0, 1.0, shape=(max_lines, self.feature_dim), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.index = 0
        self.actions: list[int] = []

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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed); self.index = 0; self.actions = []; return self._observation(), {"page_id": self.page.get("page_id")}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        line = self.lines[self.index]
        target = self.page.get("action_targets", {}).get(line["id"], 0)
        reward = 1.0 if int(action) == int(target) else -1.0
        self.actions.append(int(action)); self.index += 1
        terminated = self.index >= len(self.lines)
        return self._observation(), reward, terminated, False, {"line_id": line["id"], "target_action": target}
