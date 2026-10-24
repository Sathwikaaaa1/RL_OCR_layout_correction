"""One-step page-level selection between native and fixed gated orders."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rlocr.ppo_gate import index_tau


PAGE_FEATURE_NAMES = (
    "log1p_confident_disagreements_div5", "max_confident_disagreement_margin",
    "gated_native_tau", "native_xycut_tau", "log1p_line_count_div5",
    "fraction_narrow_edge_lines",
)


def page_gate_features(page: dict[str, Any], pair_threshold: float) -> np.ndarray:
    """Six OCR/model-only features; no template labels or ground truth.

    A narrow edge line has width <=25% of page width and horizontal center
    <=20% or >=80% of page width. Margins are opposite-order p minus 0.5,
    restricted to strict native disagreements passing the fixed pair gate.
    """
    rank = page["native_rank"]
    probabilities = page["probabilities"]
    opposite = np.where(rank[:, None] < rank[None, :], 1 - probabilities, probabilities)
    confident = np.triu((opposite > 0.5) & (opposite >= pair_threshold), k=1)
    count = int(confident.sum())
    max_margin = float((opposite[confident] - 0.5).max()) if count else 0.0
    narrow_edge = (page["widths"] <= 0.25) & ((page["x_centers"] <= 0.20) | (page["x_centers"] >= 0.80))
    return np.asarray([
        np.log1p(count) / 5.0, max_margin, index_tau(page["gated"], rank),
        page["xy_tau"], np.log1p(len(rank)) / 5.0,
        float(narrow_edge.mean()) if len(rank) else 0.0,
    ], dtype=np.float32)


class PageGateEnv(gym.Env):
    """One page, one action, one terminal improvement reward."""

    def __init__(self, page: dict[str, Any], pair_threshold: float = 0.65) -> None:
        super().__init__()
        self.page = page
        self.features = page_gate_features(page, pair_threshold)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(PAGE_FEATURE_NAMES),), np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.finished = False
        self.order: list[int] = []
        return self.features.copy(), {"page_key": self.page.get("key")}

    def step(self, action: int):
        if self.finished:
            raise RuntimeError("Reset before starting another page episode")
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0 (native) or 1 (gated decoder)")
        self.order = list(self.page["gated"] if action else self.page["native"])
        tau = index_tau(self.order, self.page["gt_rank"])
        reward = tau - self.page["native_tau"]
        self.finished = True
        return self.features.copy(), reward, True, False, {"order": self.order,
            "action": int(action), "tau": tau, "native_tau": self.page["native_tau"]}
