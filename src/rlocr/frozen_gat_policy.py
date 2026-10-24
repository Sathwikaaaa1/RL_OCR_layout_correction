"""Frozen GAT state features and a masked RETAIN/REORDER editing MDP."""

from typing import Any

from gymnasium import spaces
import numpy as np
import torch

from rlocr.order_actions import RETAIN, REORDER
from rlocr.ppo_gate import index_tau
from rlocr.ppo_gate_v2 import OverrideNativeV2Env


def inferred_columns(page: dict[str, Any]) -> np.ndarray:
    """OCR-only left-edge clusters; no GT template or column_index inputs.

    Consecutive left edges are clustered at gaps <= max(3.5% page width,
    twice median OCR line height). Cluster indices run left to right.
    This is deliberately a simple heuristic, not a learned column detector.
    """
    width = np.median([(line["bbox"][2] - line["bbox"][0]) / relative
                       for line, relative in zip(page["lines"], page["widths"]) if relative > 0])
    heights = [(line["bbox"][3] - line["bbox"][1]) / width for line in page["lines"]]
    tolerance = max(0.035, 2 * float(np.median(heights)))
    lefts = page["x_centers"] - page["widths"] / 2
    labels = np.zeros(len(lefts), dtype=np.int64)
    ordered = np.argsort(lefts, kind="stable")
    current = 0
    for i, index in enumerate(ordered):
        if i and lefts[index] - lefts[ordered[i - 1]] > tolerance:
            current += 1
        labels[index] = current
    return labels


def mask_logits(logits: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    if not bool(action_masks.any(dim=-1).all()):
        raise ValueError("Every live policy state needs an available action")
    return logits.masked_fill(~action_masks, -1e9)


class FrozenGATPolicyEnv(OverrideNativeV2Env):
    """Frozen embeddings are cached; only the MLP policy head is optimized."""

    def __init__(self, page: dict[str, Any]) -> None:
        super().__init__(page)
        dimension = 3 * page["node_embedding"].shape[1] + len(page["lookahead_normalized"]) + 4
        self.observation_space = spaces.Box(-np.inf, np.inf, (dimension,), np.float32)

    def reset(self, *, seed=None, options=None):
        self.last_decision = -1
        self.invalid_actions = 0
        self.reorder_attempts = 0
        return super().reset(seed=seed, options=options)

    def available_block_mask(self) -> np.ndarray:
        return self.remaining.copy()

    def mask_for_candidates(self, candidates) -> np.ndarray:
        return np.asarray([0 <= index < len(self.remaining) and self.remaining[index] for index in candidates], dtype=bool)

    def action_masks(self) -> np.ndarray:
        if not self.remaining.any():
            return np.zeros(2, dtype=bool)
        return self.mask_for_candidates(self._proposals())

    def invalid_reorder_jump(self, native: int, gat: int) -> bool:
        if not self.order:
            return False
        columns = self.page["inferred_columns"]
        current_column = columns[self.order[-1]]
        return bool(columns[native] == current_column and abs(int(columns[gat]) - int(current_column)) > 1)

    def _observation(self):
        dimension = self.observation_space.shape[0]
        if not self.remaining.any():
            return np.zeros(dimension, dtype=np.float32)
        native, gat = self._proposals()
        last = np.zeros(3, dtype=np.float32)  # no decision yet, RETAIN, REORDER
        last[self.last_decision + 1] = 1.0
        return np.concatenate([
            self.page["node_embedding"][native], self.page["node_embedding"][gat],
            self.page["pooled_embedding"], self.page["lookahead_normalized"],
            np.asarray([len(self.order) / max(1, len(self.remaining))], dtype=np.float32), last,
        ]).astype(np.float32)

    def step(self, action):
        if self.finished:
            raise RuntimeError("Reset a completed page")
        if not self.action_space.contains(action):
            raise ValueError("Use RETAIN (0) or REORDER (1)")
        native, gat = self._proposals()
        candidates = (native, gat)
        chosen = candidates[int(action)]
        invalid = not self.mask_for_candidates(candidates)[int(action)]
        invalid = bool(invalid or (action == REORDER and self.invalid_reorder_jump(native, gat)))
        self.last_decision = int(action)
        self.disagreements += 1
        self.reorder_attempts += int(action == REORDER)
        self.invalid_actions += int(invalid)
        # A rejected edit executes RETAIN, consuming a valid native line so
        # repeated invalid edits cannot create unbounded episodes.
        if invalid:
            chosen = next(int(i) for i in self.page["native"] if self.remaining[i])
        self.overrides += int(action == REORDER and not invalid)
        self._place(chosen)
        self._auto_place()
        self.finished = not self.remaining.any()
        reward = -0.005 * int(action == REORDER) - 0.05 * int(invalid)
        if self.finished:
            reward += 20 * (index_tau(self.order, self.page["gt_rank"]) - self.page["native_tau"])
        self.episode_return += reward
        info = self._info()
        if info["terminal_reward"] is not None:
            info["terminal_reward"] *= 20.0
        info.update({"invalid_action": invalid, "invalid_actions": self.invalid_actions,
                     "reorder_attempts": self.reorder_attempts, "action_name": self.action_names[int(action)]})
        return self._observation(), float(reward), self.finished, False, info
