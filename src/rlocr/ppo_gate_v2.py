"""Lookahead observations, terminal-only reward, and gated-teacher projection."""

from functools import lru_cache
from typing import Any

from gymnasium import spaces
import numpy as np
import torch
from torch.nn import functional as F

from rlocr.ppo_gate import OBSERVATION_NAMES, OverrideNativeEnv, index_tau


V2_OBSERVATION_NAMES = OBSERVATION_NAMES + (
    "log1p_confident_disagreements_div5", "max_native_disagreement_margin",
    "gat_native_page_tau", "native_xycut_page_tau",
)


def lookahead_features(page: dict[str, Any], pair_threshold: float) -> np.ndarray:
    """Compute once, from OCR and model outputs only; count also stored raw."""
    probabilities = page["probabilities"]
    rank = page["native_rank"]
    count = 0
    margins = []
    for i in range(len(rank)):
        for j in range(i + 1, len(rank)):
            opposite = 1.0 - probabilities[i, j] if rank[i] < rank[j] else probabilities[i, j]
            if opposite > 0.5:
                margins.append(float(opposite - 0.5))
                count += int(opposite >= pair_threshold)
    page["confident_disagreement_count"] = count
    return np.asarray([np.log1p(count) / 5.0, max(margins, default=0.0),
                       index_tau(page["raw"], rank), page["xy_tau"]], dtype=np.float32)


class OverrideNativeV2Env(OverrideNativeEnv):
    def __init__(self, page: dict[str, Any]) -> None:
        super().__init__(page)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(V2_OBSERVATION_NAMES),), np.float32)

    def _observation(self):
        return np.concatenate([super()._observation(), self.page["lookahead"]]).astype(np.float32)

    def _place(self, index: int) -> float:
        self.remaining[index] = False
        self.order.append(index)
        return 0.0

    def step(self, action: int):
        if self.finished:
            raise RuntimeError("Reset a completed episode")
        if not self.action_space.contains(action):
            raise ValueError("Action must be RETAIN (0) or REORDER (1)")
        native, gat = self._proposals()
        self.disagreements += 1
        self.overrides += int(action)
        self._place(gat if action else native)
        self._auto_place()
        self.finished = not self.remaining.any()
        reward = -0.005 * int(action)
        if self.finished:
            reward += 20.0 * (index_tau(self.order, self.page["gt_rank"]) - self.page["native_tau"])
        self.episode_return += reward
        info = self._info()
        info["terminal_reward"] = 20.0 * info["terminal_reward"] if info["terminal_reward"] is not None else None
        return self._observation(), float(reward), self.finished, False, info


def gated_teacher_action(env: OverrideNativeEnv) -> int:
    """Project the full gated order onto the two currently available actions.

    Prefer whichever proposal occurs earlier in the teacher's target order.
    This exactly reproduces the teacher when that order is reachable, but
    the binary action space cannot express every pairwise-gated permutation.
    """
    native, gat = env._proposals()
    return int(env.page["gated_rank"][gat] < env.page["gated_rank"][native])


def fit_imitation(model, observations, actions, *, seed: int, epochs: int = 30,
                  learning_rate: float = 1e-3) -> list[dict[str, float]]:
    """Plain supervised cross-entropy on training-page teacher actions only."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        losses = []
        for indices in torch.randperm(len(actions), generator=generator).split(256):
            logits, _ = model(observations[indices])
            loss = F.cross_entropy(logits, actions[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    return history


def reachable_tau_oracle(page: dict[str, Any]) -> dict[str, Any]:
    """Exact tau ceiling for the binary action space, using GT for diagnostics.

    Placed sets are unions of native and GAT prefixes: at most (n+1)^2
    distinct states. Dynamic programming minimizes total pair inversions.
    The greedy true-next oracle is generally not a global tau ceiling.
    """
    n = len(page["native"])
    full = (1 << n) - 1
    ranks = page["gt_rank"]
    lower_masks = [sum(1 << j for j in range(n) if ranks[j] < ranks[i]) for i in range(n)]

    @lru_cache(None)
    def solve(mask: int):
        if mask == full:
            return 0, ()
        native = next(int(i) for i in page["native"] if not mask & (1 << int(i)))
        gat = next(int(i) for i in page["raw"] if not mask & (1 << int(i)))
        best = None
        for candidate in dict.fromkeys((native, gat)):
            remaining_cost, tail = solve(mask | (1 << candidate))
            inversions = (lower_masks[candidate] & (full ^ mask)).bit_count() + remaining_cost
            value = (inversions, (candidate,) + tail)
            if best is None or value[0] < best[0]:
                best = value
        return best

    cost, order = solve(0)
    return {"tau": 1.0 - 4.0 * cost / max(1, n * (n - 1)),
            "order": list(order), "states": solve.cache_info().currsize}
