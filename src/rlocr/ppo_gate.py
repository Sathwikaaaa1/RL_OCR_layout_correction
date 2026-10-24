"""Cheap binary override environment and CPU MLP PPO with GAE and clipping."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rlocr.ppo import PPOConfig
from rlocr.order_actions import ACTION_NAMES


OBSERVATION_NAMES = (
    "native_candidate_probability", "gat_candidate_probability",
    "native_candidate_margin", "gat_candidate_margin", "native_rank_difference",
    "native_xycut_tau", "disagreements_so_far", "native_relative_width",
    "gat_relative_width", "native_horizontal_position", "gat_horizontal_position",
    "step_fraction", "pair_probability_native_before_gat",
)


def index_tau(order: list[int], ranks: np.ndarray) -> float:
    count = len(order)
    if count < 2:
        return 1.0
    values = ranks[order]
    inversions = int(np.count_nonzero(np.triu(values[:, None] > values[None, :], 1)))
    return 1.0 - 4.0 * inversions / (count * (count - 1))


class OverrideNativeEnv(gym.Env):
    """One OCR page per episode; decisions occur only when proposals differ.

    Existing evaluation units are OCR lines, called blocks in the request.
    GAT proposes the highest remaining canonical page score (prior=4).
    Ground truth is consulted only for reward and reported metrics.
    """

    action_names = ACTION_NAMES

    def __init__(self, page: dict[str, Any]) -> None:
        super().__init__()
        self.page = page
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(OBSERVATION_NAMES),), np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.remaining = np.ones(len(self.page["native"]), dtype=bool)
        self.order: list[int] = []
        self.overrides = 0
        self.disagreements = 0
        self.dense_total = 0.0
        self.episode_return = 0.0
        self.pending_reward = self._auto_place()
        self.finished = not self.remaining.any()
        if self.finished:
            self.episode_return = self.pending_reward
        return self._observation(), self._info()

    def _proposals(self) -> tuple[int, int]:
        native = next(int(i) for i in self.page["native"] if self.remaining[i])
        gat = next(int(i) for i in self.page["raw"] if self.remaining[i])
        return native, gat

    def _place(self, index: int) -> float:
        true_next = int(np.argmin(np.where(self.remaining, self.page["gt_rank"], np.inf)))
        reward = 0.1 if index == true_next else -0.1
        self.remaining[index] = False
        self.order.append(index)
        self.dense_total += reward
        return reward

    def _auto_place(self) -> float:
        reward = 0.0
        while self.remaining.any():
            native, gat = self._proposals()
            if native != gat:
                break
            reward += self._place(native)
        return reward

    def _observation(self) -> np.ndarray:
        if not self.remaining.any():
            return np.zeros(len(OBSERVATION_NAMES), dtype=np.float32)
        a, b = self._proposals()
        remaining = np.flatnonzero(self.remaining)
        scores = self.page["scores"] / max(1, len(self.remaining))
        weights = np.exp(scores[remaining] - np.max(scores[remaining]))
        probabilities = weights / weights.sum()
        probability_by_id = dict(zip(remaining, probabilities))
        margin_a = scores[a] - max(scores[i] for i in remaining if i != a)
        margin_b = scores[b] - max(scores[i] for i in remaining if i != b)
        n = len(self.remaining)
        return np.asarray([
            probability_by_id[a], probability_by_id[b], margin_a, margin_b,
            (self.page["native_rank"][a] - self.page["native_rank"][b]) / max(1, n - 1),
            self.page["xy_tau"], (self.disagreements + 1) / max(1, n),
            self.page["widths"][a], self.page["widths"][b],
            self.page["x_centers"][a], self.page["x_centers"][b],
            len(self.order) / max(1, n), self.page["probabilities"][a, b],
        ], dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        terminal_delta = index_tau(self.order, self.page["gt_rank"]) - self.page["native_tau"] if not self.remaining.any() else None
        return {"automatic_terminal": not self.remaining.any(), "terminal_reward": terminal_delta,
                "overrides": self.overrides, "dense_reward": self.dense_total,
                "episode_return": self.episode_return, "order": list(self.order)}

    def step(self, action: int):
        if self.finished:
            raise RuntimeError("Episode already complete; reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError("Action must be RETAIN (0) or REORDER (1)")
        native, gat = self._proposals()
        self.disagreements += 1
        self.overrides += int(action)
        reward = self.pending_reward + self._place(gat if action else native) - 0.02 * int(action)
        self.pending_reward = 0.0
        reward += self._auto_place()
        self.finished = not self.remaining.any()
        if self.finished:
            reward += index_tau(self.order, self.page["gt_rank"]) - self.page["native_tau"]
        self.episode_return += reward
        return self._observation(), float(reward), self.finished, False, self._info()


class OverrideActorCritic(nn.Module):
    def __init__(self, input_dim: int = len(OBSERVATION_NAMES)) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)

    def forward(self, observations: torch.Tensor):
        hidden = self.body(observations)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def gae(rewards, dones, values, bootstrap, config: PPOConfig):
    """Time x environment tensors; bootstrap unfinished rollout tails."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(bootstrap)
    for time in reversed(range(len(rewards))):
        next_value = bootstrap if time == len(rewards) - 1 else values[time + 1]
        continuation = 1.0 - dones[time]
        residual = rewards[time] + config.gamma * next_value * continuation - values[time]
        running = residual + config.gamma * config.gae_lambda * continuation * running
        advantages[time] = running
    return advantages, advantages + values


def ppo_update(model, optimizer, observations, actions, old_log_probs, advantages, returns, config, generator, action_masks=None):
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    losses = []
    for _ in range(config.epochs):
        for indices in torch.randperm(len(actions), generator=generator).split(256):
            logits, values = model(observations[indices])
            if action_masks is not None:
                logits = logits.masked_fill(~action_masks[indices], -1e9)
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(actions[indices]) - old_log_probs[indices]).exp()
            policy_loss = -torch.minimum(ratio * advantages[indices], ratio.clamp(1-config.clip_ratio, 1+config.clip_ratio) * advantages[indices]).mean()
            loss = policy_loss + config.value_coef * (values - returns[indices]).square().mean() - config.entropy_coef * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))
