from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from rlocr.model import OCRGATEncoder


def ring_edges(nodes: int, device: torch.device | None = None) -> Tensor:
    source = torch.arange(nodes, device=device)
    target = torch.roll(source, -1)
    values = torch.cat([source, target, source])
    return torch.stack([values, torch.cat([target, source, source])])


def spatial_edges(observation: np.ndarray, device: torch.device | None = None, k: int = 4) -> Tensor:
    """Build symmetric kNN edges from normalized bbox centers."""
    points = observation[:, :2]
    pairs: set[tuple[int, int]] = {(i, i) for i in range(len(points))}
    for i in range(len(points)):
        distances = np.sum((points - points[i]) ** 2, axis=1); distances[i] = np.inf
        for j in np.argsort(distances)[:min(k, max(1, len(points) - 1))]:
            pairs.add((i, int(j))); pairs.add((int(j), i))
    return torch.tensor(list(pairs), dtype=torch.long, device=device).T


class GATActorCritic(nn.Module):
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 32, actions: int = 4) -> None:
        super().__init__()
        self.encoder = OCRGATEncoder(feature_dim, hidden_dim, heads=2)
        self.policy = nn.Linear(hidden_dim, actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, observations: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
        embeddings = torch.stack([self.encoder(item, edge_index) for item in observations])
        return self.policy(embeddings), self.value(embeddings).squeeze(-1)


@dataclass
class PPOConfig:
    epochs: int = 4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95


class PPOTrainer:
    def __init__(self, model: GATActorCritic, config: PPOConfig | None = None, seed: int = 1337) -> None:
        self.model = model
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        torch.manual_seed(seed); np.random.seed(seed)

    def update(self, observations: Tensor, nodes: Tensor, actions: Tensor, old_log_probs: Tensor, rewards: Tensor, dones: Tensor, edge_index: Tensor) -> float:
        dones = dones.float()
        with torch.no_grad():
            _, values = self.model(observations, edge_index)
            values = values[torch.arange(len(nodes)), nodes]
            advantages = torch.zeros_like(rewards); running = torch.tensor(0.0)
            for i in range(len(rewards) - 1, -1, -1):
                next_value = values[i + 1] if i + 1 < len(values) else torch.tensor(0.0)
                delta = rewards[i] + self.config.gamma * next_value * (1 - dones[i]) - values[i]
                running = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[i]) * running
                advantages[i] = running
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_value = 0.0
        for _ in range(self.config.epochs):
            logits, predicted_values = self.model(observations, edge_index)
            distribution = Categorical(logits=logits[torch.arange(len(nodes)), nodes])
            log_probs = distribution.log_prob(actions); ratio = (log_probs - old_log_probs).exp()
            policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages).mean()
            value_loss = (predicted_values[torch.arange(len(nodes)), nodes] - returns).pow(2).mean()
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * distribution.entropy().mean()
            self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step(); loss_value = float(loss.detach())
        return loss_value

    def behavior_clone(self, observations: Tensor, targets: Tensor, edge_index: Tensor, epochs: int = 1, valid_mask: Tensor | None = None) -> float:
        """Warm-start on alignment labels with inverse-frequency weighting."""
        if valid_mask is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
        flat_targets = targets.reshape(-1)[valid_mask.reshape(-1)]
        weights = torch.bincount(flat_targets, minlength=4).float().clamp_min(1).reciprocal()
        weights = weights / weights.mean(); loss_value = 0.0
        for _ in range(epochs):
            logits, _ = self.model(observations, edge_index)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 4)[valid_mask.reshape(-1)], flat_targets, weight=weights)
            self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step(); loss_value = float(loss.detach())
        return loss_value


def collect_episode(env: object, model: GATActorCritic, edge_index: Tensor, device: torch.device) -> tuple[list[Tensor], list[int], list[Tensor], list[float], list[bool], float]:
    observation, _ = env.reset(); states: list[Tensor] = []; nodes: list[int] = []; actions: list[Tensor] = []; log_probs: list[Tensor] = []; rewards: list[float] = []; dones: list[bool] = []
    total = 0.0
    while True:
        state = torch.from_numpy(observation).float().to(device); logits, _ = model(state.unsqueeze(0), edge_index); node = env.index
        distribution = Categorical(logits=logits[0, node]); action = distribution.sample(); next_observation, reward, terminated, _, _ = env.step(int(action))
        states.append(state); nodes.append(node); actions.append(action); log_probs.append(distribution.log_prob(action).detach()); rewards.append(float(reward)); dones.append(terminated); total += float(reward); observation = next_observation
        if terminated: return states, nodes, actions, log_probs, rewards, total
