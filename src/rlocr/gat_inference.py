"""Single, shared inference path for the pairwise GAT."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import torch

from rlocr.order_features import decode_pair_probabilities


@dataclass(frozen=True)
class GATInference:
    """Pair logits, probabilities, and the native-prior decoded order."""

    logits: torch.Tensor
    probabilities: dict[tuple[int, int], float]
    order: list[str]


def gat_inference_from_logits(
    logits: torch.Tensor,
    lines: list[dict[str, Any]],
    native_prior: float,
    noise_std: float = 0.0,
    seed: int = 0,
) -> GATInference:
    """Convert model logits to canonical pair probabilities and reading order."""
    probabilities = {
        (i, j): float(torch.sigmoid(logits[i, j]).item())
        for i, j in combinations(range(len(lines)), 2)
    }
    order = decode_pair_probabilities(
        lines,
        probabilities,
        native_prior=native_prior,
        noise_std=noise_std,
        rng=np.random.default_rng(seed),
    )
    return GATInference(logits=logits, probabilities=probabilities, order=order)


def infer_gat(
    model: torch.nn.Module,
    features: torch.Tensor,
    edges: torch.Tensor,
    lines: list[dict[str, Any]],
    native_prior: float,
    noise_std: float = 0.0,
    seed: int = 0,
) -> GATInference:
    """Run a GAT once and decode it through the canonical native-aware path."""
    with torch.no_grad():
        logits = model(features, edges)
    return gat_inference_from_logits(logits, lines, native_prior, noise_std, seed)
