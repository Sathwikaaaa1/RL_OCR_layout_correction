"""Native-retaining decoding and paired test statistics."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np

from rlocr.ocr.alignment import kendall_tau


def native_order(lines: list[dict[str, Any]]) -> list[str]:
    return [line["id"] for line in sorted(lines, key=lambda item: item.get("native_order", 0))]


def _pair_disagreement(left: dict[str, Any], right: dict[str, Any], probability: float, threshold: float) -> bool:
    native_before = left.get("native_order", 0) < right.get("native_order", 0)
    model_before = probability >= 0.5
    confidence_opposite = (1.0 - probability) if native_before else probability
    return model_before != native_before and confidence_opposite >= threshold


def raw_gat_order(lines: list[dict[str, Any]], probabilities: dict[tuple[int, int], float]) -> list[str]:
    scores = np.zeros(len(lines), dtype=np.float64)
    native_rank = {index: rank for rank, index in enumerate(sorted(range(len(lines)), key=lambda i: lines[i].get("native_order", i)))}
    for i, j in combinations(range(len(lines)), 2):
        probability = float(np.clip(probabilities.get((i, j), 0.5), 0.0, 1.0))
        scores[i] += probability
        scores[j] += 1.0 - probability
    return [lines[index]["id"] for index in sorted(range(len(lines)), key=lambda i: (-scores[i], native_rank[i]))]


def gated_order(lines: list[dict[str, Any]], probabilities: dict[tuple[int, int], float], pair_threshold: float, page_threshold: int) -> tuple[list[str], int, bool]:
    """Return native order unless enough confident pair disagreements exist."""
    native = native_order(lines)
    disagreements = [(i, j) for i, j in combinations(range(len(lines)), 2) if _pair_disagreement(lines[i], lines[j], probabilities.get((i, j), 0.5), pair_threshold)]
    if len(disagreements) <= page_threshold:
        return native, len(disagreements), False
    disagreement_set = set(disagreements)
    scores = np.zeros(len(lines), dtype=np.float64)
    native_rank = {index: rank for rank, index in enumerate(sorted(range(len(lines)), key=lambda i: lines[i].get("native_order", i)))}
    for i, j in combinations(range(len(lines)), 2):
        if (i, j) in disagreement_set:
            probability = float(np.clip(probabilities.get((i, j), 0.5), 0.0, 1.0))
            scores[i] += probability
            scores[j] += 1.0 - probability
        else:
            native_before = lines[i].get("native_order", i) < lines[j].get("native_order", j)
            scores[i if native_before else j] += 1.0
    decoded = [lines[index]["id"] for index in sorted(range(len(lines)), key=lambda i: (-scores[i], native_rank[i]))]
    return decoded, len(disagreements), decoded != native


def paired_bootstrap(values: Sequence[float], samples: int = 10_000, seed: int = 90210) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    means = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(samples)])
    return {"mean": float(values.mean()), "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975))}


def sign_flip_pvalue(values: Sequence[float], samples: int = 10_000, seed: int = 90210) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    simulated = np.abs((rng.choice(np.array([-1.0, 1.0]), size=(samples, len(values))) * values).mean(axis=1))
    return float((1 + np.sum(simulated >= observed)) / (samples + 1))


def paired_significance(method: Sequence[float], native: Sequence[float], samples: int = 10_000, seed: int = 90210) -> dict[str, Any]:
    differences = [float(a - b) for a, b in zip(method, native)]
    return {"difference": paired_bootstrap(differences, samples, seed), "permutation_p_value": sign_flip_pvalue(differences, samples, seed), "better_pages": sum(value > 1e-9 for value in differences), "worse_pages": sum(value < -1e-9 for value in differences), "same_pages": sum(abs(value) <= 1e-9 for value in differences), "differences": differences}
