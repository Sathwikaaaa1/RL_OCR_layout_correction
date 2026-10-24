"""Statistics for frozen out-of-fold reading-order predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


def template_weighted_bootstrap(
    differences: dict[str, list[float]],
    target_mass: dict[str, float],
    samples: int = 10_000,
    seed: int = 271828,
) -> dict[str, Any]:
    """Resample paired pages within templates, holding target proportions fixed.

    Each page in template t has weight target_mass[t] / page_count[t].
    The interval is conditional on the frozen models and target mixture.
    """
    if samples < 1 or any(value < 0 for value in target_mass.values()):
        raise ValueError("Require positive samples and nonnegative target masses")
    total = sum(target_mass.values())
    if total <= 0:
        raise ValueError("Target masses must have positive total")
    mass = {key: value / total for key, value in target_mass.items()}
    rng = np.random.default_rng(seed)
    draws = np.zeros(samples, dtype=np.float64)
    mean = 0.0
    for template, weight in sorted(mass.items()):
        if weight == 0:
            continue
        values = np.asarray(differences.get(template, []), dtype=np.float64)
        if not len(values):
            raise ValueError(f"No pages for positive-weight template {template}")
        mean += weight * float(values.mean())
        # Bound memory even for larger corpora.
        for start in range(0, samples, 256):
            end = min(start + 256, samples)
            indices = rng.integers(0, len(values), size=(end - start, len(values)))
            draws[start:end] += weight * values[indices].mean(axis=1)
    return {
        "mean": mean,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "target_template_mass": mass,
        "page_weight_by_template": {
            key: weight / len(differences[key]) if weight else 0.0
            for key, weight in mass.items()
        },
    }
