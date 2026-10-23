"""Deterministic page-grouped cross-validation splits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Fold:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def grouped_folds(
    page_templates: Sequence[tuple[str, str]],
    folds: int = 5,
    validation_fraction: float = 0.15,
    seed: int = 271828,
) -> list[Fold]:
    """Stratify page groups by template and test every page exactly once."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    keys = [key for key, _ in page_templates]
    if len(keys) != len(set(keys)):
        raise ValueError("page group keys must be unique")
    by_template: dict[str, list[str]] = defaultdict(list)
    for key, template in page_templates:
        by_template[template].append(key)

    test_buckets: list[list[str]] = [[] for _ in range(folds)]
    for template_index, template in enumerate(sorted(by_template)):
        group = sorted(by_template[template])
        np.random.default_rng(seed + template_index).shuffle(group)
        offset = template_index % folds
        for index, key in enumerate(group):
            test_buckets[(index + offset) % folds].append(key)

    all_keys = set(keys)
    result: list[Fold] = []
    template_by_key = dict(page_templates)
    for fold_index, test_values in enumerate(test_buckets):
        test = set(test_values)
        remaining_by_template: dict[str, list[str]] = defaultdict(list)
        for key in sorted(all_keys - test):
            remaining_by_template[template_by_key[key]].append(key)
        validation: set[str] = set()
        for template_index, template in enumerate(sorted(remaining_by_template)):
            group = remaining_by_template[template]
            np.random.default_rng(seed + 10_000 * (fold_index + 1) + template_index).shuffle(group)
            count = max(1, int(round(len(group) * validation_fraction))) if group else 0
            validation.update(group[:count])
        train = all_keys - test - validation
        result.append(Fold(tuple(sorted(train)), tuple(sorted(validation)), tuple(sorted(test))))
    return result
