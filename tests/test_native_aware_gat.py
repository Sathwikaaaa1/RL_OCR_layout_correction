from __future__ import annotations

import numpy as np

from rlocr.order_features import decode_pair_probabilities


def test_native_prior_can_override_weak_model_evidence() -> None:
    lines = [
        {"id": "a", "native_order": 0, "bbox": [0, 0, 10, 10]},
        {"id": "b", "native_order": 1, "bbox": [20, 0, 30, 10]},
    ]
    # The model weakly prefers the wrong direction; a native prior restores it.
    probabilities = {(0, 1): 0.45}
    assert decode_pair_probabilities(lines, probabilities, native_prior=0.0) == ["b", "a"]
    assert decode_pair_probabilities(lines, probabilities, native_prior=1.0, rng=np.random.default_rng(2)) == ["a", "b"]
