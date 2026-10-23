from __future__ import annotations

from rlocr.gated_decoder import gated_order, paired_bootstrap, sign_flip_pvalue


def test_high_thresholds_return_native_exactly() -> None:
    lines = [{"id": "a", "native_order": 0}, {"id": "b", "native_order": 1}, {"id": "c", "native_order": 2}]
    order, disagreements, changed = gated_order(lines, {(0, 1): 0.01, (0, 2): 0.01, (1, 2): 0.01}, 1.01, 0)
    assert order == ["a", "b", "c"]
    assert disagreements == 0 and not changed


def test_bootstrap_and_sign_flip_are_deterministic_on_toy_data() -> None:
    values = [1.0, -1.0, 0.5]
    result = paired_bootstrap(values, samples=100, seed=7)
    assert result["mean"] == 1 / 6
    assert result["low"] <= result["mean"] <= result["high"]
    p_value = sign_flip_pvalue(values, samples=100, seed=7)
    assert 0.0 < p_value <= 1.0
    assert p_value == sign_flip_pvalue(values, samples=100, seed=7)
