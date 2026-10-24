import pytest

from rlocr.diagnostics import template_weighted_bootstrap


def test_template_weights_hold_target_mix_and_pairing_fixed() -> None:
    # Different template sample sizes must not change equal-template mass.
    differences = {"sidebar": [0.2] * 7, "table": [-0.1] * 2}
    equal = template_weighted_bootstrap(differences, {"sidebar": 1, "table": 1}, samples=100, seed=7)
    assert equal["mean"] == pytest.approx(0.05)
    assert equal["low"] == pytest.approx(0.05)
    assert equal["high"] == pytest.approx(0.05)
    old = template_weighted_bootstrap(differences, {"sidebar": 1, "table": 0}, samples=100, seed=7)
    assert old["mean"] == pytest.approx(0.2)


def test_weighted_bootstrap_reproducible_and_rejects_missing_strata() -> None:
    data = {"a": [-0.1, 0.2, 0.3], "b": [-0.4, 0.5]}
    first = template_weighted_bootstrap(data, {"a": 0.25, "b": 0.75}, samples=200, seed=12)
    assert first == template_weighted_bootstrap(data, {"a": 0.25, "b": 0.75}, samples=200, seed=12)
    assert first["low"] < first["mean"] < first["high"]
    with pytest.raises(ValueError, match="No pages"):
        template_weighted_bootstrap(data, {"missing": 1}, samples=10)
