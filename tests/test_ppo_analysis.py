import pytest

from rlocr.ppo_analysis import compare_pages, holm_adjust


def test_holm_known_example_and_original_order():
    assert holm_adjust([.01,.04,.03]) == pytest.approx([.03,.06,.06])
    assert holm_adjust([1.,.01,.01]) == pytest.approx([1.,.03,.03])
    assert holm_adjust([]) == []
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_paired_comparison_zero_signal_and_reproducibility():
    result = compare_pages([.3,.8,1.],[.3,.8,1.])
    assert result["ci95"] == [0.,0.]
    assert result["sign_flip_p"] == 1.
    assert result["same"] == 3
    assert compare_pages([1.,1.],[.5,.5]) == compare_pages([1.,1.],[.5,.5])
