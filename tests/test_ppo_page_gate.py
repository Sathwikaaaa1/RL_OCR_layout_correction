import numpy as np
import pytest

from rlocr.ppo_page_gate import PageGateEnv, page_gate_features


def toy_page():
    return {"key": "toy", "native": [0, 1, 2], "gated": [1, 0, 2],
            "native_rank": np.arange(3), "gt_rank": np.array([1, 0, 2]),
            "native_tau": 1 / 3, "probabilities": np.array([[0.5, 0.1, 0.9], [0.9, 0.5, 0.9], [0.1, 0.1, 0.5]]),
            "xy_tau": 1.0, "widths": np.array([0.2, 0.5, 0.2]), "x_centers": np.array([0.1, 0.5, 0.9])}


def test_always_action_zero_reproduces_native_with_zero_reward():
    page = toy_page()
    env = PageGateEnv(page)
    observation, _ = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    _, reward, done, truncated, info = env.step(0)
    assert done and not truncated
    assert info["order"] == page["native"]
    assert reward == pytest.approx(0.0, abs=1e-15)


def test_action_one_returns_full_gated_order_and_exact_tau_gain():
    env = PageGateEnv(toy_page())
    env.reset()
    _, reward, done, _, info = env.step(1)
    assert done and info["order"] == [1, 0, 2]
    assert reward == pytest.approx(2 / 3)
    with pytest.raises(RuntimeError):
        env.step(1)


def test_features_have_no_ground_truth_dependency_and_define_edge_fraction():
    page = toy_page()
    original = page_gate_features(page, 0.65)
    assert original[-1] == pytest.approx(2 / 3)
    assert original[1] == pytest.approx(0.4)
    page["gt_rank"] = np.arange(3)
    page["native_tau"] = 1.0
    assert np.array_equal(original, page_gate_features(page, 0.65))
