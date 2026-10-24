import numpy as np
import pytest
import torch

from rlocr.ppo import PPOConfig
from rlocr.ppo_gate import OverrideNativeEnv, gae


def toy_page(raw=(2, 0, 1), truth=(1, 0, 2)):
    native = [0, 1, 2]
    ranks = np.argsort(truth)
    from rlocr.ppo_gate import index_tau
    return {"native": native, "raw": list(raw), "gt_rank": ranks,
            "native_rank": np.arange(3), "native_tau": index_tau(native, ranks),
            "scores": np.array([1., 0., 2.]), "probabilities": np.full((3, 3), 0.5),
            "xy_tau": 1., "widths": np.ones(3) * 0.3, "x_centers": np.ones(3) * 0.5}


def test_always_native_order_and_zero_terminal_improvement():
    env = OverrideNativeEnv(toy_page())
    observation, info = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    terminated = info["automatic_terminal"]
    while not terminated:
        _, _, terminated, truncated, info = env.step(0)
        assert not truncated
    assert env.order == [0, 1, 2]
    assert info["terminal_reward"] == 0.0
    assert info["overrides"] == 0
    # Dense reward is intentionally additional to the zero terminal delta.
    assert env.episode_return == pytest.approx(0.1)


def test_agreement_is_automatic_and_rewards_do_not_leak_into_observation():
    env = OverrideNativeEnv(toy_page(raw=(0, 1, 2), truth=(0, 1, 2)))
    _, info = env.reset()
    assert info["automatic_terminal"]
    assert env.order == [0, 1, 2]
    assert info["terminal_reward"] == 0
    assert info["episode_return"] == pytest.approx(0.3)
    first = OverrideNativeEnv(toy_page(truth=(0, 1, 2)))
    second = OverrideNativeEnv(toy_page(truth=(2, 1, 0)))
    assert np.array_equal(first.reset()[0], second.reset()[0])


def test_override_penalty_dense_and_terminal_reward():
    env = OverrideNativeEnv(toy_page(raw=(2, 0, 1), truth=(2, 0, 1)))
    env.reset()
    _, reward, done, _, info = env.step(1)
    assert done
    assert env.order == [2, 0, 1]
    assert reward == pytest.approx(0.3 - 0.02 + 1 - env.page["native_tau"])
    assert info["overrides"] == 1


def test_gae_bootstraps_rollout_tail_but_not_terminal_episode():
    config = PPOConfig(gamma=0.9, gae_lambda=1.0)
    rewards = torch.tensor([[1., 1.], [2., 2.]])
    dones = torch.tensor([[0., 1.], [0., 1.]])
    advantages, returns = gae(rewards, dones, torch.zeros_like(rewards), torch.tensor([10., 10.]), config)
    assert torch.allclose(returns, torch.tensor([[10.9, 1.], [11., 2.]]))
    assert torch.equal(returns, advantages)
