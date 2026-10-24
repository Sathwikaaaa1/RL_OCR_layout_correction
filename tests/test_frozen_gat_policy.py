import numpy as np
import pytest
import torch

from rlocr.frozen_gat_policy import FrozenGATPolicyEnv, mask_logits
from rlocr.order_actions import RETAIN, REORDER


def toy_page():
    return {"native": [0, 1, 2, 3], "raw": [3, 0, 1, 2], "native_tau": 1.0,
            "gt_rank": np.arange(4), "node_embedding": np.arange(16, dtype=np.float32).reshape(4, 4),
            "pooled_embedding": np.ones(4, dtype=np.float32), "lookahead_normalized": np.zeros(4, dtype=np.float32),
            "inferred_columns": np.array([0, 0, 1, 2])}


def test_always_retain_reproduces_native_and_zero_reward():
    env = FrozenGATPolicyEnv(toy_page())
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert np.array_equal(observation[-3:], [1, 0, 0])
    done, total = info["automatic_terminal"], 0.0
    while not done:
        observation, reward, done, _, info = env.step(int(RETAIN))
        total += reward
    assert env.order == env.page["native"]
    assert total == 0.0 and info["terminal_reward"] == 0.0
    assert int(RETAIN) == 0 and int(REORDER) == 1


def test_placed_line_is_masked_and_state_updates():
    env = FrozenGATPolicyEnv(toy_page())
    before, _ = env.reset()
    after, _, _, _, _ = env.step(int(RETAIN))
    assert not env.available_block_mask()[0]
    assert env.mask_for_candidates([0, 1]).tolist() == [False, True]
    assert env.action_masks().all()  # both newly proposed lines are unplaced
    assert not np.array_equal(before[:4], after[:4])
    assert after[-4] == pytest.approx(0.25)
    assert np.array_equal(after[-3:], [0, 1, 0])
    logits = mask_logits(torch.tensor([[100.0, 0.0]]), torch.tensor([[False, True]]))
    assert int(logits.argmax(-1)) == int(REORDER)


def test_invalid_cross_column_reorder_is_penalized_and_executes_retain():
    env = FrozenGATPolicyEnv(toy_page())
    env.reset()
    env.step(int(RETAIN))
    _, reward, done, _, info = env.step(int(REORDER))
    assert not done and info["invalid_action"]
    assert reward == pytest.approx(-0.055)
    assert env.order == [0, 1] and env.overrides == 0
    assert env.invalid_actions == 1 and env.reorder_attempts == 1
