import numpy as np
import pytest
import torch

from rlocr.gated_decoder import gated_order
from rlocr.order_features import pair_probability_scores
from rlocr.ppo_gate import OverrideActorCritic, index_tau
from rlocr.ppo_gate_v2 import (V2_OBSERVATION_NAMES, OverrideNativeV2Env, fit_imitation,
                               gated_teacher_action, lookahead_features, reachable_tau_oracle)


def page_from_probability(probability):
    lines = [{"id": "a", "native_order": 0, "bbox": [0, 0, 30, 10]},
             {"id": "b", "native_order": 1, "bbox": [0, 20, 30, 30]}]
    probabilities = {(0, 1): probability}
    scores = pair_probability_scores(lines, probabilities, native_prior=4.0)
    gated, _, _ = gated_order(lines, probabilities, pair_threshold=0.99, page_threshold=0)
    ids = {line["id"]: i for i, line in enumerate(lines)}
    page = {"native": [0, 1], "raw": np.argsort(-scores, kind="stable").tolist(),
            "gt_rank": np.arange(2), "native_rank": np.arange(2), "native_tau": 1.0,
            "scores": scores, "probabilities": np.array([[0.5, probability], [1-probability, 0.5]]),
            "xy_tau": 1.0, "widths": np.array([0.3, 0.3]), "x_centers": np.array([0.5, 0.5]),
            "gated": [ids[key] for key in gated]}
    page["gated_rank"] = np.argsort(page["gated"])
    page["lookahead"] = lookahead_features(page, 0.99)
    return page


def test_imitation_reproduces_actual_gated_decoder_choices_on_reachable_examples():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    # Both have a different raw-GAT proposal; only one passes the teacher gate.
    pages = [page_from_probability(0.015), page_from_probability(0.00001)]
    observations, actions = [], []
    for page in pages:
        env = OverrideNativeV2Env(page)
        observation, info = env.reset()
        assert not info["automatic_terminal"]
        observations.append(observation)
        actions.append(gated_teacher_action(env))
    assert actions == [0, 1]
    model = OverrideActorCritic(len(V2_OBSERVATION_NAMES))
    fit_imitation(model, torch.from_numpy(np.asarray(observations)), torch.tensor(actions),
                  seed=7, epochs=120, learning_rate=0.01)
    for page in pages:
        env = OverrideNativeV2Env(page)
        observation, _ = env.reset()
        with torch.no_grad():
            action = int(model(torch.from_numpy(observation).unsqueeze(0))[0].argmax())
        assert action == gated_teacher_action(env)
        _, _, done, _, _ = env.step(action)
        assert done and env.order == page["gated"]


def test_v2_zero_native_reward_and_scaled_override_terminal_reward():
    page = page_from_probability(0.00001)
    env = OverrideNativeV2Env(page)
    env.reset()
    _, reward, done, _, _ = env.step(0)
    assert done and reward == 0 and env.episode_return == 0
    env.reset()
    _, reward, done, _, info = env.step(1)
    assert done and reward == pytest.approx(20 * (-1 - 1) - 0.005)
    assert info["dense_reward"] == 0
    assert info["terminal_reward"] == -40


def test_lookahead_does_not_depend_on_ground_truth():
    page = page_from_probability(0.00001)
    original = lookahead_features(page, 0.99)
    page["gt_rank"] = np.array([1, 0])
    page["native_tau"] = -1.0
    assert np.array_equal(original, lookahead_features(page, 0.99))


def test_exact_reachable_oracle_matches_exhaustive_binary_decisions():
    page = {"native": [0, 1, 2, 3], "raw": [3, 2, 1, 0], "gt_rank": np.argsort([2, 0, 3, 1])}
    def enumerate_orders(prefix):
        if len(prefix) == 4:
            return [prefix]
        candidates = {next(i for i in order if i not in prefix) for order in (page["native"], page["raw"])}
        return [order for candidate in candidates for order in enumerate_orders(prefix + [candidate])]
    exact = reachable_tau_oracle(page)
    assert exact["tau"] == max(index_tau(order, page["gt_rank"]) for order in enumerate_orders([]))
    assert exact["states"] <= 25
