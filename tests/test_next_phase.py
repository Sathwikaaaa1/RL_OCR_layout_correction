from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from rlocr.env import OCRCorrectionEnv
from rlocr.model import OCRGATEncoder
from rlocr.model import PairwiseOrderModel
from rlocr.ppo import GATActorCritic, ring_edges, spatial_edges
from rlocr.evaluation import corrected_order, decode_pairwise_order, pair_f1, xy_cut_order
from rlocr.order_features import pair_feature_row
from rlocr.ocr.alignment import kendall_tau


def test_env_episode() -> None:
    page = {"page_id": "toy", "ocr_lines": [{"id": "a", "bbox": [0, 0, 20, 20], "text": "a", "confidence": 90, "native_order": 0}], "action_targets": {"a": 0}}
    env = OCRCorrectionEnv(page, max_lines=4); observation, _ = env.reset(seed=4); assert observation.shape == (4, 8)
    _, reward, terminated, _, _ = env.step(0); assert reward == 1 and terminated


def test_gat_shape() -> None:
    model = OCRGATEncoder(8, 4, 2); features = torch.rand(3, 8); edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    assert model(features, edges).shape == (3, 4)


def test_actor_critic_shape() -> None:
    model = GATActorCritic(); observations = torch.rand(2, 4, 8); edges = ring_edges(4)
    logits, values = model(observations, edges); assert logits.shape == (2, 4, 4); assert values.shape == (2, 4)


def test_spatial_edges_and_env_state_change() -> None:
    page = {"ocr_lines": [{"id": "a", "bbox": [0, 0, 20, 20], "text": "a"}], "merge_pairs": [["a", "b"]]}
    env = OCRCorrectionEnv(page, max_lines=4); observation, _ = env.reset(); assert spatial_edges(observation).shape[0] == 2; env.step(1); assert "a" in env.merged


def test_action_order_and_f1() -> None:
    lines = [{"id": "a", "bbox": [100, 0, 120, 10], "native_order": 0}, {"id": "b", "bbox": [0, 0, 20, 10], "native_order": 1}]
    assert corrected_order(lines, {"a": 3, "b": 0}) == ["a", "b"]; assert pair_f1({("a", "b")}, {("a", "b")}) == 1.0


def test_behavior_clone_step() -> None:
    from rlocr.ppo import PPOTrainer
    model = GATActorCritic(); trainer = PPOTrainer(model); observations = torch.rand(1, 4, 8); targets = torch.tensor([[0, 1, 2, 3]])
    assert trainer.behavior_clone(observations, targets, ring_edges(4), epochs=1) >= 0


def test_link_target_uses_per_line_order_mismatch() -> None:
    page = {"ocr_lines": [{"id": "a", "bbox": [0, 0, 10, 10], "native_order": 0}, {"id": "b", "bbox": [0, 20, 10, 30], "native_order": 1}], "gt_order": ["b", "a"], "tesseract_order": ["a", "b"]}
    env = OCRCorrectionEnv(page); assert env.page["action_targets"]["a"] == 3 and env.page["action_targets"]["b"] == 3


def test_link_target_precedes_merge_target() -> None:
    page = {"ocr_lines": [{"id": "a", "bbox": [0, 0, 10, 10], "native_order": 0}, {"id": "b", "bbox": [0, 20, 10, 30], "native_order": 1}], "gt_order": ["b", "a"], "tesseract_order": ["a", "b"], "merge_pairs": [["a", "b"]]}
    assert OCRCorrectionEnv(page).page["action_targets"]["a"] == 3


def test_pairwise_order_shape() -> None:
    model = PairwiseOrderModel(input_dim=8, hidden_dim=4); scores = model(torch.rand(4, 8), ring_edges(4)); assert scores.shape == (4, 4)


def test_oracle_pairwise_decoder() -> None:
    ids = ["a", "b", "c"]; pairs = {("a", "b"), ("a", "c"), ("b", "c")}; assert decode_pairwise_order(ids, pairs) == ids


def test_kendall_matches_scipy() -> None:
    from scipy.stats import kendalltau
    a = ["a", "b", "c"]; b = ["b", "a", "c"]; ours = kendall_tau(a, b); scipy_value = float(kendalltau([0, 1, 2], [1, 0, 2]).statistic); assert abs(ours - scipy_value) < 1e-9


def test_xy_cut_orders_wide_two_columns() -> None:
    lines = [
        {"id": "a0", "bbox": [0, 0, 40, 10], "native_order": 0},
        {"id": "a1", "bbox": [0, 20, 40, 30], "native_order": 1},
        {"id": "b0", "bbox": [100, 0, 140, 10], "native_order": 2},
        {"id": "b1", "bbox": [100, 20, 140, 30], "native_order": 3},
    ]
    assert xy_cut_order(lines) == ["a0", "a1", "b0", "b1"]


def test_pair_features_include_native_rank_difference_and_same_column() -> None:
    left = {"bbox": [0, 0, 40, 10], "native_order": 0, "block_num": 1}
    right = {"bbox": [0, 20, 40, 30], "native_order": 2, "block_num": 1}
    row = pair_feature_row(left, right, 4)
    assert len(row) == 13
    assert row[9] == 1.0
    assert row[10] == -0.5
    assert row[12] == 1.0
