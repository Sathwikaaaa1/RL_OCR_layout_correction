from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from rlocr.env import OCRCorrectionEnv
from rlocr.model import OCRGATEncoder
from rlocr.ppo import GATActorCritic, ring_edges, spatial_edges
from rlocr.evaluation import corrected_order, pair_f1


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
