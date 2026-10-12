from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from rlocr.env import OCRCorrectionEnv
from rlocr.model import OCRGATEncoder
from rlocr.ppo import GATActorCritic, ring_edges


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
