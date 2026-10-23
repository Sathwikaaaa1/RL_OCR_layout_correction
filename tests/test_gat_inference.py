from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rlocr.gat_inference import gat_inference_from_logits, infer_gat


class FixedModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        return self.logits


def test_shared_gat_inference_matches_logits_path() -> None:
    lines = [
        {"id": "a", "native_order": 0},
        {"id": "b", "native_order": 1},
        {"id": "c", "native_order": 2},
    ]
    logits = torch.tensor([[0.0, -2.0, 1.0], [2.0, 0.0, -0.5], [-1.0, 0.5, 0.0]])
    direct = gat_inference_from_logits(logits, lines, native_prior=4.0)
    model_path = infer_gat(FixedModel(logits), torch.zeros((3, 1)), torch.zeros((2, 0), dtype=torch.long), lines, native_prior=4.0)
    assert model_path.probabilities == direct.probabilities
    assert model_path.order == direct.order
    assert model_path.order == ["a", "b", "c"]


@pytest.mark.skipif(
    not Path("runs/native_aware_gat_full.json").exists()
    or not Path("runs/gated_decoder/report.json").exists(),
    reason="saved full-experiment artifacts are not available",
)
def test_saved_trainer_and_gated_paths_match_on_test_split() -> None:
    trainer = json.loads(Path("runs/native_aware_gat_full.json").read_text(encoding="utf-8"))
    gated = json.loads(Path("runs/gated_decoder/report.json").read_text(encoding="utf-8"))
    trainer_tau = trainer["test"]["overall"]["model_tau"]["mean"]
    gated_tau = gated["test"]["table"]["raw_gat"]["overall_tau"]
    assert gated["native_prior"] == trainer["validation_prior"] == 4.0
    assert gated_tau == pytest.approx(trainer_tau, abs=1e-12)
