import importlib.util
from pathlib import Path

import pytest


def test_followup_reports_all_seeds_and_averages_without_selection():
    path=Path(__file__).parents[1]/"scripts"/"evaluate_sidebar_followup.py"
    spec=importlib.util.spec_from_file_location("sidebar_followup",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tau={"Native":1.,"XY-cut":.5,"Raw GAT":.6,"Gated GAT":.7}
    for seed,value in zip(module.SEEDS,[.5,.6,.7]):
        tau[f"PPO v2 {seed}"]=value
        tau[f"Page PPO {seed}"]=value+.2
    result=module.methods(tau)
    assert result["Sequential PPO v2"] == pytest.approx(.6)
    assert result["Page-level PPO"] == pytest.approx(.8)
    assert result["Page-level PPO / 1337"] == pytest.approx(.7)
    assert result["Page-level PPO / 90210"] == pytest.approx(.9)
    assert result["Native"] == 1
    assert len(result) == 12
