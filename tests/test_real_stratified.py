import pytest
import torch

from rlocr.model import PairwiseOrderModel
from rlocr.real_stratified import paired_summary, strata_for, switched_summary


def test_paired_bootstrap_tiny_example_and_seed_reproducibility():
    result = paired_summary([1.,-1.],[0.,0.],seed=7)
    assert result["mean_difference"] == 0
    assert result["ci95"] == [-1.,1.]
    assert (result["better"],result["worse"],result["same"]) == (1,1,0)
    assert result["worst_single_page_loss"] == 1
    assert result == paired_summary([1.,-1.],[0.,0.],seed=7)
    constant = paired_summary([.5,.5],[.25,.25])
    assert constant["ci95"] == [.25,.25]
    assert constant["worst_single_page_loss"] == 0
    assert paired_summary([],[])["ci95"] is None
    with pytest.raises(ValueError):
        paired_summary([1],[])


def test_bootstrap_is_paired_not_independent():
    result = paired_summary([.25,.5,.75],[.25,.5,.75])
    assert result["mean_difference"] == 0
    assert result["ci95"] == [0,0]
    assert result["same"] == 3


def test_strata_do_not_depend_on_learned_model_results():
    row = {"layout_type":"double_column","number_of_lines":97,"native_tau":1.,"native_xy_tau":.98,
           "scores":{"Gated GAT":0.5}}
    initial = strata_for(row)
    row["scores"]["Gated GAT"] = -1
    assert strata_for(row) == initial
    assert initial["line_count"] == ">96"
    assert initial["native_outcome"] == "native-correct"
    assert initial["native_xy_agreement"] == ">=0.98"


def test_switched_denominator_and_fixed_vs_improved():
    result = switched_summary([.5,1.,.5,1.],[.75,.5,1.,1.],[True,True,True,False])
    assert result["pages_switched"] == 3
    assert result["fraction_improved"] == pytest.approx(2/3)
    assert result["net_improved_minus_worsened"] == 1
    assert result["fully_fixed"] == 1
    assert result["native_correct_broken"] == 1
    assert result["net_pages_fixed_minus_broken"] == 0
    assert switched_summary([1],[1],[False])["fraction_same"] is None


def test_gat_accepts_more_than_96_nodes_without_architecture_change():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = PairwiseOrderModel(hidden_dim=4).eval()
        count = 101
        edges = torch.arange(count).repeat(2,1)
        with torch.no_grad():
            logits = model(torch.zeros(count,8),edges)
        assert logits.shape == (count,count)
        assert torch.isfinite(logits).all()
    finally:
        torch.set_num_threads(threads)
