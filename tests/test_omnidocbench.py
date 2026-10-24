import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlocr.omnidocbench import align_text, image_filename, select_pages


def block(index, box, order=None):
    x0,y0,x1,y1 = box
    return {"anno_id":str(index),"category_type":"text_block","ignore":False,
            "text":f"text {index}","order":index if order is None else order,
            "poly":[x0,y0,x1,y0,x1,y1,x0,y1]}


def page(index=0, layout="double_column", language="english"):
    return {"page_info":{"width":100,"height":100,"image_path":f"{index}.png",
                         "page_attribute":{"layout":layout,"language":language,"data_source":"book"}},
            "layout_dets":[block(0,[0,0,40,20]),block(1,[0,30,40,50]),block(2,[60,0,100,20])]}


def line(index, box, text="text"):
    return {"id":str(index),"bbox":box,"text":text}


def test_selection_seed_eligibility_and_budget():
    pages = [page(i) for i in range(12)] + [page(20,language="en_ch_mixed"),page(21,layout="single_column")]
    sizes = {f"images/{i}.png":10 for i in range(22)}
    selected = select_pages(pages,sizes,seed=5,limit=8,budget=55,annotation_bytes=20)
    assert len(selected) == 3
    assert selected == select_pages(pages,sizes,seed=5,limit=8,budget=55,annotation_bytes=20)
    assert selected != select_pages(pages,sizes,seed=8,limit=8,budget=55,annotation_bytes=20)
    assert all(row["annotation_index"] < 12 for row in selected)
    assert 20+sum(row["bytes"] for row in selected) <= 55


def test_selection_requires_three_ordered_text_blocks():
    sample=page()
    sample["layout_dets"][0]["ignore"]=True
    assert not select_pages([sample],{"images/0.png":10})


@pytest.mark.parametrize("path",["../secret", "/outside.png", "C:\\outside.png"])
def test_unsafe_image_path_rejected(path):
    with pytest.raises(ValueError):
        image_filename(path)


def test_alignment_scaling_order_and_no_disjoint_text_match():
    sample=page()
    lines=[line("right",[122,2,198,38]),line("bottom",[2,62,78,98]),
           line("top",[2,2,78,38]),line("disjoint",[90,120,110,140],"text 0")]
    result=align_text(lines,sample,(200,200))
    assert result["gt_order"] == ["top","bottom","right"]
    assert result["matched_blocks"] == 3
    assert "disjoint" not in result["mapping"]


def test_cross_column_and_duplicate_order_lines_are_ambiguous():
    sample=page()
    result=align_text([line(0,[0,0,100,20])],sample,(100,100))
    assert result["ambiguous_lines"] == ["0"]
    sample["layout_dets"][2]["order"]=0
    result=align_text([line(0,[0,0,40,20])],sample,(100,100))
    assert not result["gt_order"]


@pytest.mark.parametrize("filename",["0.png", "页面.png"])
def test_downloader_resumes_without_network(tmp_path, monkeypatch, filename):
    spec=importlib.util.spec_from_file_location("prepare_omnidocbench",Path(__file__).parents[1]/"scripts"/"prepare_omnidocbench.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sample=page()
    sample["page_info"]["image_path"]=filename
    (tmp_path/"OmniDocBench.json").write_text(json.dumps([sample]),encoding="utf8")
    selected=select_pages([sample],{f"images/{filename}":3})
    manifest={"revision":module.REVISION,"seed":1337,"limit":150,"budget_mb":200,"pages":selected}
    (tmp_path/"selected_pages.json").write_text(json.dumps(manifest),encoding="utf8")
    (tmp_path/"images").mkdir()
    (tmp_path/"images"/filename).write_bytes(b"abc")
    def unexpected(*args,**kwargs):
        pytest.fail("Completed download must not access the network")
    monkeypatch.setattr(module,"hf_hub_download",unexpected)
    monkeypatch.setattr(module,"HfApi",unexpected)
    monkeypatch.setattr("sys.argv",["prepare","--root",str(tmp_path)])
    module.main()


def test_downloader_stops_before_exceeding_hard_limit(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("prepare_budget",Path(__file__).parents[1]/"scripts"/"prepare_omnidocbench.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/"OmniDocBench.json").write_text(json.dumps([page()]),encoding="utf8")
    selected=select_pages([page()],{"images/0.png":3})
    manifest={"revision":module.REVISION,"seed":1337,"limit":150,"budget_mb":200,"pages":selected}
    (tmp_path/"selected_pages.json").write_text(json.dumps(manifest),encoding="utf8")
    monkeypatch.setattr(module,"directory_bytes",lambda root:299_999_999)
    monkeypatch.setattr("sys.argv",["prepare","--root",str(tmp_path)])
    with pytest.raises(RuntimeError,match="300 MB"):
        module.main()


def test_annotation_preflight_respects_hard_limit(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("prepare_annotation_budget",Path(__file__).parents[1]/"scripts"/"prepare_omnidocbench.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata=SimpleNamespace(siblings=[SimpleNamespace(rfilename="OmniDocBench.json",size=42_000_000)])
    monkeypatch.setattr(module,"HfApi",lambda:SimpleNamespace(dataset_info=lambda *args,**kwargs:metadata))
    monkeypatch.setattr(module,"directory_bytes",lambda root:280_000_000)
    def unexpected(*args,**kwargs):
        pytest.fail("Must check annotation size before any file download")
    monkeypatch.setattr(module,"hf_hub_download",unexpected)
    monkeypatch.setattr("sys.argv",["prepare","--root",str(tmp_path)])
    with pytest.raises(RuntimeError,match="300 MB"):
        module.main()
